#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_gen.py

Potts simulator that saves RAW final spin configurations (no Euler, no correlations).
Analysis-side script computes correlations identically to the older generator.

Outputs (new IO style):
  potts_data/<YYYYMMDD_HHMMSSZ>/potts_sims_q{q}_{H}x{W}.h5
  potts_data/<YYYYMMDD_HHMMSSZ>/potts_sims_q{q}_{H}x{W}.json

H5 layout:
  parameters/temperature            (N,)
  parameters/fraction_initial       (N,)   # fraction of phase 0 in initialization
  states/final_spins                (N, R, H, W) int8
  attrs include config + created_utc

Notes:
  - Initial conditions match the older logic: phase 0 occupies `fraction_initial`,
    remaining sites distributed across phases 1..q-1 via scaled uniform.
  - Simulation uses the same batched Metropolis-like update as before.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _utc_now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_folder_name_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")


@dataclass(frozen=True)
class Config:
    q: int = 3
    grid_size: int = 128

    n_param_draws: int = 1000
    n_repeats: int = 100
    nsteps: int = 100

    # Simulation chunking: each chunk simulates param_batch * repeat_chunk samples
    param_batch: int = 16
    sim_batch_size: int = 512  # max samples per chunk; repeat_chunk derived from this and param_batch

    temp_range: Tuple[float, float] = (0.4, 1.2)
    fraction_range: Tuple[float, float] = (0.1, 0.9)
    seed_params: int = 42

    periodic: bool = True
    remove_spurious: bool = False

    out_root: str = "potts_data"
    tag: str = "potts_sims"

    device: str = "cuda"


@torch.no_grad()
def potts_step(
    spins: torch.Tensor,
    beta: torch.Tensor,  # (B,)
    q: int,
    periodic: bool = False,
    remove_spurious: bool = False,
) -> torch.Tensor:
    if periodic:
        up = torch.roll(spins, 1, dims=-2)
        down = torch.roll(spins, -1, dims=-2)
        left = torch.roll(spins, 1, dims=-1)
        right = torch.roll(spins, -1, dims=-1)
    else:
        spins_padded = F.pad(spins, (1, 1, 1, 1), mode="replicate")
        up = spins_padded[:, :, :-2, 1:-1]
        down = spins_padded[:, :, 2:, 1:-1]
        left = spins_padded[:, :, 1:-1, :-2]
        right = spins_padded[:, :, 1:-1, 2:]

    aligned = (
        (spins == up).byte()
        + (spins == down).byte()
        + (spins == left).byte()
        + (spins == right).byte()
    ).float()

    new_spins = torch.randint_like(spins, q, dtype=torch.int8)

    new_aligned = (
        (new_spins == up).byte()
        + (new_spins == down).byte()
        + (new_spins == left).byte()
        + (new_spins == right).byte()
    ).float()

    delta = new_aligned - aligned
    beta_expanded = beta.view(-1, 1, 1, 1)
    acceptance_probs = torch.exp(beta_expanded * delta)

    if remove_spurious:
        flips = (torch.rand_like(acceptance_probs) < acceptance_probs) & (new_aligned > 0)
    else:
        flips = torch.rand_like(acceptance_probs) < acceptance_probs

    return torch.where(flips, new_spins, spins)


@torch.no_grad()
def simulate_potts(
    spins: torch.Tensor, temperatures: torch.Tensor, steps: int, q: int, periodic: bool, remove_spurious: bool
) -> torch.Tensor:
    beta = 1.0 / temperatures
    for _ in range(int(steps)):
        spins = potts_step(spins, beta, q=q, periodic=periodic, remove_spurious=remove_spurious)
    return spins


@torch.no_grad()
def create_initial_states(
    batch_size: int,
    grid_size: int,
    fractions: torch.Tensor,  # (B,) fraction of phase 0
    q: int,
    seeds: torch.Tensor,      # (B,)
    device: torch.device,
) -> torch.Tensor:
    states = torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.int8, device=device)
    for i, (fraction, seed) in enumerate(zip(fractions, seeds)):
        rand_grid = torch.rand(grid_size, grid_size, device=device)
        mask_0 = rand_grid < fraction
        states[i, 0][mask_0] = 0

        remaining_mask = ~mask_0
        if int(remaining_mask.sum().item()) > 0:
            remaining_rand = (rand_grid[remaining_mask] - fraction) / (1.0 - fraction)
            other_states = 1 + (remaining_rand * float(q - 1)).to(torch.int8)
            states[i, 0][remaining_mask] = other_states
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default=Config.out_root)
    ap.add_argument("--tag", type=str, default=Config.tag)

    ap.add_argument("--q", type=int, default=Config.q)
    ap.add_argument("--grid_size", type=int, default=Config.grid_size)

    ap.add_argument("--n_param_draws", type=int, default=Config.n_param_draws)
    ap.add_argument("--n_repeats", type=int, default=Config.n_repeats)
    ap.add_argument("--nsteps", type=int, default=Config.nsteps)

    ap.add_argument("--param_batch", type=int, default=Config.param_batch)
    ap.add_argument("--sim_batch_size", type=int, default=Config.sim_batch_size)

    ap.add_argument("--T_lo", type=float, default=Config.temp_range[0])
    ap.add_argument("--T_hi", type=float, default=Config.temp_range[1])
    ap.add_argument("--f_lo", type=float, default=Config.fraction_range[0])
    ap.add_argument("--f_hi", type=float, default=Config.fraction_range[1])

    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--remove_spurious", action="store_true")

    ap.add_argument("--seed_params", type=int, default=Config.seed_params)
    ap.add_argument("--device", type=str, default=Config.device)

    args = ap.parse_args()

    cfg = Config(
        q=int(args.q),
        grid_size=int(args.grid_size),
        n_param_draws=int(args.n_param_draws),
        n_repeats=int(args.n_repeats),
        nsteps=int(args.nsteps),
        param_batch=int(args.param_batch),
        sim_batch_size=int(args.sim_batch_size),
        temp_range=(float(args.T_lo), float(args.T_hi)),
        fraction_range=(float(args.f_lo), float(args.f_hi)),
        periodic=bool(args.periodic),
        remove_spurious=bool(args.remove_spurious),
        seed_params=int(args.seed_params),
        out_root=str(args.out_root),
        tag=str(args.tag),
        device=str(args.device),
    )

    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")
    print(f"[potts_gen] device={device}")

    torch.manual_seed(int(cfg.seed_params))

    N = int(cfg.n_param_draws)
    R = int(cfg.n_repeats)
    H = W = int(cfg.grid_size)
    q = int(cfg.q)

    temperatures = torch.empty((N,), device="cpu").uniform_(float(cfg.temp_range[0]), float(cfg.temp_range[1]))
    fractions_param = torch.empty((N,), device="cpu").uniform_(float(cfg.fraction_range[0]), float(cfg.fraction_range[1]))

    out_root = Path(cfg.out_root)
    run_dir = out_root / _run_folder_name_utc()
    run_dir.mkdir(parents=True, exist_ok=True)

    base = f"{cfg.tag}_q{q}_{H}x{W}"
    h5_path = run_dir / f"{base}.h5"
    js_path = run_dir / f"{base}.json"

    # Chunking: per parameter-batch, simulate repeat_chunk repeats at a time
    P = max(1, int(cfg.param_batch))
    maxB = max(1, int(cfg.sim_batch_size))
    repeat_chunk = max(1, maxB // P)

    created_utc = _utc_now_z()

    with h5py.File(h5_path, "w") as f:
        f.attrs["created_utc"] = created_utc
        f.attrs["config_json"] = json.dumps(asdict(cfg))
        f.attrs["n_parameters"] = N
        f.attrs["n_repeats"] = R
        f.attrs["grid_size"] = H
        f.attrs["q"] = q
        f.attrs["layout"] = "states/final_spins is (N, R, H, W) int8"

        grp_p = f.create_group("parameters")
        grp_p.create_dataset("temperature", data=temperatures.numpy().astype(np.float32))
        grp_p.create_dataset("fraction_initial", data=fractions_param.numpy().astype(np.float32))

        grp_s = f.create_group("states")
        dset = grp_s.create_dataset(
            "final_spins",
            shape=(N, R, H, W),
            dtype=np.int8,
            chunks=(1, min(R, max(1, repeat_chunk)), H, W),
            compression="gzip",
        )

        outer = range(0, N, P)
        if tqdm:
            outer = tqdm(outer, desc="potts_gen param-batches", total=(N + P - 1) // P)

        for pb in outer:
            pe = min(N, pb + P)
            Pb = pe - pb

            t_pb = temperatures[pb:pe].to(torch.float32)  # CPU
            f_pb = fractions_param[pb:pe].to(torch.float32)

            # Repeat chunks
            for rb in range(0, R, repeat_chunk):
                re = min(R, rb + repeat_chunk)
                Rc = re - rb
                B = Pb * Rc

                temps = t_pb.repeat_interleave(Rc).to(device=device)
                fracs = f_pb.repeat_interleave(Rc).to(device=device)

                # deterministic per-sample init seeds (for the init only)
                # seed space avoids overlap across (pb,rb)
                base_seed = (pb * R + rb) + 1000
                seeds = torch.arange(base_seed, base_seed + B, dtype=torch.int64, device=device)

                states0 = create_initial_states(
                    batch_size=B,
                    grid_size=H,
                    fractions=fracs,
                    q=q,
                    seeds=seeds,
                    device=device,
                )
                final_states = simulate_potts(
                    states0,
                    temperatures=temps,
                    steps=int(cfg.nsteps),
                    q=q,
                    periodic=bool(cfg.periodic),
                    remove_spurious=bool(cfg.remove_spurious),
                )
                final_cpu = final_states[:, 0].detach().cpu().numpy().astype(np.int8, copy=False)  # (B,H,W)

                # reshape back to (Pb,Rc,H,W) aligned with (pb:pe, rb:re)
                final_cpu = final_cpu.reshape(Pb, Rc, H, W)
                dset[pb:pe, rb:re, :, :] = final_cpu

                del states0, final_states
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    meta = {
        "created_utc": created_utc,
        "run_dir": str(run_dir),
        "h5": str(h5_path),
        "config": asdict(cfg),
        "description": {
            "parameters/temperature": "(N,) sampled temperatures",
            "parameters/fraction_initial": "(N,) sampled init fraction for phase 0",
            "states/final_spins": "(N,R,H,W) int8 raw final spins for each parameter draw and repeat",
        },
    }
    js_path.write_text(json.dumps(meta, indent=2))

    print(f"[potts_gen] wrote: {h5_path}")
    print(f"[potts_gen] wrote: {js_path}")


if __name__ == "__main__":
    main()
