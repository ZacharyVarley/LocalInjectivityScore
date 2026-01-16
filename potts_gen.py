#!/usr/bin/env python3
"""
Potts (q=3) Dataset Generator (Warp + Repeats)
=============================================

Generates q=3 Potts-model microstructures (final spin configurations) with
controls optionally passed through toy warps to induce controlled non-injectivity.

Default behavior:
  - Runs each warp in {none, fold, ribbon, pinch}
  - For each warp, runs both modes in {fixedseed, repeated}
  - Stores, per run:
      * original parameters (unwarped) : (f0, T) in physical units
      * warped controls (used in simulation): (f0, f1, T) and derived f2
      * final spin configurations "spins" as uint8, with optional repeats

Warp semantics:
  - Warps are applied in normalized coordinates on [-1,1]^2 to the original (f0, T).
  - For none/fold/pinch: the warped control remains effectively 2D; f1 is set to the
    "usual" value f1=f2=(1-f0)/2 for initialization (i.e., no independent f1 control).
  - For ribbon: the warped control is treated as genuinely 3D: (f0, f1, T), with
    f2 = 1 - f0 - f1 ensured nonnegative via a stick-breaking map.

Stochasticity control:
  - fixedseed: one run per parameter draw with a fixed RNG seed for both initialization
    and Monte Carlo updates (reproducible).
  - repeated: R independent runs per parameter draw, with per-repeat RNG seeds.

Outputs:
  outdir/
    potts__{warp}__{mode}.h5
    potts__{warp}__{mode}.json

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


WarpName = Literal["none", "fold", "ribbon", "pinch"]
SeedMode = Literal["fixedseed", "repeated"]


def _utc_now_z() -> str:
    # RFC3339-ish with trailing Z; timezone-aware (no utcnow())
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Config:
    # Lattice
    grid_size: int = 128
    q: int = 3

    # Dataset size
    n_draws: int = 1000
    n_repeats: int = 50

    # Controls (original sampling range)
    T_range: Tuple[float, float] = (0.4, 1.2)
    f0_range: Tuple[float, float] = (0.1, 0.9)

    # For ribbon (3D fractions) we enforce a minimum phase fraction
    min_phase_frac: float = 0.02

    # Dynamics
    steps: int = 200
    batch_size: int = 64
    periodic: bool = False
    remove_spurious: bool = False

    # Warps
    warp_strength: float = 1.0

    # Reproducibility
    seed_controls: int = 2468
    seed_fixed: int = 13579
    seed_repeat_base: int = 80000

    # Storage (no compression; chunking retained for streaming writes)
    chunk_sims: int = 1

    # Device
    device: str = "cuda"


# ----------------------------- Toy warps (normalized domain) -----------------------------

@torch.no_grad()
def warp_ribbon(p: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    y_prime = t * (0.75 * torch.pi) * (p[:, 1] + 1.0) + torch.pi / 4
    curl = math.sin(t * torch.pi / 2)
    x = p[:, 0]
    y = t * p[:, 1] * torch.cos(y_prime * curl) + (1 - t) * p[:, 1]
    z = p[:, 1] * torch.sin(y_prime * curl) + 0.5 * t**2
    return torch.stack((x, y, z), dim=1)


@torch.no_grad()
def warp_fold_sheet(p: torch.Tensor, t: float) -> torch.Tensor:
    x, y = p[:, 0], p[:, 1]
    mask = y > -x + 1
    x_new = x.clone()
    y_new = y.clone()
    z_new = torch.zeros_like(x)

    if torch.any(mask):
        px, py = x[mask], y[mask]
        dx, dy = px - 0.5, py - 0.5
        theta = math.pi * t
        axis = torch.tensor([1.0, -1.0, 0.0], device=p.device)
        axis = axis / torch.norm(axis)

        points_3d = torch.stack((dx, dy, torch.zeros_like(dx)), dim=1)
        k = axis
        cos_t = torch.cos(torch.tensor(theta, device=p.device))
        sin_t = torch.sin(torch.tensor(theta, device=p.device))

        rotated = (
            points_3d * cos_t
            + torch.cross(k.expand_as(points_3d), points_3d, dim=1) * sin_t
            + k * torch.sum(points_3d * k, dim=1, keepdim=True) * (1 - cos_t)
        )

        x_new[mask] = rotated[:, 0] + 0.5
        y_new[mask] = rotated[:, 1] + 0.5

    return torch.stack((x_new, y_new, z_new), dim=1)


@torch.no_grad()
def warp_pinch(p: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    x, y = p[:, 0], p[:, 1]
    return torch.stack((x, y * torch.abs(x) ** (2 * t), torch.zeros_like(x)), dim=1)


def apply_warp_2d(u2: torch.Tensor, warp: WarpName, t: float) -> torch.Tensor:
    if warp == "none":
        return u2
    if warp == "fold":
        return warp_fold_sheet(u2, t)[:, :2]
    if warp == "pinch":
        return warp_pinch(u2, t)[:, :2]
    if warp == "ribbon":
        return warp_ribbon(u2, t)  # (N,3)
    raise ValueError(warp)


# ----------------------------- Parameter sampling + warp pipeline -----------------------------

def _to_unit_interval(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (x - lo) / (hi - lo)


def _from_unit_interval(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + u * (hi - lo)


def _to_pm1(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return _to_unit_interval(x, lo, hi) * 2.0 - 1.0


def _from_pm1(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return _from_unit_interval((u + 1.0) * 0.5, lo, hi)


@torch.no_grad()
def _stick_break_fractions(u0: torch.Tensor, u1: torch.Tensor, min_frac: float) -> torch.Tensor:
    """
    Map (u0,u1) in [-1,1]^2 to (f0,f1,f2) with all >= min_frac and sum=1.
    """
    # Sharper sigmoid for better spread from normalized coords
    v0 = torch.sigmoid(2.0 * u0)
    v1 = torch.sigmoid(2.0 * u1)

    # Reserve min fractions for all 3 phases
    min_total = 3.0 * float(min_frac)
    if min_total >= 1.0:
        raise ValueError("min_phase_frac too large")

    f0 = float(min_frac) + (1.0 - min_total) * v0
    rem = 1.0 - f0 - 2.0 * float(min_frac)
    f1 = float(min_frac) + rem * v1
    f2 = 1.0 - f0 - f1
    return torch.stack((f0, f1, f2), dim=1)


@torch.no_grad()
def sample_controls_potts(cfg: Config, warp: WarpName, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      orig:  (N,2) -> (f0,T) in physical units
      warped_controls: (N,4) -> (f0,f1,f2,T) used for simulation
      warped_pm1: tensor of the warp output in normalized coords (for debugging / record)
    """
    N = cfg.n_draws
    f0_lo, f0_hi = cfg.f0_range
    T_lo, T_hi = cfg.T_range

    g = torch.Generator(device=device).manual_seed(int(cfg.seed_controls))
    f0 = f0_lo + (f0_hi - f0_lo) * torch.rand((N,), device=device, generator=g)
    T = T_lo + (T_hi - T_lo) * torch.rand((N,), device=device, generator=g)
    orig = torch.stack((f0, T), dim=1)

    u_f0 = _to_pm1(f0, f0_lo, f0_hi)
    u_T = _to_pm1(T, T_lo, T_hi)
    u2 = torch.stack((u_f0, u_T), dim=1)

    t = float(cfg.warp_strength)

    if warp == "ribbon":
        u3 = apply_warp_2d(u2, warp="ribbon", t=t)  # (N,3) in [-1,1] (clipped later)
        u3 = u3.clamp(-1.0, 1.0)
        u0, u1, uT = u3[:, 0], u3[:, 1], u3[:, 2]

        fracs = _stick_break_fractions(u0, u1, min_frac=cfg.min_phase_frac)  # (N,3)
        T_w = _from_pm1(uT, T_lo, T_hi)
        warped_controls = torch.cat((fracs, T_w[:, None]), dim=1)  # (N,4)
        return orig, warped_controls, u3

    # 2D warps: compute warped (f0,T); set f1=f2=(1-f0)/2
    u2_w = apply_warp_2d(u2, warp=warp, t=t).clamp(-1.0, 1.0)
    f0_w = _from_pm1(u2_w[:, 0], f0_lo, f0_hi)
    T_w = _from_pm1(u2_w[:, 1], T_lo, T_hi)
    f1_w = 0.5 * (1.0 - f0_w)
    f2_w = 0.5 * (1.0 - f0_w)
    warped_controls = torch.stack((f0_w, f1_w, f2_w, T_w), dim=1)
    return orig, warped_controls, u2_w


# ----------------------------- Potts simulation core -----------------------------

@torch.no_grad()
def create_initial_states(
    fractions: torch.Tensor,  # (B,3) sums to 1
    grid_size: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """
    Vectorized initial condition: sample a uniform random field and assign states
    by cumulative thresholds from fractions.
    Returns spins as int8 with shape (B,1,H,W).
    """
    B = fractions.shape[0]
    u = torch.rand((B, 1, grid_size, grid_size), device=device, generator=gen)
    c0 = fractions[:, 0].view(B, 1, 1, 1)
    c1 = (fractions[:, 0] + fractions[:, 1]).view(B, 1, 1, 1)

    spins = torch.empty((B, 1, grid_size, grid_size), device=device, dtype=torch.int8)
    spins[u < c0] = 0
    spins[(u >= c0) & (u < c1)] = 1
    spins[u >= c1] = 2
    return spins


@torch.no_grad()
def potts_step(
    spins: torch.Tensor,
    beta: torch.Tensor,   # (B,)
    q: int,
    gen: torch.Generator,
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

    new_spins = torch.randint(0, q, spins.shape, device=spins.device, generator=gen, dtype=torch.int8)

    new_aligned = (
        (new_spins == up).byte()
        + (new_spins == down).byte()
        + (new_spins == left).byte()
        + (new_spins == right).byte()
    ).float()

    delta = new_aligned - aligned
    beta_expanded = beta.view(-1, 1, 1, 1)
    acceptance_probs = torch.exp(beta_expanded * delta)

    u = torch.rand(acceptance_probs.shape, device=spins.device, generator=gen)

    if remove_spurious:
        flips = (u < acceptance_probs) & (new_aligned > 0)
    else:
        flips = (u < acceptance_probs)

    spins = torch.where(flips, new_spins, spins)
    return spins


@torch.no_grad()
def simulate_potts(
    spins: torch.Tensor,
    temperatures: torch.Tensor,  # (B,)
    steps: int,
    q: int,
    gen: torch.Generator,
    periodic: bool,
    remove_spurious: bool,
) -> torch.Tensor:
    beta = 1.0 / temperatures
    for _ in range(steps):
        spins = potts_step(spins, beta, q=q, gen=gen, periodic=periodic, remove_spurious=remove_spurious)
    return spins


# ----------------------------- HDF5 IO -----------------------------

def _h5_create_dataset(h5: h5py.File, name: str, shape: tuple, dtype, cfg: Config):
    # No compression by design (publication-ready reproducibility + predictable I/O)
    kwargs = {}
    if len(shape) == 3:
        kwargs["chunks"] = (min(cfg.chunk_sims, shape[0]), shape[1], shape[2])
    elif len(shape) == 4:
        kwargs["chunks"] = (min(cfg.chunk_sims, shape[0]), 1, shape[2], shape[3])
    return h5.create_dataset(name, shape=shape, dtype=dtype, **kwargs)


@torch.no_grad()
def simulate_and_write(
    cfg: Config,
    device: torch.device,
    warped_controls: torch.Tensor,  # (N,4) (f0,f1,f2,T)
    h5_spins: h5py.Dataset,
    mode: SeedMode,
    progress_desc: str,
):
    N = int(warped_controls.shape[0])
    H = W = int(cfg.grid_size)

    fracs = warped_controls[:, :3].to(device=device, dtype=torch.float32, non_blocking=True)
    temps = warped_controls[:, 3].to(device=device, dtype=torch.float32, non_blocking=True)

    n_runs = cfg.n_repeats if mode == "repeated" else 1
    n_batches = (N + int(cfg.batch_size) - 1) // int(cfg.batch_size)
    total_batches = n_runs * n_batches

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_batches, desc=progress_desc, unit="batch", dynamic_ncols=True)

    processed = 0
    for r in range(n_runs):
        seed_r = (cfg.seed_repeat_base + r) if mode == "repeated" else cfg.seed_fixed
        gen = torch.Generator(device=device).manual_seed(int(seed_r))

        for start in range(0, N, cfg.batch_size):
            end = min(N, start + cfg.batch_size)
            B = end - start

            spins0 = create_initial_states(fracs[start:end], grid_size=H, gen=gen, device=device)
            spinsF = simulate_potts(
                spins0,
                temperatures=temps[start:end],
                steps=int(cfg.steps),
                q=int(cfg.q),
                gen=gen,
                periodic=bool(cfg.periodic),
                remove_spurious=bool(cfg.remove_spurious),
            )

            # store as (H,W) per sample; strip channel dim
            out = spinsF[:, 0].detach().to("cpu", dtype=torch.uint8).numpy()

            if mode == "fixedseed":
                h5_spins[start:end, :, :] = out
            else:
                h5_spins[start:end, r, :, :] = out

            processed += 1
            if pbar is not None:
                pbar.update(1)
            elif processed % 10 == 0 or processed == total_batches:
                print(f"[{progress_desc}] {processed}/{total_batches} batches")

    if pbar is not None:
        pbar.close()


def _write_run(cfg: Config, outdir: Path, warp: WarpName, mode: SeedMode):
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    orig, warped_controls, warped_pm1 = sample_controls_potts(cfg, warp=warp, device=device)

    tag = f"potts__{warp}__{mode}"
    h5_path = outdir / f"{tag}.h5"
    js_path = outdir / f"{tag}.json"

    N = cfg.n_draws
    R = cfg.n_repeats if mode == "repeated" else 1
    H = W = cfg.grid_size

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["created_utc"] = _utc_now_z()
        h5.attrs["warp"] = warp
        h5.attrs["mode"] = mode
        h5.attrs["q"] = int(cfg.q)

        h5.create_dataset("params_original", data=orig.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("controls_warped", data=warped_controls.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("controls_warped_pm1", data=warped_pm1.detach().cpu().numpy().astype(np.float32))

        if mode == "fixedseed":
            spins = _h5_create_dataset(h5, "spins", shape=(N, H, W), dtype=np.uint8, cfg=cfg)
        else:
            spins = _h5_create_dataset(h5, "spins", shape=(N, R, H, W), dtype=np.uint8, cfg=cfg)

        simulate_and_write(
            cfg,
            device=device,
            warped_controls=warped_controls.cpu(),
            h5_spins=spins,
            mode=mode,
            progress_desc=f"Potts {warp} {mode}",
        )

    meta = dict(
        created_utc=_utc_now_z(),
        script=str(Path(__file__).name),
        warp=warp,
        mode=mode,
        config=asdict(cfg),
        outputs=dict(h5=str(h5_path)),
    )
    js_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {h5_path}")


# ----------------------------- CLI -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="potts_data")
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--n-draws", type=int, default=Config.n_draws)
    p.add_argument("--n-repeats", type=int, default=Config.n_repeats)
    p.add_argument("--grid-size", type=int, default=Config.grid_size)
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--warp-strength", type=float, default=Config.warp_strength)

    p.add_argument("--T-lo", type=float, default=Config.T_range[0])
    p.add_argument("--T-hi", type=float, default=Config.T_range[1])
    p.add_argument("--f0-lo", type=float, default=Config.f0_range[0])
    p.add_argument("--f0-hi", type=float, default=Config.f0_range[1])
    p.add_argument("--min-phase-frac", type=float, default=Config.min_phase_frac)

    p.add_argument("--periodic", action="store_true")
    p.add_argument("--remove-spurious", action="store_true")

    p.add_argument("--only-warp", type=str, default=None, choices=[None, "none", "fold", "ribbon", "pinch"])
    p.add_argument("--only-mode", type=str, default=None, choices=[None, "fixedseed", "repeated"])
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = Config(
        device=args.device,
        n_draws=int(args.n_draws),
        n_repeats=int(args.n_repeats),
        grid_size=int(args.grid_size),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        warp_strength=float(args.warp_strength),
        T_range=(float(args.T_lo), float(args.T_hi)),
        f0_range=(float(args.f0_lo), float(args.f0_hi)),
        min_phase_frac=float(args.min_phase_frac),
        periodic=bool(args.periodic),
        remove_spurious=bool(args.remove_spurious),
    )

    outdir = Path(args.outdir)
    warps: list[WarpName] = ["none", "fold", "ribbon", "pinch"]
    modes: list[SeedMode] = ["fixedseed", "repeated"]

    if args.only_warp is not None:
        warps = [args.only_warp]  # type: ignore[assignment]
    if args.only_mode is not None:
        modes = [args.only_mode]  # type: ignore[assignment]

    for w in warps:
        for m in modes:
            _write_run(cfg, outdir=outdir, warp=w, mode=m)


if __name__ == "__main__":
    main()
