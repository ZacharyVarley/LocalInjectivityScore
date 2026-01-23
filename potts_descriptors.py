#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_descriptors.py

Computes descriptors from Potts simulation data:
  1) Reads RAW spins from potts_gen output
  2) Computes correlations_2d and correlations_radial with whitening
  3) Computes phase fractions
  4) Saves descriptors to H5 file

Outputs:
  <input_dir>/<descriptor_kind>/
    <input_stem>_<descriptor_kind>.h5
    <input_stem>_<descriptor_kind>.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import h5py
import numpy as np
import torch


DescKind = Literal["radial1d", "corr2d"]


# ----------------------------- time / run discovery -----------------------------

def _utc_now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_run_dir_name(name: str) -> float | None:
    try:
        dt = _dt.datetime.strptime(name, "%Y%m%d_%H%M%SZ")
        return dt.replace(tzinfo=_dt.timezone.utc).timestamp()
    except Exception:
        return None


def find_latest_h5_under(data_root: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    if data_root.exists():
        for p in data_root.iterdir():
            if p.is_dir():
                ts = _parse_run_dir_name(p.name)
                if ts is None:
                    ts = p.stat().st_mtime
                for h in sorted(p.glob("*.h5")):
                    candidates.append((ts, h))
            elif p.is_file() and p.suffix.lower() == ".h5":
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(f"No .h5 found under {data_root}")
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class Config:
    potts_data_dir: str = "potts_data"

    descriptor: DescKind = "corr2d"
    n_radial_bins: int = 64
    
    # Whitening regularization for correlations
    whiten_eps: float = 0.0

    # Batching for memory management
    batch_size: int = 128

    device: str = "cuda"


def _pair_labels(q: int) -> List[str]:
    return [f"{i}-{j}" for i in range(q) for j in range(i, q)]


# ----------------------------- Potts correlation ops -----------------------------

@torch.no_grad()
def compute_correlations_2d(spins: torch.Tensor, q: int, whiten_eps: float = 0.0) -> torch.Tensor:
    """
    spins: (B,1,H,W) int8
    returns: (B, n_pairs, H, W) float32 with whitening applied
    """
    B, _, H, W = spins.shape
    n_pairs = q * (q + 1) // 2
    out = torch.zeros((B, n_pairs, H, W), device=spins.device, dtype=torch.float32)

    pair_idx = 0
    for i in range(q):
        for j in range(i, q):
            ind_i = (spins[:, 0] == i).float()
            ind_j = (spins[:, 0] == j).float()

            ind_i -= ind_i.mean(dim=(-2, -1), keepdim=True)
            ind_j -= ind_j.mean(dim=(-2, -1), keepdim=True)

            fft_i = torch.fft.rfft2(ind_i)
            fft_j = torch.fft.rfft2(ind_j)
            cross_power = fft_i * torch.conj(fft_j)
            corr = torch.fft.irfft2(cross_power, s=(H, W)) / float(H * W)

            out[:, pair_idx] = torch.fft.fftshift(corr, dim=(-2, -1))
            pair_idx += 1
    
    # Apply whitening: add small constant
    out = out + whiten_eps

    return out


@torch.no_grad()
def compute_radial_average(correlations_2d: torch.Tensor, n_bins: int, whiten_eps: float = 0.0) -> torch.Tensor:
    """
    correlations_2d: (B,n_pairs,H,W)
    returns: (B,n_pairs,n_bins) with whitening applied
    """
    B, n_pairs, H, W = correlations_2d.shape
    center = H // 2
    max_r = H // 2

    y, x = torch.meshgrid(
        torch.arange(H, device=correlations_2d.device, dtype=torch.float32),
        torch.arange(W, device=correlations_2d.device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - float(center)) ** 2 + (y - float(center)) ** 2)

    bin_edges = torch.linspace(0, float(max_r), int(n_bins) + 1, device=correlations_2d.device)
    bin_indices = torch.searchsorted(bin_edges[:-1], r, right=False)
    bin_indices = torch.clamp(bin_indices, 0, int(n_bins) - 1)

    flat_indices = bin_indices.flatten()
    flat_corr = correlations_2d.view(B * n_pairs, H * W)

    sums = torch.zeros((B * n_pairs, int(n_bins)), device=correlations_2d.device, dtype=torch.float32)
    counts = torch.zeros_like(sums)

    expanded = flat_indices.unsqueeze(0).expand(B * n_pairs, -1)
    sums.scatter_add_(1, expanded, flat_corr)
    counts.scatter_add_(1, expanded, torch.ones_like(flat_corr))

    radial = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
    radial = radial.view(B, n_pairs, int(n_bins))
    
    # Apply whitening: add small constant
    radial = radial + whiten_eps
    
    return radial


@torch.no_grad()
def compute_phase_fractions(final_states: torch.Tensor, q: int) -> torch.Tensor:
    """final_states: (B,1,H,W) int8 -> (B,q) float32 fractions."""
    spins = final_states[:, 0].long()
    one_hot = torch.nn.functional.one_hot(spins, num_classes=q).float()
    return one_hot.view(one_hot.shape[0], -1, q).mean(dim=1)


# ----------------------------- descriptor computation -----------------------------

def compute_descriptors(
    in_h5: Path,
    out_h5: Path,
    out_json: Path,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Computes descriptors from spins and saves to H5.
    Returns metadata dictionary.
    """
    ensure_dir(out_h5.parent)

    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    created_utc = _utc_now_z()

    print("[potts_descriptors] Loading data from H5...")
    with h5py.File(str(in_h5), "r") as fin:
        temps = np.array(fin["parameters/temperature"], dtype=np.float32)
        fracs_init = np.array(fin["parameters/fraction_initial"], dtype=np.float32)
        spins_data = np.array(fin["states/final_spins"], dtype=np.int8)  # (N,R,H,W)

        N, R, H, W = spins_data.shape
        q = int(fin.attrs.get("q", 3))
        
    n_pairs = q * (q + 1) // 2
    pair_labels = _pair_labels(q)

    # Radial bin centers
    max_r = H // 2
    edges = torch.linspace(0, float(max_r), int(cfg.n_radial_bins) + 1)
    radial_bins = ((edges[:-1] + edges[1:]) * 0.5).cpu().numpy().astype(np.float32)

    print(f"[potts_descriptors] Computing descriptors for {N} parameters, {R} repeats...")
    print(f"[potts_descriptors] Descriptor: {cfg.descriptor}, grid: {H}x{W}, q: {q}")
    
    # Allocate output arrays
    mean2d = np.zeros((N, n_pairs, H, W), dtype=np.float32)
    mean1d = np.zeros((N, n_pairs, int(cfg.n_radial_bins)), dtype=np.float32)
    std1d = np.zeros((N, n_pairs, int(cfg.n_radial_bins)), dtype=np.float32)
    meanph = np.zeros((N, q), dtype=np.float32)
    stdph = np.zeros((N, q), dtype=np.float32)
    
    # Batch over N to reduce memory usage
    batch_size = min(int(cfg.batch_size), N)
    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        batch_n = i_end - i
        
        if i % (batch_size * 4) == 0 or i == 0:
            print(f"[potts_descriptors] Processing parameters {i}/{N}...")
        
        # Accumulate statistics over R dimension in batches to avoid VRAM overflow
        max_elements = 50 * 1024 * 1024  # ~200MB / 4 bytes
        repeat_batch = max(1, min(R, max_elements // (batch_n * H * W)))
        
        sum2d = np.zeros((batch_n, n_pairs, H, W), dtype=np.float64)
        sum1d = np.zeros((batch_n, n_pairs, int(cfg.n_radial_bins)), dtype=np.float64)
        sumsq1d = np.zeros((batch_n, n_pairs, int(cfg.n_radial_bins)), dtype=np.float64)
        sumph = np.zeros((batch_n, q), dtype=np.float64)
        sumsqph = np.zeros((batch_n, q), dtype=np.float64)
        
        for r in range(0, R, repeat_batch):
            r_end = min(r + repeat_batch, R)
            r_batch = r_end - r
            
            # Load only this repeat batch to device
            spins_batch = torch.as_tensor(spins_data[i:i_end, r:r_end], device=device)
            spins_batch = spins_batch.reshape(batch_n * r_batch, 1, H, W)
            
            # Compute correlations
            corr2d_batch = compute_correlations_2d(spins_batch, q=q, whiten_eps=cfg.whiten_eps)
            rad_batch = compute_radial_average(corr2d_batch, n_bins=int(cfg.n_radial_bins), whiten_eps=cfg.whiten_eps)
            ph_batch = compute_phase_fractions(spins_batch, q=q)
            
            # Reshape to separate batch_n and r_batch
            corr2d_batch = corr2d_batch.reshape(batch_n, r_batch, n_pairs, H, W)
            rad_batch = rad_batch.reshape(batch_n, r_batch, n_pairs, int(cfg.n_radial_bins))
            ph_batch = ph_batch.reshape(batch_n, r_batch, q)
            
            # Accumulate statistics
            sum2d += corr2d_batch.sum(dim=1).cpu().numpy()
            sum1d += rad_batch.sum(dim=1).cpu().numpy()
            sumsq1d += (rad_batch ** 2).sum(dim=1).cpu().numpy()
            sumph += ph_batch.sum(dim=1).cpu().numpy()
            sumsqph += (ph_batch ** 2).sum(dim=1).cpu().numpy()
            
            # Clean up GPU memory
            del spins_batch, corr2d_batch, rad_batch, ph_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Compute statistics from accumulated sums
        mean2d[i:i_end] = (sum2d / R).astype(np.float32)
        mean1d[i:i_end] = (sum1d / R).astype(np.float32)
        var1d = (sumsq1d / R) - (sum1d / R) ** 2
        std1d[i:i_end] = np.sqrt(np.maximum(var1d, 0.0)).astype(np.float32)
        meanph[i:i_end] = (sumph / R).astype(np.float32)
        varph = (sumsqph / R) - (sumph / R) ** 2
        stdph[i:i_end] = np.sqrt(np.maximum(varph, 0.0)).astype(np.float32)
    
    print("[potts_descriptors] Computing statistics... done")

    # Write descriptors H5
    print(f"[potts_descriptors] Writing descriptors to {out_h5}...")
    with h5py.File(str(out_h5), "w") as fout:
        fout.attrs["created_utc"] = created_utc
        fout.attrs["input_h5"] = str(in_h5)
        fout.attrs["descriptor"] = cfg.descriptor
        fout.attrs["config_json"] = json.dumps(asdict(cfg))
        fout.attrs["n_parameters"] = N
        fout.attrs["n_repeats"] = R
        fout.attrs["grid_size"] = H
        fout.attrs["q"] = q
        fout.attrs["n_correlation_pairs"] = n_pairs
        fout.attrs["whiten_eps"] = float(cfg.whiten_eps)

        gp = fout.create_group("parameters")
        gp.create_dataset("temperature", data=temps)
        gp.create_dataset("fraction_initial", data=fracs_init)

        gm = fout.create_group("metadata")
        gm.create_dataset("radial_bins", data=radial_bins)
        gm.create_dataset("pair_labels", data=np.array([s.encode() for s in pair_labels], dtype="S"))
        gm.create_dataset("phase_labels", data=np.arange(q, dtype=np.int32))

        gc = fout.create_group("correlations")
        gc.create_dataset("correlations_2d_mean", data=mean2d)
        gc.create_dataset("correlations_radial_mean", data=mean1d)
        gc.create_dataset("correlations_radial_std", data=std1d)

        gpz = fout.create_group("phases")
        gpz.create_dataset("final_fraction_mean", data=meanph)
        gpz.create_dataset("final_fraction_std", data=stdph)

    meta = dict(
        created_utc=created_utc,
        input_h5=str(in_h5),
        output_h5=str(out_h5),
        descriptor=str(cfg.descriptor),
        config=asdict(cfg),
        shapes=dict(
            correlations_2d_mean=[int(N), int(n_pairs), int(H), int(W)],
            correlations_radial_mean=[int(N), int(n_pairs), int(cfg.n_radial_bins)],
            correlations_radial_std=[int(N), int(n_pairs), int(cfg.n_radial_bins)],
            final_fraction_mean=[int(N), int(q)],
            final_fraction_std=[int(N), int(q)],
        ),
        pair_labels=pair_labels,
        radial_bins=radial_bins.tolist(),
    )
    out_json.write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Potts descriptors from simulation data")

    ap.add_argument("--h5", type=str, default="", 
                    help="Input HDF5 from potts_gen. If empty, uses latest under --potts_data_dir.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)

    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"])
    ap.add_argument("--n_radial_bins", type=int, default=Config.n_radial_bins)
    ap.add_argument("--whiten_eps", type=float, default=Config.whiten_eps)

    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--device", type=str, default=Config.device)

    args = ap.parse_args()

    cfg = Config(
        potts_data_dir=str(args.potts_data_dir),
        descriptor=str(args.descriptor),  # type: ignore
        n_radial_bins=int(args.n_radial_bins),
        whiten_eps=float(args.whiten_eps),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    data_root = Path(cfg.potts_data_dir)
    if str(args.h5).strip():
        in_h5 = Path(str(args.h5)).expanduser().resolve()
    else:
        in_h5 = find_latest_h5_under(data_root)

    # Output directory: <input_dir>/<descriptor>/
    descriptor_dir = ensure_dir(in_h5.parent / cfg.descriptor)
    
    out_h5 = descriptor_dir / f"{in_h5.stem}_{cfg.descriptor}.h5"
    out_json = descriptor_dir / f"{in_h5.stem}_{cfg.descriptor}.json"

    print(f"[potts_descriptors] Input: {in_h5}")
    print(f"[potts_descriptors] Output: {out_h5}")
    
    meta = compute_descriptors(
        in_h5=in_h5,
        out_h5=out_h5,
        out_json=out_json,
        cfg=cfg,
    )

    print(f"[potts_descriptors] Wrote: {out_h5}")
    print(f"[potts_descriptors] Wrote: {out_json}")
    print("[potts_descriptors] Done!")


if __name__ == "__main__":
    main()
