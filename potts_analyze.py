#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze.py

Simplified Potts analysis:
  1) Reads RAW spins from potts_gen output
  2) Computes correlations_2d and correlations_radial with whitening (1e-3)
  3) Builds descriptor Y from ensemble means
  4) Computes local injectivity diagnostics
  5) Generates plots

Outputs:
  potts_analysis/<YYYYMMDD_HHMMSSZ>/<input_stem>/<descriptor_kind>/
    potts_stats_q{q}_{H}x{W}.h5
    potts_stats_q{q}_{H}x{W}.json
    potts_local_explainedcov_injectivity.csv
    figs/*.png + *.pdf
    metadata_local_explainedcov_injectivity.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


DescKind = Literal["radial1d", "corr2d"]


# ----------------------------- time / run discovery -----------------------------

def _utc_now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_folder_name_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")


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
    potts_analysis_dir: str = "potts_analysis"
    tag: str = "potts_stats"

    descriptor: DescKind = "corr2d"
    n_radial_bins: int = 64

    prepend_phase_fractions_to_Y: bool = False
    
    # Whitening regularization for correlations
    whiten_eps: float = 1e-2

    # Batching for memory management
    batch_size: int = 256

    # Injectivity metric knobs
    standardize_X: bool = True
    standardize_Y: bool = True

    kY: int = 15

    use_weights: bool = False
    eps_tau: float = 1e-12

    ridge_y: float = 1e-3
    ridge_x: float = 1e-8
    eps_trace: float = 1e-18

    device: str = "cuda"

    # Plot controls
    dpi: int = 250
    save_scatter: bool = True
    save_heatmaps: bool = True

    hm_bins_temp: int = 60
    hm_bins_frac: int = 60
    hm_sigma_px: float = 1.0
    hm_clip: Tuple[float, float] = (1.0, 99.0)


def _pair_labels(q: int) -> List[str]:
    return [f"{i}-{j}" for i in range(q) for j in range(i, q)]


# ----------------------------- old-correct Potts correlation ops -----------------------------

@torch.no_grad()
def compute_correlations_2d(spins: torch.Tensor, q: int, whiten_eps: float = 1e-3) -> torch.Tensor:
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
    
    # Apply whitening: add small constant to diagonal
    out = out + whiten_eps

    return out


@torch.no_grad()
def compute_radial_average(correlations_2d: torch.Tensor, n_bins: int, whiten_eps: float = 1e-3) -> torch.Tensor:
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


# ----------------------------- plotting helpers (old style) -----------------------------

def scatter_tf(temp: np.ndarray, frac: np.ndarray, vals: np.ndarray,
               title: str, outpath_base: Path, dpi: int = 250, cmap: str = "viridis",
               vmin=None, vmax=None) -> None:
    plt.figure(figsize=(5.2, 4.2), dpi=dpi)
    sc = plt.scatter(temp, frac, c=vals, s=8, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("temperature")
    plt.ylabel("fraction_initial")
    plt.title(title)
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(str(outpath_base) + ".png", bbox_inches="tight", dpi=dpi)
    plt.savefig(str(outpath_base) + ".pdf", bbox_inches="tight")
    plt.close()


def _gaussian_kernel1d(sigma_px: float) -> np.ndarray:
    sigma = float(max(sigma_px, 1e-6))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _pad_reflect(arr: np.ndarray, pad: int, axis: int) -> np.ndarray:
    pw = [(0, 0)] * arr.ndim
    pw[axis] = (pad, pad)
    return np.pad(arr, pw, mode="reflect")


def _conv1d_reflect(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = k.size // 2
    x = _pad_reflect(arr, pad, axis)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=axis, arr=x)


def _smooth_nan(img: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px is None or sigma_px <= 0:
        return img
    k = _gaussian_kernel1d(sigma_px)
    val = img.copy()
    mask = np.isfinite(val).astype(np.float64)
    val[~np.isfinite(val)] = 0.0
    val = _conv1d_reflect(val, k, axis=0)
    val = _conv1d_reflect(val, k, axis=1)
    msk = _conv1d_reflect(mask, k, axis=0)
    msk = _conv1d_reflect(msk, k, axis=1)
    out = np.divide(val, np.maximum(msk, 1e-12), where=(msk > 1e-12))
    out[msk < 1e-12] = np.nan
    return out


def heatmap_binned_tf(temp: np.ndarray,
                      frac: np.ndarray,
                      Z: np.ndarray,
                      title: str,
                      outbase: Path,
                      bins_t: int = 60,
                      bins_f: int = 60,
                      sigma_px: float = 1.0,
                      clip=(1, 99),
                      cmap="viridis",
                      dpi: int = 250) -> None:
    temp = temp.astype(np.float64)
    frac = frac.astype(np.float64)
    Z = Z.astype(np.float64)

    tmin, tmax = float(np.min(temp)), float(np.max(temp))
    fmin, fmax = float(np.min(frac)), float(np.max(frac))

    sum_w, tx, fx = np.histogram2d(temp, frac,
                                  bins=[bins_t, bins_f],
                                  range=[[tmin, tmax], [fmin, fmax]],
                                  weights=Z)
    cnt, _, _ = np.histogram2d(temp, frac,
                              bins=[tx, fx],
                              range=[[tmin, tmax], [fmin, fmax]])

    with np.errstate(invalid="ignore", divide="ignore"):
        img = sum_w / cnt
    img[cnt == 0] = np.nan
    img_s = _smooth_nan(img, sigma_px)

    if clip is not None:
        vmin, vmax = np.nanpercentile(img_s, clip[0]), np.nanpercentile(img_s, clip[1])
    else:
        vmin = vmax = None

    fig, axp = plt.subplots(figsize=(5.6, 4.8), dpi=dpi)
    im = axp.imshow(img_s.T,
                    origin="lower",
                    extent=[tx[0], tx[-1], fx[0], fx[-1]],
                    aspect="auto",
                    cmap=cmap,
                    interpolation="bilinear",
                    vmin=vmin,
                    vmax=vmax)
    fig.colorbar(im, ax=axp, pad=0.02, fraction=0.05)
    axp.set_xlabel("temperature")
    axp.set_ylabel("fraction_initial")
    axp.set_title(title)
    fig.tight_layout()
    fig.savefig(str(outbase) + ".png", bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf")
    plt.close(fig)


# ----------------------------- injectivity core (old script logic) -----------------------------

def standardize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def to_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute k nearest neighbors in Y."""
    device = Y.device
    N = Y.shape[0]
    k = min(int(k), N - 1)
    
    # Compute all pairwise distances
    Dc = torch.cdist(Y, Y, p=2.0)  # (N,N)
    
    # Set self-distances to infinity
    Dc.fill_diagonal_(float("inf"))
    
    # Get top k nearest neighbors
    vals, idx = torch.topk(Dc, k=k, largest=False, sorted=True)
    
    return idx, vals


@torch.no_grad()
def local_explainedcov_metrics_LOO(
    X: torch.Tensor,          # (N,p)
    Y: torch.Tensor,          # (N,qy)
    idxY: torch.Tensor,       # (N,kY) excluding self
    dY: torch.Tensor,         # (N,kY)
    use_weights: bool,
    eps_tau: float,
    ridge_y: float,
    ridge_x: float,
    eps_trace: float,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Kernel ridge in neighborhood, scored by LOO residuals.
    Batched over N to reduce memory.
    """
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    expl = torch.empty((N,), device=device, dtype=torch.float32)
    trX = torch.empty((N,), device=device, dtype=torch.float32)
    trR = torch.empty((N,), device=device, dtype=torch.float32)

    worst_unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    worst_ret = torch.empty((N,), device=device, dtype=torch.float32)

    unexpl_coord_max = torch.empty((N,), device=device, dtype=torch.float32)
    unexpl_coord_0 = torch.empty((N,), device=device, dtype=torch.float32)
    unexpl_coord_1 = torch.empty((N,), device=device, dtype=torch.float32) if p >= 2 else None

    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

    # Process in batches to reduce memory
    for i0 in range(0, N, int(batch_size)):
        i1 = min(N, i0 + int(batch_size))
        B = i1 - i0
        
        centers = torch.arange(i0, i1, device=device, dtype=torch.int64)
        neigh = torch.cat([centers[:, None], idxY[i0:i1]], dim=1)  # (B,k)

        dn = torch.cat([
            torch.zeros((B, 1), device=device, dtype=torch.float32),
            dY[i0:i1].to(torch.float32)
        ], dim=1)
        avg_dy[i0:i1] = dn[:, 1:].mean(dim=1)

        Xn = X[neigh]  # (B,k,p)
        Yn = Y[neigh]  # (B,k,qy)

        if use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(eps_tau))
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)  # (B,k)
        sw = torch.sqrt(w).to(torch.float32)

        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * Yn).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Yc = Yn.to(torch.float32) - muY[:, None, :]

        Xs = Xc * sw[:, :, None]  # (B,k,p)
        Ys = Yc * sw[:, :, None]  # (B,k,qy)

        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))  # (B,k,k)
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(ridge_y) * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))  # (B,k,k)
        alpha = torch.bmm(Hinv, Xs)                                       # (B,k,p)

        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)

        u = (trR_b / (trX_b + float(eps_trace))).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexpl[i0:i1] = u
        expl[i0:i1] = e

        varX = (Xs * Xs).sum(dim=1)              # (B,p)
        varR = (Rloo * Rloo).sum(dim=1)          # (B,p)
        ucoord = (varR / (varX + float(eps_trace))).clamp(0.0, 1.0)
        unexpl_coord_0[i0:i1] = ucoord[:, 0]
        if p >= 2:
            unexpl_coord_1[i0:i1] = ucoord[:, 1]
        unexpl_coord_max[i0:i1] = ucoord.max(dim=1).values

        # Directional worst-case via generalized eig of (SigmaR, SigmaX + gam I)
        SigmaX = torch.bmm(Xs.transpose(1, 2), Xs).to(torch.float32)
        SigmaR = torch.bmm(Rloo.transpose(1, 2), Rloo).to(torch.float32)

        gam = (float(ridge_x) * trX_b / float(max(p, 1))).to(torch.float32)
        SigmaXr = SigmaX + gam[:, None, None] * I_p[None, :, :]

        L = torch.linalg.cholesky(SigmaXr)
        Z = torch.linalg.solve_triangular(L, SigmaR, upper=False)
        M = torch.linalg.solve_triangular(L.transpose(1, 2), Z, upper=True)
        M = 0.5 * (M + M.transpose(1, 2))
        ev = torch.linalg.eigvalsh(M)
        wmax = ev[:, -1].clamp(0.0, 1.0)

        worst_unexpl[i0:i1] = wmax
        worst_ret[i0:i1] = (1.0 - wmax).clamp(0.0, 1.0)

    out: Dict[str, np.ndarray] = dict(
        unexplained_frac=unexpl.detach().cpu().numpy(),
        explained_frac=expl.detach().cpu().numpy(),
        trX=trX.detach().cpu().numpy(),
        trR=trR.detach().cpu().numpy(),
        worst_unexplained_ratio=worst_unexpl.detach().cpu().numpy(),
        worst_retention=worst_ret.detach().cpu().numpy(),
        unexplained_coord0=unexpl_coord_0.detach().cpu().numpy(),
        unexplained_coord_max=unexpl_coord_max.detach().cpu().numpy(),
        avg_dY=avg_dy.detach().cpu().numpy(),
    )
    if p >= 2 and unexpl_coord_1 is not None:
        out["unexplained_coord1"] = unexpl_coord_1.detach().cpu().numpy()
    return out


def build_stats_and_Y(
    in_h5: Path,
    out_h5: Path,
    out_json: Path,
    cfg: Config,
    root_desc: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes stats H5 and returns (X,Y) for injectivity.
    Simplified - no batching or chunking.
    """
    ensure_dir(out_h5.parent)

    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    created_utc = _utc_now_z()

    print("[potts_analyze] Loading data from H5...")
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

    # Decide Y dimension
    if cfg.descriptor == "radial1d":
        base_dim = n_pairs * int(cfg.n_radial_bins)
    else:
        base_dim = n_pairs * H * W
    ydim = (q + base_dim) if cfg.prepend_phase_fractions_to_Y else base_dim

    X = np.stack([temps, fracs_init], axis=1).astype(np.float32)  # (N,2)
    Y = np.empty((N, ydim), dtype=np.float32)

    print(f"[potts_analyze] Computing correlations for {N} parameters, {R} repeats...")
    
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
            print(f"[potts_analyze] Processing parameters {i}/{N}...")
        
        # Accumulate statistics over R dimension in batches to avoid VRAM overflow
        # Determine batch size for repeats - be conservative with VRAM
        # Aim for ~200MB per batch: batch_n * repeat_batch * H * W * 4 bytes * n_pairs * 2 < 200MB
        # Factor of 2 for intermediate tensors, n_pairs for correlation outputs
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
            spins_batch = torch.as_tensor(spins_data[i:i_end, r:r_end], device=device)  # (batch_n, r_batch, H, W)
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
    
    print("[potts_analyze] Computing statistics... done")

    # Write stats H5
    print(f"[potts_analyze] Writing stats to {out_h5}...")
    with h5py.File(str(out_h5), "w") as fout:
        fout.attrs["created_utc"] = created_utc
        fout.attrs["input_h5"] = str(in_h5)
        fout.attrs["config_json"] = json.dumps(asdict(cfg))
        fout.attrs["n_parameters"] = N
        fout.attrs["n_repeats"] = R
        fout.attrs["grid_size"] = H
        fout.attrs["q"] = q
        fout.attrs["n_correlation_pairs"] = n_pairs

        gp = fout.create_group("parameters")
        gp.create_dataset("temperature", data=temps)
        gp.create_dataset("fraction_initial", data=fracs_init)

        gm = fout.create_group("metadata")
        gm.create_dataset("radial_bins", data=radial_bins)
        gm.create_dataset("pair_labels", data=np.array([s.encode() for s in pair_labels], dtype="S"))
        gm.create_dataset("phase_labels", data=np.arange(q, dtype=np.int32))

        gc = fout.create_group("correlations")
        gc.create_dataset("correlations_2d_mean", data=mean2d, compression="gzip")
        gc.create_dataset("correlations_radial_mean", data=mean1d, compression="gzip")
        gc.create_dataset("correlations_radial_std", data=std1d, compression="gzip")

        gpz = fout.create_group("phases")
        gpz.create_dataset("final_fraction_mean", data=meanph)
        gpz.create_dataset("final_fraction_std", data=stdph)

        # Build Y from ensemble means
        if cfg.descriptor == "radial1d":
            feat = mean1d.reshape(N, -1)
        else:
            feat = mean2d.reshape(N, -1)

        if cfg.prepend_phase_fractions_to_Y:
            Y = np.concatenate([meanph, feat], axis=1)
        else:
            Y = feat

        gy = fout.create_group("Y")
        gy.attrs["descriptor"] = cfg.descriptor
        gy.attrs["prepend_phase_fractions_to_Y"] = int(cfg.prepend_phase_fractions_to_Y)
        gy.create_dataset("Y_mean_descriptor", data=Y, compression="gzip")

    meta = dict(
        created_utc=created_utc,
        input_h5=str(in_h5),
        output_h5=str(out_h5),
        config=asdict(cfg),
        shapes=dict(
            correlations_2d_mean=[int(N), int(n_pairs), int(H), int(W)],
            correlations_radial_mean=[int(N), int(n_pairs), int(cfg.n_radial_bins)],
            correlations_radial_std=[int(N), int(n_pairs), int(cfg.n_radial_bins)],
            final_fraction_mean=[int(N), int(q)],
            final_fraction_std=[int(N), int(q)],
            Y=[int(N), int(ydim)],
        ),
        pair_labels=pair_labels,
        radial_bins=radial_bins.tolist(),
    )
    out_json.write_text(json.dumps(meta, indent=2))
    return X, Y, meta


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--h5", type=str, default="", help="Input HDF5 from potts_gen. If empty, uses latest under --potts_data_dir.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    ap.add_argument("--tag", type=str, default=Config.tag)

    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"])
    ap.add_argument("--n_radial_bins", type=int, default=Config.n_radial_bins)
    ap.add_argument("--whiten_eps", type=float, default=Config.whiten_eps)

    ap.add_argument("--prepend_phase_fractions_to_Y", action="store_true")
    ap.add_argument("--no-prepend_phase_fractions_to_Y", dest="prepend_phase_fractions_to_Y", action="store_false")
    ap.set_defaults(prepend_phase_fractions_to_Y=False)

    ap.add_argument("--standardize_X", type=bool, default=True)
    ap.add_argument("--standardize_Y", type=bool, default=True)

    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true", help="If set, use Gaussian weights in neighborhoods (default False).")
    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)

    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--device", type=str, default=Config.device)

    ap.add_argument("--dpi", type=int, default=Config.dpi)
    ap.add_argument("--no_scatter", action="store_true")
    ap.add_argument("--no_heatmaps", action="store_true")

    ap.add_argument("--hm_bins_temp", type=int, default=Config.hm_bins_temp)
    ap.add_argument("--hm_bins_frac", type=int, default=Config.hm_bins_frac)
    ap.add_argument("--hm_sigma_px", type=float, default=Config.hm_sigma_px)
    ap.add_argument("--hm_clip_lo", type=float, default=Config.hm_clip[0])
    ap.add_argument("--hm_clip_hi", type=float, default=Config.hm_clip[1])

    args = ap.parse_args()

    cfg = Config(
        potts_data_dir=str(args.potts_data_dir),
        potts_analysis_dir=str(args.potts_analysis_dir),
        tag=str(args.tag),
        descriptor=str(args.descriptor),  # type: ignore
        n_radial_bins=int(args.n_radial_bins),
        whiten_eps=float(args.whiten_eps),
        batch_size=int(args.batch_size),
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),
        standardize_X=bool(args.standardize_X),
        standardize_Y=bool(args.standardize_Y),
        kY=int(args.kY),
        use_weights=bool(args.use_weights),
        ridge_y=float(args.ridge_y),
        ridge_x=float(args.ridge_x),
        device=str(args.device),
        dpi=int(args.dpi),
        save_scatter=not bool(args.no_scatter),
        save_heatmaps=not bool(args.no_heatmaps),
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        hm_clip=(float(args.hm_clip_lo), float(args.hm_clip_hi)),
    )

    data_root = Path(cfg.potts_data_dir)
    if str(args.h5).strip():
        in_h5 = Path(str(args.h5)).expanduser().resolve()
    else:
        in_h5 = find_latest_h5_under(data_root)

    # output directories
    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / in_h5.stem
    root_desc = ensure_dir(session_dir / cfg.descriptor)

    out_base_stub = None
    with h5py.File(str(in_h5), "r") as fin:
        q = int(fin.attrs.get("q", 3))
        H = int(fin["states/final_spins"].shape[2])
        W = int(fin["states/final_spins"].shape[3])
        out_base_stub = f"{cfg.tag}_q{q}_{H}x{W}"

    out_h5 = session_dir / f"{out_base_stub}.h5"
    out_json = session_dir / f"{out_base_stub}.json"

    # Compute stats and Y (no caching)
    print(f"[potts_analyze] Input: {in_h5}")
    print(f"[potts_analyze] Output: {out_h5}")
    X, Y, meta = build_stats_and_Y(
        in_h5=in_h5,
        out_h5=out_h5,
        out_json=out_json,
        cfg=cfg,
        root_desc=root_desc,
    )

    # Injectivity computation
    print("[potts_analyze] Computing injectivity metrics...")
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    X_use = X.copy()
    Y_use = Y.copy()
    if cfg.standardize_X:
        X_use = standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = standardize_np(Y_use)

    Xt = to_t(X_use, device=device)
    Yt = to_t(Y_use, device=device)

    print(f"[potts_analyze] Finding {cfg.kY} nearest neighbors...")
    idxY_t, dY_t = knn_in_y(Yt, k=int(cfg.kY))

    print("[potts_analyze] Computing local explained covariance metrics...")
    metrics = local_explainedcov_metrics_LOO(
        X=Xt,
        Y=Yt,
        idxY=idxY_t,
        dY=dY_t,
        use_weights=bool(cfg.use_weights),
        eps_tau=float(cfg.eps_tau),
        ridge_y=float(cfg.ridge_y),
        ridge_x=float(cfg.ridge_x),
        eps_trace=float(cfg.eps_trace),
        batch_size=int(cfg.batch_size),
    )

    # Save CSV
    out_root = root_desc
    figs_dir = ensure_dir(out_root / "figs")
    csv_path = out_root / "potts_local_explainedcov_injectivity.csv"

    header_cols = [
        "temperature", "fraction_initial",
        "unexplained_frac", "explained_frac",
        "worst_unexplained_ratio", "worst_retention",
        "trX", "trR",
        "avg_dY",
        "unexplained_coord0",
        "unexplained_coord_max",
    ]
    if "unexplained_coord1" in metrics:
        header_cols.append("unexplained_coord1")

    cols = [
        X[:, 0], X[:, 1],
        metrics["unexplained_frac"], metrics["explained_frac"],
        metrics["worst_unexplained_ratio"], metrics["worst_retention"],
        metrics["trX"], metrics["trR"],
        metrics["avg_dY"],
        metrics["unexplained_coord0"],
        metrics["unexplained_coord_max"],
    ]
    if "unexplained_coord1" in metrics:
        cols.append(metrics["unexplained_coord1"])

    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

    # Plots
    print("[potts_analyze] Generating plots...")
    temp = X[:, 0].astype(np.float64, copy=False)
    frac = X[:, 1].astype(np.float64, copy=False)

    if cfg.save_scatter:
        scatter_tf(temp, frac, metrics["unexplained_frac"],
                   f"{cfg.descriptor}: unexplained fraction tr(R)/tr(X) (higher worse)",
                   figs_dir / "scatter_unexplained_frac", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["explained_frac"],
                   f"{cfg.descriptor}: explained fraction 1 - tr(R)/tr(X) (higher better)",
                   figs_dir / "scatter_explained_frac", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["worst_retention"],
                   f"{cfg.descriptor}: worst retention (directional, ridge-stabilized)",
                   figs_dir / "scatter_worst_retention", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, np.log10(metrics["trX"] + 1e-30),
                   f"{cfg.descriptor}: log10 local X energy tr(X) (excitation)",
                   figs_dir / "scatter_log10_trX", dpi=cfg.dpi)
        scatter_tf(temp, frac, metrics["unexplained_coord_max"],
                   f"{cfg.descriptor}: max coord unexplained (catches single-parameter collapse)",
                   figs_dir / "scatter_unexplained_coord_max", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        if "unexplained_coord1" in metrics:
            scatter_tf(temp, frac, metrics["unexplained_coord1"],
                       f"{cfg.descriptor}: unexplained fraction_initial coordinate",
                       figs_dir / "scatter_unexplained_fraction_coord", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        trXv = metrics["trX"]
        exc = trXv >= np.quantile(trXv, 0.25)
        flag_ok = ((metrics["explained_frac"] >= 0.9) & exc).astype(np.float64)
        scatter_tf(temp, frac, flag_ok,
                   f"{cfg.descriptor}: indicator explained_frac>=0.9 & trX>=Q25",
                   figs_dir / "scatter_ok_indicator", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

    if cfg.save_heatmaps:
        heatmap_binned_tf(temp, frac, metrics["unexplained_frac"],
                          f"{cfg.descriptor}: heatmap unexplained fraction",
                          figs_dir / "heatmap_unexplained_frac",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["explained_frac"],
                          f"{cfg.descriptor}: heatmap explained fraction",
                          figs_dir / "heatmap_explained_frac",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["worst_retention"],
                          f"{cfg.descriptor}: heatmap worst retention",
                          figs_dir / "heatmap_worst_retention",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

    # Metadata JSON (injectivity)
    summary = dict(
        input_h5=str(in_h5),
        output_stats_h5=str(out_h5),
        descriptor=str(cfg.descriptor),
        N=int(X.shape[0]),
        y_dim=int(Y.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        whiten_eps=float(cfg.whiten_eps),
        stats=dict(
            unexpl_median=float(np.median(metrics["unexplained_frac"])),
            unexpl_q90=float(np.quantile(metrics["unexplained_frac"], 0.90)),
            expl_median=float(np.median(metrics["explained_frac"])),
            worst_ret_median=float(np.median(metrics["worst_retention"])),
        ),
    )
    inj_meta = dict(
        created_utc=_utc_now_z(),
        config=asdict(cfg),
        files=dict(
            csv=str(csv_path),
            figs=str(figs_dir),
        ),
        summary=summary,
        stats_meta=meta,
    )
    meta_path = out_root / "metadata_local_explainedcov_injectivity.json"
    meta_path.write_text(json.dumps(inj_meta, indent=2))
    print(f"[potts_analyze] wrote: {meta_path}")
    print("[potts_analyze] Done!")


if __name__ == "__main__":
    main()
