#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch_injectivity_analysis.py

CH local injectivity diagnostics via local explained covariance (CCA-inspired),
LOO-scored kernel ridge in Y-neighborhoods.

Key constraints enforced:
  - Injectivity is evaluated against UNWARPED parameters X (original draw in [-1,1]x[-1,1]),
    not the warped/latent variables.
  - Uses the same torch-based LOO kernel ridge + eps/ridge conventions as your working scripts.
  - Repeats-mode descriptors are scalar-field descriptors (no pairings, no cross-correlations of masks).

Auto-discovery:
  - Recursively finds every subdirectory under ch_data_root that contains *.h5.
  - For each such directory (interpreted as one "warp + repeat/fixed combination"), loads ONLY the most recent *.h5.
Output layout:
  out_root/
    <combo_relpath>/                # combo_relpath is the relative path under ch_data_root, "/" -> "__"
      latest/                       # only most recent file is analyzed
        fixed/raw_flat/...
        repeats/radial1d/...
        repeats/corr2d/...
        repeats/euler/...

Modes:
  - fixed:   fields is (N,H,W) and Y = flattened (optionally downsampled) raw fields.
  - repeats: fields is (N,R,H,W) and Y is one of:
      radial1d: scalar autocorr corr2d mean over repeats -> radial average -> (N,n_radial_bins)
      corr2d:   scalar autocorr corr2d mean over repeats -> (optional avgpool ds) -> flatten
      euler:    thresholds -> dilations (euler_radii) -> chi(V-E+F) per repeat -> mean over repeats

Plots:
  - publication-style explained_frac maps with BOTH:
      * adaptive range [min,max]
      * fixed range [0,1]
    (scatter + binned heatmap) over xdim0, xdim1.

Dependencies: numpy, torch, h5py, matplotlib
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------- defaults (keep eps/ridge conventions) ----------------------

CFG: Dict[str, Any] = dict(
    ch_data_root="ch_ab_controls_data",
    out_root="ch_ab_analysis",

    # X selection:
    # - ds_X="auto" will prefer any dataset path that looks unwarped/prewarp/original and matches (N,2) if available
    ds_X="auto",
    ds_Y="fields",

    standardize_X=False,
    standardize_Y=False,

    # fixed mode (raw fields)
    downsample_fixed=1,

    # repeats mode (descriptor selection)
    descriptor="all",  # radial1d | corr2d | euler | all

    thresholds=(0.4, 0.45, 0.50, 0.55, 0.60),
    n_radial_bins=64,
    corr2d_downsample=4,
    euler_radii=(0, 1, 2, 4, 8),

    # kNN in Y
    kY=15,
    knn_chunk=512,

    # neighborhood weighting
    use_weights=False,
    eps_tau=1e-12,

    # ridge (DO NOT change defaults)
    ridge_y=1e-3,
    ridge_x=1e-8,
    eps_trace=1e-18,

    batch_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=1337,
    max_N=None,

    # plotting
    xdim0=0,
    xdim1=1,
    dpi=300,
    hm_bins0=60,
    hm_bins1=60,
    hm_sigma_px=1.0,
)


# ---------------------- misc utils ----------------------

def standardize_np(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (A - mu) / sd

def json_sanitize(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    if isinstance(x, dict): return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_sanitize(v) for v in x]
    return x

def maybe_subsample_idx(N: int, max_N: Optional[int], seed: int) -> np.ndarray:
    if max_N is None or N <= int(max_N):
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(N, size=int(max_N), replace=False)
    idx.sort()
    return idx.astype(np.int64)

def read_h5_attrs(f: h5py.File) -> Dict[str, Any]:
    out = {}
    for k, v in f.attrs.items():
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8")
            except Exception:
                v = str(v)
        if k == "config" and isinstance(v, str):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                out[k] = {"raw_config_str": v}
                continue
        out[k] = v
    return out


# ---------------------- dataset discovery + output naming ----------------------

def find_most_recent_timestamp(ch_root: Path) -> Optional[Path]:
    """
    Find the most recent timestamp directory under ch_root.
    Expected structure: ch_data/{timestamp}/
    """
    if not ch_root.exists():
        return None

    timestamp_dirs = []
    for d in ch_root.iterdir():
        if d.is_dir() and len(d.name) == 15 and "_" in d.name:
            try:
                parts = d.name.split("_")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    timestamp_dirs.append(d)
            except Exception:
                continue

    if not timestamp_dirs:
        return None

    timestamp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return timestamp_dirs[0]

def discover_h5_files_in_timestamp(timestamp_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Find all h5 files in a timestamp directory.
    Returns list of (warp, mode, h5_path) tuples.
    Expected names: {warp}_{mode}.h5
    """
    files = []
    for h5_path in timestamp_dir.glob("*.h5"):
        name = h5_path.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            warp, mode = parts
            files.append((warp, mode, h5_path))

    files.sort(key=lambda t: (t[0], t[1]))
    return files


# ---------------------- X selection: prefer unwarped/prewarp/original ----------------------

_UNWARP_HINTS = (
    "unwarp", "unwarped", "prewarp", "pre_warp", "before_warp", "original",
    "raw_draw", "rawdraw", "draw_xy", "draw_x", "draw_y", "source_xy", "src_xy",
    "base_xy", "latent_input", "input_xy"
)
_WARP_HINTS_BAD = (
    "warp", "warped", "latent", "hidden", "transformed"
)

def _list_datasets(f: h5py.File) -> List[str]:
    paths: List[str] = []
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    f.visititems(_visit)
    return paths

def _score_x_path(path: str) -> int:
    s = path.lower()
    score = 0
    for h in _UNWARP_HINTS:
        if h in s:
            score += 10
    for h in _WARP_HINTS_BAD:
        if h in s:
            score -= 5
    if "param" in s:
        score += 2
    if s.endswith("x") or s.endswith("y") or "xy" in s:
        score += 1
    return score

def pick_X_dataset_path(f: h5py.File, N_expected: int, ds_X_cfg: str) -> str:
    """
    If ds_X_cfg != "auto": use it.
    If "auto": choose best candidate that matches (N_expected,2) if possible, else (N_expected,p).
    """
    if str(ds_X_cfg).strip().lower() != "auto":
        return str(ds_X_cfg)

    paths = _list_datasets(f)
    cands: List[Tuple[int, str, Tuple[int, ...]]] = []
    for pth in paths:
        try:
            d = f[pth]
            shp = tuple(int(x) for x in d.shape)
        except Exception:
            continue
        if len(shp) != 2:
            continue
        if shp[0] != int(N_expected):
            continue
        sc = _score_x_path(pth)
        if shp[1] == 2:
            sc += 50
        cands.append((sc, pth, shp))

    if cands:
        cands.sort(key=lambda t: t[0], reverse=True)
        return cands[0][1]

    if "controlling_temperatures" in f:
        d = f["controlling_temperatures"]
        if d.ndim == 2 and d.shape[0] == int(N_expected):
            return "controlling_temperatures"

    raise KeyError("Could not auto-select an X dataset path matching N. Set --ds_X explicitly.")


# ---------------------- plotting ----------------------

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

def _save_both(fig, base: Path):
    png_dir = base.parent / "png"
    pdf_dir = base.parent / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / (base.name + ".png"), bbox_inches="tight")
    fig.savefig(pdf_dir / (base.name + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def scatter_2d(x0, x1, z, xlabel, ylabel, title, base: Path, dpi: int, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5.3, 4.4), dpi=dpi)
    if vmin is None: vmin = float(np.nanmin(z))
    if vmax is None: vmax = float(np.nanmax(z))
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = 1.0
    if vmin == vmax: vmax = vmin + 1e-6
    sc = ax.scatter(x0, x1, c=z, s=8, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.95, linewidths=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    fig.tight_layout()
    _save_both(fig, base)

def heatmap_binned_2d(x0, x1, z, xlabel, ylabel, title, base: Path,
                      bins0: int, bins1: int, sigma_px: float, dpi: int,
                      vmin=None, vmax=None, cmap="viridis"):
    x0 = x0.astype(np.float64); x1 = x1.astype(np.float64); z = z.astype(np.float64)
    xmin, xmax = float(np.min(x0)), float(np.max(x0))
    ymin, ymax = float(np.min(x1)), float(np.max(x1))

    sum_w, xedges, yedges = np.histogram2d(
        x0, x1, bins=[bins0, bins1],
        range=[[xmin, xmax], [ymin, ymax]],
        weights=z,
    )
    cnt, _, _ = np.histogram2d(
        x0, x1, bins=[xedges, yedges],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        img = sum_w / cnt
    img[cnt == 0] = np.nan
    img_s = _smooth_nan(img, sigma_px)

    if vmin is None: vmin = float(np.nanmin(img_s)) if np.isfinite(np.nanmin(img_s)) else 0.0
    if vmax is None: vmax = float(np.nanmax(img_s)) if np.isfinite(np.nanmax(img_s)) else 1.0
    if vmin == vmax: vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=dpi)
    im = ax.imshow(
        img_s.T, origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap=cmap, interpolation="bilinear",
        vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    _save_both(fig, base)


# ---------------------- kNN in Y (chunked) ----------------------

@torch.no_grad()
def knn_in_y_chunked(Y: torch.Tensor, k: int, chunk: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    device = Y.device
    N = Y.shape[0]
    if k >= N:
        k = N - 1
    idx_out = torch.empty((N, k), device=device, dtype=torch.int64)
    d_out = torch.empty((N, k), device=device, dtype=torch.float32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Dc = torch.cdist(Y[s:e], Y, p=2.0)

        r = torch.arange(e - s, device=device)
        c = torch.arange(s, e, device=device)
        Dc[r, c] = float("inf")

        vals, idx = torch.topk(Dc, k=k, largest=False, sorted=False)
        ord_local = torch.argsort(vals, dim=1)
        vals_s = torch.gather(vals, dim=1, index=ord_local)
        idx_s = torch.gather(idx, dim=1, index=ord_local)

        idx_out[s:e] = idx_s
        d_out[s:e] = vals_s

    return idx_out, d_out


# ---------------------- local explained covariance (LOO) ----------------------

@torch.no_grad()
def local_explainedcov_metrics_batched_LOO(
    X: torch.Tensor,
    Y: torch.Tensor,
    idxY: torch.Tensor,
    dY: torch.Tensor,
    use_weights: bool,
    eps_tau: float,
    ridge_y: float,
    ridge_x: float,
    eps_trace: float,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
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
    unexpl_coords = torch.empty((N, p), device=device, dtype=torch.float32)
    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        B = i1 - i0

        centers = torch.arange(i0, i1, device=device, dtype=torch.int64)
        neigh = torch.cat([centers[:, None], idxY[i0:i1]], dim=1)

        dn = torch.cat(
            [torch.zeros((B, 1), device=device, dtype=torch.float32), dY[i0:i1].to(torch.float32)],
            dim=1,
        )
        avg_dy[i0:i1] = dn[:, 1:].mean(dim=1)

        Xn = X[neigh]
        Yn = Y[neigh]

        if use_weights:
            tau = dn.max(dim=1).values.clamp_min(eps_tau)
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)
        sw = torch.sqrt(w).to(torch.float32)

        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * Yn).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Yc = Yn.to(torch.float32) - muY[:, None, :]

        Xs = Xc * sw[:, :, None]
        Ys = Yc * sw[:, :, None]

        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (ridge_y * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))
        alpha = torch.bmm(Hinv, Xs)

        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)

        u = (trR_b / (trX_b + eps_trace)).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexpl[i0:i1] = u
        expl[i0:i1] = e

        varX = (Xs * Xs).sum(dim=1)
        varR = (Rloo * Rloo).sum(dim=1)
        ucoord = (varR / (varX + eps_trace)).clamp(0.0, 1.0)
        unexpl_coords[i0:i1, :] = ucoord
        unexpl_coord_max[i0:i1] = ucoord.max(dim=1).values

        SigmaX = torch.bmm(Xs.transpose(1, 2), Xs)
        SigmaR = torch.bmm(Rloo.transpose(1, 2), Rloo)

        gam = (ridge_x * trX_b / float(max(p, 1))).to(torch.float32)
        SigmaXr = SigmaX + gam[:, None, None] * I_p[None, :, :]

        L = torch.linalg.cholesky(SigmaXr)
        Z = torch.linalg.solve_triangular(L, SigmaR, upper=False)
        M = torch.linalg.solve_triangular(L.transpose(1, 2), Z, upper=True)
        M = 0.5 * (M + M.transpose(1, 2))
        ev = torch.linalg.eigvalsh(M)
        wmax = ev[:, -1].clamp(0.0, 1.0)

        worst_unexpl[i0:i1] = wmax
        worst_ret[i0:i1] = (1.0 - wmax).clamp(0.0, 1.0)

    return dict(
        unexplained_frac=unexpl.detach().cpu().numpy(),
        explained_frac=expl.detach().cpu().numpy(),
        trX=trX.detach().cpu().numpy(),
        trR=trR.detach().cpu().numpy(),
        worst_unexplained_ratio=worst_unexpl.detach().cpu().numpy(),
        worst_retention=worst_ret.detach().cpu().numpy(),
        unexplained_coord_max=unexpl_coord_max.detach().cpu().numpy(),
        unexplained_coords=unexpl_coords.detach().cpu().numpy(),
        avg_dY=avg_dy.detach().cpu().numpy(),
    )


# ---------------------- CH scalar autocorrelation + Euler descriptors ----------------------

@torch.no_grad()
def corr2d_scalar_autocorr_fft(x: torch.Tensor) -> torch.Tensor:
    B, H, W = x.shape
    z = x - x.mean(dim=(-2, -1), keepdim=True)
    f = torch.fft.rfft2(z)
    p = f * torch.conj(f)
    corr = torch.fft.irfft2(p, s=(H, W)) / float(H * W)
    corr = torch.fft.fftshift(corr, dim=(-2, -1))
    return corr.to(torch.float32)

@torch.no_grad()
def radial_average_scalar(corr2d: torch.Tensor, n_bins: int) -> torch.Tensor:
    B, H, W = corr2d.shape
    center = H // 2
    max_r = H // 2

    y, x = torch.meshgrid(
        torch.arange(H, device=corr2d.device, dtype=torch.float32),
        torch.arange(W, device=corr2d.device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - center) ** 2 + (y - center) ** 2)

    bin_edges = torch.linspace(0, max_r, int(n_bins) + 1, device=corr2d.device)
    bin_indices = torch.searchsorted(bin_edges[:-1], r, right=False)
    bin_indices = torch.clamp(bin_indices, 0, int(n_bins) - 1)

    flat_idx = bin_indices.flatten()
    flat_val = corr2d.reshape(B, H * W)

    sums = torch.zeros((B, int(n_bins)), device=corr2d.device, dtype=torch.float32)
    cnts = torch.zeros((B, int(n_bins)), device=corr2d.device, dtype=torch.float32)

    expanded = flat_idx.unsqueeze(0).expand(B, -1)
    sums.scatter_add_(1, expanded, flat_val.to(torch.float32))
    cnts.scatter_add_(1, expanded, torch.ones_like(flat_val, dtype=torch.float32))

    out = torch.where(cnts > 0, sums / cnts, torch.zeros_like(sums))
    return out

@torch.no_grad()
def euler_from_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    B, H, W = mask.shape

    V = mask.view(B, -1).sum(dim=1)

    horiz = mask[:, :, :-1] * mask[:, :, 1:]
    vert = mask[:, :-1, :] * mask[:, 1:, :]
    E = horiz.view(B, -1).sum(dim=1) + vert.view(B, -1).sum(dim=1)

    f00 = mask[:, :-1, :-1]
    f01 = mask[:, :-1, 1:]
    f10 = mask[:, 1:, :-1]
    f11 = mask[:, 1:, 1:]
    Fq = (f00 * f01 * f10 * f11).view(B, -1).sum(dim=1)

    return V - E + Fq

@torch.no_grad()
def dilate_maxpool_iters(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    x = mask
    for _ in range(r):
        x = F.max_pool2d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    return x


# ---------------------- Y builders ----------------------

@torch.no_grad()
def build_Y_fixed_raw(fields: np.ndarray, downsample: int, device_pool: torch.device) -> np.ndarray:
    N, H, W = fields.shape
    ds = int(max(1, downsample))
    if ds > 1 and ((H % ds) != 0 or (W % ds) != 0):
        raise RuntimeError(f"downsample_fixed={ds} must divide H,W exactly (H={H}, W={W})")

    out = []
    chunk = 256
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        t = torch.from_numpy(fields[s:e]).to(device_pool, dtype=torch.float32)
        if ds > 1:
            t4 = t.unsqueeze(1)
            t4 = F.avg_pool2d(t4, kernel_size=ds, stride=ds)
            t = t4.squeeze(1)
        out.append(t.reshape(t.shape[0], -1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)

@torch.no_grad()
def build_Y_repeats_descriptor(
    fields: np.ndarray,
    descriptor: str,
    thresholds: Tuple[float, ...],
    n_radial_bins: int,
    corr2d_downsample: int,
    euler_radii: Tuple[int, ...],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    N, R, H, W = fields.shape
    meta: Dict[str, Any] = {
        "descriptor": descriptor,
        "N": int(N), "R": int(R), "H": int(H), "W": int(W),
    }

    if descriptor == "radial1d":
        nb = int(n_radial_bins)
        meta["n_radial_bins"] = nb
        Y = np.empty((N, nb), dtype=np.float32)

        for i in range(N):
            x = torch.from_numpy(fields[i]).to(device=device, dtype=torch.float32)
            c2 = corr2d_scalar_autocorr_fft(x)
            c2m = c2.mean(dim=0, keepdim=True)
            rad = radial_average_scalar(c2m, n_bins=nb)
            Y[i] = rad[0].detach().cpu().numpy().astype(np.float32)

        meta["Y_shape"] = [int(N), int(nb)]
        return Y, meta

    if descriptor == "corr2d":
        ds = int(max(1, corr2d_downsample))
        if ds > 1 and ((H % ds) != 0 or (W % ds) != 0):
            raise RuntimeError(f"corr2d_downsample={ds} must divide H,W exactly (H={H}, W={W})")
        meta["corr2d_downsample"] = ds
        H2, W2 = (H // ds, W // ds) if ds > 1 else (H, W)
        meta["corr2d_shape"] = [int(H2), int(W2)]

        Y = np.empty((N, H2 * W2), dtype=np.float32)

        for i in range(N):
            x = torch.from_numpy(fields[i]).to(device=device, dtype=torch.float32)
            c2 = corr2d_scalar_autocorr_fft(x)
            cmean = c2.mean(dim=0)

            if ds > 1:
                c4 = cmean.unsqueeze(0).unsqueeze(0)
                c4 = F.avg_pool2d(c4, kernel_size=ds, stride=ds)
                cmean = c4[0, 0]

            Y[i] = cmean.reshape(-1).detach().cpu().numpy().astype(np.float32)

        meta["Y_shape"] = [int(N), int(H2 * W2)]
        return Y, meta

    if descriptor == "euler":
        thr = tuple(float(t) for t in thresholds)
        radii = [int(r) for r in euler_radii]
        meta["thresholds"] = list(thr)
        meta["euler_radii"] = radii

        n_thr = len(thr)
        n_r = len(radii)
        Y = np.empty((N, n_thr * n_r), dtype=np.float32)

        for i in range(N):
            x = torch.from_numpy(fields[i]).to(device=device, dtype=torch.float32)

            curves = torch.zeros((R, n_thr, n_r), device=device, dtype=torch.float32)
            for ti, t in enumerate(thr):
                base = (x > t).float()
                for ri, r in enumerate(radii):
                    mr = dilate_maxpool_iters(base, r)
                    chi = euler_from_mask_batch(mr)
                    curves[:, ti, ri] = chi

            m = curves.mean(dim=0)
            Y[i] = m.reshape(-1).detach().cpu().numpy().astype(np.float32)

        meta["Y_shape"] = [int(N), int(n_thr * n_r)]
        return Y, meta

    raise KeyError(f"Unknown descriptor '{descriptor}' (expected radial1d|corr2d|euler)")


# ---------------------- pipeline ----------------------

def run_one_combo(h5_path: Path, warp: str, mode: str, timestamp: str, cfg: Dict[str, Any]) -> None:
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    device = torch.device(str(cfg["device"]))
    pool_device = device if device.type == "cuda" else torch.device("cpu")

    with h5py.File(str(h5_path.resolve()), "r") as f:
        if "prewarp_coordinates_normalized" in f:
            x_path = "prewarp_coordinates_normalized"
            X0 = np.array(f[x_path], dtype=np.float32)
        else:
            if cfg["ds_Y"] not in f:
                raise KeyError(f"Dataset '{cfg['ds_Y']}' not found in {h5_path}")
            dY = f[cfg["ds_Y"]]
            N_expected = int(dY.shape[0])
            x_path = "prewarp_coordinates_normalized"
            X0 = np.array(f[x_path], dtype=np.float32)

        if cfg["ds_Y"] not in f:
            raise KeyError(f"Dataset '{cfg['ds_Y']}' not found in {h5_path}")
        dY = f[cfg["ds_Y"]]
        if dY.ndim not in (3, 4):
            raise RuntimeError(f"Expected Y='{cfg['ds_Y']}' to be (N,H,W) or (N,R,H,W); got shape {dY.shape}")

        Y0 = np.array(dY, dtype=np.float32)
        attrs = read_h5_attrs(f)

    if X0.ndim != 2:
        raise RuntimeError(f"Expected X dataset '{x_path}' to be (N,p); got {X0.shape}")
    if int(X0.shape[0]) != int(Y0.shape[0]):
        raise RuntimeError(f"X/Y mismatch: X {X0.shape} vs Y {Y0.shape} (x_path='{x_path}')")

    detected_mode = "fixed" if Y0.ndim == 3 else "repeats"

    idx = maybe_subsample_idx(X0.shape[0], cfg["max_N"], cfg["seed"])
    X = X0[idx]
    Yraw = Y0[idx]

    if bool(cfg["standardize_X"]):
        X = standardize_np(X)

    if detected_mode == "fixed":
        Yemb = build_Y_fixed_raw(Yraw, downsample=int(cfg["downsample_fixed"]), device_pool=pool_device)
        y_meta = {
            "mode": "fixed",
            "Y_raw_shape": list(Yraw.shape),
            "Y_shape": list(Yemb.shape),
            "downsample_fixed": int(cfg["downsample_fixed"]),
        }
        if bool(cfg["standardize_Y"]):
            Yemb = standardize_np(Yemb)

        run_descriptor_block(
            h5_path=h5_path,
            warp=warp,
            mode=mode,
            timestamp=timestamp,
            cfg=cfg,
            desc_key="raw_flat",
            X=X,
            Y=Yemb,
            attrs=attrs,
            y_meta=y_meta,
            x_path=x_path,
        )
        return

    desc_req = str(cfg["descriptor"]).strip().lower()
    if desc_req == "all":
        desc_list = ["radial1d", "corr2d", "euler"]
    else:
        desc_list = [desc_req]

    for desc in desc_list:
        Yemb, y_meta = build_Y_repeats_descriptor(
            fields=Yraw,
            descriptor=desc,
            thresholds=tuple(cfg["thresholds"]),
            n_radial_bins=int(cfg["n_radial_bins"]),
            corr2d_downsample=int(cfg["corr2d_downsample"]),
            euler_radii=tuple(cfg["euler_radii"]),
            device=device,
        )
        if bool(cfg["standardize_Y"]):
            Yemb = standardize_np(Yemb)

        y_meta["mode"] = "repeats"
        y_meta["Y_raw_shape"] = list(Yraw.shape)

        run_descriptor_block(
            h5_path=h5_path,
            warp=warp,
            mode=mode,
            timestamp=timestamp,
            cfg=cfg,
            desc_key=desc,
            X=X,
            Y=Yemb,
            attrs=attrs,
            y_meta=y_meta,
            x_path=x_path,
        )

def run_descriptor_block(
    h5_path: Path,
    warp: str,
    mode: str,
    timestamp: str,
    cfg: Dict[str, Any],
    desc_key: str,
    X: np.ndarray,
    Y: np.ndarray,
    attrs: Dict[str, Any],
    y_meta: Dict[str, Any],
    x_path: str,
) -> None:
    device = torch.device(str(cfg["device"]))

    N, p = X.shape
    q = Y.shape[1]

    outdir = Path(cfg["out_root"]) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    base_name = f"{warp}_{mode}_{desc_key}"

    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    Y_t = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

    idxY_t, dY_t = knn_in_y_chunked(Y_t, k=int(cfg["kY"]), chunk=int(cfg["knn_chunk"]))

    metrics = local_explainedcov_metrics_batched_LOO(
        X=X_t,
        Y=Y_t,
        idxY=idxY_t,
        dY=dY_t,
        use_weights=bool(cfg["use_weights"]),
        eps_tau=float(cfg["eps_tau"]),
        ridge_y=float(cfg["ridge_y"]),
        ridge_x=float(cfg["ridge_x"]),
        eps_trace=float(cfg["eps_trace"]),
        batch_size=int(cfg["batch_size"]),
    )

    csv_path = outdir / f"{base_name}.csv"
    x_cols = [f"x_dim{j}" for j in range(p)]
    cols = x_cols + [
        "unexplained_frac",
        "explained_frac",
        "worst_unexplained_ratio",
        "worst_retention",
        "unexplained_coord_max",
        "trX",
        "trR",
        "avg_dY",
    ] + [f"unexplained_coord{j}" for j in range(p)]

    M = np.concatenate(
        [
            X,
            metrics["unexplained_frac"][:, None],
            metrics["explained_frac"][:, None],
            metrics["worst_unexplained_ratio"][:, None],
            metrics["worst_retention"][:, None],
            metrics["unexplained_coord_max"][:, None],
            metrics["trX"][:, None],
            metrics["trR"][:, None],
            metrics["avg_dY"][:, None],
            metrics["unexplained_coords"],
        ],
        axis=1,
    )
    np.savetxt(csv_path, M, delimiter=",", header=",".join(cols), comments="")

    xdim0 = int(cfg["xdim0"])
    xdim1 = int(cfg["xdim1"])
    if not (0 <= xdim0 < p and 0 <= xdim1 < p):
        raise ValueError(f"xdim0/xdim1 must be in [0,{p-1}] (got {xdim0},{xdim1})")
    x0 = X[:, xdim0]
    x1 = X[:, xdim1]
    xlabel = f"X_dim{xdim0}"
    ylabel = f"X_dim{xdim1}"

    expl = metrics["explained_frac"].astype(np.float64)
    zmin = float(np.nanmin(expl)) if np.isfinite(np.nanmin(expl)) else 0.0
    zmax = float(np.nanmax(expl)) if np.isfinite(np.nanmax(expl)) else 1.0
    if zmin == zmax:
        zmax = zmin + 1e-6

    scatter_2d(
        x0, x1, expl,
        xlabel, ylabel,
        f"CH explained fraction (adaptive) | {warp} | {mode} | Y={desc_key} | q={q}",
        outdir / f"{base_name}_scatter_adaptive",
        dpi=int(cfg["dpi"]),
        vmin=zmin, vmax=zmax,
    )
    heatmap_binned_2d(
        x0, x1, expl,
        xlabel, ylabel,
        f"CH explained fraction heatmap (adaptive) | {warp} | {mode} | Y={desc_key}",
        outdir / f"{base_name}_heatmap_adaptive",
        bins0=int(cfg["hm_bins0"]), bins1=int(cfg["hm_bins1"]),
        sigma_px=float(cfg["hm_sigma_px"]), dpi=int(cfg["dpi"]),
        vmin=zmin, vmax=zmax,
    )

    scatter_2d(
        x0, x1, expl,
        xlabel, ylabel,
        f"CH explained fraction (fixed [0,1]) | {warp} | {mode} | Y={desc_key} | q={q}",
        outdir / f"{base_name}_scatter_fixed01",
        dpi=int(cfg["dpi"]),
        vmin=0.0, vmax=1.0,
    )
    heatmap_binned_2d(
        x0, x1, expl,
        xlabel, ylabel,
        f"CH explained fraction heatmap (fixed [0,1]) | {warp} | {mode} | Y={desc_key}",
        outdir / f"{base_name}_heatmap_fixed01",
        bins0=int(cfg["hm_bins0"]), bins1=int(cfg["hm_bins1"]),
        sigma_px=float(cfg["hm_sigma_px"]), dpi=int(cfg["dpi"]),
        vmin=0.0, vmax=1.0,
    )

    summary = dict(
        input_file=str(h5_path.resolve()),
        timestamp=str(timestamp),
        warp=str(warp),
        mode=mode,
        descriptor=desc_key,
        N=int(N),
        p=int(p),
        y_dim=int(q),
        device=str(device),
        ds_Y=str(cfg["ds_Y"]),
        x_dataset_path=str(x_path),
        kY=int(cfg["kY"]),
        use_weights=bool(cfg["use_weights"]),
        ridge_y=float(cfg["ridge_y"]),
        ridge_x=float(cfg["ridge_x"]),
        eps_trace=float(cfg["eps_trace"]),
        plot_dims=dict(xdim0=int(xdim0), xdim1=int(xdim1)),
        stats=dict(
            expl_median=float(np.median(metrics["explained_frac"])),
            expl_q10=float(np.quantile(metrics["explained_frac"], 0.10)),
            expl_q90=float(np.quantile(metrics["explained_frac"], 0.90)),
            worst_ret_median=float(np.median(metrics["worst_retention"])),
        ),
    )
    meta_out = dict(config=cfg, summary=summary, h5_attrs=attrs, y_meta=y_meta)
    meta_path = outdir / f"{base_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(json_sanitize(meta_out), f, indent=2)


# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="CH injectivity analysis (LOO explained covariance) with fixed/repeats descriptors; unwarped X enforced.")
    p.add_argument("--ch_data_root", type=str, default=CFG["ch_data_root"])
    p.add_argument("--out_root", type=str, default=CFG["out_root"])
    p.add_argument("--h5", type=str, default=None, help="Optional explicit H5. If set, combo_key='manual'.")
    p.add_argument("--device", type=str, default=CFG["device"])
    p.add_argument("--max_N", type=int, default=None)

    p.add_argument("--ds_X", type=str, default=CFG["ds_X"], help="HDF5 dataset path for UNWARPED X, or 'auto'.")
    p.add_argument("--ds_Y", type=str, default=CFG["ds_Y"])

    p.add_argument("--descriptor", type=str, default=CFG["descriptor"], help="radial1d | corr2d | euler | all (repeats mode only)")
    p.add_argument("--downsample_fixed", type=int, default=CFG["downsample_fixed"])
    p.add_argument("--corr2d_downsample", type=int, default=CFG["corr2d_downsample"])
    p.add_argument("--n_radial_bins", type=int, default=CFG["n_radial_bins"])
    p.add_argument("--euler_radii", type=str, default="(0,1,2,4,8)")
    p.add_argument("--thresholds", type=str, default="(0.4,0.45,0.5,0.55,0.6)")

    p.add_argument("--kY", type=int, default=CFG["kY"])
    p.add_argument("--knn_chunk", type=int, default=CFG["knn_chunk"])
    p.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    p.add_argument("--use_weights", type=int, default=int(CFG["use_weights"]))

    p.add_argument("--ridge_y", type=float, default=CFG["ridge_y"])
    p.add_argument("--ridge_x", type=float, default=CFG["ridge_x"])

    p.add_argument("--standardize_X", type=int, default=int(CFG["standardize_X"]))
    p.add_argument("--standardize_Y", type=int, default=int(CFG["standardize_Y"]))

    p.add_argument("--xdim0", type=int, default=CFG["xdim0"])
    p.add_argument("--xdim1", type=int, default=CFG["xdim1"])
    p.add_argument("--dpi", type=int, default=CFG["dpi"])
    p.add_argument("--hm_bins0", type=int, default=CFG["hm_bins0"])
    p.add_argument("--hm_bins1", type=int, default=CFG["hm_bins1"])
    p.add_argument("--hm_sigma_px", type=float, default=CFG["hm_sigma_px"])

    p.add_argument("--seed", type=int, default=CFG["seed"])
    return p.parse_args()

def main():
    args = parse_args()
    cfg = dict(CFG)

    cfg["ch_data_root"] = args.ch_data_root
    cfg["out_root"] = args.out_root
    cfg["device"] = args.device
    cfg["max_N"] = args.max_N

    cfg["ds_X"] = str(args.ds_X)
    cfg["ds_Y"] = str(args.ds_Y)

    cfg["descriptor"] = str(args.descriptor).strip().lower()
    cfg["downsample_fixed"] = int(args.downsample_fixed)
    cfg["corr2d_downsample"] = int(args.corr2d_downsample)
    cfg["n_radial_bins"] = int(args.n_radial_bins)

    er_strs = str(args.euler_radii).replace("(", "").replace(")", "").split(",")
    cfg["euler_radii"] = tuple(int(s.strip()) for s in er_strs if s.strip())

    thr_strs = str(args.thresholds).replace("(", "").replace(")", "").split(",")
    cfg["thresholds"] = tuple(float(s.strip()) for s in thr_strs if s.strip())
    if len(cfg["thresholds"]) < 1:
        raise ValueError("thresholds must contain at least one value")

    cfg["kY"] = int(args.kY)
    cfg["knn_chunk"] = int(args.knn_chunk)
    cfg["batch_size"] = int(args.batch_size)
    cfg["use_weights"] = bool(int(args.use_weights))
    cfg["ridge_y"] = float(args.ridge_y)
    cfg["ridge_x"] = float(args.ridge_x)

    cfg["standardize_X"] = bool(int(args.standardize_X))
    cfg["standardize_Y"] = bool(int(args.standardize_Y))

    cfg["xdim0"] = int(args.xdim0)
    cfg["xdim1"] = int(args.xdim1)
    cfg["dpi"] = int(args.dpi)
    cfg["hm_bins0"] = int(args.hm_bins0)
    cfg["hm_bins1"] = int(args.hm_bins1)
    cfg["hm_sigma_px"] = float(args.hm_sigma_px)

    cfg["seed"] = int(args.seed)

    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    if args.h5 is not None:
        h5_path = Path(args.h5).expanduser().resolve()
        name = h5_path.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            warp, mode = parts
        else:
            warp, mode = "unknown", "unknown"
        timestamp = "manual"
        run_one_combo(h5_path, warp=warp, mode=mode, timestamp=timestamp, cfg=cfg)
        return

    ch_root = Path(cfg["ch_data_root"]).expanduser().resolve()
    timestamp_dir = find_most_recent_timestamp(ch_root)
    if timestamp_dir is None:
        raise FileNotFoundError(f"No timestamp directories found under {ch_root}")

    timestamp = timestamp_dir.name
    print(f"Processing most recent data from: {timestamp}")

    h5_files = discover_h5_files_in_timestamp(timestamp_dir)
    if not h5_files:
        raise FileNotFoundError(f"No *.h5 files found in {timestamp_dir}")

    for warp, mode, h5_path in h5_files:
        print(f"Processing {warp}_{mode}...")
        run_one_combo(h5_path, warp=warp, mode=mode, timestamp=timestamp, cfg=cfg)

if __name__ == "__main__":
    main()
