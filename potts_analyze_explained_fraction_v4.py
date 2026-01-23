#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction.py  (REWRITTEN)

Replaces the broken "inverse-from-Y neighborhoods" explained-fraction diagnostic with a
principled local forward linearization test aligned with controllability.

Core diagnostic (p=2 controls):
  - Build neighborhoods in control space X (kNN in X).
  - Fit local forward ridge: Y ~ a + B^T X on each neighborhood.
  - Score generalization using exact LOO multivariate R^2 (Frobenius energy).
  - Enforce 2D controllability: require two independent control directions in Y via
    canonical explained fractions (λ1 >= λ2) computed from local covariances.

Outputs:
  potts_analysis/<YYYYMMDD_HHMMSSZ>/<input_stem>/<descriptor_kind>/
    potts_local_linear_controllability.csv
    figs/*.png + *.pdf
    metadata_local_linear_controllability.json

No random projections. No random subsets. Deterministic given data and numeric libs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import h5py
import numpy as np
import torch
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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_latest_descriptor_h5(potts_data_dir: Path, descriptor: str) -> Path:
    """
    Searches potts_data/<timestamp>/<descriptor>/*.h5 and returns the latest by timestamp dir name.
    """
    candidates: List[Tuple[float, Path]] = []
    if not potts_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {potts_data_dir}")

    for run_dir in potts_data_dir.iterdir():
        if not run_dir.is_dir():
            continue
        desc_dir = run_dir / descriptor
        if not desc_dir.exists() or not desc_dir.is_dir():
            continue

        ts = _parse_run_dir_name(run_dir.name)
        for h5_file in desc_dir.glob("*.h5"):
            t = ts if ts is not None else h5_file.stat().st_mtime
            candidates.append((t, h5_file))

    if not candidates:
        raise FileNotFoundError(
            f"No descriptor files found for '{descriptor}' under {potts_data_dir}\n"
            f"Run potts_descriptors.py first with --descriptor {descriptor}"
        )

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


# ----------------------------- plotting helpers -----------------------------

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
    msk = _conv1d_reflect(mask, k, axis=1)
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


# ----------------------------- math / utilities -----------------------------

def make_corr2d_weight_np(H: int, W: int, power: float) -> np.ndarray:
    """
    Weight matrix w(dx,dy) = 1 / max(r,1)^power in fftshifted spatial-lag coordinates.
    """
    cy = H // 2
    cx = W // 2
    y, x = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    r = np.sqrt((x - float(cx)) ** 2 + (y - float(cy)) ** 2)
    r = np.maximum(r, 1.0)
    w = 1.0 / (r ** float(power))
    return w.astype(np.float32)


def standardize_np(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd


def knn_in_X_np(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact kNN in 2D (or low-dim) control space using full pairwise distances.
    Deterministic. O(N^2) but N~1e3 is fine.
    Returns:
      idx: (N,k) excluding self
      d:   (N,k)
    """
    X64 = X.astype(np.float64, copy=False)
    N = X64.shape[0]
    k = min(int(k), N - 1)

    diff = X64[:, None, :] - X64[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)

    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    row = np.arange(N)[:, None]
    dist = D[row, idx]
    ords = np.argsort(dist, axis=1)
    idx = idx[row, ords]
    dist = dist[row, ords]

    return idx.astype(np.int64), dist.astype(np.float64)


@torch.no_grad()
def inv_sqrt_2x2_batch(S: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Inverse square root of SPD 2x2 matrices. S: (B,2,2) float64/float32.
    Returns: (B,2,2) same dtype.
    """
    w, V = torch.linalg.eigh(S)
    w = torch.clamp(w, min=float(eps))
    inv_sqrt_w = torch.diag_embed(1.0 / torch.sqrt(w))
    return V @ inv_sqrt_w @ V.transpose(1, 2)


# ----------------------------- descriptor loading -----------------------------

def load_descriptors_and_build_Y(
    desc_h5: Path,
    prepend_phase_fractions_to_Y: bool,
    corr2d_weight_power: float,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """
    Loads descriptor H5 and returns:
      X: (N,2) [temperature, fraction_initial]
      Y: (N,D) flattened descriptor (radial or corr2d), optionally prefixed by phase fractions
      descriptor_kind: 'radial1d' or 'corr2d'
      metadata
    """
    print(f"[potts_analyze] Loading descriptors from {desc_h5}...")

    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)

        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))

        mean2d = np.array(f["correlations/correlations_2d_mean"], dtype=np.float32)
        mean1d = np.array(f["correlations/correlations_radial_mean"], dtype=np.float32)
        meanph = np.array(f["phases/final_fraction_mean"], dtype=np.float32)

    X = np.stack([temps, fracs], axis=1).astype(np.float32)

    if descriptor == "radial1d":
        feat = mean1d.reshape(N, -1)
    else:
        # Apply radial weighting to corr2d
        # mean2d can be (N,H,W) or (N,C,H,W) - use last two dimensions for spatial
        H, W = mean2d.shape[-2], mean2d.shape[-1]
        if corr2d_weight_power > 0.0:
            weight = make_corr2d_weight_np(H, W, power=corr2d_weight_power)  # (H,W)
            # Broadcast weight to match mean2d shape
            weight_bc = weight.reshape((1,) * (mean2d.ndim - 2) + (H, W))
            feat = (mean2d * weight_bc).reshape(N, -1)
        else:
            feat = mean2d.reshape(N, -1)

    if prepend_phase_fractions_to_Y:
        Y = np.concatenate([meanph, feat], axis=1).astype(np.float32, copy=False)
    else:
        Y = feat.astype(np.float32, copy=False)

    meta = {
        "descriptor": descriptor,
        "q": q,
        "N": N,
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "corr2d_weight_power": float(corr2d_weight_power) if descriptor == "corr2d" else None,
    }

    print(f"[potts_analyze] Loaded: N={N}, descriptor={descriptor}, Y_dim={Y.shape[1]}")
    return X, Y, descriptor, meta


# ----------------------------- local forward linear controllability -----------------------------

@torch.no_grad()
def local_forward_linear_controllability(
    X_use: np.ndarray,             # (N,2) float32
    Y_use: np.ndarray,             # (N,D) float32
    idxX: np.ndarray,              # (N,k) int64
    dX: np.ndarray,                # (N,k) float64
    ridge: float,
    eps: float,
    batch_size: int,
    device: torch.device,
    y_on_gpu: bool,
) -> Dict[str, np.ndarray]:
    """
    For each i, neighborhood J_i = {i} ∪ kNN_X(i).

    Fits forward ridge: Y ~ a + B^T X with intercept unpenalized.
    Computes:
      - R2_loo (multivariate Frobenius R^2 via exact LOO using hat diagonal)
      - SST, SSE_loo
      - lambda1, lambda2 (canonical explained fractions of Y energy attributable to 2 control directions)
      - balance = 2*lambda2/(lambda1+lambda2)
      - score = max(0,R2_loo) * balance
      - avg_dX

    Returns numpy arrays length N.
    """
    X_t = torch.as_tensor(X_use, device=device, dtype=torch.float64)

    Y_cpu = torch.as_tensor(Y_use, device="cpu", dtype=torch.float32)
    if y_on_gpu:
        Y_t = Y_cpu.to(device=device, non_blocking=False)
    else:
        Y_t = Y_cpu  # stays on CPU; will be moved per-batch

    N, p = X_use.shape
    assert p == 2, "This diagnostic assumes p=2 controls."
    k = int(idxX.shape[1])
    k1 = k + 1

    # outputs
    R2_loo = np.empty((N,), dtype=np.float64)
    SST = np.empty((N,), dtype=np.float64)
    SSE_loo = np.empty((N,), dtype=np.float64)

    lam1 = np.empty((N,), dtype=np.float64)
    lam2 = np.empty((N,), dtype=np.float64)
    lam_sum = np.empty((N,), dtype=np.float64)
    balance = np.empty((N,), dtype=np.float64)
    score = np.empty((N,), dtype=np.float64)

    hmax = np.empty((N,), dtype=np.float64)
    avg_dx = dX.mean(axis=1).astype(np.float64)

    # ridge penalty matrix for [1, x1, x2] with no penalty on intercept
    R = torch.diag(torch.tensor([0.0, float(ridge), float(ridge)], device=device, dtype=torch.float64))  # (3,3)

    # batching over anchors
    bs = max(1, int(batch_size))
    for i0 in range(0, N, bs):
        i1 = min(N, i0 + bs)
        B = i1 - i0

        centers = np.arange(i0, i1, dtype=np.int64)
        neigh = np.empty((B, k1), dtype=np.int64)
        neigh[:, 0] = centers
        neigh[:, 1:] = idxX[i0:i1]

        neigh_t = torch.as_tensor(neigh, device=device, dtype=torch.int64)  # (B,k1)

        Xn = X_t[neigh_t]  # (B,k1,2) float64
        ones = torch.ones((B, k1, 1), device=device, dtype=torch.float64)
        Z = torch.cat([ones, Xn], dim=2)  # (B,k1,3) float64

        # fetch Y neighborhood
        if y_on_gpu:
            Yn = Y_t.index_select(0, neigh_t.reshape(-1)).reshape(B, k1, -1).to(dtype=torch.float64)
        else:
            # gather on CPU then move
            Yn_cpu = Y_t.index_select(0, neigh_t.reshape(-1).to("cpu")).reshape(B, k1, -1)
            Yn = Yn_cpu.to(device=device, dtype=torch.float64, non_blocking=False)

        D = int(Yn.shape[2])

        # ---- LOO multivariate R^2 for forward ridge ----
        # A = Z^T Z + R
        Zt = Z.transpose(1, 2)                               # (B,3,k1)
        ZTZ = torch.bmm(Zt, Z)                               # (B,3,3)
        A = ZTZ + R[None, :, :]                              # (B,3,3)
        Ainv = torch.linalg.inv(A)                           # (B,3,3)

        # T = Z^T Y, U = Ainv T, Pred = Z U
        T = torch.bmm(Zt, Yn)                                # (B,3,D)
        U = torch.bmm(Ainv, T)                               # (B,3,D)
        Pred = torch.bmm(Z, U)                               # (B,k1,D)
        Res = Yn - Pred                                      # (B,k1,D)

        # hat diagonal h = diag(Z Ainv Z^T)
        ZA = torch.bmm(Z, Ainv)                              # (B,k1,3)
        h = torch.sum(ZA * Z, dim=2).clamp(min=-1e6, max=1e6) # (B,k1)
        den = (1.0 - h).clamp_min(1e-8)                      # (B,k1)

        # row residual norms in Y
        res2 = torch.sum(Res * Res, dim=2)                   # (B,k1)
        Eloo2 = res2 / (den * den)                            # (B,k1)
        sse = torch.sum(Eloo2, dim=1)                        # (B,)

        # SST = sum ||y_t||^2 - ||sum y_t||^2 / k1
        Ysum = torch.sum(Yn, dim=1)                          # (B,D)
        Ysum_sq = torch.sum(Ysum * Ysum, dim=1)              # (B,)
        Ysq_sum = torch.sum(Yn * Yn, dim=(1, 2))             # (B,)
        sst = (Ysq_sum - (Ysum_sq / float(k1))).clamp_min(0.0) + float(eps)

        r2 = 1.0 - (sse / sst)

        # ---- 2D controllability via canonical explained fractions ----
        # Xc = X - meanX. Then TT = Xc^T Yn (since sum Xc = 0) gives same as Xc^T Yc.
        Xmean = torch.mean(Xn, dim=1, keepdim=True)          # (B,1,2)
        Xc = Xn - Xmean                                      # (B,k1,2)

        Sxx = torch.bmm(Xc.transpose(1, 2), Xc)              # (B,2,2)
        Axs = inv_sqrt_2x2_batch(Sxx, eps=float(eps))        # (B,2,2)

        TT = torch.bmm(Xc.transpose(1, 2), Yn)               # (B,2,D)
        G = torch.bmm(TT, TT.transpose(1, 2))                # (B,2,2)

        M = torch.bmm(Axs, torch.bmm(G, Axs)) / sst[:, None, None]  # (B,2,2)
        w = torch.linalg.eigvalsh(0.5 * (M + M.transpose(1, 2)))    # (B,2), ascending
        l2 = torch.clamp(w[:, 0], 0.0, 1.0)
        l1 = torch.clamp(w[:, 1], 0.0, 1.0)
        lsum = torch.clamp(l1 + l2, 0.0, 1.0)

        bal = torch.clamp((2.0 * l2) / (lsum + float(eps)), 0.0, 1.0)
        scr = torch.clamp(r2, min=0.0) * bal

        # ---- store ----
        R2_loo[i0:i1] = r2.detach().cpu().numpy()
        SST[i0:i1] = sst.detach().cpu().numpy()
        SSE_loo[i0:i1] = sse.detach().cpu().numpy()

        lam1[i0:i1] = l1.detach().cpu().numpy()
        lam2[i0:i1] = l2.detach().cpu().numpy()
        lam_sum[i0:i1] = lsum.detach().cpu().numpy()
        balance[i0:i1] = bal.detach().cpu().numpy()
        score[i0:i1] = scr.detach().cpu().numpy()

        hmax[i0:i1] = torch.max(h, dim=1).values.detach().cpu().numpy()

        # memory hygiene
        del Xn, Z, Yn, Pred, Res, ZA, h, den, res2, Eloo2, sse, Ysum, Ysum_sq, Ysq_sum, sst, r2, Xc, Sxx, Axs, TT, G, M, w, l1, l2, lsum, bal, scr

        if device.type == "cuda":
            torch.cuda.empty_cache()

    out: Dict[str, np.ndarray] = dict(
        R2_loo=R2_loo.astype(np.float32),
        SST=SST.astype(np.float32),
        SSE_loo=SSE_loo.astype(np.float32),
        lambda1=lam1.astype(np.float32),
        lambda2=lam2.astype(np.float32),
        lambda_sum=lam_sum.astype(np.float32),
        balance=balance.astype(np.float32),
        score=score.astype(np.float32),
        avg_dX=avg_dx.astype(np.float32),
        hmax=hmax.astype(np.float32),
    )
    return out


# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class Config:
    potts_data_dir: str = "potts_data"
    potts_analysis_dir: str = "potts_analysis"
    h5: str = ""  # if empty, auto-find latest under potts_data/<timestamp>/<descriptor>/

    descriptor: DescKind = "corr2d"
    prepend_phase_fractions_to_Y: bool = True
    corr2d_weight_power: float = 0.0  # w = 1/max(r,1)^power for corr2d weighting

    # Local forward linearization knobs
    standardize_X: bool = False     # recommended for kNN in X + conditioning
    standardize_Y: bool = False     # recommended to avoid amplitude dominating variance energy
    kX: int = 15                  # neighborhood size in X (excluding self)

    ridge: float = 1e-2            # slope ridge (intercept unpenalized)
    eps: float = 1e-18

    batch_size: int = 8
    device: str = "cuda"

    # Device residency control
    y_on_gpu: bool = True
    max_y_gb: float = 8.0          # if y_on_gpu, only move Y to GPU if Y_bytes <= max_y_gb

    # Plot controls
    dpi: int = 250
    save_scatter: bool = True
    save_heatmaps: bool = True

    hm_bins_temp: int = 60
    hm_bins_frac: int = 60
    hm_sigma_px: float = 1.0
    hm_clip: Tuple[float, float] = (1.0, 99.0)


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Potts descriptors for local forward linear controllability")

    ap.add_argument("--h5", type=str, default=Config.h5, help="Input descriptor HDF5. If empty, uses latest.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"])

    ap.add_argument("--prepend_phase_fractions_to_Y", action="store_true")
    ap.add_argument("--no_prepend_phase_fractions_to_Y", dest="prepend_phase_fractions_to_Y", action="store_false")
    ap.set_defaults(prepend_phase_fractions_to_Y=Config.prepend_phase_fractions_to_Y)

    ap.add_argument("--corr2d_weight_power", type=float, default=Config.corr2d_weight_power,
                    help="Radial weighting power for corr2d: w = 1/max(r,1)^power")

    ap.add_argument("--standardize_X", action="store_true")
    ap.add_argument("--no_standardize_X", dest="standardize_X", action="store_false")
    ap.set_defaults(standardize_X=Config.standardize_X)

    ap.add_argument("--standardize_Y", action="store_true")
    ap.add_argument("--no_standardize_Y", dest="standardize_Y", action="store_false")
    ap.set_defaults(standardize_Y=Config.standardize_Y)

    ap.add_argument("--kX", type=int, default=Config.kX)
    ap.add_argument("--ridge", type=float, default=Config.ridge)
    ap.add_argument("--eps", type=float, default=Config.eps)

    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--device", type=str, default=Config.device)

    ap.add_argument("--y_on_gpu", action="store_true")
    ap.add_argument("--no_y_on_gpu", dest="y_on_gpu", action="store_false")
    ap.set_defaults(y_on_gpu=Config.y_on_gpu)
    ap.add_argument("--max_y_gb", type=float, default=Config.max_y_gb)

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
        h5=str(args.h5),
        descriptor=str(args.descriptor),  # type: ignore
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),
        corr2d_weight_power=float(args.corr2d_weight_power),
        standardize_X=bool(args.standardize_X),
        standardize_Y=bool(args.standardize_Y),
        kX=int(args.kX),
        ridge=float(args.ridge),
        eps=float(args.eps),
        batch_size=int(args.batch_size),
        device=str(args.device),
        y_on_gpu=bool(args.y_on_gpu),
        max_y_gb=float(args.max_y_gb),
        dpi=int(args.dpi),
        save_scatter=not bool(args.no_scatter),
        save_heatmaps=not bool(args.no_heatmaps),
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        hm_clip=(float(args.hm_clip_lo), float(args.hm_clip_hi)),
    )

    # Find descriptor H5 file
    if str(cfg.h5).strip():
        desc_h5 = Path(str(cfg.h5)).expanduser().resolve()
    else:
        desc_h5 = find_latest_descriptor_h5(Path(cfg.potts_data_dir), str(cfg.descriptor))

    print(f"[potts_analyze] Input: {desc_h5}")

    # Load descriptors and build X,Y
    X, Y, descriptor_kind, load_meta = load_descriptors_and_build_Y(
        desc_h5, 
        cfg.prepend_phase_fractions_to_Y,
        cfg.corr2d_weight_power,
    )

    # Standardize
    X_use = X.copy()
    Y_use = Y.copy()
    norm_meta: Dict[str, Any] = {}

    if cfg.standardize_X:
        X_use, muX, sdX = standardize_np(X_use)
        norm_meta["X_mu"] = muX.astype(np.float64).tolist()
        norm_meta["X_sd"] = sdX.astype(np.float64).tolist()
    else:
        norm_meta["X_mu"] = [0.0, 0.0]
        norm_meta["X_sd"] = [1.0, 1.0]

    if cfg.standardize_Y:
        Y_use, muY, sdY = standardize_np(Y_use)
        # avoid dumping huge vectors; store scalars about scale
        norm_meta["Y_sd_median"] = float(np.median(sdY.astype(np.float64)))
        norm_meta["Y_sd_q10"] = float(np.quantile(sdY.astype(np.float64), 0.10))
        norm_meta["Y_sd_q90"] = float(np.quantile(sdY.astype(np.float64), 0.90))
    else:
        norm_meta["Y_sd_median"] = 1.0
        norm_meta["Y_sd_q10"] = 1.0
        norm_meta["Y_sd_q90"] = 1.0

    # Build neighborhoods in X
    print(f"[potts_analyze] Finding kNN in X with k={cfg.kX}...")
    idxX, dX = knn_in_X_np(X_use, k=int(cfg.kX))

    # Device
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")
    print(f"[potts_analyze] device={device}")

    # Decide whether to place Y on GPU
    y_bytes = int(Y_use.nbytes)
    y_gb = float(y_bytes) / (1024.0 ** 3)
    y_on_gpu = bool(cfg.y_on_gpu) and (device.type == "cuda") and (y_gb <= float(cfg.max_y_gb))
    print(f"[potts_analyze] Y size ~{y_gb:.3f} GB; y_on_gpu={y_on_gpu}")

    # Compute metrics
    print("[potts_analyze] Computing local forward linear controllability metrics...")
    metrics = local_forward_linear_controllability(
        X_use=X_use.astype(np.float32, copy=False),
        Y_use=Y_use.astype(np.float32, copy=False),
        idxX=idxX,
        dX=dX,
        ridge=float(cfg.ridge),
        eps=float(cfg.eps),
        batch_size=int(cfg.batch_size),
        device=device,
        y_on_gpu=y_on_gpu,
    )

    # Output directories
    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    root_desc = ensure_dir(session_dir / descriptor_kind)
    figs_dir = ensure_dir(root_desc / "figs")

    # Save CSV
    csv_path = root_desc / "potts_local_linear_controllability.csv"
    header_cols = [
        "temperature", "fraction_initial",
        "R2_loo",
        "lambda1", "lambda2", "lambda_sum",
        "balance",
        "score",
        "avg_dX",
        "SST", "SSE_loo",
        "hmax",
    ]
    cols = [
        X[:, 0], X[:, 1],
        metrics["R2_loo"],
        metrics["lambda1"], metrics["lambda2"], metrics["lambda_sum"],
        metrics["balance"],
        metrics["score"],
        metrics["avg_dX"],
        metrics["SST"], metrics["SSE_loo"],
        metrics["hmax"],
    ]
    data = np.column_stack(cols).astype(np.float64, copy=False)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

    # Plots
    print("[potts_analyze] Generating plots...")
    temp = X[:, 0].astype(np.float64, copy=False)
    frac = X[:, 1].astype(np.float64, copy=False)

    R2_pos = np.clip(metrics["R2_loo"].astype(np.float64), 0.0, 1.0)
    balance = np.clip(metrics["balance"].astype(np.float64), 0.0, 1.0)
    score = np.clip(metrics["score"].astype(np.float64), 0.0, 1.0)
    lam2 = np.clip(metrics["lambda2"].astype(np.float64), 0.0, 1.0)
    lam_sum = np.clip(metrics["lambda_sum"].astype(np.float64), 0.0, 1.0)

    if cfg.save_scatter:
        scatter_tf(temp, frac, R2_pos,
                   f"{descriptor_kind}: forward LOO R^2 (clipped) for local linearization",
                   figs_dir / "scatter_R2loo_clipped", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, lam_sum,
                   f"{descriptor_kind}: total explainable Y-energy by 2D control span (lambda1+lambda2)",
                   figs_dir / "scatter_lambda_sum", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, lam2,
                   f"{descriptor_kind}: 2nd-direction explained fraction lambda2 (2D controllability gate)",
                   figs_dir / "scatter_lambda2", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, balance,
                   f"{descriptor_kind}: balance = 2*lambda2/(lambda1+lambda2)",
                   figs_dir / "scatter_balance", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, score,
                   f"{descriptor_kind}: score = max(0,R2_loo) * balance",
                   figs_dir / "scatter_score", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, np.log10(metrics["SST"].astype(np.float64) + 1e-30),
                   f"{descriptor_kind}: log10 local Y energy SST",
                   figs_dir / "scatter_log10_SST", dpi=cfg.dpi)

    if cfg.save_heatmaps:
        heatmap_binned_tf(temp, frac, R2_pos,
                          f"{descriptor_kind}: heatmap forward LOO R^2 (clipped)",
                          figs_dir / "heatmap_R2loo_clipped",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, lam2,
                          f"{descriptor_kind}: heatmap lambda2 (2D controllability gate)",
                          figs_dir / "heatmap_lambda2",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, score,
                          f"{descriptor_kind}: heatmap score",
                          figs_dir / "heatmap_score",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

    # Metadata JSON
    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor_kind,
        N=int(X.shape[0]),
        y_dim=int(Y_use.shape[1]),
        kX=int(cfg.kX),
        ridge=float(cfg.ridge),
        standardize_X=bool(cfg.standardize_X),
        standardize_Y=bool(cfg.standardize_Y),
        y_on_gpu=bool(y_on_gpu),
        stats=dict(
            score_median=float(np.median(score)),
            score_q10=float(np.quantile(score, 0.10)),
            score_q90=float(np.quantile(score, 0.90)),
            R2loo_median=float(np.median(metrics["R2_loo"].astype(np.float64))),
            R2loo_q10=float(np.quantile(metrics["R2_loo"].astype(np.float64), 0.10)),
            R2loo_q90=float(np.quantile(metrics["R2_loo"].astype(np.float64), 0.90)),
            lambda2_median=float(np.median(lam2)),
            lambda2_q10=float(np.quantile(lam2, 0.10)),
            lambda2_q90=float(np.quantile(lam2, 0.90)),
        ),
        normalization=norm_meta,
    )

    out_meta = dict(
        created_utc=_utc_now_z(),
        config=asdict(cfg),
        files=dict(
            csv=str(csv_path),
            figs=str(figs_dir),
        ),
        summary=summary,
        load_metadata=load_meta,
    )

    meta_path = root_desc / "metadata_local_linear_controllability.json"
    meta_path.write_text(json.dumps(out_meta, indent=2))
    print(f"[potts_analyze] wrote: {meta_path}")
    print("[potts_analyze] Done!")


if __name__ == "__main__":
    main()
