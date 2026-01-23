#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction_laplacian.py

Analyzes Potts descriptors for injectivity using neighborhood KRR with
graph-Laplacian regularization in *sample space*, and correct LOO for
linear smoothers.

Key change vs baseline:
  Solve (K + lam I + eta L) alpha = Xs, with Xhat = K alpha,
  where L is a Laplacian built from pairwise distances within the k-point neighborhood.
  Compute LOO residuals via:
      Rloo[t,:] = (Xs[t,:] - Xhat[t,:]) / (1 - S_tt),
  where S = K (K + lam I + eta L)^(-1).

Outputs:
  potts_analysis/<YYYYMMDD_HHMMSSZ>/<input_stem>/<descriptor_kind>/
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
from typing import Any, Dict, List, Literal, Tuple

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt


DescKind = Literal["radial1d", "corr2d"]
LapSigmaMode = Literal["median", "tau"]


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


# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class Config:
    potts_data_dir: str = "potts_data"
    potts_analysis_dir: str = "potts_analysis"
    h5: str = "potts_data/20260123_161715Z/corr2d/potts_sims_q3_128x128_corr2d.h5"
    descriptor: str = "corr2d"

    prepend_phase_fractions_to_Y: bool = True

    # Injectivity metric knobs
    standardize_X: bool = True
    standardize_Y: bool = True

    kY: int = 15

    use_weights: bool = False
    eps_tau: float = 1e-10

    # Kernel / regularization scales
    ridge_y: float = 1e-3
    ridge_x: float = 1e-8
    eps_trace: float = 1e-18

        # ---------------- Graph-Laplacian regularization (sample-space) ----------------
    compare_laplacian: bool = True          # compute both baseline and laplacian columns
    use_laplacian: bool = True              # if False, only baseline is computed (even if compare=True)

    # eta selection:
    #   lap_eta_rel >= 0: fixed eta = lap_eta_rel * (trK/k)
    #   lap_eta_rel < 0:  auto-eta by df targeting (tr(S) <= df_target)
    lap_eta_rel: float = -1.0
    lap_df_target: float = 4.0
    lap_bisect_iters: int = 12
    lap_eta_min_rel: float = 1e-6
    lap_eta_max_rel: float = 1e6

    # Laplacian construction
    lap_sigma_mode: str = "median"          # "median" or "tau"
    lap_normalized: bool = True
    lap_w_floor_rel: float = 1e-3           # ensures connectivity; relative to mean(W_offdiag)

    # IMPORTANT: build graph on RAW (unstandardized) Y to avoid noise-geometry issues
    lap_graph_on_raw_Y: bool = True


    batch_size: int = 8
    device: str = "cuda"

    # Plot controls
    dpi: int = 250
    save_scatter: bool = True
    save_heatmaps: bool = True

    hm_bins_temp: int = 60
    hm_bins_frac: int = 60
    hm_sigma_px: float = 1.0
    hm_clip: Tuple[float, float] = (1.0, 99.0)


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


# ----------------------------- injectivity core -----------------------------

def standardize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def to_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute k nearest neighbors in Y (brute force)."""
    device = Y.device
    N = Y.shape[0]
    k = min(int(k), N - 1)

    Dc = torch.cdist(Y, Y, p=2.0)  # (N,N)
    Dc.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(Dc, k=k, largest=False, sorted=True)
    return idx, vals


def _offdiag_median(D: torch.Tensor) -> torch.Tensor:
    # D: (k,k)
    k = D.shape[0]
    mask = ~torch.eye(k, device=D.device, dtype=torch.bool)
    v = D[mask]
    v_sorted, _ = torch.sort(v)
    return v_sorted[v_sorted.numel() // 2].clamp_min(1e-12)


@torch.no_grad()
def _build_laplacian_from_Y(
    Yg: torch.Tensor,        # (k,q) float32
    tau: float,
    sigma_mode: str,
    normalized: bool,
    w_floor_rel: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      L: (k,k) float64 Laplacian (normalized if requested)
      lambda2: float64 second-smallest eigenvalue (algebraic connectivity proxy)
    """
    k = Yg.shape[0]
    D = torch.cdist(Yg, Yg, p=2.0)  # (k,k) float32

    if sigma_mode == "tau":
        sigma = float(max(tau, 1e-12))
        sigma_t = torch.tensor(sigma, device=Yg.device, dtype=Yg.dtype)
    else:
        sigma_t = _offdiag_median(D).to(Yg.dtype)

    W = torch.exp(-0.5 * (D / sigma_t).pow(2))
    W.fill_diagonal_(0.0)

    # Floor weights to avoid (near-)disconnected graphs in noisy / anisotropic neighborhoods.
    # This is the common failure mode that makes Laplacian regularization ineffective exactly where you want it.
    mask = ~torch.eye(k, device=Yg.device, dtype=torch.bool)
    w_mean = W[mask].mean().clamp_min(1e-12)
    w_floor = float(w_floor_rel) * float(w_mean.item())
    if w_floor > 0.0:
        W = W + w_floor
        W.fill_diagonal_(0.0)

    deg = W.sum(dim=1).clamp_min(1e-12)

    if normalized:
        inv_sqrt = torch.rsqrt(deg)
        S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
        L = torch.eye(k, device=Yg.device, dtype=Yg.dtype) - S
    else:
        L = torch.diag(deg) - W

    L = 0.5 * (L + L.T)
    ev = torch.linalg.eigvalsh(L.to(torch.float64))
    lambda2 = ev[1] if k >= 2 else ev[0]
    return L.to(torch.float64), lambda2


@torch.no_grad()
def _choose_eta_by_df(
    K: torch.Tensor,      # (k,k) float64
    L: torch.Tensor,      # (k,k) float64
    lam: float,
    df_target: float,
    eta_min: float,
    eta_max: float,
    iters: int,
) -> Tuple[float, float]:
    """
    Choose eta so that df(eta)=tr(K (K+lam I+eta L)^(-1)) <= df_target.
    Returns (eta, df_at_eta).
    """
    k = K.shape[0]
    I = torch.eye(k, device=K.device, dtype=K.dtype)

    def df_of(eta: float) -> float:
        A = K + float(lam) * I + float(eta) * L
        Ainv = torch.linalg.solve(A, I)
        S = K @ Ainv
        return float(torch.trace(S).clamp_min(0.0).item())

    df0 = df_of(0.0)
    if df0 <= df_target:
        return 0.0, df0

    # bracket
    hi = float(max(eta_min, 1e-30))
    df_hi = df_of(hi)
    while df_hi > df_target and hi < eta_max:
        hi *= 10.0
        df_hi = df_of(hi)

    if df_hi > df_target:
        return float(eta_max), df_hi

    lo = float(max(eta_min, 1e-30))
    for _ in range(int(iters)):
        mid = math.sqrt(lo * hi)
        df_mid = df_of(mid)
        if df_mid > df_target:
            lo = mid
        else:
            hi = mid
            df_hi = df_mid

    return hi, df_hi



def _pairwise_offdiag_median(D: torch.Tensor) -> torch.Tensor:
    """
    D: (k,k) or (B,k,k)
    returns median of off-diagonal entries along last two dims.
    """
    if D.ndim == 2:
        k = D.shape[0]
        mask = ~torch.eye(k, device=D.device, dtype=torch.bool)
        v = D[mask]
        v_sorted, _ = torch.sort(v)
        return v_sorted[v_sorted.numel() // 2].clamp_min(1e-12)
    elif D.ndim == 3:
        B, k, _ = D.shape
        mask = ~torch.eye(k, device=D.device, dtype=torch.bool)[None, :, :]
        v = D[mask].reshape(B, k * (k - 1))
        v_sorted, _ = torch.sort(v, dim=1)
        return v_sorted[:, v_sorted.shape[1] // 2].clamp_min(1e-12)
    else:
        raise ValueError("D must be (k,k) or (B,k,k)")


def _choose_eta_by_df_target(
    K: torch.Tensor,         # (k,k) float64
    L: torch.Tensor,         # (k,k) float64
    lam: float,
    df_target: float,
    eta_min: float,
    eta_max: float,
    iters: int,
) -> Tuple[float, float]:
    """
    Choose eta so that df = tr( K (K + lam I + eta L)^(-1) ) <= df_target.
    Uses log-bisection; returns (eta, df_at_eta).
    Assumes df decreases with eta (true for PSD L in practice here).
    """
    k = K.shape[0]
    I = torch.eye(k, device=K.device, dtype=K.dtype)

    def df_of(eta: float) -> float:
        A = K + lam * I + float(eta) * L
        Hinv = torch.linalg.solve(A, I)
        S = K @ Hinv
        return float(torch.trace(S).clamp_min(0.0).item())

    df0 = df_of(0.0)
    if df0 <= df_target:
        return 0.0, df0

    # Ensure upper bound achieves df <= target
    hi = float(max(eta_min, 1e-30))
    df_hi = df_of(hi)
    while df_hi > df_target and hi < eta_max:
        hi *= 10.0
        df_hi = df_of(hi)
    if df_hi > df_target:
        return float(eta_max), df_hi

    lo = float(max(eta_min, 1e-30))
    for _ in range(int(iters)):
        mid = math.sqrt(lo * hi)
        df_mid = df_of(mid)
        if df_mid > df_target:
            lo = mid
        else:
            hi = mid
            df_hi = df_mid
    return hi, df_hi


@torch.no_grad()
def local_explainedcov_metrics_LOO_laplacian(
    X: torch.Tensor,          # (N,p)
    Y_kernel: torch.Tensor,   # (N,qy) used for K
    Y_graph: torch.Tensor,    # (N,qg) used for graph distances (often RAW Y)
    idxY: torch.Tensor,       # (N,kY)
    dY: torch.Tensor,         # (N,kY)
    cfg: Config,
    batch_size: int = 256,
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
    unexpl_coord_0 = torch.empty((N,), device=device, dtype=torch.float32)
    unexpl_coord_1 = torch.empty((N,), device=device, dtype=torch.float32) if p >= 2 else None

    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    lap_eta = torch.empty((N,), device=device, dtype=torch.float32)
    lap_df = torch.empty((N,), device=device, dtype=torch.float32)
    lap_lambda2 = torch.empty((N,), device=device, dtype=torch.float32)

    I_k32 = torch.eye(k, device=device, dtype=torch.float32)
    I_k64 = torch.eye(k, device=device, dtype=torch.float64)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

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

        Xn = X[neigh]                 # (B,k,p)
        YnK = Y_kernel[neigh]         # (B,k,q)
        YnG = Y_graph[neigh]          # (B,k,qg)

        if cfg.use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(cfg.eps_tau))
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            tau = dn.max(dim=1).values.clamp_min(float(cfg.eps_tau))
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)  # (B,k)
        sw = torch.sqrt(w).to(torch.float32)

        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * YnK).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Yc = YnK.to(torch.float32) - muY[:, None, :]

        Xs = Xc * sw[:, :, None]     # (B,k,p)
        Ys = Yc * sw[:, :, None]     # (B,k,q)

        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))          # (B,k,k) float32
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(cfg.ridge_y) * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k32[None, :, :]

        # Per-neighborhood loop: add eta L and use correct LOO for linear smoother
        for b in range(B):
            Kb32 = Kmat[b]
            Kb = Kb32.to(torch.float64)
            lam_b = float(lam[b].item())
            Kreg_b = Kreg[b].to(torch.float64)

            # Laplacian from graph-space Y
            Lb, l2 = _build_laplacian_from_Y(
                Yg=YnG[b].to(torch.float32),
                tau=float(tau[b].item()),
                sigma_mode=str(cfg.lap_sigma_mode),
                normalized=bool(cfg.lap_normalized),
                w_floor_rel=float(cfg.lap_w_floor_rel),
            )
            lap_lambda2[i0 + b] = float(l2)

            base = float((trK[b].item() / float(k)) if trK[b].item() > 0 else 1.0)

            if cfg.lap_eta_rel >= 0.0:
                eta_b = float(cfg.lap_eta_rel) * base
                A = Kreg_b + eta_b * Lb
                Ainv = torch.linalg.solve(A, I_k64)
                S = Kb @ Ainv
                df_b = float(torch.trace(S).clamp_min(0.0).item())
            else:
                df_target = float(max(2.0, min(cfg.lap_df_target, float(k - 1))))
                eta_min = float(cfg.lap_eta_min_rel) * base
                eta_max = float(cfg.lap_eta_max_rel) * base
                eta_b, df_b = _choose_eta_by_df(
                    K=Kb, L=Lb, lam=lam_b,
                    df_target=df_target,
                    eta_min=eta_min, eta_max=eta_max,
                    iters=int(cfg.lap_bisect_iters),
                )
                A = Kreg_b + eta_b * Lb
                Ainv = torch.linalg.solve(A, I_k64)
                S = Kb @ Ainv

            lap_eta[i0 + b] = float(eta_b)
            lap_df[i0 + b] = float(df_b)

            Xsb = Xs[b].to(torch.float64)   # (k,p)

            # alpha = A^{-1} Xs
            alpha = Ainv @ Xsb              # (k,p)

            # predictions use K (not A)
            Xhat = Kb @ alpha               # (k,p)

            # correct LOO for linear smoother
            sdiag = torch.diagonal(S).clamp_min(-1e6).clamp_max(1.0 - 1e-6)
            denom = (1.0 - sdiag).clamp_min(1e-6)          # (k,)
            Rloo = (Xsb - Xhat) / denom[:, None]           # (k,p)

            trX_b = (Xsb * Xsb).sum().clamp_min(0.0)
            trR_b = (Rloo * Rloo).sum().clamp_min(0.0)

            u = (trR_b / (trX_b + float(cfg.eps_trace))).clamp(0.0, 1.0)
            e = (1.0 - u).clamp(0.0, 1.0)

            trX[i0 + b] = trX_b.to(torch.float32)
            trR[i0 + b] = trR_b.to(torch.float32)
            unexpl[i0 + b] = u.to(torch.float32)
            expl[i0 + b] = e.to(torch.float32)

            varX = (Xsb * Xsb).sum(dim=0)                  # (p,)
            varR = (Rloo * Rloo).sum(dim=0)                # (p,)
            ucoord = (varR / (varX + float(cfg.eps_trace))).clamp(0.0, 1.0)

            unexpl_coord_0[i0 + b] = ucoord[0].to(torch.float32)
            if p >= 2 and unexpl_coord_1 is not None:
                unexpl_coord_1[i0 + b] = ucoord[1].to(torch.float32)
            unexpl_coord_max[i0 + b] = ucoord.max().to(torch.float32)

            # Directional worst-case (keep your ridge-stabilized generalized eig)
            Rf = Rloo.to(torch.float32)
            Xf = Xs[b].to(torch.float32)
            SigmaX = (Xf.transpose(0, 1) @ Xf).to(torch.float32)
            SigmaR = (Rf.transpose(0, 1) @ Rf).to(torch.float32)

            gam = (float(cfg.ridge_x) * float(trX_b.item()) / float(max(p, 1)))
            SigmaXr = SigmaX + gam * I_p

            Lchol = torch.linalg.cholesky(SigmaXr)
            Z = torch.linalg.solve_triangular(Lchol, SigmaR, upper=False)
            M = torch.linalg.solve_triangular(Lchol.transpose(0, 1), Z, upper=True)
            M = 0.5 * (M + M.transpose(0, 1))
            ev = torch.linalg.eigvalsh(M)
            wmax = ev[-1].clamp(0.0, 1.0)

            worst_unexpl[i0 + b] = wmax
            worst_ret[i0 + b] = (1.0 - wmax).clamp(0.0, 1.0)

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
        lap_eta=lap_eta.detach().cpu().numpy(),
        lap_df=lap_df.detach().cpu().numpy(),
        lap_lambda2=lap_lambda2.detach().cpu().numpy(),
    )
    if p >= 2 and unexpl_coord_1 is not None:
        out["unexplained_coord1"] = unexpl_coord_1.detach().cpu().numpy()
    return out



@torch.no_grad()
def local_explainedcov_metrics_LOO_laplacian(
    X: torch.Tensor,          # (N,p)
    Y: torch.Tensor,          # (N,qy)
    idxY: torch.Tensor,       # (N,kY) excluding self
    dY: torch.Tensor,         # (N,kY)
    cfg: Config,
) -> Dict[str, np.ndarray]:
    """
    Neighborhood KRR with graph-Laplacian regularization in sample space.
    Uses correct LOO for linear smoothers.

    Returns per-anchor metrics (N,).
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

    eta_used = torch.empty((N,), device=device, dtype=torch.float32)
    df_eff = torch.empty((N,), device=device, dtype=torch.float32)

    I_k_f64 = torch.eye(k, device=device, dtype=torch.float64)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

    bs = int(cfg.batch_size)

    for i0 in range(0, N, bs):
        i1 = min(N, i0 + bs)
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

        if cfg.use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(cfg.eps_tau))
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            tau = dn.max(dim=1).values.clamp_min(float(cfg.eps_tau))
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)  # (B,k)
        sw = torch.sqrt(w).to(torch.float32)

        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * Yn).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Yc = Yn.to(torch.float32) - muY[:, None, :]

        Xs = Xc * sw[:, :, None]  # (B,k,p)
        Ys = Yc * sw[:, :, None]  # (B,k,qy)

        # Gram matrix
        Kmat = torch.bmm(Ys, Ys.transpose(1, 2)).to(torch.float64)  # (B,k,k)
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)  # (B,)
        lam = (float(cfg.ridge_y) * trK / float(k)).to(torch.float64)   # (B,)

        # Per-neighborhood solves (eta can vary per neighborhood)
        for b in range(B):
            Kb = Kmat[b]
            lam_b = float(lam[b].item())

            # Build Laplacian from within-neighborhood Y distances (use standardized Yn, not centered/weighted Ys)
            Lb = _build_laplacian_from_Y(
                Yg=Yn[b].to(torch.float32),
                tau=float(tau[b].item()),
                sigma_mode=cfg.lap_sigma_mode,
                normalized=cfg.lap_normalized,
                w_floor_rel=cfg.lap_w_floor_rel,
            ).to(torch.float64)

            base = float((trK[b].item() / float(k)) if trK[b].item() > 0 else 1.0)

            if cfg.laplacian_eta_rel >= 0.0:
                eta_b = float(cfg.laplacian_eta_rel) * base
                # df for reporting
                A = Kb + lam_b * I_k_f64 + eta_b * Lb
                Hinv = torch.linalg.solve(A, I_k_f64)
                S = Kb @ Hinv
                df_b = float(torch.trace(S).clamp_min(0.0).item())
            else:
                # auto-eta by df targeting
                df_target = float(max(2.0, min(cfg.laplacian_df_target, float(k - 1))))
                eta_min = float(cfg.laplacian_eta_min_rel) * base
                eta_max = float(cfg.laplacian_eta_max_rel) * base
                eta_b, df_b = _choose_eta_by_df_target(
                    K=Kb,
                    L=Lb,
                    lam=lam_b,
                    df_target=df_target,
                    eta_min=eta_min,
                    eta_max=eta_max,
                    iters=int(cfg.laplacian_bisect_iters),
                )
                A = Kb + lam_b * I_k_f64 + eta_b * Lb
                Hinv = torch.linalg.solve(A, I_k_f64)

            # alpha, predictions, smoother diag
            Xsb = Xs[b].to(torch.float64)                 # (k,p)
            alpha = Hinv @ Xsb                            # (k,p)
            Xhat = Kb @ alpha                             # (k,p)
            S = Kb @ Hinv                                 # (k,k)
            sdiag = torch.diagonal(S).to(torch.float64)

            denom = (1.0 - sdiag).clamp_min(1e-6)         # (k,)
            Rloo = (Xsb - Xhat) / denom[:, None]          # (k,p)

            # Energies
            trX_b = (Xsb * Xsb).sum().clamp_min(0.0)
            trR_b = (Rloo * Rloo).sum().clamp_min(0.0)

            u = (trR_b / (trX_b + float(cfg.eps_trace))).clamp(0.0, 1.0)
            e = (1.0 - u).clamp(0.0, 1.0)

            trX[i0 + b] = trX_b.to(torch.float32)
            trR[i0 + b] = trR_b.to(torch.float32)
            unexpl[i0 + b] = u.to(torch.float32)
            expl[i0 + b] = e.to(torch.float32)

            eta_used[i0 + b] = float(eta_b)
            df_eff[i0 + b] = float(df_b)

            # Coordinate-wise
            varX = (Xsb * Xsb).sum(dim=0)                 # (p,)
            varR = (Rloo * Rloo).sum(dim=0)               # (p,)
            ucoord = (varR / (varX + float(cfg.eps_trace))).clamp(0.0, 1.0)

            unexpl_coord_0[i0 + b] = ucoord[0].to(torch.float32)
            if p >= 2 and unexpl_coord_1 is not None:
                unexpl_coord_1[i0 + b] = ucoord[1].to(torch.float32)
            unexpl_coord_max[i0 + b] = ucoord.max().to(torch.float32)

            # Directional worst-case via generalized eig of (SigmaR, SigmaX + gam I)
            Rf = Rloo.to(torch.float32)
            Xf = Xs[b].to(torch.float32)

            SigmaX = (Xf.transpose(0, 1) @ Xf).to(torch.float32)   # (p,p)
            SigmaR = (Rf.transpose(0, 1) @ Rf).to(torch.float32)   # (p,p)

            gam = (float(cfg.ridge_x) * float(trX_b.item()) / float(max(p, 1)))
            SigmaXr = SigmaX + gam * I_p

            # Stable symmetric generalized eigenvalue via Cholesky
            Lchol = torch.linalg.cholesky(SigmaXr)
            Z = torch.linalg.solve_triangular(Lchol, SigmaR, upper=False)
            M = torch.linalg.solve_triangular(Lchol.transpose(0, 1), Z, upper=True)
            M = 0.5 * (M + M.transpose(0, 1))
            ev = torch.linalg.eigvalsh(M)
            wmax = ev[-1].clamp(0.0, 1.0)

            worst_unexpl[i0 + b] = wmax
            worst_ret[i0 + b] = (1.0 - wmax).clamp(0.0, 1.0)

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
        lap_eta=eta_used.detach().cpu().numpy(),
        lap_df=df_eff.detach().cpu().numpy(),
    )
    if p >= 2 and unexpl_coord_1 is not None:
        out["unexplained_coord1"] = unexpl_coord_1.detach().cpu().numpy()
    return out


# ----------------------------- load descriptors and build Y -----------------------------

def load_descriptors_and_build_Y(
    desc_h5: Path,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """
    Load descriptors from H5 and build Y vector.
    Returns: (X, Y, descriptor_kind, metadata)
    """
    print(f"[potts_analyze] Loading descriptors from {desc_h5}...")

    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)

        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))

        mean2d = np.array(f["correlations/correlations_2d_mean"])
        mean1d = np.array(f["correlations/correlations_radial_mean"])
        meanph = np.array(f["phases/final_fraction_mean"])

    X = np.stack([temps, fracs], axis=1).astype(np.float32)  # (N,2)

    if descriptor == "radial1d":
        feat = mean1d.reshape(N, -1)
    else:
        feat = mean2d.reshape(N, -1)

    if cfg.prepend_phase_fractions_to_Y:
        Y = np.concatenate([meanph, feat], axis=1)
    else:
        Y = feat

    metadata = {
        "descriptor": descriptor,
        "q": q,
        "N": N,
        "X_shape": X.shape,
        "Y_shape": Y.shape,
    }

    print(f"[potts_analyze] Loaded: N={N}, descriptor={descriptor}, Y_dim={Y.shape[1]}")
    return X, Y, descriptor, metadata


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Potts descriptors for injectivity (graph-Laplacian regularized)")

    ap.add_argument("--h5", type=str, default=Config.h5, help="Input descriptor HDF5.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)

    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"],
                    help="Descriptor type (used to find latest file if --h5 is empty).")

    ap.add_argument("--prepend_phase_fractions_to_Y", action=argparse.BooleanOptionalAction,
                    default=Config.prepend_phase_fractions_to_Y)

    ap.add_argument("--standardize_X", action=argparse.BooleanOptionalAction, default=Config.standardize_X)
    ap.add_argument("--standardize_Y", action=argparse.BooleanOptionalAction, default=Config.standardize_Y)

    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true", help="Use Gaussian weights from center-to-neighbor distances.")
    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)

    ap.add_argument("--compare_laplacian", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_laplacian", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--lap_eta_rel", type=float, default=-1.0)
    ap.add_argument("--lap_df_target", type=float, default=4.0)
    ap.add_argument("--lap_bisect_iters", type=int, default=12)
    ap.add_argument("--lap_eta_min_rel", type=float, default=1e-6)
    ap.add_argument("--lap_eta_max_rel", type=float, default=1e6)

    ap.add_argument("--lap_sigma_mode", type=str, default="median", choices=["median", "tau"])
    ap.add_argument("--lap_normalized", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lap_w_floor_rel", type=float, default=1e-3)

    ap.add_argument("--lap_graph_on_raw_Y", action=argparse.BooleanOptionalAction, default=True)

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
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),
        standardize_X=bool(args.standardize_X),
        standardize_Y=bool(args.standardize_Y),
        kY=int(args.kY),
        use_weights=bool(args.use_weights),
        ridge_y=float(args.ridge_y),
        ridge_x=float(args.ridge_x),
        compare_laplacian=bool(args.compare_laplacian),
        use_laplacian=bool(args.use_laplacian),

        lap_eta_rel=float(args.lap_eta_rel),
        lap_df_target=float(args.lap_df_target),
        lap_bisect_iters=int(args.lap_bisect_iters),
        lap_eta_min_rel=float(args.lap_eta_min_rel),
        lap_eta_max_rel=float(args.lap_eta_max_rel),

        lap_sigma_mode=str(args.lap_sigma_mode),
        lap_normalized=bool(args.lap_normalized),
        lap_w_floor_rel=float(args.lap_w_floor_rel),

        lap_graph_on_raw_Y=bool(args.lap_graph_on_raw_Y),

        batch_size=int(args.batch_size),
        device=str(args.device),
        dpi=int(args.dpi),
        save_scatter=not bool(args.no_scatter),
        save_heatmaps=not bool(args.no_heatmaps),
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        hm_clip=(float(args.hm_clip_lo), float(args.hm_clip_hi)),
    )

    # Determine input file
    if str(args.h5).strip():
        desc_h5 = Path(str(args.h5)).expanduser().resolve()
    else:
        data_root = Path(cfg.potts_data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")

        candidates: List[Tuple[float, Path]] = []
        for run_dir in data_root.iterdir():
            if run_dir.is_dir():
                desc_dir = run_dir / args.descriptor
                if desc_dir.exists() and desc_dir.is_dir():
                    for h5_file in desc_dir.glob("*.h5"):
                        ts = _parse_run_dir_name(run_dir.name)
                        if ts is None:
                            ts = h5_file.stat().st_mtime
                        candidates.append((ts, h5_file))

        if not candidates:
            raise FileNotFoundError(
                f"No descriptor files found for '{args.descriptor}' under {data_root}\n"
                f"Run potts_descriptors.py first with --descriptor {args.descriptor}"
            )

        candidates.sort(key=lambda t: t[0], reverse=True)
        desc_h5 = candidates[0][1]

    print(f"[potts_analyze] Input: {desc_h5}")

    X, Y, descriptor, load_meta = load_descriptors_and_build_Y(desc_h5, cfg)

    # Output dirs
    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    root_desc = ensure_dir(session_dir / descriptor)

    print("[potts_analyze] Computing injectivity metrics (graph-Laplacian regularized)...")
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    X_use = X.copy()
    Y_use = Y.copy()

    if cfg.standardize_X:
        X_use = standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = standardize_np(Y_use)

    Xt = to_t(X_use, device=device)

    # Kernel-space Y (as before)
    Yt_kernel = to_t(Y_use, device=device)

    # Graph-space Y: raw (unstandardized) to avoid building the graph on inflated noise geometry
    if cfg.lap_graph_on_raw_Y:
        Y_graph_use = Y.copy()
        # (optional) do NOT prepend phases differently; use the exact same Y layout as kernel-space
        Yt_graph = to_t(Y_graph_use, device=device)
    else:
        Yt_graph = Yt_kernel


    idxY_t, dY_t = knn_in_y(Yt_kernel, k=int(cfg.kY))

    metrics_base = local_explainedcov_metrics_LOO_laplacian(
        X=Xt,
        Y=Yt_kernel,
        idxY=idxY_t,
        dY=dY_t,
        cfg=cfg,
    )

    metrics_lap = None
    if cfg.use_laplacian:
        metrics_lap = local_explainedcov_metrics_LOO_laplacian(
            X=Xt,
            Y_kernel=Yt_kernel,
            Y_graph=Yt_graph,
            idxY=idxY_t,
            dY=dY_t,
            cfg=cfg,
            batch_size=int(cfg.batch_size),
        )


    # Save CSV
    figs_dir = ensure_dir(root_desc / "figs")
    csv_path = root_desc / "potts_local_explainedcov_injectivity.csv"


    header_cols = [
        "temperature", "fraction_initial",

        # baseline
        "unexplained_frac", "explained_frac",
        "worst_unexplained_ratio", "worst_retention",
        "trX", "trR",
        "avg_dY",
        "unexplained_coord0",
        "unexplained_coord_max",
    ]
    if "unexplained_coord1" in metrics_base:
        header_cols.append("unexplained_coord1")

    cols = [
        X[:, 0], X[:, 1],
        metrics_base["unexplained_frac"], metrics_base["explained_frac"],
        metrics_base["worst_unexplained_ratio"], metrics_base["worst_retention"],
        metrics_base["trX"], metrics_base["trR"],
        metrics_base["avg_dY"],
        metrics_base["unexplained_coord0"],
        metrics_base["unexplained_coord_max"],
    ]
    if "unexplained_coord1" in metrics_base:
        cols.append(metrics_base["unexplained_coord1"])

    # Laplacian columns (suffixed)
    if cfg.use_laplacian and (metrics_lap is not None) and cfg.compare_laplacian:
        header_cols += [
            "unexplained_frac_lap", "explained_frac_lap",
            "worst_unexplained_ratio_lap", "worst_retention_lap",
            "trX_lap", "trR_lap",
            "unexplained_coord0_lap",
            "unexplained_coord_max_lap",
            "lap_eta", "lap_df", "lap_lambda2",
        ]
        cols += [
            metrics_lap["unexplained_frac"], metrics_lap["explained_frac"],
            metrics_lap["worst_unexplained_ratio"], metrics_lap["worst_retention"],
            metrics_lap["trX"], metrics_lap["trR"],
            metrics_lap["unexplained_coord0"],
            metrics_lap["unexplained_coord_max"],
            metrics_lap["lap_eta"], metrics_lap["lap_df"], metrics_lap["lap_lambda2"],
        ]
        if "unexplained_coord1" in metrics_lap:
            header_cols.append("unexplained_coord1_lap")
            cols.append(metrics_lap["unexplained_coord1"])


    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

    # Plots
    print("[potts_analyze] Generating plots...")
    temp = X[:, 0].astype(np.float64, copy=False)
    frac = X[:, 1].astype(np.float64, copy=False)

    if cfg.save_scatter:
        scatter_tf(temp, frac, metrics_base["unexplained_frac"],
                   f"{descriptor}: unexplained fraction (LOO) tr(R)/tr(X)",
                   figs_dir / "scatter_unexplained_frac", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics_base["explained_frac"],
                   f"{descriptor}: explained fraction (LOO) 1 - tr(R)/tr(X)",
                   figs_dir / "scatter_explained_frac", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics_base["worst_retention"],
                   f"{descriptor}: worst retention (directional, ridge-stabilized)",
                   figs_dir / "scatter_worst_retention", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, np.log10(metrics_base["trX"] + 1e-30),
                   f"{descriptor}: log10 local X energy tr(X)",
                   figs_dir / "scatter_log10_trX", dpi=cfg.dpi)

        scatter_tf(temp, frac, metrics_base["unexplained_coord_max"],
                   f"{descriptor}: max coord unexplained (collapse detector)",
                   figs_dir / "scatter_unexplained_coord_max", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, np.log10(metrics_base["lap_eta"] + 1e-30),
                   f"{descriptor}: log10 eta used (Laplacian)",
                   figs_dir / "scatter_log10_lap_eta", dpi=cfg.dpi)

        scatter_tf(temp, frac, metrics_base["lap_df"],
                   f"{descriptor}: df = tr(S) (effective dof)",
                   figs_dir / "scatter_lap_df", dpi=cfg.dpi)

    if cfg.save_heatmaps:
        heatmap_binned_tf(temp, frac, metrics_base["unexplained_frac"],
                          f"{descriptor}: heatmap unexplained fraction",
                          figs_dir / "heatmap_unexplained_frac",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics_base["explained_frac"],
                          f"{descriptor}: heatmap explained fraction",
                          figs_dir / "heatmap_explained_frac",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics_base["worst_retention"],
                          f"{descriptor}: heatmap worst retention",
                          figs_dir / "heatmap_worst_retention",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, np.log10(metrics_base["lap_eta"] + 1e-30),
                          f"{descriptor}: heatmap log10 eta used",
                          figs_dir / "heatmap_log10_lap_eta",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        
    if cfg.use_laplacian and (metrics_lap is not None) and cfg.compare_laplacian:
        scatter_tf(temp, frac, metrics_lap["explained_frac"],
                   f"{descriptor}: explained fraction (Laplacian)",
                   figs_dir / "scatter_explained_frac_lap", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics_lap["lap_df"],
                   f"{descriptor}: df=tr(S) (Laplacian)",
                   figs_dir / "scatter_lap_df", dpi=cfg.dpi)
        scatter_tf(temp, frac, np.log10(metrics_lap["lap_eta"] + 1e-30),
                   f"{descriptor}: log10 eta (Laplacian)",
                   figs_dir / "scatter_log10_lap_eta", dpi=cfg.dpi)
        scatter_tf(temp, frac, metrics_lap["lap_lambda2"],
                   f"{descriptor}: lambda2(L) (connectivity proxy)",
                   figs_dir / "scatter_lap_lambda2", dpi=cfg.dpi)

        diff = (metrics_lap["explained_frac"] - metrics_base["explained_frac"]).astype(np.float64)
        scatter_tf(temp, frac, diff,
                   f"{descriptor}: explained_frac_lap - explained_frac_base",
                   figs_dir / "scatter_explained_frac_diff", dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics_lap["explained_frac"],
                          f"{descriptor}: heatmap explained fraction (Laplacian)",
                          figs_dir / "heatmap_explained_frac_lap",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, diff,
                          f"{descriptor}: heatmap (lap - base) explained fraction",
                          figs_dir / "heatmap_explained_frac_diff",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)


    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor,
        N=int(X.shape[0]),
        y_dim=int(Y.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        laplacian=dict(
            eta_rel=float(cfg.laplacian_eta_rel),
            df_target=float(cfg.laplacian_df_target),
            sigma_mode=str(cfg.laplacian_sigma_mode),
            normalized=bool(cfg.laplacian_normalized),
        ),
        stats=dict(
            unexpl_median=float(np.median(metrics_base["unexplained_frac"])),
            expl_median=float(np.median(metrics_base["explained_frac"])),
            worst_ret_median=float(np.median(metrics_base["worst_retention"])),
            lap_eta_median=float(np.median(metrics_base["lap_eta"])),
            lap_df_median=float(np.median(metrics_base["lap_df"])),
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
        load_metadata=load_meta,
    )
    meta_path = root_desc / "metadata_local_explainedcov_injectivity.json"
    meta_path.write_text(json.dumps(inj_meta, indent=2))
    print(f"[potts_analyze] wrote: {meta_path}")
    print("[potts_analyze] Done!")


if __name__ == "__main__":
    main()
