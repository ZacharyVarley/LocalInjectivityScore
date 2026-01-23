#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction.py

Analyzes Potts descriptors for injectivity:
  1) Loads descriptors from potts_descriptors.py output
  2) Builds descriptor Y vector
  3) Computes local injectivity diagnostics
  4) Generates plots

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

    prepend_phase_fractions_to_Y: bool = True

    # Injectivity metric knobs
    standardize_X: bool = False
    standardize_Y: bool = False

    kY: int = 15

    use_weights: bool = False
    eps_tau: float = 1e-10

    ridge_y: float = 1e-3
    ridge_x: float = 1e-8
    eps_trace: float = 1e-18

    # Rank-cap kernel capacity (fixes high-q overcapacity pathology)
    rank_cap: int = 10
    ridge_y_rankcap: float = 1e-8

    # SNR weighting (uses raw Y scale + repeat noise proxy)
    snr_eps: float = 1e-30
    snr_use_rawY: bool = True

    # Control-space covariance volume diagnostic
    ridge_cx: float = 1e-8

    batch_size: int = 16
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

def stable_rank_lam(lam: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    # lam: (B,k) nonnegative eigenvalues
    s1 = lam.sum(dim=1)
    s2 = (lam * lam).sum(dim=1)
    return (s1 * s1) / (s2 + float(eps))


def effective_rank_lam(lam: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    s1 = lam.sum(dim=1).clamp_min(float(eps))
    p = lam / s1[:, None]
    H = -(p * (p + float(eps)).log()).sum(dim=1)
    return H.exp()


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
    Y_raw: torch.Tensor | None = None,
    noise_varsum: torch.Tensor | None = None,
    rank_cap: int = 6,
    ridge_y_rankcap: float = 1e-3,
    snr_eps: float = 1e-30,
    snr_use_rawY: bool = True,
    ridge_cx: float = 1e-8,
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

    expl_rankcap = torch.empty((N,), device=device, dtype=torch.float32)
    w_snr_out = torch.empty((N,), device=device, dtype=torch.float32)
    score_rankcap_snr = torch.empty((N,), device=device, dtype=torch.float32)

    stable_rank_out = torch.empty((N,), device=device, dtype=torch.float32)
    effective_rank_out = torch.empty((N,), device=device, dtype=torch.float32)
    logdet_cx_out = torch.empty((N,), device=device, dtype=torch.float32)

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

        # ---------- intrinsic-dimension diagnostics from eig(K) ----------
        lam_e, U = torch.linalg.eigh(Kmat)  # lam ascending, (B,k), U (B,k,k)
        lam_e = lam_e.clamp_min(0.0)

        stable_rank_out[i0:i1] = stable_rank_lam(lam_e).clamp_min(0.0)
        effective_rank_out[i0:i1] = effective_rank_lam(lam_e).clamp_min(0.0)

        # ---------- rank-capped kernel ridge (capacity control) ----------
        rcap = int(min(max(1, int(rank_cap)), k - 1))
        # take largest rcap eigpairs (ascending -> last rcap)
        Ur = U[:, :, -rcap:]                 # (B,k,rcap)
        lamr = lam_e[:, -rcap:]              # (B,rcap)

        Kr = torch.bmm(Ur * lamr[:, None, :], Ur.transpose(1, 2))  # (B,k,k)
        trKr = Kr.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam_rc = (float(ridge_y_rankcap) * trKr / float(k)).to(torch.float32)

        Kreg_r = Kr + lam_rc[:, None, None] * I_k[None, :, :]
        Hinv_r = torch.linalg.solve(Kreg_r, I_k[None, :, :].expand(B, k, k))
        alpha_r = torch.bmm(Hinv_r, Xs)  # (B,k,p)

        hdiag_r = Hinv_r.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo_r = alpha_r / hdiag_r[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trRr = (Rloo_r * Rloo_r).sum(dim=(1, 2)).clamp_min(0.0)
        ur = (trRr / (trX_b + float(eps_trace))).clamp(0.0, 1.0)
        er = (1.0 - ur).clamp(0.0, 1.0)
        expl_rankcap[i0:i1] = er

        # ---------- SNR weight (raw-scale if enabled) ----------
        if snr_use_rawY and (Y_raw is not None) and (noise_varsum is not None):
            Yn_raw = Y_raw[neigh]  # (B,k,qy)
            muY_raw = (w[:, :, None] * Yn_raw).sum(dim=1)
            Yc_raw = Yn_raw.to(torch.float32) - muY_raw[:, None, :]
            Ys_raw = Yc_raw * sw[:, :, None]
            E_sig = (Ys_raw * Ys_raw).sum(dim=(1, 2)).clamp_min(0.0)  # ||Ys_raw||_F^2

            nv = noise_varsum[neigh].to(torch.float32)
            E_noise = (w * nv).sum(dim=1).clamp_min(0.0)  # weighted mean of sum-variances

            snr = E_sig / (E_noise + float(snr_eps))
        else:
            # fallback: use centered standardized energy; still useful as a collapse indicator
            trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
            snr = trK / (trK.mean().clamp_min(1e-12))

        wsnr = (snr / (1.0 + snr)).clamp(0.0, 1.0)
        w_snr_out[i0:i1] = wsnr

        score_rankcap_snr[i0:i1] = (er * wsnr).clamp(0.0, 1.0)

        # ---------- logdet(Cx) on local weighted covariance ----------
        # Xs already includes sqrt(w) and centering -> SigmaX = Xs^T Xs is weighted covariance.
        SigmaX = torch.bmm(Xs.transpose(1, 2), Xs)  # (B,p,p)
        trSX = SigmaX.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        gam = (float(ridge_cx) * trSX / float(max(p, 1))).to(torch.float32)
        SigmaXr = SigmaX + gam[:, None, None] * I_p[None, :, :]

        sign, logabsdet = torch.linalg.slogdet(SigmaXr)
        # if sign <= 0, store nan
        logdet = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, float("nan")))
        logdet_cx_out[i0:i1] = logdet

        # ---------- original full-rank ridge ----------
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(ridge_y) * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))  # (B,k,k)
        alpha = torch.bmm(Hinv, Xs)                                       # (B,k,p)

        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

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
        SigmaXw = torch.bmm(Xs.transpose(1, 2), Xs).to(torch.float32)
        SigmaR = torch.bmm(Rloo.transpose(1, 2), Rloo).to(torch.float32)

        gam = (float(ridge_x) * trX_b / float(max(p, 1))).to(torch.float32)
        SigmaXr_w = SigmaXw + gam[:, None, None] * I_p[None, :, :]

        L = torch.linalg.cholesky(SigmaXr_w)
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
        explained_rankcap=expl_rankcap.detach().cpu().numpy(),
        w_snr=w_snr_out.detach().cpu().numpy(),
        score_rankcap_snr=score_rankcap_snr.detach().cpu().numpy(),
        stable_rank=stable_rank_out.detach().cpu().numpy(),
        effective_rank=effective_rank_out.detach().cpu().numpy(),
        logdet_Cx=logdet_cx_out.detach().cpu().numpy(),
    )
    if p >= 2 and unexpl_coord_1 is not None:
        out["unexplained_coord1"] = unexpl_coord_1.detach().cpu().numpy()
    return out


# ----------------------------- load descriptors and build Y -----------------------------

def load_descriptors_and_build_Y(
    desc_h5: Path,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """
    Load descriptors from H5 and build Y vector.
    Returns: (X, Y, descriptor_kind, metadata)
    """
    print(f"[potts_analyze] Loading descriptors from {desc_h5}...")
    
    with h5py.File(str(desc_h5), "r") as f:
        # Parameters
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)
        
        # Metadata
        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))
        
        # Descriptors
        mean2d = np.array(f["correlations/correlations_2d_mean"])
        mean1d = np.array(f["correlations/correlations_radial_mean"])
        meanph = np.array(f["phases/final_fraction_mean"])
        std1d = np.array(f["correlations/correlations_radial_std"])
        stdph = np.array(f["phases/final_fraction_std"])
    
    # Build X
    X = np.stack([temps, fracs], axis=1).astype(np.float32)  # (N,2)
    
    # Build Y based on descriptor type
    if descriptor == "radial1d":
        feat = mean1d.reshape(N, -1)
    else:  # corr2d
        feat = mean2d.reshape(N, -1)
    
    if cfg.prepend_phase_fractions_to_Y:
        Y = np.concatenate([meanph, feat], axis=1)
    else:
        Y = feat
    
    # --- noise proxy for SNR (per-sample scalar) ---
    # Goal: match the squared-norm scale of centered Ys (sum over dims), so use SUM of variances.
    var_ph_sum = (stdph.astype(np.float64) ** 2).sum(axis=1)  # (N,)

    # Always available: radial std gives a defensible noise floor proxy.
    var_rad_sum = (std1d.reshape(N, -1).astype(np.float64) ** 2).sum(axis=1)  # (N,)

    if descriptor == "radial1d":
        var_feat_sum = var_rad_sum
    else:
        # corr2d: no per-feature std for 2D correlation surface stored; use mean(radial var) * Y_feature_dim
        y_feat_dim = int(feat.reshape(N, -1).shape[1])
        var_feat_sum = (var_rad_sum / float(std1d.reshape(N, -1).shape[1])) * float(y_feat_dim)

    if cfg.prepend_phase_fractions_to_Y:
        noise_varsum = var_ph_sum + var_feat_sum
    else:
        noise_varsum = var_feat_sum

    noise_varsum = noise_varsum.astype(np.float32)
    
    metadata = {
        "descriptor": descriptor,
        "q": q,
        "N": N,
        "X_shape": X.shape,
        "Y_shape": Y.shape,
    }
    
    print(f"[potts_analyze] Loaded: N={N}, descriptor={descriptor}, Y_dim={Y.shape[1]}")
    
    return X, Y, noise_varsum, descriptor, metadata


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Potts descriptors for injectivity")

    ap.add_argument("--h5", type=str, default="potts_data/20260123_025553Z/corr2d/potts_sims_q3_128x128_corr2d.h5", help="Input descriptor HDF5.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    
    ap.add_argument("--descriptor", type=str, default="corr2d", choices=["radial1d", "corr2d"],
                    help="Descriptor type (used to find latest file if --h5 not specified)")

    ap.add_argument("--prepend_phase_fractions_to_Y", type=bool, default=True, help="Whether to prepend phase fractions to Y.")

    ap.add_argument("--standardize_X", type=bool, default=True)
    ap.add_argument("--standardize_Y", type=bool, default=True)

    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true", help="If set, use Gaussian weights in neighborhoods (default False).")
    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)

    ap.add_argument("--rank_cap", type=int, default=Config.rank_cap)
    ap.add_argument("--ridge_y_rankcap", type=float, default=Config.ridge_y_rankcap)
    ap.add_argument("--snr_eps", type=float, default=Config.snr_eps)

    ap.add_argument("--snr_use_rawY", action="store_true")
    ap.add_argument("--no_snr_use_rawY", dest="snr_use_rawY", action="store_false")
    ap.set_defaults(snr_use_rawY=Config.snr_use_rawY)

    ap.add_argument("--ridge_cx", type=float, default=Config.ridge_cx)

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
        rank_cap=int(args.rank_cap),
        ridge_y_rankcap=float(args.ridge_y_rankcap),
        snr_eps=float(args.snr_eps),
        snr_use_rawY=bool(args.snr_use_rawY),
        ridge_cx=float(args.ridge_cx),
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

    # Find descriptor H5 file
    if str(args.h5).strip():
        desc_h5 = Path(str(args.h5)).expanduser().resolve()
    else:
        # Look for latest descriptor file in potts_data/<timestamp>/<descriptor>/ structure
        data_root = Path(cfg.potts_data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        
        # Search through all timestamped directories for descriptor subdirectories
        candidates: List[Tuple[float, Path]] = []
        for run_dir in data_root.iterdir():
            if run_dir.is_dir():
                desc_dir = run_dir / args.descriptor
                if desc_dir.exists() and desc_dir.is_dir():
                    # Find H5 files in this descriptor directory
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

    # Load descriptors and build Y
    X, Y, noise_varsum, descriptor, load_meta = load_descriptors_and_build_Y(desc_h5, cfg)

    # Output directories
    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    root_desc = ensure_dir(session_dir / descriptor)

    # Injectivity computation
    print("[potts_analyze] Computing injectivity metrics...")
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    X_use = X.copy()
    Y_use = Y.copy()
    Y_raw_np = Y.copy().astype(np.float32, copy=False)
    noise_varsum_np = noise_varsum.astype(np.float32, copy=False)

    if cfg.standardize_X:
        X_use = standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = standardize_np(Y_use)

    Xt = to_t(X_use, device=device)
    Yt = to_t(Y_use, device=device)
    Yrawt = to_t(Y_raw_np, device=device)
    noise_varsum_t = torch.as_tensor(noise_varsum_np, device=device, dtype=torch.float32)

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
        Y_raw=Yrawt,
        noise_varsum=noise_varsum_t,
        rank_cap=int(cfg.rank_cap),
        ridge_y_rankcap=float(cfg.ridge_y_rankcap),
        snr_eps=float(cfg.snr_eps),
        snr_use_rawY=bool(cfg.snr_use_rawY),
        ridge_cx=float(cfg.ridge_cx),
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
    header_cols.extend([
        "explained_rankcap",
        "w_snr",
        "score_rankcap_snr",
        "stable_rank",
        "effective_rank",
        "logdet_Cx",
    ])

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
    cols.extend([
        metrics["explained_rankcap"],
        metrics["w_snr"],
        metrics["score_rankcap_snr"],
        metrics["stable_rank"],
        metrics["effective_rank"],
        metrics["logdet_Cx"],
    ])

    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

    # Plots
    print("[potts_analyze] Generating plots...")
    temp = X[:, 0].astype(np.float64, copy=False)
    frac = X[:, 1].astype(np.float64, copy=False)

    if cfg.save_scatter:
        scatter_tf(temp, frac, metrics["unexplained_frac"],
                   f"{descriptor}: unexplained fraction tr(R)/tr(X) (higher worse)",
                   figs_dir / "scatter_unexplained_frac", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["explained_frac"],
                   f"{descriptor}: explained fraction 1 - tr(R)/tr(X) (higher better)",
                   figs_dir / "scatter_explained_frac", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["worst_retention"],
                   f"{descriptor}: worst retention (directional, ridge-stabilized)",
                   figs_dir / "scatter_worst_retention", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, np.log10(metrics["trX"] + 1e-30),
                   f"{descriptor}: log10 local X energy tr(X) (excitation)",
                   figs_dir / "scatter_log10_trX", dpi=cfg.dpi)
        scatter_tf(temp, frac, metrics["unexplained_coord_max"],
                   f"{descriptor}: max coord unexplained (catches single-parameter collapse)",
                   figs_dir / "scatter_unexplained_coord_max", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        if "unexplained_coord1" in metrics:
            scatter_tf(temp, frac, metrics["unexplained_coord1"],
                       f"{descriptor}: unexplained fraction_initial coordinate",
                       figs_dir / "scatter_unexplained_fraction_coord", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        trXv = metrics["trX"]
        exc = trXv >= np.quantile(trXv, 0.25)
        flag_ok = ((metrics["explained_frac"] >= 0.9) & exc).astype(np.float64)
        scatter_tf(temp, frac, flag_ok,
                   f"{descriptor}: indicator explained_frac>=0.9 & trX>=Q25",
                   figs_dir / "scatter_ok_indicator", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, metrics["explained_rankcap"],
                   f"{descriptor}: explained fraction (rank-capped)",
                   figs_dir / "scatter_explained_rankcap", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, metrics["w_snr"],
                   f"{descriptor}: w_snr = SNR/(1+SNR)",
                   figs_dir / "scatter_w_snr", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, metrics["score_rankcap_snr"],
                   f"{descriptor}: score = explained_rankcap * w_snr",
                   figs_dir / "scatter_score_rankcap_snr", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, metrics["stable_rank"],
                   f"{descriptor}: stable rank of local K",
                   figs_dir / "scatter_stable_rank", dpi=cfg.dpi)

        scatter_tf(temp, frac, metrics["effective_rank"],
                   f"{descriptor}: effective rank (entropy) of local K",
                   figs_dir / "scatter_effective_rank", dpi=cfg.dpi)

        scatter_tf(temp, frac, metrics["logdet_Cx"],
                   f"{descriptor}: logdet local Cov(X) (ridge-stabilized)",
                   figs_dir / "scatter_logdet_Cx", dpi=cfg.dpi)

    if cfg.save_heatmaps:
        heatmap_binned_tf(temp, frac, metrics["unexplained_frac"],
                          f"{descriptor}: heatmap unexplained fraction",
                          figs_dir / "heatmap_unexplained_frac",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["explained_frac"],
                          f"{descriptor}: heatmap explained fraction",
                          figs_dir / "heatmap_explained_frac",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["worst_retention"],
                          f"{descriptor}: heatmap worst retention",
                          figs_dir / "heatmap_worst_retention",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["explained_rankcap"],
                          f"{descriptor}: heatmap explained_rankcap",
                          figs_dir / "heatmap_explained_rankcap",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["w_snr"],
                          f"{descriptor}: heatmap w_snr",
                          figs_dir / "heatmap_w_snr",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["score_rankcap_snr"],
                          f"{descriptor}: heatmap score_rankcap_snr",
                          figs_dir / "heatmap_score_rankcap_snr",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["stable_rank"],
                          f"{descriptor}: heatmap stable_rank",
                          figs_dir / "heatmap_stable_rank",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["effective_rank"],
                          f"{descriptor}: heatmap effective_rank",
                          figs_dir / "heatmap_effective_rank",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["logdet_Cx"],
                          f"{descriptor}: heatmap logdet_Cx",
                          figs_dir / "heatmap_logdet_Cx",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

    # Metadata JSON (injectivity)
    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor,
        N=int(X.shape[0]),
        y_dim=int(Y.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        rank_cap=int(cfg.rank_cap),
        ridge_y_rankcap=float(cfg.ridge_y_rankcap),
        snr_use_rawY=bool(cfg.snr_use_rawY),
        stats=dict(
            unexpl_median=float(np.median(metrics["unexplained_frac"])),
            unexpl_q90=float(np.quantile(metrics["unexplained_frac"], 0.90)),
            expl_median=float(np.median(metrics["explained_frac"])),
            worst_ret_median=float(np.median(metrics["worst_retention"])),
            expl_rankcap_median=float(np.median(metrics["explained_rankcap"])),
            w_snr_median=float(np.median(metrics["w_snr"])),
            score_rankcap_snr_median=float(np.median(metrics["score_rankcap_snr"])),
            stable_rank_median=float(np.nanmedian(metrics["stable_rank"])),
            effective_rank_median=float(np.nanmedian(metrics["effective_rank"])),
            logdet_Cx_median=float(np.nanmedian(metrics["logdet_Cx"])),
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
    meta_path = out_root / "metadata_local_explainedcov_injectivity.json"
    meta_path.write_text(json.dumps(inj_meta, indent=2))
    print(f"[potts_analyze] wrote: {meta_path}")
    print("[potts_analyze] Done!")


if __name__ == "__main__":
    main()
