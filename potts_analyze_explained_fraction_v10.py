#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction_abs_snr_gated.py

Goal: keep your (pathological-but-useful) inverse-KRR LOO "explained_frac",
but force the reported score to 0 in neighborhoods where the OUTCOME SPACE
locally collapses (flat / noise-floor / effectively 1D), WITHOUT any relative
quantile/median normalization.

We do this by multiplying the baseline explained_frac by TWO ABSOLUTE gates
computed from RAW (unstandardized) outcomes + repeat noise statistics saved in the H5:

  score_gated = explained_frac * gate_energy_abs * gate_dim_abs

Definitions (per neighborhood of size k):
  - Raw weighted-centered block:
      Y_s_raw = diag(sqrt(w)) (Y_raw - 1 mu^T)

  - Raw sample-kernel:
      K_raw = Y_s_raw Y_s_raw^T  (k x k)

  - Observed raw excitation energy:
      E_obs = tr(K_raw) = ||Y_s_raw||_F^2

  - Expected noise-only energy (ABSOLUTE, from repeats):
      E_noise = (1 - ||w||_2^2) * sum_d sigma_d^2
    where sigma_d^2 are per-feature within-repeat variances estimated from saved std arrays.
    For uniform weights: (1 - ||w||^2) = 1 - 1/k.

  - SNR:
      SNR = E_obs / (E_noise + eps)

  - Absolute energy gate:
      gate_energy_abs = SNR / (SNR + gamma)
    gamma=1 => gate=0.5 at SNR=1.

  - Noise-debiased effective rank:
      eig = eigvals(K_raw) (>=0)
      eig_sig = max(eig - E_noise/k, 0)
      effrank_sig = exp( -sum p log p ), p = eig_sig / sum(eig_sig)
    (if sum(eig_sig) ~ 0 => effrank_sig=0)

  - Absolute dimension gate (tied to control dimension p=dim(X)=2):
      gate_dim_abs = clip((effrank_sig - 1) / (p - 1), 0, 1)
    For p=2 => gate_dim_abs = clip(effrank_sig - 1, 0, 1).

Important practical detail:
  - If descriptor == corr2d and your H5 does NOT contain correlations_2d_std,
    we cannot build an absolute noise floor for corr2d features.
    In that case, this script gates using PHASE FRACTIONS ONLY (meanph/stdph),
    which is still sufficient to zero "phase-collapsed" regions reliably.

Outputs:
  potts_analysis/<run>/<input_stem>/<descriptor>/
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

    # Baseline injectivity metric knobs
    standardize_X: bool = True
    standardize_Y: bool = True
    kY: int = 15

    use_weights: bool = False
    eps_tau: float = 1e-10

    ridge_y: float = 1e-3
    ridge_x: float = 1e-8
    eps_trace: float = 1e-18

    batch_size: int = 8
    device: str = "cuda"

    # Absolute SNR gate parameters
    gate_gamma: float = 1.0          # gate_energy = SNR / (SNR + gamma)
    gate_use_dim: bool = False        # include dimension gate
    gate_use_energy: bool = True     # include energy gate

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


# ----------------------------- standardization / tensors -----------------------------

def standardize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def to_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


# ----------------------------- kNN -----------------------------

@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N = Y.shape[0]
    k = min(int(k), N - 1)
    Dc = torch.cdist(Y, Y, p=2.0)
    Dc.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(Dc, k=k, largest=False, sorted=True)
    return idx, vals


# ----------------------------- baseline inverse-KRR LOO + ABS SNR gates -----------------------------

@torch.no_grad()
def local_metrics_LOO_and_abs_gates(
    X: torch.Tensor,          # (N,p) standardized if chosen
    Y_std: torch.Tensor,      # (N,q) standardized if chosen (for kNN + KRR)
    Y_gate_raw: torch.Tensor, # (N,qg) raw features used for absolute gates
    sigma2_sum: float,        # sum_d sigma_d^2 over the qg gating features (absolute)
    idxY: torch.Tensor,       # (N,kY)
    dY: torch.Tensor,         # (N,kY)
    cfg: Config,
) -> Dict[str, np.ndarray]:
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    explained = torch.empty((N,), device=device, dtype=torch.float32)
    unexplained = torch.empty((N,), device=device, dtype=torch.float32)
    trX = torch.empty((N,), device=device, dtype=torch.float32)
    trR = torch.empty((N,), device=device, dtype=torch.float32)

    # absolute gate diagnostics
    E_obs = torch.empty((N,), device=device, dtype=torch.float32)
    E_noise = torch.empty((N,), device=device, dtype=torch.float32)
    snr = torch.empty((N,), device=device, dtype=torch.float32)
    gate_energy = torch.empty((N,), device=device, dtype=torch.float32)
    effrank_sig = torch.empty((N,), device=device, dtype=torch.float32)
    gate_dim = torch.empty((N,), device=device, dtype=torch.float32)
    score_gated = torch.empty((N,), device=device, dtype=torch.float32)

    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)

    sigma2_sum_t = torch.tensor(float(sigma2_sum), device=device, dtype=torch.float32).clamp_min(0.0)
    gamma_t = torch.tensor(float(cfg.gate_gamma), device=device, dtype=torch.float32).clamp_min(1e-12)

    for i0 in range(0, N, int(cfg.batch_size)):
        i1 = min(N, i0 + int(cfg.batch_size))
        B = i1 - i0

        centers = torch.arange(i0, i1, device=device, dtype=torch.int64)
        neigh = torch.cat([centers[:, None], idxY[i0:i1]], dim=1)  # (B,k)

        dn = torch.cat([
            torch.zeros((B, 1), device=device, dtype=torch.float32),
            dY[i0:i1].to(torch.float32)
        ], dim=1)
        avg_dy[i0:i1] = dn[:, 1:].mean(dim=1)

        Xn = X[neigh]               # (B,k,p)
        Yn_std = Y_std[neigh]       # (B,k,q)
        Yn_gate = Y_gate_raw[neigh] # (B,k,qg)

        if cfg.use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(cfg.eps_tau))
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)  # (B,k)
        sw = torch.sqrt(w).to(torch.float32)

        # -------- baseline inverse-KRR LOO (same as your original metric) --------
        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * Yn_std).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Yc = Yn_std.to(torch.float32) - muY[:, None, :]

        Xs = Xc * sw[:, :, None]  # (B,k,p)
        Ys = Yc * sw[:, :, None]  # (B,k,q)

        K = torch.bmm(Ys, Ys.transpose(1, 2))  # (B,k,k)
        trK = K.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(cfg.ridge_y) * trK / float(k)).to(torch.float32)
        Kreg = K + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))  # (B,k,k)
        alpha = torch.bmm(Hinv, Xs)                                       # (B,k,p)

        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)
        u = (trR_b / (trX_b + float(cfg.eps_trace))).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexplained[i0:i1] = u
        explained[i0:i1] = e

        # -------- absolute gates on RAW outcomes (noise-calibrated) --------
        muG = (w[:, :, None] * Yn_gate).sum(dim=1)
        Gc = Yn_gate.to(torch.float32) - muG[:, None, :]
        Gs = Gc * sw[:, :, None]                            # (B,k,qg)
        Kraw = torch.bmm(Gs, Gs.transpose(1, 2))             # (B,k,k)

        eig = torch.linalg.eigvalsh(Kraw).clamp_min(0.0)     # (B,k), ascending
        Eobs_b = eig.sum(dim=1).clamp_min(0.0)               # (B,)
        E_obs[i0:i1] = Eobs_b

        wnorm2 = (w * w).sum(dim=1).clamp(0.0, 1.0)          # (B,)
        one_minus = (1.0 - wnorm2).clamp_min(0.0)            # (B,)
        En_b = (one_minus * sigma2_sum_t).clamp_min(0.0)     # (B,)
        E_noise[i0:i1] = En_b

        snr_b = (Eobs_b / (En_b + 1e-30)).clamp_min(0.0)
        snr[i0:i1] = snr_b

        if cfg.gate_use_energy:
            ge = (snr_b / (snr_b + gamma_t)).clamp(0.0, 1.0)
        else:
            ge = torch.ones_like(snr_b)
        gate_energy[i0:i1] = ge

        # noise-debiased effective rank
        if cfg.gate_use_dim and p >= 2:
            # subtract isotropic noise energy per mode
            eig_sig = (eig - (En_b / float(k))[:, None]).clamp_min(0.0)
            s_sig = eig_sig.sum(dim=1).clamp_min(0.0)

            # if s_sig ~ 0 => effrank=0
            s_safe = s_sig.clamp_min(1e-30)
            pi = eig_sig / s_safe[:, None]
            ent = -(pi * (pi + 1e-30).log()).sum(dim=1)      # (B,)
            er = torch.exp(ent)                               # (B,)
            er = torch.where(s_sig <= 1e-20, torch.zeros_like(er), er)
            effrank_sig[i0:i1] = er

            gd = ((er - 1.0) / float(p - 1)).clamp(0.0, 1.0)
        else:
            effrank_sig[i0:i1] = torch.zeros((B,), device=device, dtype=torch.float32)
            gd = torch.ones((B,), device=device, dtype=torch.float32)

        gate_dim[i0:i1] = gd

        score_gated[i0:i1] = (e * ge * gd).clamp(0.0, 1.0)

    out: Dict[str, np.ndarray] = dict(
        unexplained_frac=unexplained.detach().cpu().numpy(),
        explained_frac=explained.detach().cpu().numpy(),
        score_gated=score_gated.detach().cpu().numpy(),

        gate_energy_abs=gate_energy.detach().cpu().numpy(),
        gate_dim_abs=gate_dim.detach().cpu().numpy(),

        E_obs=E_obs.detach().cpu().numpy(),
        E_noise=E_noise.detach().cpu().numpy(),
        snr=snr.detach().cpu().numpy(),
        effrank_sig=effrank_sig.detach().cpu().numpy(),

        trX=trX.detach().cpu().numpy(),
        trR=trR.detach().cpu().numpy(),
        avg_dY=avg_dy.detach().cpu().numpy(),
    )
    return out


# ----------------------------- load descriptors and build Y + noise stats -----------------------------

def load_all_needed(
    desc_h5: Path,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str, Dict[str, Any]]:
    """
    Returns:
      X_raw: (N,2) unstandardized controls
      Y_raw: (N,q_full) unstandardized descriptor used for KRR (phases optionally prepended)
      Y_gate_raw: (N,q_gate) unstandardized descriptor used for ABS gates
      sigma2_sum: scalar sum_d sigma_d^2 over gate features
      descriptor_kind: "radial1d" or "corr2d"
      metadata: dict
    """
    print(f"[potts_analyze] Loading descriptors from {desc_h5}...")

    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)

        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))
        R = int(f.attrs.get("n_repeats", -1))

        mean2d = np.array(f["correlations/correlations_2d_mean"])
        mean1d = np.array(f["correlations/correlations_radial_mean"])
        std1d = np.array(f["correlations/correlations_radial_std"])  # always saved in your snippet

        meanph = np.array(f["phases/final_fraction_mean"])
        stdph = np.array(f["phases/final_fraction_std"])

        # Optional: corr2d std if present (recommended to add in writer)
        has_std2d = ("correlations/correlations_2d_std" in f)
        std2d = np.array(f["correlations/correlations_2d_std"]) if has_std2d else None

    X_raw = np.stack([temps, fracs], axis=1).astype(np.float32)

    # Full Y used for KRR metric
    if descriptor == "radial1d":
        feat_mean = mean1d.reshape(N, -1)
    else:
        feat_mean = mean2d.reshape(N, -1)

    if cfg.prepend_phase_fractions_to_Y:
        Y_raw = np.concatenate([meanph, feat_mean], axis=1)
    else:
        Y_raw = feat_mean

    # Gate features + sigma2:
    # Always include phase fractions if they are prepended (or if you want them as gate-only fallback).
    sigma2_list: List[np.ndarray] = []
    gate_parts: List[np.ndarray] = []

    # Phase stats always exist
    sigma2_phase = np.mean(stdph.astype(np.float64) ** 2, axis=0)  # (q,)
    if cfg.prepend_phase_fractions_to_Y:
        gate_parts.append(meanph.astype(np.float32))
        sigma2_list.append(sigma2_phase)

    # Add correlation-feature stats if available
    if descriptor == "radial1d":
        # std1d exists
        feat_gate = mean1d.reshape(N, -1).astype(np.float32)
        sigma2_feat = np.mean(std1d.reshape(N, -1).astype(np.float64) ** 2, axis=0)
        if not cfg.prepend_phase_fractions_to_Y:
            gate_parts.append(feat_gate)
            sigma2_list.append(sigma2_feat)
        else:
            gate_parts.append(feat_gate)
            sigma2_list.append(sigma2_feat)
    else:
        # corr2d: absolute noise floor only if std2d exists; otherwise gate using phases only
        if has_std2d and (std2d is not None):
            feat_gate = mean2d.reshape(N, -1).astype(np.float32)
            sigma2_feat = np.mean(std2d.reshape(N, -1).astype(np.float64) ** 2, axis=0)
            if not cfg.prepend_phase_fractions_to_Y:
                gate_parts.append(feat_gate)
                sigma2_list.append(sigma2_feat)
            else:
                gate_parts.append(feat_gate)
                sigma2_list.append(sigma2_feat)
        else:
            # fallback: phases only (still absolute, because stdph exists)
            pass

    if not gate_parts:
        raise RuntimeError("No gate features available; cannot compute absolute noise floor.")

    Y_gate_raw = np.concatenate(gate_parts, axis=1).astype(np.float32)
    sigma2_sum = float(np.sum(np.concatenate(sigma2_list, axis=0)))

    metadata = {
        "descriptor": descriptor,
        "q": q,
        "N": N,
        "n_repeats": R,
        "X_shape": X_raw.shape,
        "Y_raw_shape": Y_raw.shape,
        "Y_gate_raw_shape": Y_gate_raw.shape,
        "has_corr2d_std": bool(has_std2d),
        "sigma2_sum_gate": sigma2_sum,
        "gate_uses_phases": bool(cfg.prepend_phase_fractions_to_Y),
    }

    print(f"[potts_analyze] Loaded: N={N}, descriptor={descriptor}, Y_dim={Y_raw.shape[1]}")
    if descriptor == "corr2d" and not has_std2d:
        print("[potts_analyze] NOTE: correlations_2d_std missing; absolute gating uses PHASE FRACTIONS ONLY.")

    return X_raw, Y_raw, Y_gate_raw, sigma2_sum, descriptor, metadata


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Potts injectivity: baseline inverse-KRR + ABS(SNR,dim) gates")

    ap.add_argument("--h5", type=str, default=Config.h5, help="Input descriptor HDF5.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"])

    ap.add_argument("--prepend_phase_fractions_to_Y", action=argparse.BooleanOptionalAction,
                    default=Config.prepend_phase_fractions_to_Y)

    ap.add_argument("--standardize_X", action=argparse.BooleanOptionalAction, default=Config.standardize_X)
    ap.add_argument("--standardize_Y", action=argparse.BooleanOptionalAction, default=Config.standardize_Y)

    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true")
    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)

    ap.add_argument("--gate_gamma", type=float, default=Config.gate_gamma)
    ap.add_argument("--gate_use_energy", action=argparse.BooleanOptionalAction, default=Config.gate_use_energy)
    ap.add_argument("--gate_use_dim", action=argparse.BooleanOptionalAction, default=Config.gate_use_dim)

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
        descriptor=str(args.descriptor),
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),
        standardize_X=bool(args.standardize_X),
        standardize_Y=bool(args.standardize_Y),
        kY=int(args.kY),
        use_weights=bool(args.use_weights),
        ridge_y=float(args.ridge_y),
        ridge_x=float(args.ridge_x),
        gate_gamma=float(args.gate_gamma),
        gate_use_energy=bool(args.gate_use_energy),
        gate_use_dim=bool(args.gate_use_dim),
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
            raise FileNotFoundError(f"No descriptor files found for '{args.descriptor}' under {data_root}")

        candidates.sort(key=lambda t: t[0], reverse=True)
        desc_h5 = candidates[0][1]

    print(f"[potts_analyze] Input: {desc_h5}")

    X_raw, Y_raw, Y_gate_raw, sigma2_sum, descriptor_kind, load_meta = load_all_needed(desc_h5, cfg)

    # Output dirs
    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    root_desc = ensure_dir(session_dir / descriptor_kind)
    figs_dir = ensure_dir(root_desc / "figs")

    # Device
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    # Standardize for baseline KRR / kNN if requested
    X_use = X_raw.copy()
    Y_use = Y_raw.copy()
    if cfg.standardize_X:
        X_use = standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = standardize_np(Y_use)

    Xt = to_t(X_use, device=device)
    Yt_std = to_t(Y_use, device=device)
    Yt_gate_raw = to_t(Y_gate_raw, device=device)

    # kNN neighborhoods in standardized Y (baseline behavior preserved)
    print(f"[potts_analyze] Finding {cfg.kY} nearest neighbors...")
    idxY_t, dY_t = knn_in_y(Yt_std, k=int(cfg.kY))

    # Compute baseline explained + absolute gates + gated score
    print("[potts_analyze] Computing baseline explained + absolute gates...")
    metrics = local_metrics_LOO_and_abs_gates(
        X=Xt,
        Y_std=Yt_std,
        Y_gate_raw=Yt_gate_raw,
        sigma2_sum=float(sigma2_sum),
        idxY=idxY_t,
        dY=dY_t,
        cfg=cfg,
    )

    # CSV
    csv_path = root_desc / "potts_local_explainedcov_injectivity.csv"

    header_cols = [
        "temperature", "fraction_initial",

        "explained_frac", "unexplained_frac",
        "score_gated",

        "gate_energy_abs", "gate_dim_abs",
        "E_obs", "E_noise", "snr", "effrank_sig",

        "trX", "trR",
        "avg_dY",
    ]
    cols = [
        X_raw[:, 0], X_raw[:, 1],

        metrics["explained_frac"], metrics["unexplained_frac"],
        metrics["score_gated"],

        metrics["gate_energy_abs"], metrics["gate_dim_abs"],
        metrics["E_obs"], metrics["E_noise"], metrics["snr"], metrics["effrank_sig"],

        metrics["trX"], metrics["trR"],
        metrics["avg_dY"],
    ]

    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

    # Plots
    print("[potts_analyze] Generating plots...")
    temp = X_raw[:, 0].astype(np.float64, copy=False)
    frac = X_raw[:, 1].astype(np.float64, copy=False)

    if cfg.save_scatter:
        scatter_tf(temp, frac, metrics["explained_frac"],
                   f"{descriptor_kind}: explained fraction (baseline inverse-KRR)",
                   figs_dir / "scatter_explained_frac_base", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["score_gated"],
                   f"{descriptor_kind}: score_gated = explained * ABS gates",
                   figs_dir / "scatter_score_gated", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["gate_energy_abs"],
                   f"{descriptor_kind}: gate_energy_abs (SNR/(SNR+gamma))",
                   figs_dir / "scatter_gate_energy_abs", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["gate_dim_abs"],
                   f"{descriptor_kind}: gate_dim_abs (effrank_sig -> [0,1])",
                   figs_dir / "scatter_gate_dim_abs", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, np.log10(np.asarray(metrics["snr"]) + 1e-30),
                   f"{descriptor_kind}: log10 SNR (E_obs / E_noise)",
                   figs_dir / "scatter_log10_snr", dpi=cfg.dpi)

    if cfg.save_heatmaps:
        heatmap_binned_tf(temp, frac, metrics["explained_frac"],
                          f"{descriptor_kind}: heatmap explained (baseline inverse-KRR)",
                          figs_dir / "heatmap_explained_frac_base",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["score_gated"],
                          f"{descriptor_kind}: heatmap score_gated (ABS collapse->0)",
                          figs_dir / "heatmap_score_gated",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["gate_energy_abs"],
                          f"{descriptor_kind}: heatmap gate_energy_abs",
                          figs_dir / "heatmap_gate_energy_abs",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["gate_dim_abs"],
                          f"{descriptor_kind}: heatmap gate_dim_abs",
                          figs_dir / "heatmap_gate_dim_abs",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

    # Metadata
    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor_kind,
        N=int(X_raw.shape[0]),
        y_dim=int(Y_raw.shape[1]),
        y_gate_dim=int(Y_gate_raw.shape[1]),
        sigma2_sum_gate=float(sigma2_sum),
        kY=int(cfg.kY),
        ridge_y=float(cfg.ridge_y),
        gate_gamma=float(cfg.gate_gamma),
        stats=dict(
            explained_median=float(np.median(metrics["explained_frac"])),
            gated_median=float(np.median(metrics["score_gated"])),
            gate_energy_median=float(np.median(metrics["gate_energy_abs"])),
            gate_dim_median=float(np.median(metrics["gate_dim_abs"])),
            snr_median=float(np.median(metrics["snr"])),
        ),
    )
    inj_meta = dict(
        created_utc=_utc_now_z(),
        config=asdict(cfg),
        files=dict(csv=str(csv_path), figs=str(figs_dir)),
        summary=summary,
        load_metadata=load_meta,
    )
    meta_path = root_desc / "metadata_local_explainedcov_injectivity.json"
    meta_path.write_text(json.dumps(inj_meta, indent=2))
    print(f"[potts_analyze] wrote: {meta_path}")
    print("[potts_analyze] Done!")


if __name__ == "__main__":
    main()
