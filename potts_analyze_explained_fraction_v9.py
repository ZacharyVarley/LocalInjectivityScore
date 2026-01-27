#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction_snr_gated.py

Goal: keep your (pathological) inverse-KRR LOO "explained" metric for continuity,
but multiply it by an ABSOLUTE (non-quantile) collapse/noise gate computed from
repeat statistics stored in the descriptor H5.

Baseline (unchanged):
  explained_base = 1 - tr(Rloo)/tr(Xs)

ABSOLUTE gating (no dataset-relative quantiles):
  - Build a raw "gate feature" vector with repeat std available:
      Y_gate_raw = [final_fraction_mean, correlations_radial_mean]  (always)
      Sigma_gate = [final_fraction_std, correlations_radial_std]    (always)
    This works for both descriptor kinds:
      * radial1d: kernel features == gate features
      * corr2d:   kernel features use corr2d_mean, but gate uses phases+radial (has std)

  - In each neighborhood with weights w (sum w = 1):
      Ys_gate = diag(sqrt(w)) * (Y_gate_raw - 1*mu_w^T)
      trK_gate = ||Ys_gate||_F^2

    Expected centered raw energy from noise alone (heteroskedastic, exact):
      E_noise = sum_d [ sum_i w_i * sigma_{i,d}^2  -  sum_i w_i^2 * sigma_{i,d}^2 ]
              = sum_d sum_i w_i(1-w_i) sigma_{i,d}^2

    Absolute SNR gate:
      gate_energy = trK_gate / (trK_gate + gamma * E_noise + eps)

    Dimensional-collapse gate via effective rank of K_gate = Ys_gate Ys_gate^T:
      eig = eigvals(K_gate) >= 0
      p_i = eig_i / sum eig
      effrank = exp( -sum p_i log p_i ) in [0, k]
      gate_dim = clip((effrank - 1)/(p-1), 0, 1)   (p = dim(X), here p=2)

  - Final score:
      score_gated = explained_base * gate_energy * gate_dim

This forces ~0 when outcomes are flat relative to repeat noise OR intrinsically 1D,
without imposing any "a fixed fraction must be low" behavior.

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

    # Baseline inverse-KRR knobs
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

    # ABSOLUTE gating parameters
    gate_gamma: float = 0.1          # stricter -> more suppression near SNR~1
    gate_eps: float = 1e-30
    use_gate_energy: bool = True
    use_gate_dim: bool = True

    # Plots
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


# ----------------------------- core: baseline inverse-KRR + absolute collapse stats -----------------------------

@torch.no_grad()
def local_metrics_LOO_with_abs_gates(
    X: torch.Tensor,              # (N,p)
    Y_std_kernel: torch.Tensor,   # (N,qk) standardized for neighborhoods + KRR
    Y_raw_gate: torch.Tensor,     # (N,qg) raw gate vector with repeat std available
    S2_raw_gate: torch.Tensor,    # (N,qg) raw noise variances = std^2
    idxY: torch.Tensor,           # (N,kY)
    dY: torch.Tensor,             # (N,kY)
    cfg: Config,
) -> Dict[str, np.ndarray]:
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    # baseline outputs
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

    # absolute gate stats
    trK_gate = torch.empty((N,), device=device, dtype=torch.float32)
    E_noise = torch.empty((N,), device=device, dtype=torch.float32)
    gate_energy = torch.empty((N,), device=device, dtype=torch.float32)
    effrank_gate = torch.empty((N,), device=device, dtype=torch.float32)
    gate_dim = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
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

        Xn = X[neigh]                       # (B,k,p)
        YnK = Y_std_kernel[neigh]           # (B,k,qk)
        YnG = Y_raw_gate[neigh]             # (B,k,qg)
        S2n = S2_raw_gate[neigh]            # (B,k,qg)

        # weights
        if cfg.use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(cfg.eps_tau))
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            w = torch.ones((B, k), device=device, dtype=torch.float32)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)
        sw = torch.sqrt(w)

        # ---------- baseline inverse KRR LOO (your metric) ----------
        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * YnK).sum(dim=1)

        Xc = Xn - muX[:, None, :]
        Yc = YnK - muY[:, None, :]

        Xs = Xc * sw[:, :, None]
        Ys = Yc * sw[:, :, None]

        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(cfg.ridge_y) * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))
        alpha = torch.bmm(Hinv, Xs)
        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)

        u = (trR_b / (trX_b + float(cfg.eps_trace))).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexpl[i0:i1] = u
        expl[i0:i1] = e

        varX = (Xs * Xs).sum(dim=1)
        varR = (Rloo * Rloo).sum(dim=1)
        ucoord = (varR / (varX + float(cfg.eps_trace))).clamp(0.0, 1.0)
        unexpl_coord_0[i0:i1] = ucoord[:, 0]
        if p >= 2 and unexpl_coord_1 is not None:
            unexpl_coord_1[i0:i1] = ucoord[:, 1]
        unexpl_coord_max[i0:i1] = ucoord.max(dim=1).values

        SigmaX = torch.bmm(Xs.transpose(1, 2), Xs)
        SigmaR = torch.bmm(Rloo.transpose(1, 2), Rloo)
        gam = (float(cfg.ridge_x) * trX_b / float(max(p, 1))).to(torch.float32)
        SigmaXr = SigmaX + gam[:, None, None] * I_p[None, :, :]
        Lc = torch.linalg.cholesky(SigmaXr)
        Z = torch.linalg.solve_triangular(Lc, SigmaR, upper=False)
        M = torch.linalg.solve_triangular(Lc.transpose(1, 2), Z, upper=True)
        M = 0.5 * (M + M.transpose(1, 2))
        ev = torch.linalg.eigvalsh(M)
        wmax = ev[:, -1].clamp(0.0, 1.0)
        worst_unexpl[i0:i1] = wmax
        worst_ret[i0:i1] = (1.0 - wmax).clamp(0.0, 1.0)

        # ---------- ABSOLUTE gating from raw outcomes + repeat noise ----------
        # Gate raw center/weight
        muG = (w[:, :, None] * YnG).sum(dim=1)
        Gc = YnG - muG[:, None, :]
        Gs = Gc * sw[:, :, None]                     # (B,k,qg)
        Kgate = torch.bmm(Gs, Gs.transpose(1, 2))     # (B,k,k)

        eig = torch.linalg.eigvalsh(Kgate).clamp_min(0.0)  # (B,k)
        s = eig.sum(dim=1).clamp_min(float(cfg.gate_eps))  # equals tr(Kgate) = ||Gs||^2
        trK_gate[i0:i1] = s

        # exact expected noise energy after weighted centering (heteroskedastic)
        # E_noise = sum_d [ sum_i w_i s2_i,d  -  sum_i w_i^2 s2_i,d ]
        term1 = (w[:, :, None] * S2n).sum(dim=1)          # (B,qg)
        term2 = ((w * w)[:, :, None] * S2n).sum(dim=1)    # (B,qg)
        En = (term1 - term2).sum(dim=1).clamp_min(0.0)    # (B,)
        E_noise[i0:i1] = En

        if cfg.use_gate_energy:
            ge = s / (s + float(cfg.gate_gamma) * En + float(cfg.gate_eps))
        else:
            ge = torch.ones_like(s)
        gate_energy[i0:i1] = ge.clamp(0.0, 1.0)

        # effective rank of Kgate (intrinsic sample-space dimension proxy)
        pi = eig / s[:, None]
        ent = -(pi * (pi + 1e-30).log()).sum(dim=1)
        er = torch.exp(ent)
        er = torch.where(En + s <= float(cfg.gate_eps), torch.zeros_like(er), er)
        effrank_gate[i0:i1] = er

        if cfg.use_gate_dim and p >= 2:
            gd = ((er - 1.0) / float(p - 1)).clamp(0.0, 1.0)
        else:
            gd = torch.ones_like(er)
        gate_dim[i0:i1] = gd

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
        trK_gate=trK_gate.detach().cpu().numpy(),
        E_noise=E_noise.detach().cpu().numpy(),
        gate_energy=gate_energy.detach().cpu().numpy(),
        effrank_gate=effrank_gate.detach().cpu().numpy(),
        gate_dim=gate_dim.detach().cpu().numpy(),
    )
    if p >= 2 and unexpl_coord_1 is not None:
        out["unexplained_coord1"] = unexpl_coord_1.detach().cpu().numpy()
    return out


# ----------------------------- load descriptors and build Y -----------------------------

def load_descriptors_and_build(
    desc_h5: Path,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """
    Returns:
      X_raw:     (N,2)
      Y_kernel:  (N,qk) for neighborhoods + KRR (phase + chosen descriptor mean)
      Y_gate:    (N,qg) for gating (phase mean + radial mean)  [has std]
      S2_gate:   (N,qg) noise variances (phase std^2 + radial std^2)
      temps, fracs: raw param arrays
      descriptor_kind: "radial1d" or "corr2d"
      metadata dict
    """
    print(f"[potts_analyze] Loading descriptors from {desc_h5}...")

    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)

        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))
        R = int(f.attrs.get("n_repeats", 0))

        mean2d = np.array(f["correlations/correlations_2d_mean"])
        mean1d = np.array(f["correlations/correlations_radial_mean"])
        std1d = np.array(f["correlations/correlations_radial_std"])
        meanph = np.array(f["phases/final_fraction_mean"])
        stdph = np.array(f["phases/final_fraction_std"])

    X_raw = np.stack([temps, fracs], axis=1).astype(np.float32)

    # kernel descriptor mean
    if descriptor == "radial1d":
        feat_kernel = mean1d.reshape(N, -1)
    else:
        feat_kernel = mean2d.reshape(N, -1)

    # gate descriptor mean/std ALWAYS uses phases + radial (std available)
    feat_gate = mean1d.reshape(N, -1)
    std_gate = std1d.reshape(N, -1)

    meanph2 = meanph.reshape(N, -1)
    stdph2 = stdph.reshape(N, -1)

    if cfg.prepend_phase_fractions_to_Y:
        Y_kernel = np.concatenate([meanph2, feat_kernel], axis=1).astype(np.float32)
        Y_gate = np.concatenate([meanph2, feat_gate], axis=1).astype(np.float32)
        S2_gate = np.concatenate([stdph2, std_gate], axis=1).astype(np.float32) ** 2
    else:
        Y_kernel = feat_kernel.astype(np.float32)
        # gating without phases is still defined, but less informative; keep it consistent
        Y_gate = feat_gate.astype(np.float32)
        S2_gate = (std_gate.astype(np.float32) ** 2)

    metadata = {
        "descriptor": descriptor,
        "q": q,
        "N": N,
        "R": R,
        "X_shape": X_raw.shape,
        "Y_kernel_shape": Y_kernel.shape,
        "Y_gate_shape": Y_gate.shape,
        "whiten_eps": float(h5py.File(str(desc_h5), "r").attrs.get("whiten_eps", np.nan)),
    }

    print(f"[potts_analyze] Loaded: N={N}, descriptor={descriptor}, Y_kernel_dim={Y_kernel.shape[1]}, Y_gate_dim={Y_gate.shape[1]}")
    return X_raw, Y_kernel, Y_gate, S2_gate, temps, fracs, descriptor, metadata


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Potts injectivity: baseline inverse-KRR + ABSOLUTE SNR/dimension gating (no quantiles)")

    ap.add_argument("--h5", type=str, default=Config.h5)
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"])

    ap.add_argument("--prepend_phase_fractions_to_Y", action=argparse.BooleanOptionalAction, default=Config.prepend_phase_fractions_to_Y)

    ap.add_argument("--standardize_X", action=argparse.BooleanOptionalAction, default=Config.standardize_X)
    ap.add_argument("--standardize_Y", action=argparse.BooleanOptionalAction, default=Config.standardize_Y)

    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true")
    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)

    ap.add_argument("--gate_gamma", type=float, default=Config.gate_gamma)
    ap.add_argument("--use_gate_energy", action=argparse.BooleanOptionalAction, default=Config.use_gate_energy)
    ap.add_argument("--use_gate_dim", action=argparse.BooleanOptionalAction, default=Config.use_gate_dim)

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
        gate_gamma=float(args.gate_gamma),
        use_gate_energy=bool(args.use_gate_energy),
        use_gate_dim=bool(args.use_gate_dim),
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

    # input file selection
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

    X_raw, Y_kernel_raw, Y_gate_raw, S2_gate_raw, temps, fracs, descriptor_kind, load_meta = load_descriptors_and_build(desc_h5, cfg)

    # output dirs
    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    root_desc = ensure_dir(session_dir / descriptor_kind)
    figs_dir = ensure_dir(root_desc / "figs")

    # device
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    # standardize for baseline KRR / neighborhood geometry (as before)
    X_use = X_raw.copy()
    Y_use = Y_kernel_raw.copy()
    if cfg.standardize_X:
        X_use = standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = standardize_np(Y_use)

    Xt = to_t(X_use, device=device)
    Yt_std_kernel = to_t(Y_use, device=device)

    # gating uses RAW (not standardized) + noise variances from repeats
    Yt_raw_gate = to_t(Y_gate_raw, device=device)
    S2t_raw_gate = to_t(S2_gate_raw, device=device)

    print(f"[potts_analyze] Finding {cfg.kY} nearest neighbors (in standardized kernel Y)...")
    idxY_t, dY_t = knn_in_y(Yt_std_kernel, k=int(cfg.kY))

    print("[potts_analyze] Computing baseline explained + ABSOLUTE gates...")
    metrics = local_metrics_LOO_with_abs_gates(
        X=Xt,
        Y_std_kernel=Yt_std_kernel,
        Y_raw_gate=Yt_raw_gate,
        S2_raw_gate=S2t_raw_gate,
        idxY=idxY_t,
        dY=dY_t,
        cfg=cfg,
    )

    explained = metrics["explained_frac"].astype(np.float64)
    ge = metrics["gate_energy"].astype(np.float64) if cfg.use_gate_energy else np.ones_like(explained)
    gd = metrics["gate_dim"].astype(np.float64) if cfg.use_gate_dim else np.ones_like(explained)
    score_gated = np.clip(explained * ge * gd, 0.0, 1.0)

    # CSV
    csv_path = root_desc / "potts_local_explainedcov_injectivity.csv"
    header_cols = [
        "temperature", "fraction_initial",
        "explained_frac_base", "unexplained_frac_base",
        "score_gated",
        "gate_energy", "gate_dim",
        "trK_gate_raw", "E_noise_raw",
        "effrank_gate",
        "worst_retention",
        "trX", "trR",
        "avg_dY",
        "unexplained_coord0",
        "unexplained_coord_max",
    ]
    cols = [
        X_raw[:, 0], X_raw[:, 1],
        metrics["explained_frac"], metrics["unexplained_frac"],
        score_gated,
        metrics["gate_energy"], metrics["gate_dim"],
        metrics["trK_gate"], metrics["E_noise"],
        metrics["effrank_gate"],
        metrics["worst_retention"],
        metrics["trX"], metrics["trR"],
        metrics["avg_dY"],
        metrics["unexplained_coord0"],
        metrics["unexplained_coord_max"],
    ]
    if "unexplained_coord1" in metrics:
        header_cols.append("unexplained_coord1")
        cols.append(metrics["unexplained_coord1"])

    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

    # Plots
    print("[potts_analyze] Generating plots...")
    temp = X_raw[:, 0].astype(np.float64, copy=False)
    frac = X_raw[:, 1].astype(np.float64, copy=False)

    if cfg.save_scatter:
        scatter_tf(temp, frac, explained,
                   f"{descriptor_kind}: explained fraction (baseline inverse-KRR)",
                   figs_dir / "scatter_explained_frac_base", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, score_gated,
                   f"{descriptor_kind}: score_gated = explained * gate_energy * gate_dim",
                   figs_dir / "scatter_score_gated", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["gate_energy"],
                   f"{descriptor_kind}: gate_energy = trK/(trK + gamma*E_noise)",
                   figs_dir / "scatter_gate_energy", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, metrics["gate_dim"],
                   f"{descriptor_kind}: gate_dim from effrank(K_gate)",
                   figs_dir / "scatter_gate_dim", dpi=cfg.dpi, vmin=0.0, vmax=1.0)
        scatter_tf(temp, frac, np.log10(metrics["trK_gate"] + 1e-30),
                   f"{descriptor_kind}: log10 trK_gate_raw",
                   figs_dir / "scatter_log10_trK_gate", dpi=cfg.dpi)
        scatter_tf(temp, frac, np.log10(metrics["E_noise"] + 1e-30),
                   f"{descriptor_kind}: log10 E_noise_raw (repeat-based)",
                   figs_dir / "scatter_log10_E_noise", dpi=cfg.dpi)
        scatter_tf(temp, frac, metrics["effrank_gate"],
                   f"{descriptor_kind}: effrank(K_gate)",
                   figs_dir / "scatter_effrank_gate", dpi=cfg.dpi)

    if cfg.save_heatmaps:
        heatmap_binned_tf(temp, frac, explained,
                          f"{descriptor_kind}: heatmap explained fraction (baseline)",
                          figs_dir / "heatmap_explained_frac_base",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, score_gated,
                          f"{descriptor_kind}: heatmap score_gated (absolute collapse/noise suppression)",
                          figs_dir / "heatmap_score_gated",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, dpi=cfg.dpi)

    # Metadata
    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor_kind,
        N=int(X_raw.shape[0]),
        y_kernel_dim=int(Y_kernel_raw.shape[1]),
        y_gate_dim=int(Y_gate_raw.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        gate_gamma=float(cfg.gate_gamma),
        stats=dict(
            explained_median=float(np.median(explained)),
            score_gated_median=float(np.median(score_gated)),
            gate_energy_median=float(np.median(metrics["gate_energy"])),
            gate_dim_median=float(np.median(metrics["gate_dim"])),
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
