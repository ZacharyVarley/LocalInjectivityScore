#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction.py

Analyzes Potts descriptors for injectivity:
  1) Loads descriptors from potts_descriptors.py output
  2) Builds descriptor Y vector
  3) Computes local injectivity diagnostics (LOO kernel ridge residual energy)
  4) Generates plots

Adds geometry-only "LOO uninformative" diagnostics per point:
  - loo_uninformative_flag
  - loo_uninformative_align
  - loo_uninformative_gap
  - loo_uninformative_dof_norm

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
    h5: str = "potts_data/20260123_161715Z/corr2d/potts_sims_q3_128x128_corr2d.h5"
    descriptor: str = "corr2d"

    prepend_phase_fractions_to_Y: bool = True

    # Injectivity metric knobs
    standardize_X: bool = True
    standardize_Y: bool = True

    kY: int = 15

    use_weights: bool = False
    eps_tau: float = 1e-10

    ridge_y: float = 1e-3
    ridge_x: float = 1e-1
    eps_trace: float = 1e-18

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

    # LOO-uninformative thresholds (geometry-only)
    loo_uninf_align_thr: float = 0.95
    loo_uninf_gap_thr: float = 1e-3
    loo_uninf_dof_thr: float = 0.95


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
                      vmin=None,
                      vmax=None,
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

    if vmin is not None:
        vmin = float(vmin)
    if vmax is not None:
        vmax = float(vmax)

    if vmin is None or vmax is None:
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
    """Compute k nearest neighbors in Y."""
    N = Y.shape[0]
    k = min(int(k), N - 1)

    Dc = torch.cdist(Y, Y, p=2.0)  # (N,N)
    Dc.fill_diagonal_(float("inf"))
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
    loo_uninf_align_thr: float = 0.95,
    loo_uninf_gap_thr: float = 1e-3,
    loo_uninf_dof_thr: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Kernel ridge in neighborhood, scored by LOO residuals.
    Batched over N to reduce memory.

    Adds geometry-only diagnostics for when LOO is structurally uninformative:
      - align: |v0^T s| where v0 is smallest-eigvec of K, s = sqrt(w)
      - gap: eta0/eta1
      - dof_norm: dof/(k-1), dof = tr(K(K+lam I)^{-1})
      - flag: align>thr & gap<thr & dof_norm>thr
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

    # LOO-uninformative diagnostics
    uninf_flag = torch.empty((N,), device=device, dtype=torch.bool)
    uninf_align = torch.empty((N,), device=device, dtype=torch.float32)
    uninf_gap = torch.empty((N,), device=device, dtype=torch.float32)
    uninf_dof_norm = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

    denom_km1 = float(max(k - 1, 1))

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
        sw = torch.sqrt(w).to(torch.float32)                 # (B,k), ||sw||_2 = 1

        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * Yn).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Yc = Yn.to(torch.float32) - muY[:, None, :]

        Xs = Xc * sw[:, :, None]  # (B,k,p)
        Ys = Yc * sw[:, :, None]  # (B,k,qy)

        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))  # (B,k,k)
        Kmat = 0.5 * (Kmat + Kmat.transpose(1, 2))  # symmetrize

        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(ridge_y) * trK / float(k)).to(torch.float32)  # (B,)
        lam_b = lam.clamp_min(1e-30)

        # ---- LOO-uninformative diagnostics (geometry-only) ----
        eta, V = torch.linalg.eigh(Kmat)  # eta ascending
        eta = eta.clamp_min(0.0)

        v0 = V[:, :, 0]  # (B,k)
        s = sw           # (B,k)
        align = torch.abs((v0 * s).sum(dim=1))  # (B,)

        eta0 = eta[:, 0]
        eta1 = eta[:, 1].clamp_min(1e-30)
        gap = (eta0 / eta1).to(torch.float32)

        dof = (eta / (eta + lam_b[:, None])).sum(dim=1)           # (B,)
        dof_norm = (dof / denom_km1).clamp(0.0, 1.0).to(torch.float32)

        loo_uninf = (align > float(loo_uninf_align_thr)) & (gap < float(loo_uninf_gap_thr)) & (dof_norm > float(loo_uninf_dof_thr))

        uninf_align[i0:i1] = align
        uninf_gap[i0:i1] = gap
        uninf_dof_norm[i0:i1] = dof_norm
        uninf_flag[i0:i1] = loo_uninf
        # ------------------------------------------------------

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

        # LOO-uninformative (geometry-only)
        loo_uninformative_flag=uninf_flag.detach().cpu().numpy().astype(np.int32),
        loo_uninformative_align=uninf_align.detach().cpu().numpy(),
        loo_uninformative_gap=uninf_gap.detach().cpu().numpy(),
        loo_uninformative_dof_norm=uninf_dof_norm.detach().cpu().numpy(),
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
    ap = argparse.ArgumentParser(description="Analyze Potts descriptors for injectivity")

    ap.add_argument("--h5", type=str, default=Config.h5, help="Input descriptor HDF5.")
    ap.add_argument("--potts_data_dir", type=str, default=Config.potts_data_dir)
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)

    ap.add_argument("--descriptor", type=str, default=Config.descriptor, choices=["radial1d", "corr2d"],
                    help="Descriptor type (used to find latest file if --h5 not specified)")

    ap.add_argument("--prepend_phase_fractions_to_Y", type=bool, default=Config.prepend_phase_fractions_to_Y,
                    help="Whether to prepend phase fractions to Y.")

    ap.add_argument("--standardize_X", type=bool, default=Config.standardize_X)
    ap.add_argument("--standardize_Y", type=bool, default=Config.standardize_Y)

    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true",
                    help="If set, use Gaussian weights in neighborhoods (default False).")
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

    # LOO-uninformative thresholds
    ap.add_argument("--loo_uninf_align_thr", type=float, default=Config.loo_uninf_align_thr)
    ap.add_argument("--loo_uninf_gap_thr", type=float, default=Config.loo_uninf_gap_thr)
    ap.add_argument("--loo_uninf_dof_thr", type=float, default=Config.loo_uninf_dof_thr)

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
        batch_size=int(args.batch_size),
        device=str(args.device),
        dpi=int(args.dpi),
        save_scatter=not bool(args.no_scatter),
        save_heatmaps=not bool(args.no_heatmaps),
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        hm_clip=(float(args.hm_clip_lo), float(args.hm_clip_hi)),
        loo_uninf_align_thr=float(args.loo_uninf_align_thr),
        loo_uninf_gap_thr=float(args.loo_uninf_gap_thr),
        loo_uninf_dof_thr=float(args.loo_uninf_dof_thr),
    )

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

    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    root_desc = ensure_dir(session_dir / descriptor)

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
        loo_uninf_align_thr=float(cfg.loo_uninf_align_thr),
        loo_uninf_gap_thr=float(cfg.loo_uninf_gap_thr),
        loo_uninf_dof_thr=float(cfg.loo_uninf_dof_thr),
    )

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

        # LOO-uninformative diagnostics
        "loo_uninformative_flag",
        "loo_uninformative_align",
        "loo_uninformative_gap",
        "loo_uninformative_dof_norm",
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

        metrics["loo_uninformative_flag"].astype(np.float64),
        metrics["loo_uninformative_align"],
        metrics["loo_uninformative_gap"],
        metrics["loo_uninformative_dof_norm"],
    ]
    if "unexplained_coord1" in metrics:
        cols.append(metrics["unexplained_coord1"])

    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[potts_analyze] wrote: {csv_path}")

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

        # ---- NEW: LOO-uninformative figures ----
        scatter_tf(temp, frac, metrics["loo_uninformative_flag"].astype(np.float64),
                   f"{descriptor}: LOO uninformative flag (1=geometry-saturated)",
                   figs_dir / "scatter_loo_uninformative_flag", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, metrics["loo_uninformative_dof_norm"],
                   f"{descriptor}: LOO uninformative dof_norm = dof/(k-1)",
                   figs_dir / "scatter_loo_uninformative_dof_norm", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, metrics["loo_uninformative_align"],
                   f"{descriptor}: LOO uninformative align = |v0^T sqrt(w)|",
                   figs_dir / "scatter_loo_uninformative_align", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        scatter_tf(temp, frac, np.log10(metrics["loo_uninformative_gap"] + 1e-30),
                   f"{descriptor}: log10 gap = log10(eta0/eta1)",
                   figs_dir / "scatter_log10_loo_uninformative_gap", dpi=cfg.dpi)
        # ---------------------------------------

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
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, vmin=0.0, vmax=1.0, dpi=cfg.dpi)
        heatmap_binned_tf(temp, frac, metrics["worst_retention"],
                          f"{descriptor}: heatmap worst retention",
                          figs_dir / "heatmap_worst_retention",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, vmin=0.0, vmax=1.0, dpi=cfg.dpi)

        # ---- NEW: LOO-uninformative heatmaps ----
        heatmap_binned_tf(temp, frac, metrics["loo_uninformative_flag"].astype(np.float64),
                          f"{descriptor}: heatmap LOO uninformative flag (1=geometry-saturated)",
                          figs_dir / "heatmap_loo_uninformative_flag",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=None, vmin=0.0, vmax=1.0, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["loo_uninformative_dof_norm"],
                          f"{descriptor}: heatmap dof_norm = dof/(k-1)",
                          figs_dir / "heatmap_loo_uninformative_dof_norm",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, vmin=0.0, vmax=1.0, dpi=cfg.dpi)

        heatmap_binned_tf(temp, frac, metrics["loo_uninformative_align"],
                          f"{descriptor}: heatmap align = |v0^T sqrt(w)|",
                          figs_dir / "heatmap_loo_uninformative_align",
                          bins_t=cfg.hm_bins_temp, bins_f=cfg.hm_bins_frac,
                          sigma_px=cfg.hm_sigma_px, clip=cfg.hm_clip, vmin=0.0, vmax=1.0, dpi=cfg.dpi)
        # ----------------------------------------

    flagged_frac = float(np.mean(metrics["loo_uninformative_flag"].astype(np.float64)))

    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor,
        N=int(X.shape[0]),
        y_dim=int(Y.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        loo_uninformative_thresholds=dict(
            align=float(cfg.loo_uninf_align_thr),
            gap=float(cfg.loo_uninf_gap_thr),
            dof_norm=float(cfg.loo_uninf_dof_thr),
        ),
        stats=dict(
            unexpl_median=float(np.median(metrics["unexplained_frac"])),
            unexpl_q90=float(np.quantile(metrics["unexplained_frac"], 0.90)),
            expl_median=float(np.median(metrics["explained_frac"])),
            worst_ret_median=float(np.median(metrics["worst_retention"])),
            loo_uninformative_flagged_frac=flagged_frac,
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
