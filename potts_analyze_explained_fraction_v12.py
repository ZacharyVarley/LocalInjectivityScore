#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_explained_fraction.py

Analyzes Potts descriptors for injectivity:
  1) Loads descriptor metadata from potts_descriptors.py output
  2) Builds controls X
  3) (Optional) Global PCA compression of outcome space Y using streaming Oja updates
     - selects m = min #components capturing >= 90% energy
  4) Computes local injectivity diagnostics (LOO KRR residual energy)
  5) Generates plots (including LOO-uninformative diagnostics)

Outputs:
  potts_analysis/<YYYYMMDD_HHMMSSZ>/<input_stem>/<descriptor_kind>/
    potts_local_explainedcov_injectivity.csv
    figs/*.png + *.pdf
    metadata_local_explainedcov_injectivity.json
    pca_mean.npy
    pca_components.pt
    pca_eigsums.npy
    pca_energy_total.npy
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Tuple

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    # Global PCA via Oja or SVD
    use_global_pca: bool = True
    pca_method: str = "auto"  # "auto", "svd", or "oja"
    pca_svd_max_gb: float = 4.0  # max GB for SVD in auto mode
    pca_energy_frac: float = 0.95
    pca_components_max: int = 100

    # if using Oja's method for PCA
    pca_epochs: int = 10
    pca_batch_size: int = 8
    pca_eta: float = 0.1
    pca_dtype: str = "float32"
    pca_device: str = "cuda" 
    pca_seed: int = 0

    # Injectivity metric knobs
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


def plot_pca_cum_energy(cum: np.ndarray, outbase: Path, dpi: int = 250) -> None:
    x = np.arange(1, cum.size + 1, dtype=np.int32)
    plt.figure(figsize=(5.4, 4.0), dpi=dpi)
    plt.plot(x, cum)
    plt.ylim(0.0, 1.01)
    plt.xlim(1, cum.size)
    plt.xlabel("n_components")
    plt.ylabel("cumulative energy fraction")
    plt.title("Global PCA cumulative energy")
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(str(outbase) + ".png", bbox_inches="tight", dpi=dpi)
    plt.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close()


# ----------------------------- numeric helpers -----------------------------

def standardize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def to_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


# ----------------------------- streaming Y from H5 -----------------------------

def _get_desc_meta(desc_h5: Path) -> Dict[str, Any]:
    with h5py.File(str(desc_h5), "r") as f:
        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))

        meanph_ds = f["phases/final_fraction_mean"]
        mean2d_ds = f["correlations/correlations_2d_mean"]
        mean1d_ds = f["correlations/correlations_radial_mean"]

        if descriptor == "radial1d":
            feat_dim = int(np.prod(mean1d_ds.shape[1:]))
        else:
            feat_dim = int(np.prod(mean2d_ds.shape[1:]))

        ph_dim = int(meanph_ds.shape[1]) if len(meanph_ds.shape) == 2 else int(meanph_ds.shape[-1])
        return dict(
            descriptor=descriptor,
            q=q,
            N=N,
            feat_dim=feat_dim,
            ph_dim=ph_dim,
        )


def iter_Y_batches(
    desc_h5: Path,
    cfg: Config,
    batch_size: int,
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Y rows are (phase_fractions || flattened_descriptor) if cfg.prepend_phase_fractions_to_Y else flattened_descriptor.
    Returns (i0, i1, Y_batch) where Y_batch is float32 (B, y_dim).
    """
    meta = _get_desc_meta(desc_h5)
    descriptor = meta["descriptor"]
    N = meta["N"]

    with h5py.File(str(desc_h5), "r") as f:
        meanph_ds = f["phases/final_fraction_mean"]
        mean2d_ds = f["correlations/correlations_2d_mean"]
        mean1d_ds = f["correlations/correlations_radial_mean"]

        for i0 in range(0, N, int(batch_size)):
            i1 = min(N, i0 + int(batch_size))
            B = i1 - i0

            meanph = np.array(meanph_ds[i0:i1], dtype=np.float32).reshape(B, -1)

            if descriptor == "radial1d":
                feat = np.array(mean1d_ds[i0:i1], dtype=np.float32).reshape(B, -1)
            else:
                feat = np.array(mean2d_ds[i0:i1], dtype=np.float32).reshape(B, -1)

            if cfg.prepend_phase_fractions_to_Y:
                Yb = np.concatenate([meanph, feat], axis=1)
            else:
                Yb = feat

            yield i0, i1, Yb


def load_X_and_basic_meta(desc_h5: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)
        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))

    X = np.stack([temps, fracs], axis=1).astype(np.float32)
    meta = dict(descriptor=descriptor, q=q, N=N, X_shape=X.shape)
    return X, meta


# ----------------------------- Oja PCA (batched, with QR) -----------------------------

class OjaPCA(torch.nn.Module):
    """
    Batched Oja update with QR re-orthogonalization.
    Uses a fixed learning rate; update is normalized by batch size to make eta less batch-size-sensitive.
    """

    def __init__(
        self,
        n_features: int,
        n_components: int,
        eta: float = 0.005,
        dtype: torch.dtype = torch.float32,
        normalize_by_batch: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_components = int(n_components)
        self.eta = float(eta)
        self.normalize_by_batch = bool(normalize_by_batch)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        Q0 = torch.randn(self.n_features, self.n_components, dtype=dtype, generator=g)
        self.register_buffer("Q", Q0)
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> None:
        """
        x: (B, n_features), assumed mean-centered.
        """
        B = int(x.shape[0])
        if self.normalize_by_batch and B > 0:
            scale = 1.0 / float(B)
        else:
            scale = 1.0

        upd = (x.T @ (x @ self.Q)) * scale
        self.Q.copy_(torch.linalg.qr(self.Q + self.eta * upd, mode="reduced")[0])
        self.step.add_(1)

    def get_components(self) -> torch.Tensor:
        return self.Q.T  # (k, n_features)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Q  # (B, k)


def compute_global_mean_Y(desc_h5: Path, cfg: Config, batch_size: int) -> np.ndarray:
    meta = _get_desc_meta(desc_h5)
    y_dim = int(meta["feat_dim"] + (meta["ph_dim"] if cfg.prepend_phase_fractions_to_Y else 0))

    mu = np.zeros((y_dim,), dtype=np.float64)
    n = 0

    for _, _, Yb in iter_Y_batches(desc_h5, cfg, batch_size=batch_size):
        B = Yb.shape[0]
        n_new = n + B
        batch_mean = Yb.mean(axis=0, dtype=np.float64)
        mu += (B / float(n_new)) * (batch_mean - mu)
        n = n_new

    return mu.astype(np.float32)


def fit_svd_pca(
    desc_h5: Path,
    cfg: Config,
    mu: np.ndarray,
    n_components: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, np.ndarray, float]:
    """
    Compute PCA using standard SVD on full data matrix.
    Returns: (Q, eig_sums, total_energy) where Q is (y_dim, n_components)
    """
    meta = _get_desc_meta(desc_h5)
    N = int(meta["N"])
    y_dim = int(meta["feat_dim"] + (meta["ph_dim"] if cfg.prepend_phase_fractions_to_Y else 0))
    
    print(f"Loading full data matrix ({N} x {y_dim})...")
    # Load full Y matrix
    Y = np.empty((N, y_dim), dtype=np.float32)
    n_batches = int(np.ceil(N / 256))
    for i0, i1, Yb in tqdm(iter_Y_batches(desc_h5, cfg, batch_size=256),
                           total=n_batches, desc="Loading Y", unit="batch"):
        Y[i0:i1, :] = Yb
    
    # Center
    print("Centering data...")
    Y = Y - mu[None, :]
    
    # Move to device and compute SVD
    print(f"Computing SVD on {device}...")
    Y_t = torch.as_tensor(Y, device=device, dtype=dtype)
    
    # Compute covariance for PCA: Y^T Y / N
    # For efficiency, compute SVD of Y directly and use singular values
    U, S, Vt = torch.linalg.svd(Y_t, full_matrices=False)
    
    # S contains singular values; eigenvalues of cov are S^2 / N
    # Principal components are rows of Vt (columns of V)
    V = Vt.T  # (y_dim, min(N, y_dim))
    
    # Take first n_components
    k = min(n_components, V.shape[1])
    Q = V[:, :k].contiguous()  # (y_dim, k)
    
    # Compute eigenvalue sums for energy calculation
    eig_vals = (S[:k].double().pow(2)).cpu().numpy()  # variance per component
    total_energy = float((S.double().pow(2).sum()).cpu())
    
    del Y, Y_t, U, S, Vt, V
    torch.cuda.empty_cache() if device.type == "cuda" else None
    
    return Q, eig_vals, total_energy


def fit_oja_pca_streaming(
    desc_h5: Path,
    cfg: Config,
    mu: np.ndarray,
    n_components: int,
    epochs: int,
    batch_size: int,
    eta: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    meta = _get_desc_meta(desc_h5)
    y_dim = int(meta["feat_dim"] + (meta["ph_dim"] if cfg.prepend_phase_fractions_to_Y else 0))
    N = int(meta["N"])
    n_batches = int(np.ceil(N / batch_size))

    model = OjaPCA(
        n_features=y_dim,
        n_components=n_components,
        eta=eta,
        dtype=dtype,
        normalize_by_batch=True,
        seed=seed,
    ).to(device)

    mu_t = torch.as_tensor(mu, device=device, dtype=dtype)

    for _ep in tqdm(range(int(epochs)), desc="PCA epochs", unit="epoch"):
        for _, _, Yb in tqdm(iter_Y_batches(desc_h5, cfg, batch_size=batch_size),
                              total=n_batches, desc=f"Epoch {_ep+1}/{epochs}", leave=False, unit="batch"):
            xb = torch.as_tensor(Yb, device=device, dtype=dtype)
            xb = xb - mu_t
            model(xb)

    return model.Q.detach().clone()  # (y_dim, n_components)


def estimate_pca_energy_streaming(
    desc_h5: Path,
    cfg: Config,
    mu: np.ndarray,
    Q: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, float]:
    """
    Returns:
      eig_sums[j] = sum over samples of (score_j^2)
      total_energy = sum over samples of ||y - mu||^2
    """
    K = int(Q.shape[1])
    meta = _get_desc_meta(desc_h5)
    N = int(meta["N"])
    n_batches = int(np.ceil(N / batch_size))
    
    mu_t = torch.as_tensor(mu, device=device, dtype=dtype)
    Qd = Q.to(device=device, dtype=dtype)

    eig_sums = torch.zeros((K,), device="cpu", dtype=torch.float64)
    total_energy = 0.0

    for _, _, Yb in tqdm(iter_Y_batches(desc_h5, cfg, batch_size=batch_size),
                         total=n_batches, desc="Computing PCA energy", unit="batch"):
        xb = torch.as_tensor(Yb, device=device, dtype=dtype)
        xb = xb - mu_t
        total_energy += float((xb.double() * xb.double()).sum().cpu())
        z = xb @ Qd  # (B,K)
        eig_sums += (z.double().pow(2).sum(dim=0)).cpu()

    return eig_sums.numpy(), float(total_energy)


def project_Y_with_Q(
    desc_h5: Path,
    cfg: Config,
    mu: np.ndarray,
    Qm: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    meta = _get_desc_meta(desc_h5)
    N = int(meta["N"])
    m = int(Qm.shape[1])

    mu_t = torch.as_tensor(mu, device=device, dtype=dtype)
    Qd = Qm.to(device=device, dtype=dtype)

    Yp = np.empty((N, m), dtype=np.float32)

    for i0, i1, Yb in iter_Y_batches(desc_h5, cfg, batch_size=batch_size):
        xb = torch.as_tensor(Yb, device=device, dtype=dtype)
        xb = xb - mu_t
        z = xb @ Qd
        Yp[i0:i1, :] = z.detach().to("cpu", dtype=torch.float32).numpy()

    return Yp


# ----------------------------- injectivity core -----------------------------

@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute k nearest neighbors in Y (dense all-pairs)."""
    N = Y.shape[0]
    k = min(int(k), N - 1)

    Dc = torch.cdist(Y, Y, p=2.0)  # (N,N)
    Dc.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(Dc, k=k, largest=False, sorted=True)
    return idx, vals


@torch.no_grad()
def local_explainedcov_metrics_LOO(
    X: torch.Tensor,
    Y: torch.Tensor,
    idxY: torch.Tensor,
    dY: torch.Tensor,
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

        Xs = Xc * sw[:, :, None]
        Ys = Yc * sw[:, :, None]

        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))
        Kmat = 0.5 * (Kmat + Kmat.transpose(1, 2))

        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(ridge_y) * trK / float(k)).to(torch.float32)
        lam_b = lam.clamp_min(1e-30)

        # LOO-uninformative diagnostics (geometry-only)
        eta, V = torch.linalg.eigh(Kmat)
        eta = eta.clamp_min(0.0)

        v0 = V[:, :, 0]
        s = sw
        align = torch.abs((v0 * s).sum(dim=1))

        eta0 = eta[:, 0]
        eta1 = eta[:, 1].clamp_min(1e-30) if k >= 2 else torch.ones_like(eta0)
        gap = (eta0 / eta1).to(torch.float32)

        dof = (eta / (eta + lam_b[:, None])).sum(dim=1)
        dof_norm = (dof / denom_km1).clamp(0.0, 1.0).to(torch.float32)

        loo_uninf = (align > float(loo_uninf_align_thr)) & (gap < float(loo_uninf_gap_thr)) & (dof_norm > float(loo_uninf_dof_thr))

        uninf_align[i0:i1] = align
        uninf_gap[i0:i1] = gap
        uninf_dof_norm[i0:i1] = dof_norm
        uninf_flag[i0:i1] = loo_uninf

        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]
        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))
        alpha = torch.bmm(Hinv, Xs)

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

        varX = (Xs * Xs).sum(dim=1)
        varR = (Rloo * Rloo).sum(dim=1)
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

        loo_uninformative_flag=uninf_flag.detach().cpu().numpy().astype(np.int32),
        loo_uninformative_align=uninf_align.detach().cpu().numpy(),
        loo_uninformative_gap=uninf_gap.detach().cpu().numpy(),
        loo_uninformative_dof_norm=uninf_dof_norm.detach().cpu().numpy(),
    )
    if p >= 2 and unexpl_coord_1 is not None:
        out["unexplained_coord1"] = unexpl_coord_1.detach().cpu().numpy()
    return out


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Potts descriptors for injectivity (with global Oja-PCA option)")

    ap.add_argument("--h5", type=str, default=Config.h5, help="Input descriptor HDF5.")
    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    ap.add_argument("--prepend_phase_fractions_to_Y", type=bool, default=Config.prepend_phase_fractions_to_Y)

    # Global PCA
    ap.add_argument("--use_global_pca", type=int, default=1 if Config.use_global_pca else 0)
    ap.add_argument("--pca_method", type=str, default=Config.pca_method, choices=["auto", "svd", "oja"])
    ap.add_argument("--pca_svd_max_gb", type=float, default=Config.pca_svd_max_gb)
    ap.add_argument("--pca_energy_frac", type=float, default=Config.pca_energy_frac)
    ap.add_argument("--pca_components_max", type=int, default=Config.pca_components_max)
    ap.add_argument("--pca_epochs", type=int, default=Config.pca_epochs)
    ap.add_argument("--pca_batch_size", type=int, default=Config.pca_batch_size)
    ap.add_argument("--pca_eta", type=float, default=Config.pca_eta)
    ap.add_argument("--pca_device", type=str, default=Config.pca_device)
    ap.add_argument("--pca_dtype", type=str, default=Config.pca_dtype, choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--pca_seed", type=int, default=Config.pca_seed)

    # Injectivity
    ap.add_argument("--standardize_X", type=bool, default=Config.standardize_X)
    ap.add_argument("--standardize_Y", type=bool, default=Config.standardize_Y)
    ap.add_argument("--kY", type=int, default=Config.kY)
    ap.add_argument("--use_weights", action="store_true")
    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)
    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--device", type=str, default=Config.device)

    # Plots
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
        potts_analysis_dir=str(args.potts_analysis_dir),
        h5=str(args.h5),
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),

        use_global_pca=bool(int(args.use_global_pca)),
        pca_method=str(args.pca_method),
        pca_svd_max_gb=float(args.pca_svd_max_gb),
        pca_energy_frac=float(args.pca_energy_frac),
        pca_components_max=int(args.pca_components_max),
        pca_epochs=int(args.pca_epochs),
        pca_batch_size=int(args.pca_batch_size),
        pca_eta=float(args.pca_eta),
        pca_device=str(args.pca_device),
        pca_dtype=str(args.pca_dtype),
        pca_seed=int(args.pca_seed),

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

    desc_h5 = Path(cfg.h5).expanduser().resolve()
    if not desc_h5.exists():
        raise FileNotFoundError(f"Input not found: {desc_h5}")

    X, xmeta = load_X_and_basic_meta(desc_h5)
    meta0 = _get_desc_meta(desc_h5)
    descriptor = meta0["descriptor"]
    N = int(meta0["N"])

    analysis_root = Path(cfg.potts_analysis_dir)
    session_dir = analysis_root / _run_folder_name_utc() / desc_h5.stem
    out_root = ensure_dir(session_dir / descriptor)
    figs_dir = ensure_dir(out_root / "figs")

    # ------------------ Global PCA (streaming or SVD) ------------------
    pca_info: Dict[str, Any] = dict(enabled=bool(cfg.use_global_pca))
    if cfg.use_global_pca:
        pca_dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[cfg.pca_dtype]
        pca_device = torch.device(cfg.pca_device)

        mu = compute_global_mean_Y(desc_h5, cfg, batch_size=cfg.pca_batch_size)
        np.save(out_root / "pca_mean.npy", mu)

        # Decide method: auto, svd, or oja
        method = cfg.pca_method
        if method == "auto":
            # Estimate memory for full Y matrix
            y_dim = len(mu)
            mem_gb = (N * y_dim * 4) / (1024**3)  # float32 = 4 bytes
            use_svd = (mem_gb <= cfg.pca_svd_max_gb) and pca_device.type == "cuda"
            method = "svd" if use_svd else "oja"
            print(f"Auto-selected PCA method: {method} (estimated data size: {mem_gb:.2f} GB)")
        
        if method == "svd":
            print(f"Using SVD-based PCA...")
            Q, eig_vals, total_energy = fit_svd_pca(
                desc_h5=desc_h5,
                cfg=cfg,
                mu=mu,
                n_components=int(cfg.pca_components_max),
                device=pca_device,
                dtype=pca_dtype,
            )
            eig_sums = eig_vals
            pca_info["method"] = "svd"
        else:
            print(f"Using Oja streaming PCA...")
            Q = fit_oja_pca_streaming(
                desc_h5=desc_h5,
                cfg=cfg,
                mu=mu,
                n_components=int(cfg.pca_components_max),
                epochs=int(cfg.pca_epochs),
                batch_size=int(cfg.pca_batch_size),
                eta=float(cfg.pca_eta),
                device=pca_device,
                dtype=pca_dtype,
                seed=int(cfg.pca_seed),
            )

            eig_sums, total_energy = estimate_pca_energy_streaming(
                desc_h5=desc_h5,
                cfg=cfg,
                mu=mu,
                Q=Q,
                batch_size=int(cfg.pca_batch_size),
                device=pca_device,
                dtype=pca_dtype,
            )
            pca_info["method"] = "oja"

        np.save(out_root / "pca_eigsums.npy", eig_sums.astype(np.float64))
        np.save(out_root / "pca_energy_total.npy", np.array([total_energy], dtype=np.float64))
        torch.save(Q.to("cpu", dtype=torch.float32), out_root / "pca_components.pt")

        cum = np.cumsum(eig_sums) / max(total_energy, 1e-30)
        plot_pca_cum_energy(cum, figs_dir / "pca_cumulative_energy", dpi=cfg.dpi)

        m = int(np.searchsorted(cum, float(cfg.pca_energy_frac)) + 1)
        m = max(1, min(m, int(cfg.pca_components_max)))
        print(f"Selected {m} PCA components to capture {cfg.pca_energy_frac*100:.1f}% energy (achieved {cum[m-1]*100:.2f}%)")

        Qm = Q[:, :m].detach().clone()

        Yp = project_Y_with_Q(
            desc_h5=desc_h5,
            cfg=cfg,
            mu=mu,
            Qm=Qm,
            batch_size=int(cfg.pca_batch_size),
            device=pca_device,
            dtype=pca_dtype,
        )

        pca_info.update(dict(
            energy_target=float(cfg.pca_energy_frac),
            components_max=int(cfg.pca_components_max),
            epochs=int(cfg.pca_epochs) if pca_info.get("method") == "oja" else None,
            batch_size=int(cfg.pca_batch_size),
            eta=float(cfg.pca_eta) if pca_info.get("method") == "oja" else None,
            dtype=str(cfg.pca_dtype),
            device=str(cfg.pca_device),
            seed=int(cfg.pca_seed) if pca_info.get("method") == "oja" else None,
            selected_components=int(m),
            achieved_energy=float(cum[m - 1]) if cum.size >= m else float("nan"),
        ))
    else:
        # Fallback: materialize full Y into RAM (same semantics as original)
        metaY = _get_desc_meta(desc_h5)
        y_dim = int(metaY["feat_dim"] + (metaY["ph_dim"] if cfg.prepend_phase_fractions_to_Y else 0))
        Yp = np.empty((N, y_dim), dtype=np.float32)
        for i0, i1, Yb in iter_Y_batches(desc_h5, cfg, batch_size=256):
            Yp[i0:i1, :] = Yb
        pca_info.update(dict(note="global_pca disabled; using raw Y"))

    # ------------------ Injectivity computation ------------------
    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    X_use = X.copy()
    Y_use = Yp.copy()

    if cfg.standardize_X:
        X_use = standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = standardize_np(Y_use)

    Xt = to_t(X_use, device=device)
    Yt = to_t(Y_use, device=device)

    idxY_t, dY_t = knn_in_y(Yt, k=int(cfg.kY))

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

    # ------------------ Save CSV ------------------
    csv_path = out_root / "potts_local_explainedcov_injectivity.csv"

    header_cols = [
        "temperature", "fraction_initial",
        "unexplained_frac", "explained_frac",
        "worst_unexplained_ratio", "worst_retention",
        "trX", "trR",
        "avg_dY",
        "unexplained_coord0",
        "unexplained_coord_max",
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

    # ------------------ Plots ------------------
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

        trXv = metrics["trX"]
        exc = trXv >= np.quantile(trXv, 0.25)
        flag_ok = ((metrics["explained_frac"] >= 0.9) & exc).astype(np.float64)
        scatter_tf(temp, frac, flag_ok,
                   f"{descriptor}: indicator explained_frac>=0.9 & trX>=Q25",
                   figs_dir / "scatter_ok_indicator", dpi=cfg.dpi, vmin=0.0, vmax=1.0)

        # LOO-uninformative
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

    flagged_frac = float(np.mean(metrics["loo_uninformative_flag"].astype(np.float64)))

    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor,
        N=int(N),
        y_dim=int(Y_use.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        global_pca=pca_info,
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
    # Merge metadata dicts carefully to avoid key conflicts
    merged_meta = dict(meta0)
    merged_meta["X_shape"] = xmeta["X_shape"]
    merged_meta["Y_projected_shape"] = Y_use.shape
    
    inj_meta = dict(
        created_utc=_utc_now_z(),
        config=asdict(cfg),
        files=dict(
            csv=str(csv_path),
            figs=str(figs_dir),
            pca_mean=str(out_root / "pca_mean.npy") if cfg.use_global_pca else None,
            pca_components=str(out_root / "pca_components.pt") if cfg.use_global_pca else None,
            pca_eigsums=str(out_root / "pca_eigsums.npy") if cfg.use_global_pca else None,
            pca_energy_total=str(out_root / "pca_energy_total.npy") if cfg.use_global_pca else None,
        ),
        summary=summary,
        load_metadata=merged_meta,
    )
    meta_path = out_root / "metadata_local_explainedcov_injectivity.json"
    meta_path.write_text(json.dumps(inj_meta, indent=2))


if __name__ == "__main__":
    main()
