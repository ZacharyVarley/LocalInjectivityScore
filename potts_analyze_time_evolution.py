#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_analyze_time_evolution.py

Analyze Potts descriptor snapshots at multiple steps and generate a 3-panel
figure showing how the explained-fraction landscape changes over time.

This script expects separate descriptor HDF5 files per snapshot (e.g., the
outputs of potts_descriptors.py for step-100, step-200, step-300 datasets).
Outputs are written under a non-timestamped folder for privacy.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from potts_descriptors import (
    _pair_labels,
    compute_correlations_2d,
    compute_phase_fractions,
    compute_radial_average,
)


DescKind = Literal["radial1d", "corr2d"]


# ----------------------------- time / io -----------------------------

def _utc_now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class Config:
    potts_analysis_dir: str = "potts_analysis_time_evolution"
    run_name: str = "time_evolution"

    # Descriptor computation (used when raw H5 is supplied)
    desc_descriptor: DescKind = "corr2d"
    desc_n_radial_bins: int = 64
    desc_whiten_eps: float = 0.0
    desc_batch_size: int = 128
    desc_device: str = "cuda"

    prepend_phase_fractions_to_Y: bool = True

    # Global PCA via Oja or SVD
    use_global_pca: bool = True
    pca_method: str = "auto"  # "auto", "svd", or "oja"
    pca_svd_max_gb: float = 4.0  # max GB for SVD in auto mode
    pca_energy_frac: float = 0.95
    pca_components_max: int = 100
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
    hm_bins_temp: int = 60
    hm_bins_frac: int = 60
    hm_sigma_px: float = 1.0
    hm_clip: Tuple[float, float] = (1.0, 99.0)


# ----------------------------- descriptor creation from raw H5 -----------------------------

def compute_descriptors_from_raw_step(
    raw_h5: Path,
    step: int,
    out_h5: Path,
    out_json: Path,
    cfg: Config,
) -> Path:
    """
    Compute descriptors from a raw Potts H5 snapshot dataset and write a descriptor H5.
    This keeps the analysis pipeline identical to descriptor-file input.
    """
    ensure_dir(out_h5.parent)

    use_cuda = torch.cuda.is_available() and str(cfg.desc_device).startswith("cuda")
    device = torch.device(cfg.desc_device if use_cuda else "cpu")

    created_utc = _utc_now_z()

    with h5py.File(str(raw_h5), "r") as fin:
        temps = np.array(fin["parameters/temperature"], dtype=np.float32)
        fracs_init = np.array(fin["parameters/fraction_initial"], dtype=np.float32)
        spins_ds = fin[f"states/final_spins_step{int(step)}"]
        N, R, H, W = spins_ds.shape
        q = int(fin.attrs.get("q", 3))

        n_pairs = q * (q + 1) // 2
        pair_labels = _pair_labels(q)

        max_r = H // 2
        edges = torch.linspace(0, float(max_r), int(cfg.desc_n_radial_bins) + 1)
        radial_bins = ((edges[:-1] + edges[1:]) * 0.5).cpu().numpy().astype(np.float32)

        mean2d = np.zeros((N, n_pairs, H, W), dtype=np.float32)
        std2d = np.zeros((N, n_pairs, H, W), dtype=np.float32)
        mean1d = np.zeros((N, n_pairs, int(cfg.desc_n_radial_bins)), dtype=np.float32)
        std1d = np.zeros((N, n_pairs, int(cfg.desc_n_radial_bins)), dtype=np.float32)
        meanph = np.zeros((N, q), dtype=np.float32)
        stdph = np.zeros((N, q), dtype=np.float32)

        batch_size = min(int(cfg.desc_batch_size), N)
        for i in range(0, N, batch_size):
            i_end = min(i + batch_size, N)
            batch_n = i_end - i

            max_elements = 50 * 1024 * 1024
            repeat_batch = max(1, min(R, max_elements // (batch_n * H * W)))

            sum2d = np.zeros((batch_n, n_pairs, H, W), dtype=np.float64)
            sumsq2d = np.zeros((batch_n, n_pairs, H, W), dtype=np.float64)
            sum1d = np.zeros((batch_n, n_pairs, int(cfg.desc_n_radial_bins)), dtype=np.float64)
            sumsq1d = np.zeros((batch_n, n_pairs, int(cfg.desc_n_radial_bins)), dtype=np.float64)
            sumph = np.zeros((batch_n, q), dtype=np.float64)
            sumsqph = np.zeros((batch_n, q), dtype=np.float64)

            for r in range(0, R, repeat_batch):
                r_end = min(r + repeat_batch, R)
                r_batch = r_end - r

                spins_batch = torch.as_tensor(spins_ds[i:i_end, r:r_end], device=device)
                spins_batch = spins_batch.reshape(batch_n * r_batch, 1, H, W)

                corr2d_batch = compute_correlations_2d(spins_batch, q=q, whiten_eps=cfg.desc_whiten_eps)
                rad_batch = compute_radial_average(
                    corr2d_batch,
                    n_bins=int(cfg.desc_n_radial_bins),
                    whiten_eps=cfg.desc_whiten_eps,
                )
                ph_batch = compute_phase_fractions(spins_batch, q=q)

                corr2d_batch = corr2d_batch.reshape(batch_n, r_batch, n_pairs, H, W)
                rad_batch = rad_batch.reshape(batch_n, r_batch, n_pairs, int(cfg.desc_n_radial_bins))
                ph_batch = ph_batch.reshape(batch_n, r_batch, q)

                sum2d += corr2d_batch.sum(dim=1).cpu().numpy()
                sum1d += rad_batch.sum(dim=1).cpu().numpy()
                sumsq1d += (rad_batch ** 2).sum(dim=1).cpu().numpy()
                sumph += ph_batch.sum(dim=1).cpu().numpy()
                sumsqph += (ph_batch ** 2).sum(dim=1).cpu().numpy()
                sumsq2d += (corr2d_batch ** 2).sum(dim=1).cpu().numpy()

                del spins_batch, corr2d_batch, rad_batch, ph_batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            mean2d[i:i_end] = (sum2d / R).astype(np.float32)
            var2d = (sumsq2d / R) - (sum2d / R) ** 2
            std2d[i:i_end] = np.sqrt(np.maximum(var2d, 0.0)).astype(np.float32)
            mean1d[i:i_end] = (sum1d / R).astype(np.float32)
            var1d = (sumsq1d / R) - (sum1d / R) ** 2
            std1d[i:i_end] = np.sqrt(np.maximum(var1d, 0.0)).astype(np.float32)
            meanph[i:i_end] = (sumph / R).astype(np.float32)
            varph = (sumsqph / R) - (sumph / R) ** 2
            stdph[i:i_end] = np.sqrt(np.maximum(varph, 0.0)).astype(np.float32)

    with h5py.File(str(out_h5), "w") as fout:
        fout.attrs["created_utc"] = created_utc
        fout.attrs["input_h5"] = str(raw_h5)
        fout.attrs["descriptor"] = cfg.desc_descriptor
        fout.attrs["config_json"] = json.dumps(
            dict(
                descriptor=str(cfg.desc_descriptor),
                n_radial_bins=int(cfg.desc_n_radial_bins),
                whiten_eps=float(cfg.desc_whiten_eps),
                batch_size=int(cfg.desc_batch_size),
                device=str(cfg.desc_device),
                source_step=int(step),
            )
        )
        fout.attrs["n_parameters"] = N
        fout.attrs["n_repeats"] = R
        fout.attrs["grid_size"] = H
        fout.attrs["q"] = q
        fout.attrs["n_correlation_pairs"] = n_pairs
        fout.attrs["whiten_eps"] = float(cfg.desc_whiten_eps)

        gp = fout.create_group("parameters")
        gp.create_dataset("temperature", data=temps)
        gp.create_dataset("fraction_initial", data=fracs_init)

        gm = fout.create_group("metadata")
        gm.create_dataset("radial_bins", data=radial_bins)
        gm.create_dataset("pair_labels", data=np.array([s.encode() for s in pair_labels], dtype="S"))
        gm.create_dataset("phase_labels", data=np.arange(q, dtype=np.int32))

        gc = fout.create_group("correlations")
        gc.create_dataset("correlations_2d_mean", data=mean2d)
        gc.create_dataset("correlations_2d_std", data=std2d)
        gc.create_dataset("correlations_radial_mean", data=mean1d)
        gc.create_dataset("correlations_radial_std", data=std1d)

        gpz = fout.create_group("phases")
        gpz.create_dataset("final_fraction_mean", data=meanph)
        gpz.create_dataset("final_fraction_std", data=stdph)

    meta = dict(
        created_utc=created_utc,
        input_h5=str(raw_h5),
        output_h5=str(out_h5),
        descriptor=str(cfg.desc_descriptor),
        shapes=dict(
            correlations_2d_mean=[int(N), int(n_pairs), int(H), int(W)],
            correlations_radial_mean=[int(N), int(n_pairs), int(cfg.desc_n_radial_bins)],
            correlations_radial_std=[int(N), int(n_pairs), int(cfg.desc_n_radial_bins)],
            correlations_2d_std=[int(N), int(n_pairs), int(H), int(W)],
            final_fraction_mean=[int(N), int(q)],
            final_fraction_std=[int(N), int(q)],
        ),
        pair_labels=pair_labels,
        radial_bins=radial_bins.tolist(),
        source_step=int(step),
    )
    out_json.write_text(json.dumps(meta, indent=2))
    return out_h5


# ----------------------------- plotting helpers -----------------------------

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
    out = np.divide(
        val,
        np.maximum(msk, 1e-12),
        out=np.zeros_like(val),
        where=(msk > 1e-12),
    )
    out[msk < 1e-12] = np.nan
    return out


def heatmap_binned_tf_ax(
    ax: plt.Axes,
    temp: np.ndarray,
    frac: np.ndarray,
    Z: np.ndarray,
    title: str,
    bins_t: int = 60,
    bins_f: int = 60,
    sigma_px: float = 1.0,
    clip=(1, 99),
    vmin=None,
    vmax=None,
    cmap="viridis",
    xlabel: str = "temperature",
    ylabel: str = "fraction_initial",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    temp = temp.astype(np.float64)
    frac = frac.astype(np.float64)
    Z = Z.astype(np.float64)

    tmin, tmax = float(np.min(temp)), float(np.max(temp))
    fmin, fmax = float(np.min(frac)), float(np.max(frac))

    sum_w, tx, fx = np.histogram2d(
        temp,
        frac,
        bins=[bins_t, bins_f],
        range=[[tmin, tmax], [fmin, fmax]],
        weights=Z,
    )
    cnt, _, _ = np.histogram2d(temp, frac, bins=[tx, fx], range=[[tmin, tmax], [fmin, fmax]])

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

    im = ax.imshow(
        img_s.T,
        origin="lower",
        extent=[tx[0], tx[-1], fx[0], fx[-1]],
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return img_s, tx, fx


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
        B = int(x.shape[0])
        if self.normalize_by_batch and B > 0:
            scale = 1.0 / float(B)
        else:
            scale = 1.0

        upd = (x.T @ (x @ self.Q)) * scale
        self.Q.copy_(torch.linalg.qr(self.Q + self.eta * upd, mode="reduced")[0])
        self.step.add_(1)

    def get_components(self) -> torch.Tensor:
        return self.Q.T

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Q


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
    meta = _get_desc_meta(desc_h5)
    N = int(meta["N"])
    y_dim = int(meta["feat_dim"] + (meta["ph_dim"] if cfg.prepend_phase_fractions_to_Y else 0))

    print(f"Loading full data matrix ({N} x {y_dim})...")
    Y = np.empty((N, y_dim), dtype=np.float32)
    n_batches = int(np.ceil(N / 256))
    for i0, i1, Yb in tqdm(iter_Y_batches(desc_h5, cfg, batch_size=256), total=n_batches, desc="Loading Y", unit="batch"):
        Y[i0:i1, :] = Yb

    print("Centering data...")
    Y = Y - mu[None, :]

    print(f"Computing SVD on {device}...")
    Y_t = torch.as_tensor(Y, device=device, dtype=dtype)
    U, S, Vt = torch.linalg.svd(Y_t, full_matrices=False)
    V = Vt.T

    k = min(n_components, V.shape[1])
    Q = V[:, :k].contiguous()

    eig_vals = (S[:k].double().pow(2)).cpu().numpy()
    total_energy = float((S.double().pow(2).sum()).cpu())

    del Y, Y_t, U, S, Vt, V
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        for _, _, Yb in tqdm(
            iter_Y_batches(desc_h5, cfg, batch_size=batch_size),
            total=n_batches,
            desc=f"Epoch {_ep+1}/{epochs}",
            leave=False,
            unit="batch",
        ):
            xb = torch.as_tensor(Yb, device=device, dtype=dtype)
            xb = xb - mu_t
            model(xb)

    return model.Q.detach().clone()


def estimate_pca_energy_streaming(
    desc_h5: Path,
    cfg: Config,
    mu: np.ndarray,
    Q: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, float]:
    K = int(Q.shape[1])
    meta = _get_desc_meta(desc_h5)
    N = int(meta["N"])
    n_batches = int(np.ceil(N / batch_size))

    mu_t = torch.as_tensor(mu, device=device, dtype=dtype)
    Qd = Q.to(device=device, dtype=dtype)

    eig_sums = torch.zeros((K,), device="cpu", dtype=torch.float64)
    total_energy = 0.0

    for _, _, Yb in tqdm(iter_Y_batches(desc_h5, cfg, batch_size=batch_size), total=n_batches, desc="Computing PCA energy", unit="batch"):
        xb = torch.as_tensor(Yb, device=device, dtype=dtype)
        xb = xb - mu_t
        total_energy += float((xb.double() * xb.double()).sum().cpu())
        z = xb @ Qd
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
    N = Y.shape[0]
    k = min(int(k), N - 1)

    Dc = torch.cdist(Y, Y, p=2.0)
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

    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

    for i0 in range(0, N, int(batch_size)):
        i1 = min(N, i0 + int(batch_size))
        B = i1 - i0

        centers = torch.arange(i0, i1, device=device, dtype=torch.int64)
        neigh = torch.cat([centers[:, None], idxY[i0:i1]], dim=1)

        dn = torch.cat([
            torch.zeros((B, 1), device=device, dtype=torch.float32),
            dY[i0:i1].to(torch.float32),
        ], dim=1)
        avg_dy[i0:i1] = dn[:, 1:].mean(dim=1)

        Xn = X[neigh]
        Yn = Y[neigh]

        if use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(eps_tau))
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
        Kmat = 0.5 * (Kmat + Kmat.transpose(1, 2))

        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (float(ridge_y) * trK / float(k)).to(torch.float32)

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
        unexplained_coord_max=unexpl_coord_max.detach().cpu().numpy(),
        avg_dY=avg_dy.detach().cpu().numpy(),
    )
    return out


# ----------------------------- analysis per snapshot -----------------------------

def analyze_snapshot(desc_h5: Path, cfg: Config, out_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    if not desc_h5.exists():
        raise FileNotFoundError(f"Input not found: {desc_h5}")

    X, xmeta = load_X_and_basic_meta(desc_h5)
    meta0 = _get_desc_meta(desc_h5)
    descriptor = meta0["descriptor"]
    N = int(meta0["N"])

    figs_dir = ensure_dir(out_root / "figs")

    # ------------------ Global PCA (streaming or SVD) ------------------
    pca_info: Dict[str, Any] = dict(enabled=bool(cfg.use_global_pca))
    if cfg.use_global_pca:
        pca_dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[cfg.pca_dtype]
        pca_device = torch.device(cfg.pca_device)

        mu = compute_global_mean_Y(desc_h5, cfg, batch_size=cfg.pca_batch_size)

        method = cfg.pca_method
        if method == "auto":
            y_dim = len(mu)
            mem_gb = (N * y_dim * 4) / (1024**3)
            use_svd = (mem_gb <= cfg.pca_svd_max_gb) and pca_device.type == "cuda"
            method = "svd" if use_svd else "oja"
            print(f"Auto-selected PCA method: {method} (estimated data size: {mem_gb:.2f} GB)")

        if method == "svd":
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

        torch.save(Q.to("cpu", dtype=torch.float32), out_root / "pca_components.pt")

        cum = np.cumsum(eig_sums) / max(total_energy, 1e-30)
        plot_pca_cum_energy(cum, figs_dir / "pca_cumulative_energy", dpi=cfg.dpi)

        m = int(np.searchsorted(cum, float(cfg.pca_energy_frac)) + 1)
        m = max(3, min(m, int(cfg.pca_components_max)))
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
    )

    # ------------------ Save CSV ------------------
    csv_path = out_root / "potts_local_explainedcov_injectivity.csv"

    header_cols = [
        "temperature",
        "fraction_initial",
        "unexplained_frac",
        "explained_frac",
        "worst_unexplained_ratio",
        "worst_retention",
        "trX",
        "trR",
        "avg_dY",
        "unexplained_coord_max",
    ]

    cols = [
        X[:, 0],
        X[:, 1],
        metrics["unexplained_frac"],
        metrics["explained_frac"],
        metrics["worst_unexplained_ratio"],
        metrics["worst_retention"],
        metrics["trX"],
        metrics["trR"],
        metrics["avg_dY"],
        metrics["unexplained_coord_max"],
    ]

    data = np.column_stack(cols).astype(np.float64)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")

    # ------------------ Metadata ------------------
    summary = dict(
        input_descriptor_h5=str(desc_h5),
        descriptor=descriptor,
        N=int(N),
        y_dim=int(Y_use.shape[1]),
        kY=int(cfg.kY),
        use_weights=bool(cfg.use_weights),
        ridge_y=float(cfg.ridge_y),
        global_pca=pca_info,
    )
    merged_meta = dict(meta0)
    merged_meta["X_shape"] = xmeta["X_shape"]
    merged_meta["Y_projected_shape"] = Y_use.shape

    inj_meta = dict(
        created_utc=_utc_now_z(),
        config=asdict(cfg),
        files=dict(
            csv=str(csv_path),
            figs=str(figs_dir),
            pca_components=str(out_root / "pca_components.pt") if cfg.use_global_pca else None,
        ),
        summary=summary,
        load_metadata=merged_meta,
    )
    meta_path = out_root / "metadata_local_explainedcov_injectivity.json"
    meta_path.write_text(json.dumps(inj_meta, indent=2))

    return X, Yp, metrics, pca_info


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Potts descriptor snapshots across time.")

    ap.add_argument(
        "--raw_h5",
        type=str,
        default="potts_data_time_evolution/time_evolution/potts_sims_q3_128x128_steps300.h5",
        help="Raw time-evolution HDF5 from potts_gen_time_evolution.py",
    )
    ap.add_argument(
        "--snapshot_steps",
        type=str,
        default="100,200,300",
        help="Comma-separated list of snapshot steps to analyze, e.g. '10,50,100,200,300'",
    )
    ap.add_argument("--h5_step100", type=str, default="", help="Descriptor HDF5 for step 100 (deprecated; use --snapshot_steps)")
    ap.add_argument("--h5_step200", type=str, default="", help="Descriptor HDF5 for step 200 (deprecated; use --snapshot_steps)")
    ap.add_argument("--h5_step300", type=str, default="", help="Descriptor HDF5 for step 300 (deprecated; use --snapshot_steps)")

    ap.add_argument("--potts_analysis_dir", type=str, default=Config.potts_analysis_dir)
    ap.add_argument("--run_name", type=str, default=Config.run_name)
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
    ap.add_argument("--hm_bins_temp", type=int, default=Config.hm_bins_temp)
    ap.add_argument("--hm_bins_frac", type=int, default=Config.hm_bins_frac)
    ap.add_argument("--hm_sigma_px", type=float, default=Config.hm_sigma_px)
    ap.add_argument("--hm_clip_lo", type=float, default=Config.hm_clip[0])
    ap.add_argument("--hm_clip_hi", type=float, default=Config.hm_clip[1])

    # Descriptor computation (only used when raw_h5 is provided)
    ap.add_argument("--desc_descriptor", type=str, default=Config.desc_descriptor, choices=["radial1d", "corr2d"])
    ap.add_argument("--desc_n_radial_bins", type=int, default=Config.desc_n_radial_bins)
    ap.add_argument("--desc_whiten_eps", type=float, default=Config.desc_whiten_eps)
    ap.add_argument("--desc_batch_size", type=int, default=Config.desc_batch_size)
    ap.add_argument("--desc_device", type=str, default=Config.desc_device)

    args = ap.parse_args()

    cfg = Config(
        potts_analysis_dir=str(args.potts_analysis_dir),
        run_name=str(args.run_name),
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),

        desc_descriptor=str(args.desc_descriptor),  # type: ignore
        desc_n_radial_bins=int(args.desc_n_radial_bins),
        desc_whiten_eps=float(args.desc_whiten_eps),
        desc_batch_size=int(args.desc_batch_size),
        desc_device=str(args.desc_device),

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
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        hm_clip=(float(args.hm_clip_lo), float(args.hm_clip_hi)),
    )

    raw_h5 = Path(args.raw_h5).expanduser().resolve() if str(args.raw_h5).strip() else None
    snap_steps = sorted({int(s.strip()) for s in args.snapshot_steps.split(",") if s.strip()})
    # Per-step h5 overrides (legacy; only used for the three original steps)
    _legacy_h5 = {
        100: str(args.h5_step100).strip(),
        200: str(args.h5_step200).strip(),
        300: str(args.h5_step300).strip(),
    }
    steps = [(s, _legacy_h5.get(s, "")) for s in snap_steps]

    out_root = ensure_dir(Path(cfg.potts_analysis_dir) / cfg.run_name)
    figs_dir = ensure_dir(out_root / "figs")

    # Analyze each snapshot
    snapshot_results = {}
    X_ref = None
    for step, h5_arg in steps:
        step_dir = ensure_dir(out_root / f"step_{step}")
        if h5_arg:
            h5_path = Path(h5_arg).expanduser().resolve()
        else:
            if raw_h5 is None:
                raise ValueError("No descriptor H5 provided and --raw_h5 is empty.")
            if not raw_h5.exists():
                raise FileNotFoundError(f"Raw H5 not found: {raw_h5}")
            desc_h5 = step_dir / f"descriptors_step{step}_{cfg.desc_descriptor}.h5"
            desc_json = step_dir / f"descriptors_step{step}_{cfg.desc_descriptor}.json"
            if not desc_h5.exists():
                print(f"\n[time_evolution] Building descriptors for step {step} from raw H5: {raw_h5}")
                compute_descriptors_from_raw_step(
                    raw_h5=raw_h5,
                    step=step,
                    out_h5=desc_h5,
                    out_json=desc_json,
                    cfg=cfg,
                )
            h5_path = desc_h5

        print(f"\n[time_evolution] Analyzing step {step}: {h5_path}")
        X, Yp, metrics, pca_info = analyze_snapshot(h5_path, cfg, step_dir)
        snapshot_results[step] = dict(X=X, Yp=Yp, metrics=metrics, pca_info=pca_info, h5=str(h5_path))
        if X_ref is None:
            X_ref = X
        else:
            if not np.allclose(X_ref, X, rtol=1e-6, atol=1e-8):
                raise ValueError("Parameter grids differ across snapshots; cannot compare heatmaps reliably.")

    # Combined N-panel figure
    steps_sorted = snap_steps
    fig, axes = plt.subplots(1, len(steps_sorted), figsize=(5.1 * len(steps_sorted), 4.8), dpi=cfg.dpi)
    if len(steps_sorted) == 1:
        axes = [axes]
    last_im = None

    for ax, step in zip(axes, steps_sorted):
        res = snapshot_results[step]
        temp = res["X"][:, 0]
        frac = res["X"][:, 1]
        Z = res["metrics"]["explained_frac"]
        img_s, _, _ = heatmap_binned_tf_ax(
            ax,
            temp,
            frac,
            Z,
            title=f"Explained Fraction (step {step})",
            bins_t=cfg.hm_bins_temp,
            bins_f=cfg.hm_bins_frac,
            sigma_px=cfg.hm_sigma_px,
            clip=cfg.hm_clip,
            vmin=0.0,
            vmax=1.0,
            xlabel="T",
            ylabel="f0",
        )
        last_im = ax.images[-1] if ax.images else None

    if last_im is not None:
        _ = last_im

    fig.tight_layout()
    fig.savefig(str(figs_dir / "heatmap_explained_frac_time_evolution") + ".png", bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(str(figs_dir / "heatmap_explained_frac_time_evolution") + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # Top-level metadata summary
    meta = dict(
        created_utc=_utc_now_z(),
        run_name=str(cfg.run_name),
        potts_analysis_dir=str(cfg.potts_analysis_dir),
        steps={str(k): v["h5"] for k, v in snapshot_results.items()},
        summary={
            str(k): {
                "median_explained_frac": float(np.median(v["metrics"]["explained_frac"])),
                "median_unexplained_frac": float(np.median(v["metrics"]["unexplained_frac"])),
            }
            for k, v in snapshot_results.items()
        },
    )
    meta_path = out_root / "metadata_time_evolution.json"
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
