#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mnist_top_bottom_injectivity_umap.py

End-to-end injectivity-style analysis on MNIST:
  - X := top half of MNIST image
  - Y := bottom half of MNIST image
  - PCA(X) -> 50 dims, PCA(Y) -> 50 dims
  - kNN in Y with k=100
  - Local LOO "explained fraction" via ridge-stabilized kernel regression residual energy
  - Visualize on 2D UMAP of X:
      (left) color by digit label
      (right) color by explained fraction

Outputs (default):
  mnist_injectivity_out/<timestamp>/
    metrics.npz
    metrics.csv
    umap_labels.png
    umap_explained_frac.png
    umap_side_by_side.png
    run_metadata.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Requires: pip install umap-learn
import umap


# ----------------------------- utilities -----------------------------

def _utc_now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_folder_name_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_t(x: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class Config:
    out_dir: str = "mnist_injectivity_out"
    seed: int = 0

    # Data
    split_row: int = 14  # MNIST is 28x28; top=rows[0:14], bottom=rows[14:28]
    use_train: bool = True
    max_samples: int = 40000  # increase if you have GPU RAM/time

    # PCA
    pca_dim: int = 200
    standardize_X: bool = False
    standardize_Y: bool = False

    # kNN in Y
    kY: int = 300  # must be > pca_dim per requirement

    # LOO KRR / metric knobs
    use_weights: bool = False
    eps_tau: float = 1e-10
    ridge_y: float = 1e-3
    ridge_x: float = 1e-8
    eps_trace: float = 1e-18
    batch_size: int = 64

    # UMAP
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.05

    # Runtime
    device: str = "cuda"  # "cuda" or "cpu"

    # Plots
    dpi: int = 250
    point_size: float = 4.0


# ----------------------------- MNIST loading -----------------------------

def load_mnist_numpy(use_train: bool, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      images: (N, 28, 28) float32 in [0,1]
      labels: (N,) int64
    """
    import torchvision
    import torchvision.transforms as T

    ds = torchvision.datasets.MNIST(
        root="./data",
        train=bool(use_train),
        download=True,
        transform=T.ToTensor(),
    )

    N = len(ds)
    idx = np.arange(N, dtype=np.int64)

    if max_samples is not None and max_samples > 0 and max_samples < N:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_samples), replace=False)

    images = np.empty((idx.size, 28, 28), dtype=np.float32)
    labels = np.empty((idx.size,), dtype=np.int64)

    for j, i in enumerate(idx):
        x, y = ds[i]  # x: (1,28,28) float
        images[j] = x.numpy()[0]
        labels[j] = int(y)

    return images, labels


def build_XY_from_halves(images: np.ndarray, split_row: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    images: (N,28,28) float32
    X_raw: flattened top half (N, 14*28)
    Y_raw: flattened bottom half (N, 14*28)
    """
    top = images[:, :split_row, :]
    bot = images[:, split_row:, :]
    X_raw = top.reshape(images.shape[0], -1).astype(np.float32)
    Y_raw = bot.reshape(images.shape[0], -1).astype(np.float32)
    return X_raw, Y_raw


def pca_project(X_raw: np.ndarray, Y_raw: np.ndarray, pca_dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    PCA(X_raw)->X_pca, PCA(Y_raw)->Y_pca, each to pca_dim.
    """
    pcaX = PCA(n_components=int(pca_dim), random_state=int(seed), svd_solver="randomized")
    pcaY = PCA(n_components=int(pca_dim), random_state=int(seed), svd_solver="randomized")

    Xp = pcaX.fit_transform(X_raw).astype(np.float32)
    Yp = pcaY.fit_transform(Y_raw).astype(np.float32)

    info = {
        "pca_dim": int(pca_dim),
        "X_explained_variance_ratio_sum": float(np.sum(pcaX.explained_variance_ratio_)),
        "Y_explained_variance_ratio_sum": float(np.sum(pcaY.explained_variance_ratio_)),
        "X_singular_values_first5": pcaX.singular_values_[:5].astype(float).tolist(),
        "Y_singular_values_first5": pcaY.singular_values_[:5].astype(float).tolist(),
    }
    return Xp, Yp, info


# ----------------------------- kNN in Y -----------------------------

@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense all-pairs kNN in Y using torch.cdist. O(N^2).
    Returns:
      idx: (N,k) neighbor indices (excluding self)
      d:   (N,k) neighbor distances
    """
    N = int(Y.shape[0])
    k = min(int(k), N - 1)
    D = torch.cdist(Y, Y, p=2.0)  # (N,N)
    D.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(D, k=k, largest=False, sorted=True)
    return idx, vals


# ----------------------------- local explained fraction (LOO) -----------------------------

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
    """
    Matches the structure of your Potts script, generalized to p-dim X.
    For each center i, builds neighborhood in Y-space (i plus its kY neighbors),
    then computes LOO residual energy ratio u = tr(R)/tr(X) and explained fraction e = 1-u.

    Returns numpy arrays for:
      unexplained_frac, explained_frac, trX, trR, worst_retention
    """
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    expl = torch.empty((N,), device=device, dtype=torch.float32)
    trX = torch.empty((N,), device=device, dtype=torch.float32)
    trR = torch.empty((N,), device=device, dtype=torch.float32)
    worst_ret = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
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

        Xn = X[neigh]  # (B,k,p)
        Yn = Y[neigh]  # (B,k,qy)

        if use_weights:
            tau = dn.max(dim=1).values.clamp_min(float(eps_tau))
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)
        sw = torch.sqrt(w)

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

        # --- robust worst_retention computation ---
        SigmaX = torch.bmm(Xs.transpose(1, 2), Xs).to(torch.float32)
        SigmaR = torch.bmm(Rloo.transpose(1, 2), Rloo).to(torch.float32)

        # Symmetrize to kill numerical skew
        SigmaX = 0.5 * (SigmaX + SigmaX.transpose(1, 2))
        SigmaR = 0.5 * (SigmaR + SigmaR.transpose(1, 2))

        # Base ridge (as before) + a hard jitter floor
        gam = (float(ridge_x) * trX_b / float(max(p, 1))).to(torch.float32)
        jitter_floor = torch.full_like(gam, 1e-6)  # hard floor; tune 1e-7..1e-4 if needed
        gam = torch.maximum(gam, jitter_floor)

        Ipb = I_p[None, :, :].expand(B, p, p)
        SigmaXr = SigmaX + gam[:, None, None] * Ipb

        # Retry Cholesky with increasing jitter
        jitter = torch.zeros_like(gam)
        max_tries = 6
        L = None
        info = None
        for t in range(max_tries):
            Sigma_try = SigmaXr + jitter[:, None, None] * Ipb
            L_try, info_try = torch.linalg.cholesky_ex(Sigma_try)
            if int(info_try.max().item()) == 0:
                L, info = L_try, info_try
                break
            # increase jitter (per-batch); start at gam and grow
            if t == 0:
                jitter = gam.clone()
            else:
                jitter = jitter * 10.0

        if L is not None:
            Z = torch.linalg.solve_triangular(L, SigmaR, upper=False)
            M = torch.linalg.solve_triangular(L.transpose(1, 2), Z, upper=True)
            M = 0.5 * (M + M.transpose(1, 2))
            ev = torch.linalg.eigvalsh(M)
            wmax = ev[:, -1].clamp(0.0, 1.0)
            worst_ret_b = (1.0 - wmax).clamp(0.0, 1.0)
        else:
            # Fallback: eigen-regularize SigmaX to SPD then compute SigmaX^{-1}SigmaR stably
            # SigmaXr might be indefinite; project to SPD by clamping eigenvalues.
            evals, evecs = torch.linalg.eigh(SigmaXr)
            evals = torch.clamp(evals, min=1e-6)  # SPD projection
            invSigmaXr = evecs @ torch.diag_embed(1.0 / evals) @ evecs.transpose(1, 2)
            M = invSigmaXr @ SigmaR
            M = 0.5 * (M + M.transpose(1, 2))
            ev = torch.linalg.eigvalsh(M)
            wmax = ev[:, -1].clamp(0.0, 1.0)
            worst_ret_b = (1.0 - wmax).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexpl[i0:i1] = u
        expl[i0:i1] = e
        worst_ret[i0:i1] = worst_ret_b

    return {
        "unexplained_frac": unexpl.detach().cpu().numpy(),
        "explained_frac": expl.detach().cpu().numpy(),
        "trX": trX.detach().cpu().numpy(),
        "trR": trR.detach().cpu().numpy(),
        "worst_retention": worst_ret.detach().cpu().numpy(),
    }


# ----------------------------- plotting -----------------------------

def save_umap_plots(
    emb2: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
    out_dir: Path,
    dpi: int,
    s: float,
) -> None:
    x = emb2[:, 0]
    y = emb2[:, 1]

    # (1) labels
    fig = plt.figure(figsize=(6.2, 5.2), dpi=dpi)
    sc = plt.scatter(x, y, c=labels, s=s, cmap="tab10", alpha=0.85, linewidths=0)
    cb = plt.colorbar(sc, fraction=0.046, pad=0.03)
    cb.set_label("digit label")
    plt.title("UMAP(X): color = digit label")
    plt.tight_layout()
    fig.savefig(out_dir / "umap_labels.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    # (2) explained fraction
    fig = plt.figure(figsize=(6.2, 5.2), dpi=dpi)
    sc = plt.scatter(x, y, c=explained, s=s, cmap="viridis", alpha=0.85, linewidths=0, vmin=0.0, vmax=1.0)
    cb = plt.colorbar(sc, fraction=0.046, pad=0.03)
    cb.set_label("explained fraction")
    plt.title("UMAP(X): color = explained fraction")
    plt.tight_layout()
    fig.savefig(out_dir / "umap_explained_frac.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    # (3) side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=dpi)
    sc0 = axes[0].scatter(x, y, c=labels, s=s, cmap="tab10", alpha=0.85, linewidths=0)
    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("digit label")
    axes[0].set_title("color = label")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    sc1 = axes[1].scatter(x, y, c=explained, s=s, cmap="viridis", alpha=0.85, linewidths=0, vmin=0.0, vmax=1.0)
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("explained fraction")
    axes[1].set_title("color = explained fraction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle("MNIST top-half X (PCA-50) UMAP, injectivity diagnostics vs bottom-half Y (PCA-50)")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_side_by_side.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    # (4) side-by-side with log-scaled explained fraction
    eps = 1e-6
    log_explained = np.log10(np.clip(explained, eps, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=dpi)
    sc0 = axes[0].scatter(x, y, c=labels, s=s, cmap="tab10", alpha=0.85, linewidths=0)
    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("digit label")
    axes[0].set_title("color = label")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    sc1 = axes[1].scatter(
        x,
        y,
        c=log_explained,
        s=s,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
        vmin=np.log10(eps),
        vmax=0.0,
    )
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("log10(explained fraction)")
    axes[1].set_title("color = log10(explained fraction)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle("MNIST top-half X (PCA-50) UMAP, injectivity diagnostics vs bottom-half Y (PCA-50)")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_side_by_side_loge.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def write_csv(path: Path, labels: np.ndarray, explained: np.ndarray, unexplained: np.ndarray, trX: np.ndarray, trR: np.ndarray, worst_ret: np.ndarray) -> None:
    header = "idx,label,explained_frac,unexplained_frac,trX,trR,worst_retention"
    idx = np.arange(labels.size, dtype=np.int64)
    data = np.column_stack([idx, labels.astype(np.int64), explained, unexplained, trX, trR, worst_ret]).astype(np.float64)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=Config.out_dir)
    ap.add_argument("--seed", type=int, default=Config.seed)

    ap.add_argument("--use_train", type=int, default=1 if Config.use_train else 0)
    ap.add_argument("--max_samples", type=int, default=Config.max_samples)
    ap.add_argument("--split_row", type=int, default=Config.split_row)

    ap.add_argument("--pca_dim", type=int, default=Config.pca_dim)
    ap.add_argument("--standardize_X", type=int, default=1 if Config.standardize_X else 0)
    ap.add_argument("--standardize_Y", type=int, default=1 if Config.standardize_Y else 0)

    ap.add_argument("--kY", type=int, default=Config.kY)

    ap.add_argument("--ridge_y", type=float, default=Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=Config.ridge_x)
    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--device", type=str, default=Config.device)

    ap.add_argument("--umap_n_neighbors", type=int, default=Config.umap_n_neighbors)
    ap.add_argument("--umap_min_dist", type=float, default=Config.umap_min_dist)

    ap.add_argument("--dpi", type=int, default=Config.dpi)
    ap.add_argument("--point_size", type=float, default=Config.point_size)

    args = ap.parse_args()

    cfg = Config(
        out_dir=str(args.out_dir),
        seed=int(args.seed),
        use_train=bool(int(args.use_train)),
        max_samples=int(args.max_samples),
        split_row=int(args.split_row),
        pca_dim=int(args.pca_dim),
        standardize_X=bool(int(args.standardize_X)),
        standardize_Y=bool(int(args.standardize_Y)),
        kY=int(args.kY),
        ridge_y=float(args.ridge_y),
        ridge_x=float(args.ridge_x),
        batch_size=int(args.batch_size),
        device=str(args.device),
        umap_n_neighbors=int(args.umap_n_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        dpi=int(args.dpi),
        point_size=float(args.point_size),
    )

    if cfg.kY <= cfg.pca_dim:
        raise ValueError(f"Requirement violated: kY ({cfg.kY}) must be > pca_dim ({cfg.pca_dim}).")

    run_dir = ensure_dir(Path(cfg.out_dir) / _run_folder_name_utc())

    # ---- load MNIST
    images, labels = load_mnist_numpy(cfg.use_train, cfg.max_samples, cfg.seed)
    X_raw, Y_raw = build_XY_from_halves(images, cfg.split_row)

    # ---- PCA(50) each
    Xp, Yp, pca_info = pca_project(X_raw, Y_raw, cfg.pca_dim, cfg.seed)

    # ---- standardize after PCA (matches your Potts default semantics)
    if cfg.standardize_X:
        Xp = StandardScaler(with_mean=True, with_std=True).fit_transform(Xp).astype(np.float32)
    if cfg.standardize_Y:
        Yp = StandardScaler(with_mean=True, with_std=True).fit_transform(Yp).astype(np.float32)

    # ---- device
    use_cuda = torch.cuda.is_available() and cfg.device.startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    Xt = to_t(Xp, device=device)
    Yt = to_t(Yp, device=device)

    # ---- kNN in Y (dense O(N^2))
    idxY, dY = knn_in_y(Yt, k=cfg.kY)

    # ---- metrics
    metrics = local_explainedcov_metrics_LOO(
        X=Xt,
        Y=Yt,
        idxY=idxY,
        dY=dY,
        use_weights=cfg.use_weights,
        eps_tau=cfg.eps_tau,
        ridge_y=cfg.ridge_y,
        ridge_x=cfg.ridge_x,
        eps_trace=cfg.eps_trace,
        batch_size=cfg.batch_size,
    )

    # ---- UMAP on X (PCA-50)
    um = umap.UMAP(
        n_components=2,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="euclidean",
        # random_state=cfg.seed, # SLOW 
    )
    emb2 = um.fit_transform(Xp).astype(np.float32)

    # ---- save artifacts
    np.savez_compressed(
        run_dir / "metrics.npz",
        labels=labels.astype(np.int64),
        X_pca=Xp.astype(np.float32),
        Y_pca=Yp.astype(np.float32),
        umap2=emb2.astype(np.float32),
        **{k: v.astype(np.float32) for k, v in metrics.items()},
    )

    write_csv(
        run_dir / "metrics.csv",
        labels=labels,
        explained=metrics["explained_frac"],
        unexplained=metrics["unexplained_frac"],
        trX=metrics["trX"],
        trR=metrics["trR"],
        worst_ret=metrics["worst_retention"],
    )

    save_umap_plots(
        emb2=emb2,
        labels=labels,
        explained=metrics["explained_frac"],
        out_dir=run_dir,
        dpi=cfg.dpi,
        s=cfg.point_size,
    )

    summary = {
        "created_utc": _utc_now_z(),
        "config": asdict(cfg),
        "data": {
            "N": int(images.shape[0]),
            "X_raw_dim": int(X_raw.shape[1]),
            "Y_raw_dim": int(Y_raw.shape[1]),
            "X_pca_dim": int(Xp.shape[1]),
            "Y_pca_dim": int(Yp.shape[1]),
        },
        "pca": pca_info,
        "stats": {
            "explained_median": float(np.median(metrics["explained_frac"])),
            "explained_q10": float(np.quantile(metrics["explained_frac"], 0.10)),
            "explained_q90": float(np.quantile(metrics["explained_frac"], 0.90)),
            "worst_retention_median": float(np.median(metrics["worst_retention"])),
        },
        "files": {
            "metrics_npz": str(run_dir / "metrics.npz"),
            "metrics_csv": str(run_dir / "metrics.csv"),
            "umap_labels_png": str(run_dir / "umap_labels.png"),
            "umap_explained_frac_png": str(run_dir / "umap_explained_frac.png"),
            "umap_side_by_side_png": str(run_dir / "umap_side_by_side.png"),
            "umap_side_by_side_loge_png": str(run_dir / "umap_side_by_side_loge.png"),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
