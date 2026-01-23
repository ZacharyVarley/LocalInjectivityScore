#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_debug_injectivity.py

Debug why explained_frac ~ 1 in regimes where Y looks flat (e.g. high T Potts).

This script runs three non-negotiable sanity checks:

(1) Permutation test:
    Permute X (controls) while keeping Y (descriptors) fixed AND keeping kNN(Y) fixed.
    If explained_frac stays high after permutation, the metric is broken (or dominated by preprocessing).

(2) Fast LOO vs brute-force LOO with FIXED preprocessing:
    Uses the exact KRR LOO identity e_t = alpha_t / A_tt with A=(K+λI)^{-1}.
    Compares to brute LOO where each point is removed but the *center/scale/λ remain fixed*.
    These MUST match (up to numerical tolerance). If not, there is a coding bug.

(3) Fast LOO vs brute-force LOO with RECENTER+REFIT:
    Removes each point, recomputes means on the remaining set, refits ridge with an intercept
    (equivalent to recentering), then predicts the left-out point.
    If this differs materially from (2), your analytic LOO is not evaluating the “true” LOO you intended.

Default descriptor for debugging is radial1d (small and sufficient to reproduce the pathology).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
import torch


# ----------------------------- utilities -----------------------------

def standardize_np(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd


def to_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Brute kNN in Y using pairwise Euclidean distances.
    Returns:
      idx: (N,k) indices of neighbors
      d:   (N,k) distances
    """
    N = Y.shape[0]
    k = min(int(k), N - 1)
    Dc = torch.cdist(Y, Y, p=2.0)  # (N,N)
    Dc.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(Dc, k=k, largest=False, sorted=True)
    return idx, vals


def pairwise_dx_stats(Xn: np.ndarray) -> Tuple[float, float]:
    """
    Simple neighborhood spread diagnostics in control space.
    Returns:
      rms_pairwise_dist, max_pairwise_dist
    """
    # Xn: (k, p)
    diffs = Xn[:, None, :] - Xn[None, :, :]
    d = np.sqrt(np.sum(diffs * diffs, axis=-1))
    # exclude diagonal
    d = d[np.triu_indices(d.shape[0], k=1)]
    if d.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(d * d))), float(np.max(d))


# ----------------------------- data loading -----------------------------

def load_descriptor_h5(
    desc_h5: Path,
    descriptor: str,
    prepend_phase_fractions_to_Y: bool,
    feature_subsample: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns:
      X: (N,2) controls [temperature, fraction_initial]
      Y: (N,D) descriptor features (possibly with phase fractions prepended)
      temps: (N,)
      meta: dict
    """
    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)

        q = int(f.attrs.get("q", 3))
        N = int(f.attrs.get("n_parameters"))

        meanph = np.array(f["phases/final_fraction_mean"], dtype=np.float32)

        if descriptor == "radial1d":
            mean1d = np.array(f["correlations/correlations_radial_mean"], dtype=np.float32)  # (N,n_pairs,n_bins)
            feat = mean1d.reshape(N, -1)
        elif descriptor == "corr2d":
            mean2d = np.array(f["correlations/correlations_2d_mean"], dtype=np.float32)      # (N,n_pairs,H,W)
            feat = mean2d.reshape(N, -1)
        else:
            raise ValueError(f"descriptor must be 'radial1d' or 'corr2d', got {descriptor}")

    if prepend_phase_fractions_to_Y:
        Y = np.concatenate([meanph, feat], axis=1).astype(np.float32)
    else:
        Y = feat.astype(np.float32)

    # Optional feature subsampling (for speed and for sensitivity tests)
    D = Y.shape[1]
    if feature_subsample > 0 and feature_subsample < D:
        rng = np.random.default_rng(int(seed))
        cols = rng.choice(D, size=int(feature_subsample), replace=False)
        cols.sort()
        Y = Y[:, cols].copy()

    X = np.stack([temps, fracs], axis=1).astype(np.float32)

    meta = dict(
        N=int(N),
        q=int(q),
        descriptor=str(descriptor),
        X_shape=list(X.shape),
        Y_shape=list(Y.shape),
        prepend_phase_fractions_to_Y=bool(prepend_phase_fractions_to_Y),
        feature_subsample=int(feature_subsample),
        seed=int(seed),
    )
    return X, Y, temps, meta


# ----------------------------- core: fast metric -----------------------------

@torch.no_grad()
def local_explainedcov_fast(
    X: torch.Tensor,          # (N,p)
    Y: torch.Tensor,          # (N,qy)
    idxY: torch.Tensor,       # (N,kY)
    dY: torch.Tensor,         # (N,kY)
    ridge_y: float,
    eps_trace: float,
    batch_size: int,
) -> dict:
    """
    Matches your current implementation (uniform weights).
    Returns numpy arrays.
    """
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    expl = torch.empty((N,), device=device, dtype=torch.float32)
    trX = torch.empty((N,), device=device, dtype=torch.float32)
    trR = torch.empty((N,), device=device, dtype=torch.float32)
    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)

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

        # uniform weights exactly as in your default path
        w = torch.full((B, k), 1.0 / float(k), device=device, dtype=torch.float32)
        sw = torch.sqrt(w)

        muX = (w[:, :, None] * Xn).sum(dim=1)
        muY = (w[:, :, None] * Yn).sum(dim=1)
        Xc = Xn - muX[:, None, :]
        Yc = Yn - muY[:, None, :]

        Xs = Xc * sw[:, :, None]
        Ys = Yc * sw[:, :, None]

        # kernel matrix in the neighborhood
        Kmat = torch.bmm(Ys, Ys.transpose(1, 2))  # (B,k,k)
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)

        lam = (float(ridge_y) * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        # A = (K + λI)^{-1}
        A = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))

        # alpha = A Xs
        alpha = torch.bmm(A, Xs)

        # LOO residuals for KRR are: e_t = alpha_t / A_tt  (elementwise for multivariate targets)
        Adiag = A.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / Adiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)

        u = (trR_b / (trX_b + float(eps_trace))).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexpl[i0:i1] = u
        expl[i0:i1] = e

    return dict(
        unexplained_frac=unexpl.detach().cpu().numpy(),
        explained_frac=expl.detach().cpu().numpy(),
        trX=trX.detach().cpu().numpy(),
        trR=trR.detach().cpu().numpy(),
        avg_dY=avg_dy.detach().cpu().numpy(),
    )


# ----------------------------- brute LOO checks (single neighborhood) -----------------------------

def _fit_krr_dual(K: np.ndarray, Xs: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve (K + lam I) alpha = Xs.
    Returns:
      alpha: (k,p)
      A:     (k,k) inverse of (K+lamI)
    """
    k = K.shape[0]
    Kreg = K + lam * np.eye(k, dtype=np.float64)
    A = np.linalg.inv(Kreg)
    alpha = A @ Xs
    return alpha, A


def neighborhood_fast_residuals_numpy(Ys: np.ndarray, Xs: np.ndarray, ridge_y: float) -> Tuple[np.ndarray, float, float]:
    """
    Fixed-preprocessing fast residuals for one neighborhood (numpy version).
    Ys: (k,qy), Xs: (k,p), both already centered & scaled.
    """
    K = (Ys @ Ys.T).astype(np.float64, copy=False)
    trK = float(np.trace(K))
    k = K.shape[0]
    lam = float(ridge_y) * trK / float(k)
    alpha, A = _fit_krr_dual(K, Xs.astype(np.float64), lam)
    Adiag = np.maximum(np.diag(A), 1e-12)[:, None]
    R = alpha / Adiag  # (k,p) LOO residuals in scaled target space
    return R.astype(np.float64), lam, trK


def neighborhood_bruteforce_LOO_fixed(
    Ys: np.ndarray, Xs: np.ndarray, lam: float
) -> np.ndarray:
    """
    True LOO by refitting after removing each row, BUT keeping preprocessing fixed:
      - same centered/scaled Ys and Xs for the remaining points
      - same λ

    Returns:
      Rbf: (k,p) residuals for each left-out point in the SAME scaled coordinate system as Xs
    """
    k, qy = Ys.shape
    p = Xs.shape[1]
    Rbf = np.zeros((k, p), dtype=np.float64)

    for t in range(k):
        mask = np.ones(k, dtype=bool)
        mask[t] = False
        Ys_tr = Ys[mask]
        Xs_tr = Xs[mask]

        # fit on k-1 points
        Ktr = (Ys_tr @ Ys_tr.T).astype(np.float64, copy=False)
        Kreg = Ktr + lam * np.eye(k - 1, dtype=np.float64)
        Atr = np.linalg.inv(Kreg)
        alpha_tr = Atr @ Xs_tr.astype(np.float64)

        # predict left-out point using kernel vector against training points
        kt = (Ys[t:t+1] @ Ys_tr.T).astype(np.float64, copy=False)  # (1,k-1)
        xhat = kt @ alpha_tr  # (1,p)

        Rbf[t] = (Xs[t] - xhat[0])

    return Rbf


def neighborhood_bruteforce_LOO_recenter_primal(
    Y_raw: np.ndarray, X_raw: np.ndarray, ridge_y: float
) -> np.ndarray:
    """
    LOO by refitting with intercept (training-set centering) each time.
    This reflects the “recenter+refit” interpretation of LOO.

    Model: X ≈ b + (Y - muY) W  with ridge on W:
      W = argmin ||Xc - Yc W||^2 + lam ||W||^2

    λ is scaled as in your method using tr(K)=tr(Yc Yc^T) on the TRAINING set.

    Returns:
      Rraw: (k,p) residuals in RAW X coordinates (not scaled by weights).
    """
    k, qy = Y_raw.shape
    p = X_raw.shape[1]
    Rraw = np.zeros((k, p), dtype=np.float64)

    for t in range(k):
        mask = np.ones(k, dtype=bool)
        mask[t] = False
        Ytr = Y_raw[mask].astype(np.float64)
        Xtr = X_raw[mask].astype(np.float64)

        muY = Ytr.mean(axis=0, keepdims=True)
        muX = Xtr.mean(axis=0, keepdims=True)
        Yc = Ytr - muY
        Xc = Xtr - muX

        # dual solve via K = Yc Yc^T
        K = (Yc @ Yc.T).astype(np.float64, copy=False)
        trK = float(np.trace(K))
        km1 = k - 1
        lam = float(ridge_y) * trK / float(max(km1, 1))

        Kreg = K + lam * np.eye(km1, dtype=np.float64)
        A = np.linalg.inv(Kreg)
        alpha = A @ Xc  # (k-1,p)

        # predict left-out
        yt = Y_raw[t:t+1].astype(np.float64)
        kt = ((yt - muY) @ Yc.T).astype(np.float64, copy=False)  # (1,k-1)
        xhat_c = kt @ alpha  # (1,p)
        xhat = muX + xhat_c  # (1,p)

        Rraw[t] = (X_raw[t].astype(np.float64) - xhat[0])

    return Rraw


# ----------------------------- main -----------------------------

@dataclass(frozen=True)
class Cfg:
    descriptor: str = "radial1d"
    kY: int = 15
    standardize_X: bool = True
    standardize_Y: bool = True
    prepend_phase_fractions_to_Y: bool = False

    ridge_y: float = 1e-3
    eps_trace: float = 1e-18

    batch_size: int = 256
    device: str = "cuda"

    feature_subsample: int = 0  # 0 => full
    seed: int = 0

    # diagnostics
    n_anchor_checks: int = 48
    highT_quantile: float = 0.90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, default="potts_data/20260120_221423Z/corr2d/potts_sims_q3_128x128_corr2d.h5", help="Descriptor H5 output from potts_descriptors.py")
    ap.add_argument("--descriptor", type=str, default="radial1d", choices=["radial1d", "corr2d"])
    ap.add_argument("--kY", type=int, default=15)
    ap.add_argument("--prepend_phase_fractions_to_Y", action="store_true")
    ap.add_argument("--standardize_X", action="store_true")
    ap.add_argument("--no_standardize_X", dest="standardize_X", action="store_false")
    ap.set_defaults(standardize_X=True)
    ap.add_argument("--standardize_Y", action="store_true")
    ap.add_argument("--no_standardize_Y", dest="standardize_Y", action="store_false")
    ap.set_defaults(standardize_Y=True)

    ap.add_argument("--ridge_y", type=float, default=1e-3)
    ap.add_argument("--eps_trace", type=float, default=1e-18)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--feature_subsample", type=int, default=0,
                    help="If >0, randomly subsample this many Y features for speed/sensitivity tests.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_anchor_checks", type=int, default=48,
                    help="How many anchors (in high-T region) to run brute LOO checks on.")
    ap.add_argument("--highT_quantile", type=float, default=0.90)

    ap.add_argument("--out_json", type=str, default="debug_injectivity_report.json")
    args = ap.parse_args()

    cfg = Cfg(
        descriptor=str(args.descriptor),
        kY=int(args.kY),
        standardize_X=bool(args.standardize_X),
        standardize_Y=bool(args.standardize_Y),
        prepend_phase_fractions_to_Y=bool(args.prepend_phase_fractions_to_Y),
        ridge_y=float(args.ridge_y),
        eps_trace=float(args.eps_trace),
        batch_size=int(args.batch_size),
        device=str(args.device),
        feature_subsample=int(args.feature_subsample),
        seed=int(args.seed),
        n_anchor_checks=int(args.n_anchor_checks),
        highT_quantile=float(args.highT_quantile),
    )

    desc_h5 = Path(str(args.h5)).expanduser().resolve()
    X, Y, temps, meta = load_descriptor_h5(
        desc_h5=desc_h5,
        descriptor=cfg.descriptor,
        prepend_phase_fractions_to_Y=cfg.prepend_phase_fractions_to_Y,
        feature_subsample=cfg.feature_subsample,
        seed=cfg.seed,
    )

    # Standardize
    if cfg.standardize_X:
        Xz, muX, sdX = standardize_np(X)
    else:
        Xz, muX, sdX = X.copy(), np.zeros((X.shape[1],), np.float32), np.ones((X.shape[1],), np.float32)

    if cfg.standardize_Y:
        Yz, muY, sdY = standardize_np(Y)
    else:
        Yz, muY, sdY = Y.copy(), np.zeros((Y.shape[1],), np.float32), np.ones((Y.shape[1],), np.float32)

    # Torch device
    use_cuda = torch.cuda.is_available() and cfg.device.startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    Xt = to_t(Xz.astype(np.float32), device=device)
    Yt = to_t(Yz.astype(np.float32), device=device)

    # kNN(Y)
    idxY_t, dY_t = knn_in_y(Yt, k=int(cfg.kY))

    # Fast metric on true pairing
    fast_true = local_explainedcov_fast(
        X=Xt, Y=Yt, idxY=idxY_t, dY=dY_t,
        ridge_y=cfg.ridge_y, eps_trace=cfg.eps_trace,
        batch_size=cfg.batch_size,
    )

    # Permutation test: shuffle X but keep same neighborhoods in Y
    rng = np.random.default_rng(int(cfg.seed) + 12345)
    perm = rng.permutation(Xz.shape[0])
    Xperm = Xz[perm].astype(np.float32)
    Xt_perm = to_t(Xperm, device=device)

    fast_perm = local_explainedcov_fast(
        X=Xt_perm, Y=Yt, idxY=idxY_t, dY=dY_t,
        ridge_y=cfg.ridge_y, eps_trace=cfg.eps_trace,
        batch_size=cfg.batch_size,
    )

    # High-T anchors for brute checks
    t_thr = float(np.quantile(temps.astype(np.float64), cfg.highT_quantile))
    highT_idx = np.where(temps >= t_thr)[0]
    if highT_idx.size == 0:
        highT_idx = np.arange(temps.shape[0], dtype=int)
    # sample anchors
    nA = min(int(cfg.n_anchor_checks), int(highT_idx.size))
    anchors = highT_idx[rng.choice(highT_idx.size, size=nA, replace=False)]
    anchors.sort()

    # Brute checks per anchor
    k = cfg.kY + 1
    p = Xz.shape[1]

    brute_rows = []
    max_abs_diff_fast_vs_bf_fixed = 0.0
    med_abs_diff_fast_vs_bf_fixed = []

    for i in anchors:
        J = np.concatenate([[i], idxY_t[i].detach().cpu().numpy()]).astype(int)
        # neighborhood in standardized coordinates
        Xn_raw = Xz[J].astype(np.float64)
        Yn_raw = Yz[J].astype(np.float64)

        # fixed preprocessing for "fast residuals"
        w = np.full((k,), 1.0 / float(k), dtype=np.float64)
        sw = np.sqrt(w)[:, None]  # (k,1)

        muXn = (w[:, None] * Xn_raw).sum(axis=0, keepdims=True)
        muYn = (w[:, None] * Yn_raw).sum(axis=0, keepdims=True)
        Xc = Xn_raw - muXn
        Yc = Yn_raw - muYn
        Xs = Xc * sw
        Ys = Yc * sw

        # Fast residuals (numpy)
        R_fast, lam_full, trK_full = neighborhood_fast_residuals_numpy(Ys=Ys, Xs=Xs, ridge_y=cfg.ridge_y)

        # Brute LOO with fixed preprocessing and fixed λ
        R_bf_fixed = neighborhood_bruteforce_LOO_fixed(Ys=Ys, Xs=Xs, lam=lam_full)

        # Brute LOO recenter+refit (returns residuals in RAW X space, not scaled)
        R_bf_recenter_raw = neighborhood_bruteforce_LOO_recenter_primal(Y_raw=Yn_raw, X_raw=Xn_raw, ridge_y=cfg.ridge_y)

        # Compare fast vs brute-fixed (they MUST match)
        diff = np.abs(R_fast - R_bf_fixed)
        max_abs = float(np.max(diff))
        max_abs_diff_fast_vs_bf_fixed = max(max_abs_diff_fast_vs_bf_fixed, max_abs)
        med_abs_diff_fast_vs_bf_fixed.append(float(np.median(diff)))

        # Convert scaled residuals to an energy ratio (same as score space)
        trX = float(np.sum(Xs * Xs))
        trR_fast = float(np.sum(R_fast * R_fast))
        trR_bf_fixed = float(np.sum(R_bf_fixed * R_bf_fixed))

        u_fast = min(max(trR_fast / (trX + cfg.eps_trace), 0.0), 1.0)
        u_bf_fixed = min(max(trR_bf_fixed / (trX + cfg.eps_trace), 0.0), 1.0)

        # For recenter case, compute denom as training-centered raw X energy for comparability
        # (not identical to your score, but detects optimism from centering choices)
        Xc_raw = Xn_raw - Xn_raw.mean(axis=0, keepdims=True)
        trX_raw = float(np.sum(Xc_raw * Xc_raw))
        trR_recenter = float(np.sum(R_bf_recenter_raw * R_bf_recenter_raw))
        u_recenter = min(max(trR_recenter / (trX_raw + 1e-18), 0.0), 1.0)

        # Neighborhood spread in X (raw standardized X space)
        dx_rms, dx_max = pairwise_dx_stats(Xn_raw)

        brute_rows.append(dict(
            anchor=int(i),
            temp=float(X[i, 0]),
            frac=float(X[i, 1]),
            t_thr=float(t_thr),
            trK=float(trK_full),
            lam=float(lam_full),
            dx_rms=float(dx_rms),
            dx_max=float(dx_max),
            u_fast=float(u_fast),
            e_fast=float(1.0 - u_fast),
            u_bf_fixed=float(u_bf_fixed),
            e_bf_fixed=float(1.0 - u_bf_fixed),
            u_recenter_raw=float(u_recenter),
            e_recenter_raw=float(1.0 - u_recenter),
            max_abs_diff_fast_vs_bf_fixed=float(max_abs),
            median_abs_diff_fast_vs_bf_fixed=float(np.median(diff)),
        ))

    report = dict(
        input_h5=str(desc_h5),
        config=dict(
            **meta,
            kY=cfg.kY,
            standardize_X=cfg.standardize_X,
            standardize_Y=cfg.standardize_Y,
            ridge_y=cfg.ridge_y,
            eps_trace=cfg.eps_trace,
            batch_size=cfg.batch_size,
            device=str(device),
            highT_quantile=cfg.highT_quantile,
            highT_threshold=t_thr,
        ),
        global_summaries=dict(
            explained_true=dict(
                median=float(np.median(fast_true["explained_frac"])),
                q10=float(np.quantile(fast_true["explained_frac"], 0.10)),
                q90=float(np.quantile(fast_true["explained_frac"], 0.90)),
            ),
            explained_perm=dict(
                median=float(np.median(fast_perm["explained_frac"])),
                q10=float(np.quantile(fast_perm["explained_frac"], 0.10)),
                q90=float(np.quantile(fast_perm["explained_frac"], 0.90)),
            ),
            highT=dict(
                n=int(highT_idx.size),
                explained_true_median=float(np.median(fast_true["explained_frac"][highT_idx])),
                explained_perm_median=float(np.median(fast_perm["explained_frac"][highT_idx])),
                trX_true_median=float(np.median(fast_true["trX"][highT_idx])),
                avg_dY_true_median=float(np.median(fast_true["avg_dY"][highT_idx])),
            ),
        ),
        loo_integrity_checks=dict(
            max_abs_diff_fast_vs_bruteforce_fixed=float(max_abs_diff_fast_vs_bf_fixed),
            median_abs_diff_fast_vs_bruteforce_fixed=float(np.median(np.array(med_abs_diff_fast_vs_bf_fixed, dtype=np.float64)))
            if med_abs_diff_fast_vs_bf_fixed else None,
        ),
        anchors_bruteforce=brute_rows,
    )

    Path(str(args.out_json)).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["global_summaries"], indent=2))
    print(json.dumps(report["loo_integrity_checks"], indent=2))
    print(f"[debug] wrote report: {args.out_json}")


if __name__ == "__main__":
    main()
