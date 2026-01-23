#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_debug_rank_pathology.py

Demonstrate why explained_frac ~ 1 even when Y is physically flat and even after permuting X.

Core claim it tests:
  With k = kY+1 neighbors and very high-dimensional Y,
  rank(Ys) = k-1 generically (after centering),
  col(Ys) = s^⊥, and Xs ∈ s^⊥ (because of centering),
  hence Xs is (nearly) always representable by Ys regardless of pairing.

Outputs:
  debug_rank_pathology.json
  debug_rank_pathology.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch


def standardize_np(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd


@torch.no_grad()
def knn_in_y(Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N = Y.shape[0]
    k = min(int(k), N - 1)
    Dc = torch.cdist(Y, Y, p=2.0)
    Dc.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(Dc, k=k, largest=False, sorted=True)
    return idx, vals


def load_descriptor_h5(desc_h5: Path, descriptor: str, prepend_phase: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(desc_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float32)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float32)
        N = int(f.attrs["n_parameters"])
        meanph = np.array(f["phases/final_fraction_mean"], dtype=np.float32)

        if descriptor == "radial1d":
            mean1d = np.array(f["correlations/correlations_radial_mean"], dtype=np.float32)
            feat = mean1d.reshape(N, -1)
        elif descriptor == "corr2d":
            mean2d = np.array(f["correlations/correlations_2d_mean"], dtype=np.float32)
            feat = mean2d.reshape(N, -1)
        else:
            raise ValueError("descriptor must be radial1d or corr2d")

    Y = np.concatenate([meanph, feat], axis=1).astype(np.float32) if prepend_phase else feat.astype(np.float32)
    X = np.stack([temps, fracs], axis=1).astype(np.float32)
    return X, Y, temps


def weighted_center_scale(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniform weights w=1/k. Returns:
      Zs = diag(sqrt(w)) (Z - 1 mu^T)
      mu
      s = sqrt(w) vector
    """
    k = Z.shape[0]
    w = np.full((k,), 1.0 / float(k), dtype=np.float64)
    s = np.sqrt(w).astype(np.float64)  # (k,)
    mu = (w[:, None] * Z).sum(axis=0, keepdims=True)
    Zc = Z - mu
    Zs = (s[:, None] * Zc).astype(np.float64)
    return Zs, mu, s


def proj_resid_fraction(Ys: np.ndarray, Xs: np.ndarray, rcond: float = 1e-12) -> float:
    """
    Compute ||Xs - Proj_col(Ys) Xs||^2 / ||Xs||^2 using least squares.
    Ys: (k, q), Xs: (k, p)
    """
    # Solve min_B ||Ys B - Xs||_F
    B, *_ = np.linalg.lstsq(Ys, Xs, rcond=rcond)
    Xhat = Ys @ B
    num = float(np.sum((Xs - Xhat) ** 2))
    den = float(np.sum(Xs ** 2)) + 1e-30
    return num / den


def loo_proj_resid_fraction(Ys: np.ndarray, Xs: np.ndarray, rcond: float = 1e-12) -> float:
    """
    LOO: for each t, fit B on rows != t, predict row t, accumulate residual energy.
    Returns total LOO residual energy / total Xs energy.
    """
    k, p = Xs.shape
    den = float(np.sum(Xs ** 2)) + 1e-30
    num = 0.0
    for t in range(k):
        mask = np.ones(k, dtype=bool)
        mask[t] = False
        Ytr = Ys[mask]
        Xtr = Xs[mask]
        B, *_ = np.linalg.lstsq(Ytr, Xtr, rcond=rcond)
        xhat_t = Ys[t:t+1] @ B  # (1,p)
        num += float(np.sum((Xs[t:t+1] - xhat_t) ** 2))
    return num / den


def svd_rank_stats(Ys: np.ndarray, tol: float = 1e-10) -> Tuple[int, float, float]:
    """
    Return:
      rank (count of svals > tol * smax),
      smin_nonzero (smallest above threshold),
      cond = smax / smin_nonzero
    """
    svals = np.linalg.svd(Ys, compute_uv=False)
    smax = float(svals[0]) if svals.size else 0.0
    thr = tol * max(smax, 1e-30)
    nz = svals[svals > thr]
    r = int(nz.size)
    smin = float(nz[-1]) if nz.size else 0.0
    cond = float(smax / max(smin, 1e-30)) if nz.size else float("inf")
    return r, smin, cond


@dataclass(frozen=True)
class Cfg:
    descriptor: str = "radial1d"
    kY: int = 15
    prepend_phase: bool = False
    standardize_X: bool = True
    standardize_Y: bool = True
    seed: int = 0
    highT_quantile: float = 0.90
    n_anchors: int = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, default="potts_data/20260120_221423Z/corr2d/potts_sims_q3_128x128_corr2d.h5", help="Descriptor H5 output from potts_descriptors.py")
    ap.add_argument("--descriptor", type=str, default="radial1d", choices=["radial1d", "corr2d"])
    ap.add_argument("--kY", type=int, default=15)
    ap.add_argument("--prepend_phase", action="store_true")
    ap.add_argument("--no_standardize_X", dest="standardize_X", action="store_false")
    ap.add_argument("--no_standardize_Y", dest="standardize_Y", action="store_false")
    ap.set_defaults(standardize_X=True, standardize_Y=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--highT_quantile", type=float, default=0.90)
    ap.add_argument("--n_anchors", type=int, default=64)
    ap.add_argument("--out_json", type=str, default="debug_rank_pathology.json")
    ap.add_argument("--out_csv", type=str, default="debug_rank_pathology.csv")
    args = ap.parse_args()

    cfg = Cfg(
        descriptor=str(args.descriptor),
        kY=int(args.kY),
        prepend_phase=bool(args.prepend_phase),
        standardize_X=bool(args.standardize_X),
        standardize_Y=bool(args.standardize_Y),
        seed=int(args.seed),
        highT_quantile=float(args.highT_quantile),
        n_anchors=int(args.n_anchors),
    )

    desc_h5 = Path(str(args.h5)).expanduser().resolve()
    X, Y, temps = load_descriptor_h5(desc_h5, cfg.descriptor, cfg.prepend_phase)

    if cfg.standardize_X:
        Xz, _, _ = standardize_np(X)
    else:
        Xz = X.copy()

    if cfg.standardize_Y:
        Yz, _, _ = standardize_np(Y)
    else:
        Yz = Y.copy()

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    Yt = torch.as_tensor(Yz, device=device, dtype=torch.float32)

    idxY, dY = knn_in_y(Yt, k=cfg.kY)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(Xz.shape[0])
    Xperm = Xz[perm].astype(np.float64)

    t_thr = float(np.quantile(temps.astype(np.float64), cfg.highT_quantile))
    highT = np.where(temps >= t_thr)[0]
    if highT.size == 0:
        highT = np.arange(temps.shape[0], dtype=int)

    nA = min(cfg.n_anchors, int(highT.size))
    anchors = np.sort(rng.choice(highT, size=nA, replace=False))

    rows = []
    k = cfg.kY + 1
    for i in anchors:
        J = np.concatenate([[i], idxY[i].detach().cpu().numpy()]).astype(int)

        # neighborhood raw (already globally standardized if enabled)
        Yn = Yz[J].astype(np.float64)
        Xn = Xz[J].astype(np.float64)
        Xn_perm = Xperm[J].astype(np.float64)

        Ys, _, s = weighted_center_scale(Yn)
        Xs, _, _ = weighted_center_scale(Xn)
        Xs_p, _, _ = weighted_center_scale(Xn_perm)

        # show centering constraints explicitly
        y_null = float(np.linalg.norm(s @ Ys))
        x_null = float(np.linalg.norm(s @ Xs))
        xp_null = float(np.linalg.norm(s @ Xs_p))

        rY, sminY, condY = svd_rank_stats(Ys)
        proj_true = proj_resid_fraction(Ys, Xs)
        proj_perm = proj_resid_fraction(Ys, Xs_p)

        loo_true = loo_proj_resid_fraction(Ys, Xs)
        loo_perm = loo_proj_resid_fraction(Ys, Xs_p)

        rows.append(dict(
            anchor=int(i),
            temp=float(X[i, 0]),
            frac=float(X[i, 1]),
            highT_threshold=float(t_thr),
            avg_dY=float(dY[i].detach().cpu().numpy().mean()),
            k=int(k),
            qY=int(Ys.shape[1]),
            rank_Ys=int(rY),
            smin_Ys=float(sminY),
            cond_Ys=float(condY),
            nullcheck_sTYs=float(y_null),
            nullcheck_sTXs=float(x_null),
            nullcheck_sTXs_perm=float(xp_null),
            proj_resid_frac_true=float(proj_true),
            proj_resid_frac_perm=float(proj_perm),
            loo_resid_frac_true=float(loo_true),
            loo_resid_frac_perm=float(loo_perm),
        ))

    # Summaries that directly answer “why permutation doesn’t hurt”
    proj_true_med = float(np.median([r["proj_resid_frac_true"] for r in rows]))
    proj_perm_med = float(np.median([r["proj_resid_frac_perm"] for r in rows]))
    loo_true_med = float(np.median([r["loo_resid_frac_true"] for r in rows]))
    loo_perm_med = float(np.median([r["loo_resid_frac_perm"] for r in rows]))
    rank_med = float(np.median([r["rank_Ys"] for r in rows]))

    report = dict(
        input_h5=str(desc_h5),
        config=cfg.__dict__,
        summary=dict(
            highT_threshold=t_thr,
            n_anchors=int(len(rows)),
            median_rank_Ys=rank_med,
            median_proj_resid_frac_true=proj_true_med,
            median_proj_resid_frac_perm=proj_perm_med,
            median_loo_resid_frac_true=loo_true_med,
            median_loo_resid_frac_perm=loo_perm_med,
        ),
        rows=rows,
    )

    Path(args.out_json).write_text(json.dumps(report, indent=2))

    # CSV
    cols = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print(json.dumps(report["summary"], indent=2))
    print(f"[debug] wrote: {args.out_json}")
    print(f"[debug] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
