#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_intro.py

Compact example showing how LOO explained fraction is computed from 20 points
in 2D under two mappings: (top) injective affine + noise, (bottom) projection
to a circle.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd


def knn_in_y(y: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(y[None, :, :] - y[:, None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx


def local_loo_explained_fraction(
    x: np.ndarray,
    y: np.ndarray,
    idx: int,
    k: int = 5,
    ridge: float = 1e-3,
    eps: float = 1e-12,
) -> dict:
    n = x.shape[0]
    k = min(int(k), n - 1)
    neigh = knn_in_y(y, k=k)[idx]
    J = np.concatenate([[idx], neigh])

    xb = x[J]
    yb = y[J]

    w = np.full((k + 1,), 1.0 / float(k + 1), dtype=np.float64)
    sw = np.sqrt(w)[:, None]

    mux = np.sum(w[:, None] * xb, axis=0)
    muy = np.sum(w[:, None] * yb, axis=0)
    xc = xb - mux
    yc = yb - muy

    xs = xc * sw
    ys = yc * sw

    K = ys @ ys.T
    lam = float(ridge) * (np.trace(K) / float(k + 1))
    H = K + lam * np.eye(k + 1, dtype=np.float64)

    alpha = np.linalg.solve(H, xs)
    G = np.linalg.inv(H)
    diagG = np.diag(G)
    rloo = alpha / diagG[:, None]

    num = float(np.sum(rloo * rloo))
    den = float(np.sum(xs * xs) + eps)
    explained = float(np.clip(1.0 - num / den, 0.0, 1.0))

    return dict(
        idx=idx,
        neigh=neigh,
        J=J,
        w=w,
        mux=mux,
        muy=muy,
        xc=xc,
        yc=yc,
        xs=xs,
        ys=ys,
        rloo=rloo,
        explained=explained,
    )


def _plot_row(
    axes: np.ndarray,
    xz: np.ndarray,
    yz: np.ndarray,
    info: dict,
    row_label: str,
) -> None:
    focus = int(info["idx"])

    ax = axes[0]
    ax.scatter(xz[:, 0], xz[:, 1], c="tab:blue", s=35, alpha=0.85)
    ax.scatter(xz[focus, 0], xz[focus, 1], c="tab:red", s=85, zorder=5)
    ax.set_title(f"{row_label} 1) X (standardized)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.scatter(yz[:, 0], yz[:, 1], c="lightgray", s=28)
    ax.scatter(yz[info["J"], 0], yz[info["J"], 1], c="tab:blue", s=55)
    ax.scatter(yz[focus, 0], yz[focus, 1], c="tab:red", s=90, zorder=5)
    for j in info["neigh"]:
        ax.plot([yz[focus, 0], yz[j, 0]], [yz[focus, 1], yz[j, 1]], color="gray", linewidth=0.8)
    ax.set_title(f"{row_label} 2) kNN in Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    ax.scatter(info["xc"][:, 0], info["xc"][:, 1], c="tab:blue", s=55)
    ax.scatter(info["xc"][0, 0], info["xc"][0, 1], c="tab:red", s=90, zorder=5)
    retracted = info["xc"] - info["rloo"]
    for p, q in zip(retracted, info["xc"]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:orange", linewidth=1.0, alpha=0.8)
    ax.scatter(retracted[:, 0], retracted[:, 1], c="tab:orange", s=35, alpha=0.9)
    ax.axhline(0.0, color="lightgray", linewidth=1.0)
    ax.axvline(0.0, color="lightgray", linewidth=1.0)
    ax.set_title(f"{row_label} 3) LOO residuals (e={info['explained']:.2f})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    rng = np.random.default_rng(4)

    n = 20
    x = rng.normal(0.0, 1.0, size=(n, 2))

    # Top row: injective affine + noise (more noise than before)
    A = np.array([[1.2, 0.3], [-0.4, 0.9]], dtype=np.float64)
    b = np.array([0.7, -0.2], dtype=np.float64)
    noise_inj = 0.12 * rng.normal(0.0, 1.0, size=(n, 2))
    y_inj = (x @ A.T) + b + noise_inj

    # Bottom row: projection onto a circle
    angles = np.arctan2(x[:, 1], x[:, 0])
    y_circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    xz = standardize(x)
    yz_inj = standardize(y_inj)
    yz_circle = standardize(y_circle)

    focus = 0
    k = 5
    info_inj = local_loo_explained_fraction(xz, yz_inj, idx=focus, k=k, ridge=1e-3)
    info_circle = local_loo_explained_fraction(xz, yz_circle, idx=focus, k=k, ridge=1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), dpi=200)
    _plot_row(axes[0], xz, yz_inj, info_inj, row_label="A)")
    _plot_row(axes[1], xz, yz_circle, info_circle, row_label="B)")

    # Match extents within each column
    def _limits(arr: np.ndarray, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)

    x_limits, y_limits = _limits(np.vstack([xz, xz]))
    ycol_limits, ycol_ylimits = _limits(np.vstack([yz_inj, yz_circle]))

    r_inj = info_inj["xc"]
    r_circle = info_circle["xc"]
    ret_inj = r_inj - info_inj["rloo"]
    ret_circle = r_circle - info_circle["rloo"]
    r_limits, r_ylimits = _limits(np.vstack([r_inj, r_circle, ret_inj, ret_circle]))

    for ax in axes[:, 0]:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    for ax in axes[:, 1]:
        ax.set_xlim(*ycol_limits)
        ax.set_ylim(*ycol_ylimits)
    for ax in axes[:, 2]:
        ax.set_xlim(*r_limits)
        ax.set_ylim(*r_ylimits)

    fig.tight_layout()
    fig.savefig("figure_intro.png", bbox_inches="tight", dpi=200)
    fig.savefig("figure_intro.pdf", bbox_inches="tight")

    print(f"Explained fraction (injective, focus={focus}): {info_inj['explained']:.4f}")
    print(f"Explained fraction (circle, focus={focus}): {info_circle['explained']:.4f}")


if __name__ == "__main__":
    main()
