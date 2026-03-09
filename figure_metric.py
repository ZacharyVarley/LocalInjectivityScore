#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_metric.py

Publication-quality figure explaining the Local Leave-One-Out (LOO) score metric
for measuring local injectivity of mappings.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize data to zero mean and unit variance."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd


def knn_in_y(y: np.ndarray, k: int) -> np.ndarray:
    """Find k-nearest neighbors for each point in Y."""
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
    """
    Compute local LOO explained fraction for a single point.
    
    The LOO score measures how well the local geometry is preserved by the mapping.
    High scores indicate locally injective (reversible) behavior.
    """
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
        xc=xc,
        yc=yc,
        rloo=rloo,
        explained=explained,
    )


def main() -> None:
    # Set publication-quality defaults
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
    })
    
    rng = np.random.default_rng(42)

    n = 25
    x = rng.normal(0.0, 1.0, size=(n, 2))
    
    # Create clearer examples:
    # Top: Nearly injective (affine + small noise)
    A = np.array([[1.3, 0.2], [-0.3, 1.1]], dtype=np.float64)
    noise_inj = 0.08 * rng.normal(0.0, 1.0, size=(n, 2))
    y_inj = (x @ A.T) + noise_inj

    # Bottom: Non-injective (projection to circle)
    angles = np.arctan2(x[:, 1], x[:, 0])
    y_circle = 1.2 * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    xz = standardize(x)
    yz_inj = standardize(y_inj)
    yz_circle = standardize(y_circle)

    focus = 0
    k = 5
    info_inj = local_loo_explained_fraction(xz, yz_inj, idx=focus, k=k, ridge=1e-3)
    info_circle = local_loo_explained_fraction(xz, yz_circle, idx=focus, k=k, ridge=1e-3)

    # Create figure with cleaner layout
    fig = plt.figure(figsize=(13, 6.5), dpi=150)
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.96, top=0.90, bottom=0.08)
    
    # Color scheme
    c_regular = '#2E5090'  # Muted blue
    c_focus = '#D62728'    # Red
    c_neighbor = '#5FA3D0' # Light blue
    c_residual = '#FF8C42' # Orange
    c_gray = '#AAAAAA'
    
    # Plot injective case (top row)
    _plot_case(fig, gs, 0, xz, yz_inj, info_inj, 
               "Locally injective", 
               c_regular, c_focus, c_neighbor, c_residual, c_gray)
    
    # Plot non-injective case (bottom row)
    _plot_case(fig, gs, 1, xz, yz_circle, info_circle,
               "Non-injective",
               c_regular, c_focus, c_neighbor, c_residual, c_gray)
    
    # Add legend at the bottom
    handles = [
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor=c_focus, markeredgecolor='white', markersize=12, label='Query point ($x_0, y_0$)'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=c_neighbor, markeredgecolor='white', markersize=8, label='k-NN neighbors'),
        plt.Line2D([0], [0], marker='x', color='none', markeredgecolor=c_residual, markersize=8, markeredgewidth=1.5, label='LOO prediction'),
        plt.Line2D([0], [0], color=c_residual, lw=1.5, label='LOO residual vector')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    
    # Add overall title
    fig.suptitle('Local Leave-One-Out (LOO) Score Metric', 
                 fontsize=14, fontweight='bold', y=0.97)
    
    # Save outputs
    fig.savefig("figure_metric.png", bbox_inches="tight", dpi=300)
    fig.savefig("figure_metric.pdf", bbox_inches="tight")
    
    print(f"✓ Saved figure_metric.png and figure_metric.pdf")
    print(f"\nLOO scores:")
    print(f"  Locally injective:  {info_inj['explained']:.3f}")
    print(f"  Non-injective:      {info_circle['explained']:.3f}")


def _plot_case(fig, gs, row, xz, yz, info, case_label,
               c_regular, c_focus, c_neighbor, c_residual, c_gray):
    """Plot a single case (one row of the figure)."""
    focus = int(info["idx"])
    
    # Panel 1: Original space X
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.scatter(xz[:, 0], xz[:, 1], c=c_regular, s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax1.scatter(xz[focus, 0], xz[focus, 1], c=c_focus, s=120, zorder=5, 
                edgecolors='white', linewidths=1.5, marker='*')
    ax1.set_title('(1) Input space $\\mathbf{X}$', fontsize=10, pad=8)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)

    # Panel 2: Full Output space Y
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.scatter(yz[:, 0], yz[:, 1], c=c_neighbor, s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax2.scatter(yz[focus, 0], yz[focus, 1], c=c_focus, s=120, zorder=5,
                edgecolors='white', linewidths=1.5, marker='*')
    ax2.set_title('(2) Output space $\\mathbf{Y}$', fontsize=10, pad=8)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    
    # Panel 3: Local Zoomed k-NN
    ax3 = fig.add_subplot(gs[row, 2])
    # Show background points faintly
    ax3.scatter(yz[:, 0], yz[:, 1], c='#E8E8E8', s=35, alpha=0.3, edgecolors='none')
    # Highlight neighborhood
    ax3.scatter(yz[info["J"], 0], yz[info["J"], 1], c=c_neighbor, s=60, alpha=0.9,
                edgecolors='white', linewidths=0.5, zorder=3)
    ax3.scatter(yz[focus, 0], yz[focus, 1], c=c_focus, s=150, zorder=5,
                edgecolors='white', linewidths=1.5, marker='*')
    # Draw connections
    for j in info["neigh"]:
        ax3.plot([yz[focus, 0], yz[j, 0]], [yz[focus, 1], yz[j, 1]], 
                color=c_gray, linewidth=1.0, alpha=0.6, zorder=1)
    
    ax3.set_title(f'(3) Local $k$-NN ($k={len(info["neigh"])}$)', 
                  fontsize=10, pad=8)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, alpha=0.15, linewidth=0.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Zoom in on neighborhood
    neigh_points = yz[info["J"]]
    center = yz[focus]
    dist = np.max(np.linalg.norm(neigh_points - center, axis=1))
    ax3.set_xlim(center[0] - 1.3*dist, center[0] + 1.3*dist)
    ax3.set_ylim(center[1] - 1.3*dist, center[1] + 1.3*dist)
    
    # Panel 4: LOO residuals
    ax4 = fig.add_subplot(gs[row, 3])
    retracted = info["xc"] - info["rloo"]
    
    # Draw residual vectors
    for p, q in zip(retracted, info["xc"]):
        ax4.annotate('', xy=q, xytext=p,
                    arrowprops=dict(arrowstyle='->', color=c_residual, 
                                  lw=1.5, alpha=0.8, shrinkA=0, shrinkB=0, mutation_scale=10))
    
    ax4.scatter(info["xc"][:, 0], info["xc"][:, 1], c=c_neighbor, s=60, 
                alpha=0.9, edgecolors='white', linewidths=0.5, zorder=3)
    ax4.scatter(info["xc"][0, 0], info["xc"][0, 1], c=c_focus, s=150, zorder=5,
                edgecolors='white', linewidths=1.5, marker='*')
    ax4.scatter(retracted[:, 0], retracted[:, 1], c=c_residual, s=45, 
                alpha=0.9, marker='x', linewidths=2.0, zorder=4)
    
    ax4.axhline(0.0, color=c_gray, linewidth=0.8, alpha=0.4, linestyle='--')
    ax4.axvline(0.0, color=c_gray, linewidth=0.8, alpha=0.4, linestyle='--')
    
    score = info['explained']
    ax4.set_title(f'(4) LOO Score: {score:.3f}', 
                  fontsize=11, fontweight='bold', pad=8)
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True, alpha=0.15, linewidth=0.5)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Set limit for residuals plot to be consistent
    r_max = np.max(np.abs(np.concatenate([info["xc"], retracted]))) * 1.2
    ax4.set_xlim(-r_max, r_max)
    ax4.set_ylim(-r_max, r_max)
    
    # Add case label
    ax1.text(-0.25, 0.5, case_label, transform=ax1.transAxes,
            fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)


if __name__ == "__main__":
    main()
