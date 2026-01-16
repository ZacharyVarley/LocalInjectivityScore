#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCA-based local injectivity diagnostics for toy examples.

Simplified version focused on publication-ready multi-panel figures.
"""

import os
import math
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

# ---------- Global Matplotlib aesthetics ----------
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------- CONFIG ----------------------
CONFIG = dict(
    outdir="toy_figures",
    device="cuda",
    seed=1337,
    n_pts=2000,  # Total number of points to sample
    scenarios=("identity", "pinch", "ribbon", "fold"),
    t=1.0,
    extra_dims=0,
    noise_sigma=0.00,
    
    # CCA neighborhood settings
    kY=15,
    knn_chunk=512,
    
    # Weighting and regularization
    use_weights=False,
    eps_tau=1e-12,
    ridge_y=1e-3,
    ridge_x=1e-8,
    eps_trace=1e-18,
    batch_size=512,
    
    # Plotting
    hm_bins=60,
    hm_sigma_px=1.0,
    elev=20,  # 3D view elevation
    azim=35,  # 3D view azimuth
)


# ---------------- Scenarios ----------------
def identity(p: Tensor, t: float = 1.0) -> Tensor:
    """Identity mapping (no deformation)."""
    return torch.stack((p[:, 0], p[:, 1], torch.zeros_like(p[:, 0])), dim=1)


def ribbon(p: Tensor, t: float = 1.0) -> Tensor:
    """Ribbon self-intersection."""
    y_prime = t * (0.75 * torch.pi) * (p[:, 1] + 1.0) + torch.pi / 4
    curl = math.sin(t * torch.pi / 2)
    x = p[:, 0]
    y = t * p[:, 1] * torch.cos(y_prime * curl) + (1 - t) * p[:, 1]
    z = p[:, 1] * torch.sin(y_prime * curl) + 0.5 * t**2
    return torch.stack((x, y, z), dim=1)


def fold_sheet(p: Tensor, t: float) -> Tensor:
    """Fold: folds a triangular portion over itself."""
    x, y = p[:, 0], p[:, 1]
    mask = y > -x + 1
    x_new = x.clone()
    y_new = y.clone()
    z_new = torch.zeros_like(x)

    if torch.any(mask):
        px, py = x[mask], y[mask]
        dx, dy = px - 0.5, py - 0.5

        theta = math.pi * t
        axis = torch.tensor([1.0, -1.0, 0.0], device=p.device)
        axis = axis / torch.norm(axis)

        points_3d = torch.stack((dx, dy, torch.zeros_like(dx)), dim=1)
        k = axis
        cos_t = torch.cos(torch.tensor(theta, device=p.device))
        sin_t = torch.sin(torch.tensor(theta, device=p.device))

        rotated = (
            points_3d * cos_t
            + torch.cross(k.expand_as(points_3d), points_3d, dim=1) * sin_t
            + k * torch.sum(points_3d * k, dim=1, keepdim=True) * (1 - cos_t)
        )

        x_final = rotated[:, 0] + 0.5
        y_final = rotated[:, 1] + 0.5

        x_new[mask] = x_final
        y_new[mask] = y_final

    return torch.stack((x_new, y_new, z_new), dim=1)


def pinch(p: Tensor, t: float = 1.0) -> Tensor:
    """Pinch: compresses x2-direction by |x1|^{2t}."""
    x, y = p[:, 0], p[:, 1]
    return torch.stack((x, y * torch.abs(x) ** (2 * t), torch.zeros_like(x)), dim=1)


# ---------------- Utilities ----------------
def make_random_points(n_pts: int, device: str, seed: int = 0) -> Tensor:
    """Draw n_pts random points in [-1,1]^2."""
    rng = torch.Generator(device=device).manual_seed(seed)
    return (torch.rand((n_pts, 2), device=device, generator=rng) * 2.0) - 1.0


def extend_with_null_noise(Y: Tensor, extra_dims: int, sigma: float,
                           rng: torch.Generator | None) -> Tensor:
    if extra_dims <= 0 or sigma <= 0.0:
        return Y
    noise = torch.randn((Y.shape[0], extra_dims), device=Y.device, generator=rng) * sigma
    return torch.cat([Y, noise], dim=1)


# ---------------------- kNN in Y (chunked) ----------------------
@torch.no_grad()
def knn_in_y_chunked(Y: torch.Tensor, k: int, chunk: int = 512):
    """Find k nearest neighbors in Y for each point (excluding self)."""
    device = Y.device
    N = Y.shape[0]
    idx_out = torch.empty((N, k), device=device, dtype=torch.int64)
    d_out = torch.empty((N, k), device=device, dtype=torch.float32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Dc = torch.cdist(Y[s:e], Y, p=2.0)
        r = torch.arange(e - s, device=device)
        c = torch.arange(s, e, device=device)
        Dc[r, c] = float("inf")

        vals, idx = torch.topk(Dc, k=k, largest=False, sorted=False)
        ord_local = torch.argsort(vals, dim=1)
        vals_s = torch.gather(vals, dim=1, index=ord_local)
        idx_s = torch.gather(idx, dim=1, index=ord_local)

        idx_out[s:e] = idx_s
        d_out[s:e] = vals_s

    return idx_out, d_out


# ---------------------- Local explained covariance (LOO version) ----------------------
@torch.no_grad()
def local_explainedcov_metrics_batched_LOO(
    X: torch.Tensor,
    Y: torch.Tensor,
    idxY: torch.Tensor,
    dY: torch.Tensor,
    use_weights: bool,
    eps_tau: float,
    ridge_y: float,
    ridge_x: float,
    eps_trace: float,
    batch_size: int = 512,
):
    """Local explained covariance with LOO cross-validation scoring."""
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    expl = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)

    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        B = i1 - i0

        centers = torch.arange(i0, i1, device=device, dtype=torch.int64)
        neigh = torch.cat([centers[:, None], idxY[i0:i1]], dim=1)

        dn = torch.cat([
            torch.zeros((B, 1), device=device, dtype=torch.float32),
            dY[i0:i1].to(torch.float32)
        ], dim=1)

        Xn = X[neigh]
        Yn = Y[neigh]

        if use_weights:
            tau = dn.max(dim=1).values.clamp_min(eps_tau)
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
        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (ridge_y * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))
        alpha = torch.bmm(Hinv, Xs)

        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)

        u = (trR_b / (trX_b + eps_trace)).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        unexpl[i0:i1] = u
        expl[i0:i1] = e

    return {
        "unexplained_frac": unexpl.detach().cpu().numpy(),
        "explained_frac": expl.detach().cpu().numpy(),
    }


# ---------------------- Smoothing utilities ----------------------
def _gaussian_kernel1d(sigma_px: float) -> np.ndarray:
    sigma = float(max(sigma_px, 1e-6))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _pad_reflect(arr: np.ndarray, pad: int, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    return np.pad(arr, pad_width, mode="reflect")


def _convolve1d_reflect(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = k.size // 2
    x = _pad_reflect(arr, pad, axis)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=axis, arr=x)


def _gaussian_smooth_nan(image: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px is None or sigma_px <= 0:
        return image
    k = _gaussian_kernel1d(sigma_px)
    val = image.copy()
    mask = np.isfinite(val).astype(np.float64)
    val[~np.isfinite(val)] = 0.0
    val = _convolve1d_reflect(val, k, axis=0)
    val = _convolve1d_reflect(val, k, axis=1)
    msk = _convolve1d_reflect(mask, k, axis=0)
    msk = _convolve1d_reflect(msk, k, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = val / np.maximum(msk, 1e-12)
    out[msk < 1e-12] = np.nan
    return out


def _heatmap_binned_core(X: np.ndarray, Z: np.ndarray, bins: int, sigma_px: float):
    """Compute smoothed binned image."""
    x = X[:, 0]
    y = X[:, 1]
    finite = np.isfinite(Z)
    x, y, w = x[finite], y[finite], Z[finite]

    sum_w, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[-1, 1], [-1, 1]], weights=w
    )
    cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=None)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = sum_w / cnt
    img[cnt == 0] = np.nan
    img_s = _gaussian_smooth_nan(img, sigma_px)
    return img_s


# ---------------------- Per-scenario runner ----------------------
def run_one(scenario: str, cfg: dict, device: torch.device, rng: torch.Generator):
    """Run CCA-based injectivity test on one toy scenario."""
    n_pts, t = cfg["n_pts"], cfg["t"]

    p = make_random_points(n_pts, device=device.type, seed=int(cfg["seed"]))

    scen_map = {
        "identity": identity,
        "ribbon": ribbon,
        "fold": fold_sheet,
        "pinch": pinch,
    }
    if scenario not in scen_map:
        raise ValueError(f"Unknown scenario: {scenario}")
    Y3 = scen_map[scenario](p, t)

    Y = extend_with_null_noise(Y3, cfg["extra_dims"], cfg["noise_sigma"], rng)

    with torch.no_grad():
        idxY, dY = knn_in_y_chunked(Y, k=int(cfg["kY"]), chunk=int(cfg["knn_chunk"]))
        metrics = local_explainedcov_metrics_batched_LOO(
            X=p,
            Y=Y,
            idxY=idxY,
            dY=dY,
            use_weights=bool(cfg["use_weights"]),
            eps_tau=float(cfg["eps_tau"]),
            ridge_y=float(cfg["ridge_y"]),
            ridge_x=float(cfg["ridge_x"]),
            eps_trace=float(cfg["eps_trace"]),
            batch_size=int(cfg["batch_size"]),
        )

    X_np = p.cpu().numpy()
    Y3_np = Y3.cpu().numpy()
    expl_np = metrics["explained_frac"]

    print(f"[OK] {scenario} | explained_frac: mean={expl_np.mean():.3f}, min={expl_np.min():.3f}, max={expl_np.max():.3f}")

    return {
        "X": X_np,
        "Y3": Y3_np,
        "explained": expl_np,
    }


# ---------------------- Publication figure ----------------------
def _create_grid_wireframe(ax, scenario_name: str, t: float, device: str, n_grid: int = 100, alpha: float = 1.0, linewidth: float = 2.0):
    """Add a wireframe showing the perimeter of the 2D input plane in 3D embedding space."""
    scen_map = {
        "identity": identity,
        "ribbon": ribbon,
        "fold": fold_sheet,
        "pinch": pinch,
    }
    
    # Create points along the perimeter of [-1, 1]^2
    u = np.linspace(-1, 1, n_grid)
    
    # Bottom edge: x2 = -1, x1 varies
    pts_bottom = torch.tensor(np.stack([u, np.full_like(u, -1)], axis=1), dtype=torch.float32, device=device)
    y3d_bottom = scen_map[scenario_name](pts_bottom, t).cpu().numpy()
    
    # Right edge: x1 = 1, x2 varies
    pts_right = torch.tensor(np.stack([np.full_like(u, 1), u], axis=1), dtype=torch.float32, device=device)
    y3d_right = scen_map[scenario_name](pts_right, t).cpu().numpy()
    
    # Top edge: x2 = 1, x1 varies (reversed)
    pts_top = torch.tensor(np.stack([u[::-1], np.full_like(u, 1)], axis=1), dtype=torch.float32, device=device)
    y3d_top = scen_map[scenario_name](pts_top, t).cpu().numpy()
    
    # Left edge: x1 = -1, x2 varies (reversed)
    pts_left = torch.tensor(np.stack([np.full_like(u, -1), u[::-1]], axis=1), dtype=torch.float32, device=device)
    y3d_left = scen_map[scenario_name](pts_left, t).cpu().numpy()
    
    # Plot each edge with high zorder to appear on top
    ax.plot(y3d_bottom[:, 0], y3d_bottom[:, 1], y3d_bottom[:, 2], 
            color='black', linewidth=linewidth, alpha=alpha, zorder=100)
    ax.plot(y3d_right[:, 0], y3d_right[:, 1], y3d_right[:, 2], 
            color='black', linewidth=linewidth, alpha=alpha, zorder=100)
    ax.plot(y3d_top[:, 0], y3d_top[:, 1], y3d_top[:, 2], 
            color='black', linewidth=linewidth, alpha=alpha, zorder=100)
    ax.plot(y3d_left[:, 0], y3d_left[:, 1], y3d_left[:, 2], 
            color='black', linewidth=linewidth, alpha=alpha, zorder=100)


def make_publication_figure(results_by_scenario: dict, cfg: dict, layout: str = "1x4"):
    """
    Create publication-ready figure showing explained fraction for all scenarios.
    
    Args:
        results_by_scenario: Dict mapping scenario name to results
        cfg: Configuration dict
        layout: Either "1x4" (heatmaps only) or "2x4" (3D + heatmaps)
    """
    scenarios = list(cfg["scenarios"])
    n_scenarios = len(scenarios)
    
    if layout == "1x4":
        nrows, ncols = 1, 4
        figsize = (12, 3)
        show_3d = False
    elif layout == "2x4":
        nrows, ncols = 2, 4
        figsize = (12, 7)
        show_3d = True
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols + 1, figure=fig, 
                          width_ratios=[1]*ncols + [0.05],
                          wspace=0.20, hspace=0.15)  # Reduced hspace from 0.30 to 0.15
    
    bins = int(cfg["hm_bins"])
    sigma_px = float(cfg["hm_sigma_px"])
    elev = cfg.get("elev", 20)
    azim = cfg.get("azim", 35)
    
    images = []
    
    for idx, scen in enumerate(scenarios):
        res = results_by_scenario[scen]
        X = res["X"]
        Y3 = res["Y3"]
        Z = res["explained"]
        
        col = idx
        
        # First row: 3D embeddings (if 2x4 layout)
        if show_3d:
            ax_3d = fig.add_subplot(gs[0, col], projection='3d')
            
            sc = ax_3d.scatter(
                Y3[:, 0], Y3[:, 1], Y3[:, 2],
                c=Z, cmap="viridis", s=4, alpha=0.7,
                linewidths=0, edgecolor="none",
                vmin=0.0, vmax=1.0,
                zorder=2,
            )
            
            # Draw perimeter wireframe LAST so it appears on top
            _create_grid_wireframe(ax_3d, scen, cfg["t"], device=device.type, n_grid=100, alpha=1.0, linewidth=1.0)
            
            ax_3d.view_init(elev=elev, azim=azim)
            ax_3d.set_title(scen.capitalize(), fontsize=11, pad=8)
            
            # Minimal ticks for 3D with reduced labelpad
            ax_3d.set_xticks([])
            ax_3d.set_yticks([])
            ax_3d.set_zticks([])
            ax_3d.set_xlabel(r"$y_1$", fontsize=9, labelpad=-8)
            ax_3d.set_ylabel(r"$y_2$", fontsize=9, labelpad=-8)
            ax_3d.set_zlabel(r"$y_3$", fontsize=9, labelpad=-8)
            
            # Adjust panes
            ax_3d.xaxis.pane.fill = False
            ax_3d.yaxis.pane.fill = False
            ax_3d.zaxis.pane.fill = False
            ax_3d.xaxis.pane.set_edgecolor('gray')
            ax_3d.yaxis.pane.set_edgecolor('gray')
            ax_3d.zaxis.pane.set_edgecolor('gray')
            ax_3d.xaxis.pane.set_alpha(0.3)
            ax_3d.yaxis.pane.set_alpha(0.3)
            ax_3d.zaxis.pane.set_alpha(0.3)
            
            images.append(sc)
        
        # Second row (or only row for 1x4): 2D heatmaps
        row_2d = 1 if show_3d else 0
        img_s = _heatmap_binned_core(X, Z, bins=bins, sigma_px=sigma_px)
        
        ax_2d = fig.add_subplot(gs[row_2d, col])
        
        im = ax_2d.imshow(
            img_s.T,
            origin="lower",
            extent=[-1, 1, -1, 1],
            aspect="equal",
            cmap="viridis",
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        
        # Only show title for 1x4, not for 2x4 (already shown in 3D)
        if not show_3d:
            ax_2d.set_title(scen.capitalize(), fontsize=11, pad=8)
        
        if col == 0:
            ax_2d.set_ylabel(r"$x_2$", fontsize=10, labelpad=2)  # Reduced labelpad
            ax_2d.set_yticks([-1, 0, 1])
        else:
            ax_2d.set_yticks([])
        
        ax_2d.set_xlabel(r"$x_1$", fontsize=10, labelpad=2)  # Reduced labelpad
        ax_2d.set_xticks([-1, 0, 1])
        
        images.append(im)
    
    # Add shared colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(images[-1], cax=cbar_ax)
    cbar.set_label("Explained Fraction", fontsize=10, labelpad=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Overall title
    title_y = 0.98 if layout == "1x4" else 0.96
    fig.suptitle("Toy Embeddings Explained Variance", fontsize=16, y=title_y)
    
    # Save
    outdir = cfg["outdir"]
    os.makedirs(outdir, exist_ok=True)
    
    png_path = os.path.join(outdir, f"toy_explained_frac_{layout.replace('x', 'by')}.png")
    pdf_path = os.path.join(outdir, f"toy_explained_frac_{layout.replace('x', 'by')}.pdf")
    
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[OK] Publication figure saved:")
    print(f"     {png_path}")
    print(f"     {pdf_path}")


# ---------------------- Main ----------------------
def main():
    cfg = CONFIG
    global device
    device = torch.device(
        cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    torch.set_grad_enabled(False)
    rng = torch.Generator(device=device).manual_seed(cfg["seed"])

    print("Running CCA-based injectivity diagnostics...")
    print(f"Device: {device}")
    print(f"Scenarios: {cfg['scenarios']}")
    print(f"Number of points: {cfg['n_pts']}")
    print()

    results_by_scenario = {}
    for scen in cfg["scenarios"]:
        results_by_scenario[scen] = run_one(scen, cfg, device, rng)

    print()
    print("Creating publication figures...")
    make_publication_figure(results_by_scenario, cfg, layout="1x4")
    make_publication_figure(results_by_scenario, cfg, layout="2x4")
    
    print()
    print(f"[COMPLETE] All outputs saved to: {cfg['outdir']}/")


if __name__ == "__main__":
    main()