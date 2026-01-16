# ch_plot_utils.py
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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


def smooth_nan(img: np.ndarray, sigma_px: float) -> np.ndarray:
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


def save_both(fig, base: Path) -> None:
    png_dir = base.parent / "png"
    pdf_dir = base.parent / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / (base.name + ".png"), bbox_inches="tight")
    fig.savefig(pdf_dir / (base.name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def scatter_2d(x0, x1, z, xlabel, ylabel, title, base: Path, dpi: int, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5.3, 4.4), dpi=dpi)
    if vmin is None:
        vmin = float(np.nanmin(z))
    if vmax is None:
        vmax = float(np.nanmax(z))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6
    sc = ax.scatter(x0, x1, c=z, s=8, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.95, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    fig.tight_layout()
    save_both(fig, base)


def heatmap_binned_2d(x0, x1, z, xlabel, ylabel, title, base: Path,
                      bins0: int, bins1: int, sigma_px: float, dpi: int,
                      vmin=None, vmax=None, cmap="viridis"):
    x0 = x0.astype(np.float64)
    x1 = x1.astype(np.float64)
    z = z.astype(np.float64)
    xmin, xmax = float(np.min(x0)), float(np.max(x0))
    ymin, ymax = float(np.min(x1)), float(np.max(x1))

    sum_w, xedges, yedges = np.histogram2d(
        x0, x1, bins=[bins0, bins1],
        range=[[xmin, xmax], [ymin, ymax]],
        weights=z,
    )
    cnt, _, _ = np.histogram2d(
        x0, x1, bins=[xedges, yedges],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        img = sum_w / cnt
    img[cnt == 0] = np.nan

    img_s = smooth_nan(img, sigma_px)

    if vmin is None:
        vmin = float(np.nanmin(img_s)) if np.isfinite(np.nanmin(img_s)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(img_s)) if np.isfinite(np.nanmax(img_s)) else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=dpi)
    im = ax.imshow(
        img_s.T, origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap=cmap, interpolation="bilinear",
        vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    save_both(fig, base)
