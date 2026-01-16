#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch_ab_publication_figures.py

Publication figures from analysis metrics.csv files, using the SAME heatmap binning/smoothing
as ch_plot_utils. No Y-embeddings. No PCA/UMAP.

Expected analysis layout (from ch_local_identifiability.py):
  analysis_root/<timestamp>/<warp>_<mode>/<descriptor>/inv/metrics.csv

Required columns in metrics.csv:
  - x_dim0, x_dim1
  - inv_explained_frac

Outputs (all under analysis_root/<timestamp>/publication/):
  1) Single-row heatmaps:
     einv_fixed_raw_flat_1by4.(png|pdf)
     einv_repeats_<descriptor>_1by4.(png|pdf)  for each repeats descriptor requested

  2) Combined 2x4:
     einv_fixed_vs_repeats_<repeats_descriptor>_2by4.(png|pdf)
       - top row: fixed/raw_flat
       - bottom row: repeats/<repeats_descriptor>

Heatmap computation mirrors ch_plot_utils.heatmap_binned_2d exactly:
  - histogram2d range uses [min(x), max(x)] per panel (adaptive bounds)
  - img = sum_w / cnt, NaN where cnt==0
  - smooth_nan reflect-padding separable Gaussian
  - extent uses bin edges (xedges, yedges)
  - we enforce vmin/vmax = [0,1] to match your e_inv semantics
    (analysis script often plotted adaptive and fixed; this is the fixed [0,1] version)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator


# ---------- Matplotlib aesthetics ----------
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


# ---------------------- timestamp discovery ----------------------
def find_most_recent_timestamp(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    timestamp_dirs = []
    for d in root.iterdir():
        if d.is_dir() and len(d.name) == 15 and "_" in d.name:
            parts = d.name.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                timestamp_dirs.append(d)
    if not timestamp_dirs:
        return None
    timestamp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return timestamp_dirs[0]


# ---------------------- ch_plot_utils-compatible smoothing ----------------------
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


def binned_image_like_ch_plot_utils(
    x0: np.ndarray,
    x1: np.ndarray,
    z: np.ndarray,
    bins0: int,
    bins1: int,
    sigma_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicates ch_plot_utils.heatmap_binned_2d core:
      - adaptive range = [min,max] for each axis
      - histogram2d -> mean per bin -> smooth_nan
    Returns:
      img_s (bins0,bins1) in x-y bin space (NOT transposed for imshow)
      xedges (bins0+1,)
      yedges (bins1+1,)
    """
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
    return img_s, xedges, yedges


# ---------------------- JSON metadata loader ----------------------
def load_param_ranges(data_root: Path, warp: str, mode: str) -> Dict[str, float]:
    """
    Load parameter ranges (alpha_min/max, beta_min/max) from the JSON metadata file.
    """
    json_path = data_root / f"{warp}_{mode}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON metadata not found: {json_path}")
    
    with open(json_path, "r") as f:
        meta = json.load(f)
    
    args = meta.get("args", {})
    return {
        "alpha_min": args.get("alpha_min", 0.7),
        "alpha_max": args.get("alpha_max", 1.3),
        "beta_min": args.get("beta_min", 0.7),
        "beta_max": args.get("beta_max", 1.3),
    }


def transform_coords_to_physical(x_norm: np.ndarray, param_min: float, param_max: float) -> np.ndarray:
    """
    Transform normalized coordinates from [-1, 1] to physical parameter range [param_min, param_max].
    """
    return param_min + (x_norm + 1.0) * (param_max - param_min) / 2.0


# ---------------------- CSV loader ----------------------
def load_metrics_csv(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "r") as f:
        header = f.readline().strip()
    if header.startswith("#"):
        header = header[1:].strip()
    cols = [c.strip() for c in header.split(",") if c.strip()]

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    out = {cols[j]: data[:, j].astype(np.float32) for j in range(len(cols))}
    required = {"x_dim0", "x_dim1", "inv_explained_frac"}
    missing = sorted(list(required.difference(out.keys())))
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    return out


# ---------------------- Path helpers ----------------------
def metrics_path(ts_dir: Path, warp: str, mode: str, descriptor: str) -> Path:
    return ts_dir / f"{warp}_{mode}" / descriptor / "inv" / "metrics.csv"


# ---------------------- Figures ----------------------
def save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_1x4_heatmaps(
    warp_order: List[str],
    metrics_by_warp: Dict[str, Dict[str, np.ndarray]],
    param_ranges_by_warp: Dict[str, Dict[str, float]],
    title: str,
    out_base: Path,
    bins0: int,
    bins1: int,
    sigma_px: float,
    cmap: str = "viridis",
):
    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.05],
        wspace=0.20, hspace=0.05
    )

    last_im = None
    for col, warp in enumerate(warp_order):
        m = metrics_by_warp[warp]
        pr = param_ranges_by_warp[warp]
        
        # Transform normalized coordinates to physical parameters
        x0_phys = transform_coords_to_physical(m["x_dim0"], pr["alpha_min"], pr["alpha_max"])
        x1_phys = transform_coords_to_physical(m["x_dim1"], pr["beta_min"], pr["beta_max"])
        z = m["inv_explained_frac"]

        img_s, xedges, yedges = binned_image_like_ch_plot_utils(
            x0=x0_phys, x1=x1_phys, z=z,
            bins0=bins0, bins1=bins1, sigma_px=sigma_px
        )

        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            img_s.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="equal",
            cmap=cmap,
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        last_im = im

        ax.set_title(warp, pad=8)
        ax.set_xlabel(r"$\alpha$", labelpad=2)
        ax.xaxis.set_major_locator(MaxNLocator(6))

        if col == 0:
            ax.set_ylabel(r"$\beta$", labelpad=2)
            ax.yaxis.set_major_locator(MaxNLocator(6))
        else:
            ax.set_yticks([])

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"$e_{\mathrm{inv}}$", labelpad=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(title, fontsize=12, y=0.98)
    save_png_pdf(fig, out_base)


def figure_2x4_fixed_vs_repeats(
    warp_order: List[str],
    fixed_metrics_by_warp: Dict[str, Dict[str, np.ndarray]],
    fixed_param_ranges_by_warp: Dict[str, Dict[str, float]],
    repeats_metrics_by_warp: Dict[str, Dict[str, np.ndarray]],
    repeats_param_ranges_by_warp: Dict[str, Dict[str, float]],
    title: str,
    out_base: Path,
    bins0: int,
    bins1: int,
    sigma_px: float,
    cmap: str = "viridis",
):
    fig = plt.figure(figsize=(12, 6.2))
    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.10],
        wspace=0.20, hspace=0.18
    )

    last_im = None
    for col, warp in enumerate(warp_order):
        # --- top row: fixed ---
        mF = fixed_metrics_by_warp[warp]
        prF = fixed_param_ranges_by_warp[warp]
        
        x0F_phys = transform_coords_to_physical(mF["x_dim0"], prF["alpha_min"], prF["alpha_max"])
        x1F_phys = transform_coords_to_physical(mF["x_dim1"], prF["beta_min"], prF["beta_max"])
        
        imgF, xedgesF, yedgesF = binned_image_like_ch_plot_utils(
            x0=x0F_phys, x1=x1F_phys, z=mF["inv_explained_frac"],
            bins0=bins0, bins1=bins1, sigma_px=sigma_px
        )
        axF = fig.add_subplot(gs[0, col])
        imF = axF.imshow(
            imgF.T,
            origin="lower",
            extent=[xedgesF[0], xedgesF[-1], yedgesF[0], yedgesF[-1]],
            aspect="equal",
            cmap=cmap,
            interpolation="bilinear",
            vmin=0.0, vmax=1.0,
        )
        last_im = imF
        axF.set_title(warp, pad=8)
        axF.set_xlabel("")  # Remove x-label on top row
        # axF.xaxis.set_major_locator(MaxNLocator(6))
        axF.set_xticklabels([])  # Remove x-tick labels on top row
        axF.set_xticks([]) # Remove x-ticks on top row

        if col == 0:
            axF.set_ylabel(r"$\beta$ (fixed)", labelpad=2)
            axF.yaxis.set_major_locator(MaxNLocator(6))
        else:
            axF.set_yticks([])

        # --- bottom row: repeats ---
        mR = repeats_metrics_by_warp[warp]
        prR = repeats_param_ranges_by_warp[warp]
        
        x0R_phys = transform_coords_to_physical(mR["x_dim0"], prR["alpha_min"], prR["alpha_max"])
        x1R_phys = transform_coords_to_physical(mR["x_dim1"], prR["beta_min"], prR["beta_max"])
        
        imgR, xedgesR, yedgesR = binned_image_like_ch_plot_utils(
            x0=x0R_phys, x1=x1R_phys, z=mR["inv_explained_frac"],
            bins0=bins0, bins1=bins1, sigma_px=sigma_px
        )
        axR = fig.add_subplot(gs[1, col])
        imR = axR.imshow(
            imgR.T,
            origin="lower",
            extent=[xedgesR[0], xedgesR[-1], yedgesR[0], yedgesR[-1]],
            aspect="equal",
            cmap=cmap,
            interpolation="bilinear",
            vmin=0.0, vmax=1.0,
        )
        last_im = imR
        axR.set_xlabel(r"$\alpha$", labelpad=2)
        axR.xaxis.set_major_locator(MaxNLocator(6))

        if col == 0:
            axR.set_ylabel(r"$\beta$ (repeats)", labelpad=2)
            axR.yaxis.set_major_locator(MaxNLocator(6))
        else:
            axR.set_yticks([])

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"$e_{\mathrm{inv}}$", labelpad=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Add horizontal partition line between rows
    fig.add_artist(plt.Line2D([0.16, 0.82], [0.49, 0.49], transform=fig.transFigure, 
                              color='black', linewidth=1.5, linestyle='-'))

    fig.suptitle(title, fontsize=12, y=0.98)
    save_png_pdf(fig, out_base)


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", type=str, default="ch_ab_analysis")
    ap.add_argument("--data_root", type=str, default="ch_ab_data")
    ap.add_argument("--timestamp", type=str, default="", help="If empty, uses most recent timestamp dir.")
    ap.add_argument("--warps", type=str, default="nowarp,pinch,ribbon,fold")

    # descriptors
    ap.add_argument("--fixed_descriptor", type=str, default="raw_flat")
    ap.add_argument("--repeats_descriptors", type=str, default="radial1d,corr2d",
                    help="Comma-separated list for 1x4 repeats figures.")
    ap.add_argument("--repeats_descriptor_for_2x4", type=str, default="radial1d",
                    help="Which repeats descriptor to use in the 2x4 fixed-vs-repeats figure.")

    # heatmap params (match analysis defaults)
    ap.add_argument("--hm_bins0", type=int, default=60)
    ap.add_argument("--hm_bins1", type=int, default=60)
    ap.add_argument("--hm_sigma_px", type=float, default=1.0)

    ap.add_argument("--cmap", type=str, default="viridis")
    args = ap.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()

    if args.timestamp.strip():
        ts_dir = analysis_root / args.timestamp.strip()
        if not ts_dir.exists():
            raise FileNotFoundError(f"timestamp dir not found: {ts_dir}")
    else:
        ts_dir = find_most_recent_timestamp(analysis_root)
        if ts_dir is None:
            raise FileNotFoundError(f"No timestamp dirs found under: {analysis_root}")

    timestamp = ts_dir.name
    outdir = ts_dir / "publication"
    outdir.mkdir(parents=True, exist_ok=True)

    warp_order = [w.strip() for w in args.warps.split(",") if w.strip()]
    fixed_desc = args.fixed_descriptor.strip()
    repeats_descs = [d.strip() for d in args.repeats_descriptors.split(",") if d.strip()]
    repeats_desc_2x4 = args.repeats_descriptor_for_2x4.strip()

    # --- fixed 1x4 ---
    fixed_metrics_by_warp: Dict[str, Dict[str, np.ndarray]] = {}
    fixed_param_ranges_by_warp: Dict[str, Dict[str, float]] = {}
    missing = []
    for warp in warp_order:
        p = metrics_path(ts_dir, warp, "fixed", fixed_desc)
        if not p.exists():
            missing.append(str(p))
        else:
            fixed_metrics_by_warp[warp] = load_metrics_csv(p)
            fixed_param_ranges_by_warp[warp] = load_param_ranges(data_root / timestamp, warp, "fixed")
    if missing:
        raise FileNotFoundError("Missing fixed metrics.csv files:\n" + "\n".join(missing))

    title_fixed = f"Fixed seed"
    out_base_fixed = outdir / f"einv_fixed_{fixed_desc}_1by4"
    figure_1x4_heatmaps(
        warp_order=warp_order,
        metrics_by_warp=fixed_metrics_by_warp,
        param_ranges_by_warp=fixed_param_ranges_by_warp,
        title=title_fixed,
        out_base=out_base_fixed,
        bins0=int(args.hm_bins0),
        bins1=int(args.hm_bins1),
        sigma_px=float(args.hm_sigma_px),
        cmap=str(args.cmap),
    )
    print(f"[OK] {out_base_fixed}.png/.pdf")

    # --- repeats 1x4 (for each repeats descriptor requested) ---
    repeats_metrics_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    repeats_param_ranges_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    for desc in repeats_descs:
        rep_metrics_by_warp: Dict[str, Dict[str, np.ndarray]] = {}
        rep_param_ranges_by_warp: Dict[str, Dict[str, float]] = {}
        missing = []
        for warp in warp_order:
            p = metrics_path(ts_dir, warp, "repeats", desc)
            if not p.exists():
                missing.append(str(p))
            else:
                rep_metrics_by_warp[warp] = load_metrics_csv(p)
                rep_param_ranges_by_warp[warp] = load_param_ranges(data_root / timestamp, warp, "repeats")
        if missing:
            raise FileNotFoundError(f"Missing repeats metrics.csv files for descriptor='{desc}':\n" + "\n".join(missing))

        repeats_metrics_cache[desc] = rep_metrics_by_warp
        repeats_param_ranges_cache[desc] = rep_param_ranges_by_warp

        title_rep = f"Repeated seeds"
        out_base_rep = outdir / f"einv_repeats_{desc}_1by4"
        figure_1x4_heatmaps(
            warp_order=warp_order,
            metrics_by_warp=rep_metrics_by_warp,
            param_ranges_by_warp=rep_param_ranges_by_warp,
            title=title_rep,
            out_base=out_base_rep,
            bins0=int(args.hm_bins0),
            bins1=int(args.hm_bins1),
            sigma_px=float(args.hm_sigma_px),
            cmap=str(args.cmap),
        )
        print(f"[OK] {out_base_rep}.png/.pdf")

    # --- combined 2x4: fixed top vs repeats bottom (both radial1d and corr2d) ---
    for desc_2x4 in ["radial1d", "corr2d"]:
        if desc_2x4 not in repeats_metrics_cache:
            # load if not already loaded
            rep_metrics_by_warp: Dict[str, Dict[str, np.ndarray]] = {}
            rep_param_ranges_by_warp: Dict[str, Dict[str, float]] = {}
            missing = []
            for warp in warp_order:
                p = metrics_path(ts_dir, warp, "repeats", desc_2x4)
                if not p.exists():
                    missing.append(str(p))
                else:
                    rep_metrics_by_warp[warp] = load_metrics_csv(p)
                    rep_param_ranges_by_warp[warp] = load_param_ranges(data_root / timestamp, warp, "repeats")
            if missing:
                raise FileNotFoundError(
                    f"Missing repeats metrics.csv files for 2x4 descriptor='{desc_2x4}':\n" + "\n".join(missing)
                )
            repeats_metrics_cache[desc_2x4] = rep_metrics_by_warp
            repeats_param_ranges_cache[desc_2x4] = rep_param_ranges_by_warp

        title_2x4 = f"Explained Variance (Fixed Seed vs Repeated Instantiations)"
        out_base_2x4 = outdir / f"einv_fixed_vs_repeats_{desc_2x4}_2by4"
        figure_2x4_fixed_vs_repeats(
            warp_order=warp_order,
            fixed_metrics_by_warp=fixed_metrics_by_warp,
            fixed_param_ranges_by_warp=fixed_param_ranges_by_warp,
            repeats_metrics_by_warp=repeats_metrics_cache[desc_2x4],
            repeats_param_ranges_by_warp=repeats_param_ranges_cache[desc_2x4],
            title=title_2x4,
            out_base=out_base_2x4,
            bins0=int(args.hm_bins0),
            bins1=int(args.hm_bins1),
            sigma_px=float(args.hm_sigma_px),
            cmap=str(args.cmap),
        )
        print(f"[OK] {out_base_2x4}.png/.pdf")

    print(f"[DONE] outputs in: {outdir}")


if __name__ == "__main__":
    main()
