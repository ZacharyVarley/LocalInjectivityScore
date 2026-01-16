#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch_make_4panel_figs.py

Minimalist 2x2 imshow grids (shared colorbar fixed to [0,1]) from
CSV outputs of ch_injectivity_analysis.py.

Fixes vs prior draft:
  - Correctly parses filenames with multi-token descriptors (e.g. *_fixed_raw_flat.csv).
  - Generates FIXED figures (raw_flat) when present.
  - Generates REPEATS figures for ALL descriptors present: radial1d, corr2d, euler.
  - Eliminates “missing bin” white-speckle artifacts by:
      * choosing bins from an occupancy target (auto bins)
      * smoothing numerator+denominator (sum and count) before division

Outputs:
  ch_figures/<timestamp>/
    fixed_explained_frac.(png/pdf)
    fixed_worst_retention.(png/pdf)
    repeats_radial1d_explained_frac.(png/pdf)
    repeats_radial1d_worst_retention.(png/pdf)
    repeats_corr2d_explained_frac.(png/pdf)
    repeats_corr2d_worst_retention.(png/pdf)
    repeats_euler_explained_frac.(png/pdf)
    repeats_euler_worst_retention.(png/pdf)
    manifest.txt

Dependencies: numpy, matplotlib
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# --------------------------- timestamp discovery ---------------------------

def find_most_recent_timestamp_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Analysis root not found: {root}")

    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if len(d.name) == 15 and "_" in d.name:
            a, b = d.name.split("_", 1)
            if a.isdigit() and b.isdigit():
                if any(d.glob("*.csv")):
                    candidates.append(d)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    any_dirs = [d for d in root.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    if not any_dirs:
        raise FileNotFoundError(f"No CSV outputs found under: {root}")
    any_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return any_dirs[0]


# --------------------------- robust filename parsing ---------------------------

def parse_analysis_csv_name(stem: str) -> Optional[Tuple[str, str, str]]:
    """
    Expected analysis naming:
      {warp}_fixed_raw_flat.csv
      {warp}_repeats_radial1d.csv
      {warp}_repeats_corr2d.csv
      {warp}_repeats_euler.csv
    Returns (warp, mode, desc) or None if not a recognized analysis CSV.
    """
    fixed_suffix = "_fixed_raw_flat"
    if stem.endswith(fixed_suffix):
        warp = stem[: -len(fixed_suffix)]
        return (warp, "fixed", "raw_flat")

    for desc in ("radial1d", "corr2d", "euler"):
        suf = f"_repeats_{desc}"
        if stem.endswith(suf):
            warp = stem[: -len(suf)]
            return (warp, "repeats", desc)

    return None


def choose_4_warps(warps: List[str]) -> List[str]:
    warps_u = sorted(set([w for w in warps if w]))
    if not warps_u:
        return ["", "", "", ""]

    def is_nowarp(w: str) -> bool:
        s = w.lower()
        return ("nowarp" in s) or ("no_warp" in s) or (s == "none") or (s == "identity")

    nowarp = [w for w in warps_u if is_nowarp(w)]
    rest = [w for w in warps_u if w not in nowarp]

    ordered: List[str] = []
    if nowarp:
        ordered.append(nowarp[0])
    ordered.extend(rest)

    ordered = ordered[:4]
    while len(ordered) < 4:
        ordered.append("")
    return ordered


# --------------------------- CSV reading ---------------------------

def read_metric_csv(csv_path: Path, xdim0: int, xdim1: int) -> Dict[str, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    x0 = arr[f"x_dim{xdim0}"]
    x1 = arr[f"x_dim{xdim1}"]
    expl = arr["explained_frac"]
    worst_ret = arr["worst_retention"]
    return dict(x0=x0, x1=x1, explained_frac=expl, worst_retention=worst_ret)


# --------------------------- smoothing helpers (no scipy) ---------------------------

def gaussian_kernel1d(sigma_px: float) -> np.ndarray:
    sigma = float(max(sigma_px, 1e-6))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def conv1d_reflect(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = k.size // 2
    pw = [(0, 0)] * arr.ndim
    pw[axis] = (pad, pad)
    x = np.pad(arr, pw, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=axis, arr=x)


def smooth2d_reflect(arr2d: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px is None or sigma_px <= 0:
        return arr2d
    k = gaussian_kernel1d(sigma_px)
    y = conv1d_reflect(arr2d, k, axis=0)
    y = conv1d_reflect(y, k, axis=1)
    return y


# --------------------------- binning with occupancy control ---------------------------

def choose_bins_auto(N: int, target_count_per_bin: int, bins_min: int, bins_max: int) -> int:
    """
    Pick bins so expected count per bin ~ target_count_per_bin.
    total bins (2D) ~ N / target
    => bins_per_axis ~ sqrt(N / target)
    """
    target = max(1, int(target_count_per_bin))
    b = int(round(math.sqrt(max(1.0, float(N) / float(target)))))
    b = max(int(bins_min), min(int(bins_max), b))
    return b


def binned_mean_smoothed(
    x0: np.ndarray,
    x1: np.ndarray,
    z: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    smooth_sigma_px: float,
    eps: float = 1e-12,
) -> np.ndarray:
    sum_w, _, _ = np.histogram2d(x0, x1, bins=[xedges, yedges], weights=z)
    cnt, _, _ = np.histogram2d(x0, x1, bins=[xedges, yedges])

    # Smooth numerator and denominator (this fills holes in a principled way)
    sum_s = smooth2d_reflect(sum_w, smooth_sigma_px)
    cnt_s = smooth2d_reflect(cnt, smooth_sigma_px)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = sum_s / np.maximum(cnt_s, eps)

    # Mask regions with effectively no support even after smoothing
    img[cnt_s < 0.5] = np.nan
    return img


# --------------------------- plotting ---------------------------

def minimalist_4panel_imshow(
    imgs: List[Optional[np.ndarray]],
    titles: List[str],
    out_base: Path,
    extent: Tuple[float, float, float, float],
    vmin: float = 0.0,
    vmax: float = 1.0,
    dpi: int = 300,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), dpi=dpi, constrained_layout=True)
    axes = axes.ravel()

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    last_im = None
    for i in range(4):
        ax = axes[i]
        img = imgs[i] if i < len(imgs) else None
        ttl = titles[i] if i < len(titles) else ""

        if img is None:
            ax.axis("off")
            continue

        m = np.ma.masked_invalid(img)
        last_im = ax.imshow(
            m.T,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.035, pad=0.02)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=9)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Make 4-panel imshow grids from ch_analysis CSV outputs.")
    ap.add_argument("--analysis_root", type=str, default="ch_tprofiles_analysis")
    ap.add_argument("--fig_root", type=str, default="ch_tprofiles_figures")
    ap.add_argument("--timestamp", type=str, default=None, help="Use ch_tprofiles_analysis/<timestamp>/ explicitly.")
    ap.add_argument("--xdim0", type=int, default=0)
    ap.add_argument("--xdim1", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=300)

    # heatmap controls
    ap.add_argument("--bins", type=int, default=0, help="If >0, force bins per axis; else auto.")
    ap.add_argument("--target_count_per_bin", type=int, default=8, help="Auto-binning occupancy target.")
    ap.add_argument("--bins_min", type=int, default=25)
    ap.add_argument("--bins_max", type=int, default=25)
    ap.add_argument("--smooth_sigma_px", type=float, default=1.0)

    args = ap.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    if args.timestamp is None:
        ts_dir = find_most_recent_timestamp_dir(analysis_root)
    else:
        ts_dir = (analysis_root / args.timestamp).expanduser().resolve()
        if not ts_dir.exists():
            raise FileNotFoundError(f"Timestamp dir not found: {ts_dir}")

    timestamp = ts_dir.name
    fig_root = Path(args.fig_root).expanduser().resolve()
    out_dir = fig_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect CSVs
    fixed_map: Dict[str, Path] = {}
    repeats_map: Dict[str, Dict[str, Path]] = {"radial1d": {}, "corr2d": {}, "euler": {}}

    for csv_path in ts_dir.glob("*.csv"):
        parsed = parse_analysis_csv_name(csv_path.stem)
        if parsed is None:
            continue
        warp, mode, desc = parsed
        if mode == "fixed" and desc == "raw_flat":
            fixed_map[warp] = csv_path
        if mode == "repeats" and desc in repeats_map:
            repeats_map[desc][warp] = csv_path

    # prefer warps from fixed if available; otherwise fall back to repeats union
    warps_for_layout = list(fixed_map.keys())
    if not warps_for_layout:
        union = set()
        for desc in repeats_map:
            union |= set(repeats_map[desc].keys())
        warps_for_layout = sorted(union)

    warp_list = choose_4_warps(warps_for_layout)

    def family_limits(paths: List[Path]) -> Tuple[float, float, float, float]:
        xs0, xs1 = [], []
        for p in paths:
            d = read_metric_csv(p, args.xdim0, args.xdim1)
            xs0.append(d["x0"])
            xs1.append(d["x1"])
        if not xs0:
            return (-1.0, 1.0, -1.0, 1.0)
        x0 = np.concatenate(xs0)
        x1 = np.concatenate(xs1)
        xmin, xmax = float(np.min(x0)), float(np.max(x0))
        ymin, ymax = float(np.min(x1)), float(np.max(x1))
        if xmin == xmax:
            xmax = xmin + 1e-6
        if ymin == ymax:
            ymax = ymin + 1e-6
        return (xmin, xmax, ymin, ymax)

    def choose_bins(N_total: int) -> int:
        if int(args.bins) > 0:
            return int(args.bins)
        return choose_bins_auto(
            N=int(N_total),
            target_count_per_bin=int(args.target_count_per_bin),
            bins_min=int(args.bins_min),
            bins_max=int(args.bins_max),
        )

    def make_family(
        warp2csv: Dict[str, Path],
        family_tag: str,
    ) -> None:
        paths = [warp2csv[w] for w in warp_list if w and (w in warp2csv)]
        xmin, xmax, ymin, ymax = family_limits(paths)

        # decide bins based on total points across included warps
        N_total = 0
        for p in paths:
            d = read_metric_csv(p, args.xdim0, args.xdim1)
            N_total += int(d["x0"].shape[0])
        bins = choose_bins(max(1, N_total))

        xedges = np.linspace(xmin, xmax, bins + 1)
        yedges = np.linspace(ymin, ymax, bins + 1)
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

        imgs_expl: List[Optional[np.ndarray]] = []
        imgs_wr: List[Optional[np.ndarray]] = []
        titles: List[str] = []

        for w in warp_list:
            if (not w) or (w not in warp2csv):
                imgs_expl.append(None)
                imgs_wr.append(None)
                titles.append("")
                continue

            d = read_metric_csv(warp2csv[w], args.xdim0, args.xdim1)

            img_expl = binned_mean_smoothed(
                d["x0"], d["x1"], d["explained_frac"],
                xedges, yedges,
                smooth_sigma_px=float(args.smooth_sigma_px),
            )
            img_wr = binned_mean_smoothed(
                d["x0"], d["x1"], d["worst_retention"],
                xedges, yedges,
                smooth_sigma_px=float(args.smooth_sigma_px),
            )

            imgs_expl.append(img_expl)
            imgs_wr.append(img_wr)
            titles.append(w)

        minimalist_4panel_imshow(
            imgs=imgs_expl,
            titles=titles,
            out_base=out_dir / f"{family_tag}_explained_frac",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            dpi=int(args.dpi),
        )
        minimalist_4panel_imshow(
            imgs=imgs_wr,
            titles=titles,
            out_base=out_dir / f"{family_tag}_worst_retention",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            dpi=int(args.dpi),
        )

    # FIXED
    if fixed_map:
        make_family(fixed_map, "fixed")

    # REPEATS: all descriptors that exist
    for desc in ("radial1d", "corr2d", "euler"):
        if repeats_map[desc]:
            make_family(repeats_map[desc], f"repeats_{desc}")

    # provenance
    manifest = out_dir / "manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"analysis_timestamp_dir: {ts_dir}\n")
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"xdim0: {args.xdim0}\n")
        f.write(f"xdim1: {args.xdim1}\n")
        f.write(f"bins_forced: {int(args.bins)}\n")
        f.write(f"target_count_per_bin: {int(args.target_count_per_bin)}\n")
        f.write(f"bins_min: {int(args.bins_min)}\n")
        f.write(f"bins_max: {int(args.bins_max)}\n")
        f.write(f"smooth_sigma_px: {float(args.smooth_sigma_px)}\n")
        f.write("\n[warps_layout]\n")
        for w in warp_list:
            f.write(f"{w}\n")
        f.write("\n[fixed]\n")
        for w, p in sorted(fixed_map.items()):
            f.write(f"{w}: {p.name}\n")
        f.write("\n[repeats]\n")
        for desc in ("radial1d", "corr2d", "euler"):
            f.write(f"\n  ({desc})\n")
            for w, p in sorted(repeats_map[desc].items()):
                f.write(f"    {w}: {p.name}\n")


if __name__ == "__main__":
    main()
