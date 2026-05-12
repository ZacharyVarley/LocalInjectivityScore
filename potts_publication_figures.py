#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_publication_figures.py

Unified publication script for the two Potts figures:
  1) heatmap_explained_frac_lines.(png|pdf)
  2) potts_control_lines_corr2d_avg_mosaic_00_01_11_12.(png|pdf)

The control-space sweeps are defined once and shared by both figures.
By default, the high-f0 and low-f0 temperature sweeps extend from T=0.6 to T=1.2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

import potts_analyze_explained_fraction_v13 as analysis
import potts_visualize_control_space_descriptors_v2 as control_viz


@dataclass(frozen=True)
class ControlSweeps:
    temps_high: np.ndarray
    fracs_high: np.ndarray
    temps_mid: np.ndarray
    fracs_mid: np.ndarray
    temps_low: np.ndarray
    fracs_low: np.ndarray

    def to_json(self) -> Dict[str, List[float]]:
        return {
            "temps_high": self.temps_high.tolist(),
            "fracs_high": self.fracs_high.tolist(),
            "temps_mid": self.temps_mid.tolist(),
            "fracs_mid": self.fracs_mid.tolist(),
            "temps_low": self.temps_low.tolist(),
            "fracs_low": self.fracs_low.tolist(),
        }


def _inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("temp_step must be positive")
    n = int(round((stop - start) / step))
    values = start + step * np.arange(n + 1, dtype=np.float64)
    if not np.isclose(values[-1], stop, atol=1e-8):
        raise ValueError("temp range must be evenly divisible by temp_step")
    values[-1] = stop
    return values.astype(np.float32)


def _parse_csv_floats(raw: str) -> np.ndarray:
    vals = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not vals:
        raise ValueError("fraction_sweep must contain at least one value")
    return np.asarray(vals, dtype=np.float32)


def build_control_sweeps(
    temp_start: float,
    temp_stop: float,
    temp_step: float,
    frac_low: float,
    frac_high: float,
    temp_mid: float,
    frac_mid_values: np.ndarray,
) -> ControlSweeps:
    temps_line = _inclusive_range(temp_start, temp_stop, temp_step)
    return ControlSweeps(
        temps_high=temps_line.copy(),
        fracs_high=np.full(temps_line.shape, frac_high, dtype=np.float32),
        temps_mid=np.full(frac_mid_values.shape, temp_mid, dtype=np.float32),
        fracs_mid=frac_mid_values.astype(np.float32, copy=False),
        temps_low=temps_line.copy(),
        fracs_low=np.full(temps_line.shape, frac_low, dtype=np.float32),
    )


def build_overlay_lines(
    sweeps: ControlSweeps,
) -> List[tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    base = dict(color="black", linewidth=3.0, alpha=0.9, zorder=5)
    points = dict(
        color="white",
        linewidth=1.8,
        alpha=0.95,
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=6,
    )
    return [
        (sweeps.temps_high, sweeps.fracs_high, dict(base)),
        (sweeps.temps_high, sweeps.fracs_high, dict(points)),
        (sweeps.temps_mid, sweeps.fracs_mid, dict(base)),
        (sweeps.temps_mid, sweeps.fracs_mid, dict(points)),
        (sweeps.temps_low, sweeps.fracs_low, dict(base)),
        (sweeps.temps_low, sweeps.fracs_low, dict(points)),
    ]


def resolve_device(device_name: str) -> torch.device:
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{device_name}' is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(device_name)


def compute_projected_Y(desc_h5: Path, cfg: analysis.Config) -> tuple[np.ndarray, Dict[str, Any]]:
    meta = analysis._get_desc_meta(desc_h5)
    n_samples = int(meta["N"])
    pca_info: Dict[str, Any] = {"enabled": bool(cfg.use_global_pca)}

    if cfg.use_global_pca:
        pca_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[cfg.pca_dtype]
        pca_device = resolve_device(cfg.pca_device)
        mu = analysis.compute_global_mean_Y(desc_h5, cfg, batch_size=cfg.pca_batch_size)

        method = cfg.pca_method
        if method == "auto":
            y_dim = len(mu)
            mem_gb = (n_samples * y_dim * 4) / (1024 ** 3)
            use_svd = (mem_gb <= cfg.pca_svd_max_gb) and pca_device.type == "cuda"
            method = "svd" if use_svd else "oja"
            print(
                f"Auto-selected PCA method: {method} "
                f"(estimated data size: {mem_gb:.2f} GB)"
            )

        if method == "svd":
            Q, eig_sums, total_energy = analysis.fit_svd_pca(
                desc_h5=desc_h5,
                cfg=cfg,
                mu=mu,
                n_components=int(cfg.pca_components_max),
                device=pca_device,
                dtype=pca_dtype,
            )
        else:
            Q = analysis.fit_oja_pca_streaming(
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
            eig_sums, total_energy = analysis.estimate_pca_energy_streaming(
                desc_h5=desc_h5,
                cfg=cfg,
                mu=mu,
                Q=Q,
                batch_size=int(cfg.pca_batch_size),
                device=pca_device,
                dtype=pca_dtype,
            )

        cum = np.cumsum(eig_sums) / max(float(total_energy), 1e-30)
        m = int(np.searchsorted(cum, float(cfg.pca_energy_frac)) + 1)
        m = max(1, min(m, int(cfg.pca_components_max)))
        Yp = analysis.project_Y_with_Q(
            desc_h5=desc_h5,
            cfg=cfg,
            mu=mu,
            Qm=Q[:, :m].detach().clone(),
            batch_size=int(cfg.pca_batch_size),
            device=pca_device,
            dtype=pca_dtype,
        )
        pca_info.update(
            {
                "method": method,
                "selected_components": int(m),
                "achieved_energy": float(cum[m - 1]) if cum.size >= m else float("nan"),
                "device": str(pca_device),
                "dtype": cfg.pca_dtype,
            }
        )
        return Yp, pca_info

    y_dim = int(meta["feat_dim"] + (meta["ph_dim"] if cfg.prepend_phase_fractions_to_Y else 0))
    Yp = np.empty((n_samples, y_dim), dtype=np.float32)
    for i0, i1, Yb in analysis.iter_Y_batches(desc_h5, cfg, batch_size=256):
        Yp[i0:i1, :] = Yb
    pca_info["note"] = "global_pca disabled; using raw Y"
    return Yp, pca_info


def compute_explained_fraction_metrics(
    desc_h5: Path,
    cfg: analysis.Config,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    X, _ = analysis.load_X_and_basic_meta(desc_h5)
    Yp, pca_info = compute_projected_Y(desc_h5, cfg)

    X_use = X.copy()
    Y_use = Yp.copy()
    if cfg.standardize_X:
        X_use = analysis.standardize_np(X_use)
    if cfg.standardize_Y:
        Y_use = analysis.standardize_np(Y_use)

    device = resolve_device(cfg.device)
    Xt = analysis.to_t(X_use, device=device)
    Yt = analysis.to_t(Y_use, device=device)
    idxY_t, dY_t = analysis.knn_in_y(Yt, k=int(cfg.kY))
    metrics = analysis.local_explainedcov_metrics_LOO(
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
        loo_uninf_align_thr=float(cfg.loo_uninf_align_thr),
        loo_uninf_gap_thr=float(cfg.loo_uninf_gap_thr),
        loo_uninf_dof_thr=float(cfg.loo_uninf_dof_thr),
    )
    return X, Yp, metrics, pca_info


def _build_binned_image(
    temp: np.ndarray,
    frac: np.ndarray,
    values: np.ndarray,
    bins_t: int,
    bins_f: int,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temp = temp.astype(np.float64)
    frac = frac.astype(np.float64)
    values = values.astype(np.float64)
    tmin, tmax = float(np.min(temp)), float(np.max(temp))
    fmin, fmax = float(np.min(frac)), float(np.max(frac))

    sum_w, tx, fx = np.histogram2d(
        temp,
        frac,
        bins=[bins_t, bins_f],
        range=[[tmin, tmax], [fmin, fmax]],
        weights=values,
    )
    cnt, _, _ = np.histogram2d(temp, frac, bins=[tx, fx], range=[[tmin, tmax], [fmin, fmax]])

    with np.errstate(invalid="ignore", divide="ignore"):
        img = sum_w / cnt
    img[cnt == 0] = np.nan
    img_s = analysis._smooth_nan(img, sigma_px)
    return img_s, tx, fx


def _label_control_lines(ax: plt.Axes) -> None:
    labels = [("A", 0.06, 0.82), ("B", 0.06, 0.50), ("C", 0.06, 0.18)]
    for text, xpos, ypos in labels:
        ax.text(
            xpos,
            ypos,
            text,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
            ),
        )


def save_explained_fraction_figure(
    outbase: Path,
    temp: np.ndarray,
    frac: np.ndarray,
    explained_frac: np.ndarray,
    Yp: np.ndarray,
    overlays: List[tuple[np.ndarray, np.ndarray, Dict[str, Any]]],
    bins_t: int,
    bins_f: int,
    sigma_px: float,
    dpi: int,
) -> None:
    img_s, tx, fx = _build_binned_image(temp, frac, explained_frac, bins_t, bins_f, sigma_px)
    vmin, vmax = 0.0, 1.0

    if Yp.shape[1] >= 3:
        pc1 = Yp[:, 0]
        pc2 = Yp[:, 1]
        pc3 = Yp[:, 2]
    elif Yp.shape[1] == 2:
        pc1 = Yp[:, 0]
        pc2 = Yp[:, 1]
        pc3 = np.zeros(Yp.shape[0], dtype=np.float32)
    else:
        pc1 = Yp[:, 0]
        pc2 = np.zeros(Yp.shape[0], dtype=np.float32)
        pc3 = np.zeros(Yp.shape[0], dtype=np.float32)

    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    pc2 = (pc2 - pc2.mean()) / (pc2.std() + 1e-8)
    pc3 = (pc3 - pc3.mean()) / (pc3.std() + 1e-8)

    fig = plt.figure(figsize=(13, 5), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.1], wspace=0.3)

    ax_hm = fig.add_subplot(gs[0, 0])
    im = ax_hm.imshow(
        img_s.T,
        origin="lower",
        extent=[tx[0], tx[-1], fx[0], fx[-1]],
        aspect="auto",
        cmap="viridis",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax_hm, pad=0.02, fraction=0.05)
    cbar.set_label("Explained Fraction", fontsize=10)
    ax_hm.set_xlabel(r"$T$", fontsize=11)
    ax_hm.set_ylabel(r"$f_0$", fontsize=11)
    ax_hm.set_title("Explained Fraction 3-State Potts", fontsize=12, fontweight="bold")
    for xline, yline, kwargs in overlays:
        ax_hm.plot(np.asarray(xline), np.asarray(yline), **kwargs)
    _label_control_lines(ax_hm)

    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.scatter(pc1, pc2, pc3, c=explained_frac, cmap="viridis", s=20, alpha=0.6, vmin=vmin, vmax=vmax)
    ax_3d.set_xlabel("PC1", fontsize=10, labelpad=8)
    ax_3d.set_ylabel("PC2", fontsize=10, labelpad=8)
    ax_3d.set_zlabel("PC3", fontsize=10, labelpad=8)
    ax_3d.set_title("3D PCA Projection", fontsize=12, fontweight="bold")
    ax_3d.view_init(elev=20, azim=45)

    fig.tight_layout()
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outbase) + ".png", bbox_inches="tight", dpi=dpi)
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _build_mosaics(
    corr_high: np.ndarray,
    corr_mid: np.ndarray,
    corr_low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pairs = int(corr_high.shape[1])
    vmin = np.zeros((n_pairs,), dtype=np.float64)
    vmax = np.zeros((n_pairs,), dtype=np.float64)
    for pair_idx in range(n_pairs):
        all_pair = np.concatenate(
            [corr_high[:, pair_idx], corr_mid[:, pair_idx], corr_low[:, pair_idx]],
            axis=0,
        )
        vmin[pair_idx] = float(np.min(all_pair))
        vmax[pair_idx] = float(np.max(all_pair))

    def make_group_mosaics(group: np.ndarray) -> np.ndarray:
        mosaics = []
        for sample_idx in range(group.shape[0]):
            tiles = [
                control_viz.normalize01(group[sample_idx, pair_idx], vmin[pair_idx], vmax[pair_idx])
                for pair_idx in range(n_pairs)
            ]
            mosaics.append(control_viz.make_mosaic_2x2(tiles[0], tiles[1], tiles[2], tiles[3], pad=2))
        return np.stack(mosaics, axis=0)

    return make_group_mosaics(corr_high), make_group_mosaics(corr_mid), make_group_mosaics(corr_low)


def _add_row_labels(
    top_ax: plt.Axes,
    bottom_ax: plt.Axes,
    panel_label: str,
    pair_names: Iterable[str],
) -> None:
    pair_names = list(pair_names)
    top_ax.text(
        -0.24,
        0.5,
        "Microstructure",
        transform=top_ax.transAxes,
        fontsize=13,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )
    bottom_ax.text(
        -0.24,
        0.5,
        f"Corr2D avg mosaic\n({pair_names[0]},{pair_names[1]}; {pair_names[2]},{pair_names[3]})",
        transform=bottom_ax.transAxes,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )
    top_ax.text(-0.33, 1.05, panel_label, transform=top_ax.transAxes, fontsize=26, fontweight="bold")


def save_corr2d_mosaic_figure(
    outbase: Path,
    sweeps: ControlSweeps,
    micro_high: np.ndarray,
    corr_high: np.ndarray,
    micro_mid: np.ndarray,
    corr_mid: np.ndarray,
    micro_low: np.ndarray,
    corr_low: np.ndarray,
    pair_names: List[str],
    dpi: int,
) -> None:
    mosaics_high, mosaics_mid, mosaics_low = _build_mosaics(corr_high, corr_mid, corr_low)
    group_specs = [
        ("A", sweeps.temps_high, sweeps.fracs_high, micro_high, mosaics_high, "high"),
        ("B", sweeps.temps_mid, sweeps.fracs_mid, micro_mid, mosaics_mid, "mid"),
        ("C", sweeps.temps_low, sweeps.fracs_low, micro_low, mosaics_low, "low"),
    ]
    max_cols = max(int(group[3].shape[0]) for group in group_specs)

    fig = plt.figure(figsize=(2.55 * max_cols + 1.8, 15.5), dpi=dpi)
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.22, left=0.08, right=0.995, top=0.97, bottom=0.05)

    for group_idx, (label, temps, fracs, micro, mosaics, mode) in enumerate(group_specs):
        n_cols = int(micro.shape[0])
        inner = gridspec.GridSpecFromSubplotSpec(2, n_cols, subplot_spec=outer[group_idx], hspace=0.03, wspace=0.03)
        top_axes: List[plt.Axes] = []
        bottom_axes: List[plt.Axes] = []

        for col_idx in range(n_cols):
            ax_top = fig.add_subplot(inner[0, col_idx])
            ax_bottom = fig.add_subplot(inner[1, col_idx])
            top_axes.append(ax_top)
            bottom_axes.append(ax_bottom)

            ax_top.imshow(micro[col_idx], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_bottom.imshow(mosaics[col_idx], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            ax_top.set_axis_off()
            ax_bottom.set_axis_off()

            if mode == "mid":
                title = f"$f_0={fracs[col_idx]:.2f}$  $T={temps[col_idx]:.1f}$"
            else:
                title = f"$T={temps[col_idx]:.1f}$  $f_0={fracs[col_idx]:.1f}$"
            ax_top.set_title(title, fontsize=12, pad=6)

        _add_row_labels(top_axes[0], bottom_axes[0], label, pair_names)

    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outbase) + ".png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate unified Potts publication figures")
    ap.add_argument("--h5", type=str, default=analysis.Config.h5, help="Input descriptor HDF5")
    ap.add_argument("--outdir", type=str, default="potts_figures/publication")
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--temp_start", type=float, default=0.6)
    ap.add_argument("--temp_stop", type=float, default=1.2)
    ap.add_argument("--temp_step", type=float, default=0.1)
    ap.add_argument("--frac_low", type=float, default=0.20)
    ap.add_argument("--frac_high", type=float, default=0.80)
    ap.add_argument("--temp_mid", type=float, default=0.60)
    ap.add_argument(
        "--fraction_sweep",
        type=str,
        default="0.25,0.35,0.45,0.55,0.65,0.75",
        help="Comma-separated f0 values for the vertical sweep",
    )

    ap.add_argument("--prepend_phase_fractions_to_Y", type=int, default=1 if analysis.Config.prepend_phase_fractions_to_Y else 0)
    ap.add_argument("--use_global_pca", type=int, default=1 if analysis.Config.use_global_pca else 0)
    ap.add_argument("--pca_method", type=str, default=analysis.Config.pca_method, choices=["auto", "svd", "oja"])
    ap.add_argument("--pca_svd_max_gb", type=float, default=analysis.Config.pca_svd_max_gb)
    ap.add_argument("--pca_energy_frac", type=float, default=analysis.Config.pca_energy_frac)
    ap.add_argument("--pca_components_max", type=int, default=analysis.Config.pca_components_max)
    ap.add_argument("--pca_epochs", type=int, default=analysis.Config.pca_epochs)
    ap.add_argument("--pca_batch_size", type=int, default=analysis.Config.pca_batch_size)
    ap.add_argument("--pca_eta", type=float, default=analysis.Config.pca_eta)
    ap.add_argument("--pca_device", type=str, default=analysis.Config.pca_device)
    ap.add_argument("--pca_dtype", type=str, default=analysis.Config.pca_dtype, choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--pca_seed", type=int, default=analysis.Config.pca_seed)
    ap.add_argument("--standardize_X", type=int, default=1 if analysis.Config.standardize_X else 0)
    ap.add_argument("--standardize_Y", type=int, default=1 if analysis.Config.standardize_Y else 0)
    ap.add_argument("--kY", type=int, default=analysis.Config.kY)
    ap.add_argument("--use_weights", action="store_true")
    ap.add_argument("--ridge_y", type=float, default=analysis.Config.ridge_y)
    ap.add_argument("--ridge_x", type=float, default=analysis.Config.ridge_x)
    ap.add_argument("--analysis_batch_size", type=int, default=analysis.Config.batch_size)
    ap.add_argument("--analysis_device", type=str, default=analysis.Config.device)
    ap.add_argument("--hm_bins_temp", type=int, default=analysis.Config.hm_bins_temp)
    ap.add_argument("--hm_bins_frac", type=int, default=analysis.Config.hm_bins_frac)
    ap.add_argument("--hm_sigma_px", type=float, default=analysis.Config.hm_sigma_px)
    ap.add_argument("--loo_uninf_align_thr", type=float, default=analysis.Config.loo_uninf_align_thr)
    ap.add_argument("--loo_uninf_gap_thr", type=float, default=analysis.Config.loo_uninf_gap_thr)
    ap.add_argument("--loo_uninf_dof_thr", type=float, default=analysis.Config.loo_uninf_dof_thr)

    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--periodic", type=int, default=1)
    ap.add_argument("--remove_spurious", type=int, default=0)
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--seed_micro_A", type=int, default=2024)
    ap.add_argument("--seed_micro_B", type=int, default=2041)
    ap.add_argument("--seed_micro_C", type=int, default=2058)
    ap.add_argument("--seed_repeat_base_A", type=int, default=9001)
    ap.add_argument("--seed_repeat_base_B", type=int, default=19001)
    ap.add_argument("--seed_repeat_base_C", type=int, default=29001)
    ap.add_argument("--corr_downsample", type=int, default=1)
    ap.add_argument("--sim_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress_every", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    desc_h5 = Path(args.h5).expanduser().resolve()
    if not desc_h5.exists():
        raise FileNotFoundError(f"Input not found: {desc_h5}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sweeps = build_control_sweeps(
        temp_start=float(args.temp_start),
        temp_stop=float(args.temp_stop),
        temp_step=float(args.temp_step),
        frac_low=float(args.frac_low),
        frac_high=float(args.frac_high),
        temp_mid=float(args.temp_mid),
        frac_mid_values=_parse_csv_floats(args.fraction_sweep),
    )
    overlays = build_overlay_lines(sweeps)

    analysis_cfg = analysis.Config(
        h5=str(desc_h5),
        prepend_phase_fractions_to_Y=bool(int(args.prepend_phase_fractions_to_Y)),
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
        standardize_X=bool(int(args.standardize_X)),
        standardize_Y=bool(int(args.standardize_Y)),
        kY=int(args.kY),
        use_weights=bool(args.use_weights),
        ridge_y=float(args.ridge_y),
        ridge_x=float(args.ridge_x),
        batch_size=int(args.analysis_batch_size),
        device=str(args.analysis_device),
        dpi=int(args.dpi),
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        loo_uninf_align_thr=float(args.loo_uninf_align_thr),
        loo_uninf_gap_thr=float(args.loo_uninf_gap_thr),
        loo_uninf_dof_thr=float(args.loo_uninf_dof_thr),
    )

    print("Computing explained-fraction publication figure...")
    X, Yp, metrics, pca_info = compute_explained_fraction_metrics(desc_h5, analysis_cfg)
    save_explained_fraction_figure(
        outbase=outdir / "heatmap_explained_frac_lines",
        temp=X[:, 0],
        frac=X[:, 1],
        explained_frac=metrics["explained_frac"],
        Yp=Yp,
        overlays=overlays,
        bins_t=int(args.hm_bins_temp),
        bins_f=int(args.hm_bins_frac),
        sigma_px=float(args.hm_sigma_px),
        dpi=int(args.dpi),
    )

    if int(args.q) != 3:
        raise ValueError("This publication script assumes q=3 with pairs 0-0, 0-1, 1-1, 1-2.")

    sim_device = resolve_device(args.sim_device)
    pair_names = ["0-0", "0-1", "1-1", "1-2"]
    pair_indices = [control_viz.pair_to_index(int(args.q), name) for name in pair_names]

    print("Simulating Corr2D publication sweeps...")
    micro_high, corr_high = control_viz.run_sweep_avg(
        temperatures=sweeps.temps_high,
        fractions0=sweeps.fracs_high,
        grid_size=int(args.grid),
        steps=int(args.steps),
        q=int(args.q),
        periodic=bool(int(args.periodic)),
        remove_spurious=bool(int(args.remove_spurious)),
        n_repeats=int(args.n_repeats),
        seed_micro=int(args.seed_micro_A),
        seed_repeat_base=int(args.seed_repeat_base_A),
        pair_indices=pair_indices,
        corr_downsample=int(args.corr_downsample),
        device=sim_device,
        progress_every=int(args.progress_every),
    )
    micro_mid, corr_mid = control_viz.run_sweep_avg(
        temperatures=sweeps.temps_mid,
        fractions0=sweeps.fracs_mid,
        grid_size=int(args.grid),
        steps=int(args.steps),
        q=int(args.q),
        periodic=bool(int(args.periodic)),
        remove_spurious=bool(int(args.remove_spurious)),
        n_repeats=int(args.n_repeats),
        seed_micro=int(args.seed_micro_B),
        seed_repeat_base=int(args.seed_repeat_base_B),
        pair_indices=pair_indices,
        corr_downsample=int(args.corr_downsample),
        device=sim_device,
        progress_every=int(args.progress_every),
    )
    micro_low, corr_low = control_viz.run_sweep_avg(
        temperatures=sweeps.temps_low,
        fractions0=sweeps.fracs_low,
        grid_size=int(args.grid),
        steps=int(args.steps),
        q=int(args.q),
        periodic=bool(int(args.periodic)),
        remove_spurious=bool(int(args.remove_spurious)),
        n_repeats=int(args.n_repeats),
        seed_micro=int(args.seed_micro_C),
        seed_repeat_base=int(args.seed_repeat_base_C),
        pair_indices=pair_indices,
        corr_downsample=int(args.corr_downsample),
        device=sim_device,
        progress_every=int(args.progress_every),
    )

    save_corr2d_mosaic_figure(
        outbase=outdir / "potts_control_lines_corr2d_avg_mosaic_00_01_11_12",
        sweeps=sweeps,
        micro_high=micro_high,
        corr_high=corr_high,
        micro_mid=micro_mid,
        corr_mid=corr_mid,
        micro_low=micro_low,
        corr_low=corr_low,
        pair_names=pair_names,
        dpi=int(args.dpi),
    )

    metadata = {
        "created_utc": analysis._utc_now_z(),
        "descriptor_h5": str(desc_h5),
        "outdir": str(outdir),
        "control_sweeps": sweeps.to_json(),
        "pca": pca_info,
        "figures": {
            "heatmap_explained_frac_lines": str(outdir / "heatmap_explained_frac_lines.png"),
            "potts_control_lines_corr2d_avg_mosaic_00_01_11_12": str(
                outdir / "potts_control_lines_corr2d_avg_mosaic_00_01_11_12.png"
            ),
        },
    }
    (outdir / "potts_publication_figures_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved unified publication figures to {outdir}")


if __name__ == "__main__":
    main()