#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_time_evolution_slice_figure.py

Build a single figure that shows the non-monotonic explained-fraction behavior
along a fixed control-space slice for the Potts time-evolution analysis.

Default figure layout:
  - Top row: explained-fraction heatmaps for steps 100, 200, and 300
             with the target slice highlighted.
  - Bottom row: explained-fraction vs temperature along the selected
                fraction-initial slice.

This uses the existing per-step CSV outputs from potts_analyze_time_evolution.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import potts_analyze_time_evolution as time_analysis


@dataclass(frozen=True)
class Config:
    analysis_root: str = "potts_analysis_time_evolution/time_evolution"
    outdir: str = "potts_analysis_time_evolution/time_evolution/figs"
    raw_h5: str = "potts_data_time_evolution/time_evolution/potts_sims_q3_128x128_steps300.h5"
    steps: Tuple[int, ...] = (10, 50, 100, 200, 300)
    target_frac: float = 0.70
    temp_lo: float = 0.70
    temp_hi: float = 1.15
    micro_step: int = 300
    micro_target_temps: Tuple[float, ...] = (
        0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.14
    )
    micro_labels: Tuple[str, ...] = (
        "Low-T", "T=0.75", "T=0.80", "T=0.85", "T=0.90",
        "T=0.95", "T=1.00", "Peak", "T=1.10", "Collapse",
    )
    micro_repeat_index: int = 0
    micro_frac_tol: float = 0.02
    hm_bins_temp: int = 60
    hm_bins_frac: int = 60
    hm_sigma_px: float = 1.0
    hm_clip: Tuple[float, float] = (1.0, 99.0)
    dpi: int = 300


def _parse_csv_floats(raw: str) -> Tuple[float, ...]:
    vals = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return tuple(vals)


def _parse_csv_strings(raw: str) -> Tuple[str, ...]:
    vals = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one label value")
    return tuple(vals)


def load_step_csv(csv_path: Path) -> np.ndarray:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No data found in {csv_path}")
    return data


def build_heatmap(
    data: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        img_s, tx, fx = time_analysis.heatmap_binned_tf_ax(
            ax=ax,
            temp=np.asarray(data["temperature"], dtype=np.float64),
            frac=np.asarray(data["fraction_initial"], dtype=np.float64),
            Z=np.asarray(data["explained_frac"], dtype=np.float64),
            title="",
            bins_t=int(cfg.hm_bins_temp),
            bins_f=int(cfg.hm_bins_frac),
            sigma_px=float(cfg.hm_sigma_px),
            clip=cfg.hm_clip,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            xlabel="T",
            ylabel="f0",
        )
    finally:
        plt.close(fig)
    return img_s, tx, fx


def extract_slice(
    img_s: np.ndarray,
    tx: np.ndarray,
    fx: np.ndarray,
    target_frac: float,
    temp_lo: float,
    temp_hi: float,
) -> Dict[str, np.ndarray | float | int]:
    temp_centers = 0.5 * (tx[:-1] + tx[1:])
    frac_centers = 0.5 * (fx[:-1] + fx[1:])

    frac_idx = int(np.argmin(np.abs(frac_centers - float(target_frac))))
    temp_mask = (temp_centers >= float(temp_lo)) & (temp_centers <= float(temp_hi))
    temps = temp_centers[temp_mask]
    values = img_s[:, frac_idx][temp_mask]

    if temps.size == 0:
        raise ValueError("Selected temperature range does not intersect the binned heatmap")
    if not np.any(np.isfinite(values)):
        raise ValueError("Selected slice contains no finite values")

    peak_local = int(np.nanargmax(values))
    return {
        "temp_centers": temp_centers,
        "frac_centers": frac_centers,
        "slice_t": temps,
        "slice_vals": values,
        "frac_idx": frac_idx,
        "frac_value": float(frac_centers[frac_idx]),
        "peak_idx": peak_local,
        "peak_t": float(temps[peak_local]),
        "peak_value": float(values[peak_local]),
        "start_value": float(values[0]),
        "end_value": float(values[-1]),
    }


def add_slice_guides(ax: plt.Axes, cfg: Config, frac_value: float) -> None:
    rect = patches.Rectangle(
        (float(cfg.temp_lo), float(frac_value) - 0.012),
        float(cfg.temp_hi - cfg.temp_lo),
        0.024,
        fill=False,
        edgecolor="white",
        linewidth=3.0,
        zorder=6,
    )
    rect2 = patches.Rectangle(
        (float(cfg.temp_lo), float(frac_value) - 0.012),
        float(cfg.temp_hi - cfg.temp_lo),
        0.024,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        zorder=7,
    )
    ax.add_patch(rect)
    ax.add_patch(rect2)
    ax.text(
        float(cfg.temp_lo) + 0.01,
        float(frac_value) + 0.032,
        rf"slice: $f_0 \approx {frac_value:.2f}$",
        color="white",
        fontsize=9,
        fontweight="bold",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, edgecolor="none"),
    )


def _select_param_index(
    temps: np.ndarray,
    fracs: np.ndarray,
    target_temp: float,
    target_frac: float,
    frac_tol: float,
) -> int:
    frac_delta = np.abs(fracs - float(target_frac))
    mask = frac_delta <= float(frac_tol)
    if np.any(mask):
        candidates = np.where(mask)[0]
        best_local = int(np.argmin(np.abs(temps[candidates] - float(target_temp))))
        return int(candidates[best_local])

    temp_scale = max(float(np.max(temps) - np.min(temps)), 1e-8)
    frac_scale = max(float(frac_tol), 1e-8)
    dist = ((temps - float(target_temp)) / temp_scale) ** 2 + ((fracs - float(target_frac)) / frac_scale) ** 2
    return int(np.argmin(dist))


def load_microstructure_payloads(raw_h5: Path, cfg: Config) -> List[Dict[str, object]]:
    payloads: List[Dict[str, object]] = []
    with h5py.File(str(raw_h5), "r") as f:
        temps = np.array(f["parameters/temperature"], dtype=np.float64)
        fracs = np.array(f["parameters/fraction_initial"], dtype=np.float64)
        reference_dataset = f[f"states/final_spins_step{int(cfg.micro_step)}"]
        q = int(f.attrs.get("q", 3))
        repeat_idx = min(int(cfg.micro_repeat_index), int(reference_dataset.shape[1]) - 1)

        for label, target_temp in zip(cfg.micro_labels, cfg.micro_target_temps):
            idx = _select_param_index(
                temps=temps,
                fracs=fracs,
                target_temp=float(target_temp),
                target_frac=float(cfg.target_frac),
                frac_tol=float(cfg.micro_frac_tol),
            )
            images_by_step: Dict[int, np.ndarray] = {}
            for step in cfg.steps:
                spins = np.array(f[f"states/final_spins_step{int(step)}"][idx, repeat_idx], dtype=np.int16)
                images_by_step[int(step)] = (
                    spins.astype(np.float32) / float(max(q - 1, 1))
                ).clip(0.0, 1.0)
            payloads.append(
                {
                    "label": str(label),
                    "target_temp": float(target_temp),
                    "param_index": int(idx),
                    "repeat_index": int(repeat_idx),
                    "actual_temp": float(temps[idx]),
                    "actual_frac": float(fracs[idx]),
                    "images_by_step": images_by_step,
                }
            )
    return payloads


def save_figure(
    outbase: Path,
    step_payloads: List[Dict[str, object]],
    micro_payloads: List[Dict[str, object]],
    cfg: Config,
) -> None:
    n_steps = len(step_payloads)
    micro_height_ratio = 1.45 * (n_steps / 3)
    fig = plt.figure(figsize=(4.5 * n_steps + 1.0, 14.0 + micro_height_ratio), dpi=cfg.dpi)
    outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.0, 1.0, micro_height_ratio],
        hspace=0.32,
    )

    top = gridspec.GridSpecFromSubplotSpec(
        1,
        n_steps + 1,
        subplot_spec=outer[0],
        width_ratios=[1] * n_steps + [0.05],
        wspace=0.22,
    )
    micro_grid = gridspec.GridSpecFromSubplotSpec(
        n_steps,
        len(micro_payloads),
        subplot_spec=outer[2],
        hspace=0.08,
        wspace=0.08,
    )

    heat_axes = [fig.add_subplot(top[0, idx]) for idx in range(n_steps)]
    cax = fig.add_subplot(top[0, n_steps])
    line_ax = fig.add_subplot(outer[1])
    micro_axes = [
        [fig.add_subplot(micro_grid[row_idx, col_idx]) for col_idx in range(len(micro_payloads))]
        for row_idx in range(n_steps)
    ]

    colors = {
        10:  "#8E44AD",
        50:  "#27AE60",
        100: "#355C7D",
        200: "#F08A24",
        300: "#C0392B",
    }

    last_im = None
    for ax, payload in zip(heat_axes, step_payloads):
        step = int(payload["step"])
        img_s = np.asarray(payload["img_s"])
        tx = np.asarray(payload["tx"])
        fx = np.asarray(payload["fx"])
        frac_value = float(payload["slice"]["frac_value"])

        last_im = ax.imshow(
            img_s.T,
            origin="lower",
            extent=[tx[0], tx[-1], fx[0], fx[-1]],
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        add_slice_guides(ax, cfg, frac_value)
        ax.set_title(f"Explained Fraction (step {step})", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("T")
        ax.set_ylabel(r"$f_0$")
        ax.set_xlim(float(np.min(tx)), float(np.max(tx)))
        ax.set_ylim(float(np.min(fx)), float(np.max(fx)))

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Explained Fraction")

    for payload in step_payloads:
        step = int(payload["step"])
        slice_info = payload["slice"]
        temps = np.asarray(slice_info["slice_t"])
        values = np.asarray(slice_info["slice_vals"])
        color = colors.get(step, None)
        lw = 3.0 if step == 300 else 2.2
        zorder = 5 if step == 300 else 3
        label = f"step {step}"
        line_ax.plot(temps, values, color=color, linewidth=lw, label=label, zorder=zorder)

    slice_300 = next(payload["slice"] for payload in step_payloads if int(payload["step"]) == 300)
    peak_t = float(slice_300["peak_t"])
    peak_value = float(slice_300["peak_value"])
    end_t = float(np.asarray(slice_300["slice_t"])[-1])
    end_value = float(slice_300["end_value"])

    line_ax.scatter([peak_t], [peak_value], color=colors[300], s=55, zorder=6)
    line_ax.scatter([end_t], [end_value], color=colors[300], s=55, facecolors="white", linewidths=1.8, zorder=6)
    line_ax.annotate(
        f"peak\n({peak_t:.2f}, {peak_value:.2f})",
        xy=(peak_t, peak_value),
        xytext=(peak_t - 0.12, min(0.98, peak_value + 0.18)),
        arrowprops=dict(arrowstyle="->", color=colors[300], lw=1.5),
        fontsize=10,
        color=colors[300],
        ha="right",
    )
    line_ax.annotate(
        f"drop by {end_t:.2f}\n({end_value:.02f})",
        xy=(end_t, end_value),
        xytext=(end_t - 0.10, 0.16),
        arrowprops=dict(arrowstyle="->", color=colors[300], lw=1.5),
        fontsize=10,
        color=colors[300],
        ha="right",
    )

    curve_t = np.asarray(slice_300["slice_t"])
    curve_vals = np.asarray(slice_300["slice_vals"])
    for idx, micro in enumerate(micro_payloads):
        actual_temp = float(micro["actual_temp"])
        curve_idx = int(np.argmin(np.abs(curve_t - actual_temp)))
        curve_val = float(curve_vals[curve_idx])
        line_ax.scatter(
            [curve_t[curve_idx]],
            [curve_val],
            s=72,
            color=colors[300],
            edgecolors="white",
            linewidths=1.4,
            zorder=7,
        )
        line_ax.text(
            curve_t[curve_idx],
            min(0.99, curve_val + 0.03),
            chr(ord("A") + idx),
            color=colors[300],
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    line_ax.set_xlim(float(cfg.temp_lo), float(cfg.temp_hi))
    line_ax.set_ylim(-0.02, 1.02)
    line_ax.set_xlabel("T", fontsize=12)
    line_ax.set_ylabel(r"Explained Fraction along $f_0 \approx 0.7$", fontsize=12)
    line_ax.set_title(
        r"Non-monotonic injectivity along the $f_0 \approx 0.7$ slice",
        fontsize=13,
        fontweight="bold",
    )
    line_ax.grid(True, alpha=0.22, linewidth=0.7)
    line_ax.legend(frameon=False, loc="upper left")
    line_ax.text(
        0.015,
        0.03,
        r"Step 300 rises sharply near $T\approx0.8$, peaks near $T\approx1.05$," + "\n"
        + r"then collapses again by $T\gtrsim1.10$.",
        transform=line_ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#BBBBBB"),
    )

    for row_idx, step in enumerate(cfg.steps):
        for col_idx, micro in enumerate(micro_payloads):
            ax = micro_axes[row_idx][col_idx]
            ax.imshow(
                np.asarray(micro["images_by_step"][int(step)]),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_axis_off()
            if row_idx == 0:
                panel = chr(ord("A") + col_idx)
                ax.set_title(
                    f"{panel}\n$T={float(micro['actual_temp']):.2f}$",
                    fontsize=7.5,
                    pad=3,
                )
            if col_idx == 0:
                ax.text(
                    -0.16,
                    0.5,
                    f"step {int(step)}",
                    transform=ax.transAxes,
                    fontsize=11.5,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                    ha="center",
                )

    if micro_axes and micro_axes[0]:
        micro_axes[0][0].text(
            -0.34,
            0.5,
            "microstructures",
            transform=micro_axes[0][0].transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )
        micro_axes[0][len(micro_payloads) // 2].text(
            0.5,
            1.18,
            rf"Microstructure evolution at the same A/B/C control points along $f_0 \approx {cfg.target_frac:.1f}$",
            transform=micro_axes[0][len(micro_payloads) // 2].transAxes,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    fig.suptitle(
        "Potts time evolution: injectivity can increase and then decrease again",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outbase) + ".png", bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot a single fixed-f0 slice through Potts time-evolution heatmaps")
    ap.add_argument("--analysis_root", type=str, default=Config.analysis_root)
    ap.add_argument("--outdir", type=str, default=Config.outdir)
    ap.add_argument("--raw_h5", type=str, default=Config.raw_h5)
    ap.add_argument("--target_frac", type=float, default=Config.target_frac)
    ap.add_argument("--temp_lo", type=float, default=Config.temp_lo)
    ap.add_argument("--temp_hi", type=float, default=Config.temp_hi)
    ap.add_argument("--micro_step", type=int, default=Config.micro_step)
    ap.add_argument("--micro_target_temps", type=str, default=",".join(f"{v:.2f}" for v in Config.micro_target_temps))
    ap.add_argument("--micro_labels", type=str, default=",".join(Config.micro_labels))
    ap.add_argument("--micro_repeat_index", type=int, default=Config.micro_repeat_index)
    ap.add_argument("--micro_frac_tol", type=float, default=Config.micro_frac_tol)
    ap.add_argument("--hm_bins_temp", type=int, default=Config.hm_bins_temp)
    ap.add_argument("--hm_bins_frac", type=int, default=Config.hm_bins_frac)
    ap.add_argument("--hm_sigma_px", type=float, default=Config.hm_sigma_px)
    ap.add_argument("--hm_clip_lo", type=float, default=Config.hm_clip[0])
    ap.add_argument("--hm_clip_hi", type=float, default=Config.hm_clip[1])
    ap.add_argument("--dpi", type=int, default=Config.dpi)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        analysis_root=str(args.analysis_root),
        outdir=str(args.outdir),
        raw_h5=str(args.raw_h5),
        target_frac=float(args.target_frac),
        temp_lo=float(args.temp_lo),
        temp_hi=float(args.temp_hi),
        micro_step=int(args.micro_step),
        micro_target_temps=_parse_csv_floats(str(args.micro_target_temps)),
        micro_labels=_parse_csv_strings(str(args.micro_labels)),
        micro_repeat_index=int(args.micro_repeat_index),
        micro_frac_tol=float(args.micro_frac_tol),
        hm_bins_temp=int(args.hm_bins_temp),
        hm_bins_frac=int(args.hm_bins_frac),
        hm_sigma_px=float(args.hm_sigma_px),
        hm_clip=(float(args.hm_clip_lo), float(args.hm_clip_hi)),
        dpi=int(args.dpi),
    )

    if len(cfg.micro_target_temps) != len(cfg.micro_labels):
        raise ValueError("micro_target_temps and micro_labels must have the same length")

    analysis_root = Path(cfg.analysis_root).expanduser().resolve()
    outdir = Path(cfg.outdir).expanduser().resolve()
    raw_h5 = Path(cfg.raw_h5).expanduser().resolve()
    outbase = outdir / "heatmap_explained_frac_slice_f0_070_time_evolution"

    payloads: List[Dict[str, object]] = []
    for step in cfg.steps:
        csv_path = analysis_root / f"step_{step}" / "potts_local_explainedcov_injectivity.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Required CSV not found: {csv_path}")

        data = load_step_csv(csv_path)
        img_s, tx, fx = build_heatmap(data, cfg)
        slice_info = extract_slice(
            img_s=img_s,
            tx=tx,
            fx=fx,
            target_frac=float(cfg.target_frac),
            temp_lo=float(cfg.temp_lo),
            temp_hi=float(cfg.temp_hi),
        )
        payloads.append(
            {
                "step": int(step),
                "csv": str(csv_path),
                "img_s": img_s,
                "tx": tx,
                "fx": fx,
                "slice": slice_info,
            }
        )

    if not raw_h5.exists():
        raise FileNotFoundError(f"Required raw H5 not found: {raw_h5}")

    micro_payloads = load_microstructure_payloads(raw_h5=raw_h5, cfg=cfg)

    save_figure(outbase=outbase, step_payloads=payloads, micro_payloads=micro_payloads, cfg=cfg)

    metadata = {
        "analysis_root": str(analysis_root),
        "raw_h5": str(raw_h5),
        "outbase": str(outbase),
        "config": asdict(cfg),
        "steps": {
            str(payload["step"]): {
                "csv": payload["csv"],
                "actual_frac_slice": float(payload["slice"]["frac_value"]),
                "peak_t": float(payload["slice"]["peak_t"]),
                "peak_value": float(payload["slice"]["peak_value"]),
                "start_value": float(payload["slice"]["start_value"]),
                "end_value": float(payload["slice"]["end_value"]),
            }
            for payload in payloads
        },
        "microstructures": [
            {
                "label": micro["label"],
                "param_index": int(micro["param_index"]),
                "repeat_index": int(micro["repeat_index"]),
                "target_temp": float(micro["target_temp"]),
                "actual_temp": float(micro["actual_temp"]),
                "actual_frac": float(micro["actual_frac"]),
                "steps_shown": [int(step) for step in cfg.steps],
            }
            for micro in micro_payloads
        ],
    }
    (outdir / "heatmap_explained_frac_slice_f0_070_time_evolution.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Saved {outbase}.png and {outbase}.pdf")


if __name__ == "__main__":
    main()