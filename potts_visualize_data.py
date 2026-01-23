#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_visualize_data.py

Visualize Potts simulation data from potts_gen.py output.
Loads the most recently generated dataset and creates diagnostic figures:
  - Sample microstructures from different parameter points
  - Multiple repeats from same parameter to check uniqueness
  - Parameter space coverage
  - Phase fraction distributions

All figures are saved (no plt.show).
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _parse_run_dir_name(name: str) -> float | None:
    """Parse timestamp from directory name."""
    try:
        dt = _dt.datetime.strptime(name, "%Y%m%d_%H%M%SZ")
        return dt.replace(tzinfo=_dt.timezone.utc).timestamp()
    except Exception:
        return None


def find_latest_h5_under(data_root: Path) -> Path:
    """Find the most recently created .h5 file under data_root."""
    candidates: List[Tuple[float, Path]] = []
    if data_root.exists():
        for p in data_root.iterdir():
            if p.is_dir():
                ts = _parse_run_dir_name(p.name)
                if ts is None:
                    ts = p.stat().st_mtime
                for h in sorted(p.glob("*.h5")):
                    candidates.append((ts, h))
            elif p.is_file() and p.suffix.lower() == ".h5":
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(f"No .h5 found under {data_root}")
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def create_potts_colormap(q: int) -> ListedColormap:
    """Create colormap for q phases."""
    if q == 3:
        colors = [
            (0.05, 0.05, 0.05),  # Phase 0: dark
            (0.20, 0.60, 1.00),  # Phase 1: blue
            (1.00, 0.40, 0.10),  # Phase 2: orange
        ]
    else:
        # Generic colormap for other q values
        colors = plt.cm.tab10(np.linspace(0, 1, q))
    return ListedColormap(colors[:q])


def plot_parameter_space(temps: np.ndarray, fracs: np.ndarray, outpath: Path, dpi: int = 200):
    """Plot the parameter space coverage."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    ax.scatter(temps, fracs, s=3, alpha=0.5, c='blue')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Fraction Initial')
    ax.set_title(f'Parameter Space Coverage (N={len(temps)})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[potts_visualize_data] Saved: {outpath}")


def plot_sample_microstructures(
    spins: np.ndarray,
    temps: np.ndarray,
    fracs: np.ndarray,
    q: int,
    outpath: Path,
    n_samples: int = 9,
    dpi: int = 200,
):
    """Plot a grid of sample microstructures from different parameter points."""
    N = spins.shape[0]
    n_samples = min(n_samples, N)
    
    # Select evenly spaced indices
    indices = np.linspace(0, N - 1, n_samples, dtype=int)
    
    # Arrange in a square grid
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=dpi)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    cmap = create_potts_colormap(q)
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        # Show first repeat for each parameter
        img = spins[ax_idx, 0, :, :]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=q-1, interpolation='nearest')
        ax.set_title(f'T={temps[ax_idx]:.3f}, f0={fracs[ax_idx]:.3f}', fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Sample Microstructures (q={q})', fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[potts_visualize_data] Saved: {outpath}")


def plot_repeat_uniqueness(
    spins: np.ndarray,
    temps: np.ndarray,
    fracs: np.ndarray,
    q: int,
    outpath: Path,
    n_params: int = 4,
    n_repeats_show: int = 6,
    dpi: int = 200,
):
    """Plot multiple repeats from same parameters to verify uniqueness."""
    N, R = spins.shape[0], spins.shape[1]
    n_params = min(n_params, N)
    n_repeats_show = min(n_repeats_show, R)
    
    # Select parameter indices
    param_indices = np.linspace(0, N - 1, n_params, dtype=int)
    
    fig, axes = plt.subplots(n_params, n_repeats_show, 
                             figsize=(2.5 * n_repeats_show, 2.5 * n_params), dpi=dpi)
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    cmap = create_potts_colormap(q)
    
    for row, param_idx in enumerate(param_indices):
        for col in range(n_repeats_show):
            ax = axes[row, col]
            img = spins[param_idx, col, :, :]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=q-1, interpolation='nearest')
            
            if col == 0:
                ax.set_ylabel(f'T={temps[param_idx]:.2f}\nf0={fracs[param_idx]:.2f}', 
                             fontsize=8)
            if row == 0:
                ax.set_title(f'Repeat {col}', fontsize=9)
            ax.axis('off')
    
    fig.suptitle(f'Repeat Uniqueness Check (q={q}, {R} total repeats)', fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[potts_visualize_data] Saved: {outpath}")


def plot_phase_fractions(
    spins: np.ndarray,
    temps: np.ndarray,
    fracs_init: np.ndarray,
    q: int,
    outpath: Path,
    dpi: int = 200,
):
    """Plot final phase fractions vs. parameters."""
    N, R, H, W = spins.shape
    
    # Compute final phase fractions
    phase_fracs = np.zeros((N, q), dtype=np.float32)
    for i in range(N):
        # Average over all repeats
        all_spins = spins[i].reshape(-1)  # (R*H*W,)
        for phase in range(q):
            phase_fracs[i, phase] = (all_spins == phase).mean()
    
    # Create plots
    fig, axes = plt.subplots(1, q, figsize=(4*q, 4), dpi=dpi)
    if q == 1:
        axes = [axes]
    
    for phase in range(q):
        ax = axes[phase]
        sc = ax.scatter(temps, fracs_init, c=phase_fracs[:, phase], 
                       s=10, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Fraction Initial')
        ax.set_title(f'Final Fraction Phase {phase}')
        plt.colorbar(sc, ax=ax, label='Fraction')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Final Phase Fractions (q={q}, averaged over {R} repeats)', fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[potts_visualize_data] Saved: {outpath}")


def plot_temperature_comparison(
    spins: np.ndarray,
    temps: np.ndarray,
    fracs: np.ndarray,
    q: int,
    outpath: Path,
    fixed_frac: float = 0.5,
    n_temps: int = 6,
    dpi: int = 200,
):
    """Plot microstructures at fixed fraction, varying temperature."""
    # Find parameters closest to fixed_frac
    frac_diffs = np.abs(fracs - fixed_frac)
    candidates = np.where(frac_diffs < 0.1)[0]  # Within 0.1 of target
    
    if len(candidates) == 0:
        print(f"[potts_visualize_data] No parameters near f0={fixed_frac}, skipping temperature comparison")
        return
    
    # Sort by temperature
    sorted_idx = candidates[np.argsort(temps[candidates])]
    
    # Select evenly spaced temperatures
    n_temps = min(n_temps, len(sorted_idx))
    selected = sorted_idx[np.linspace(0, len(sorted_idx)-1, n_temps, dtype=int)]
    
    fig, axes = plt.subplots(1, n_temps, figsize=(2.5*n_temps, 3), dpi=dpi)
    if n_temps == 1:
        axes = [axes]
    
    cmap = create_potts_colormap(q)
    
    for idx, param_idx in enumerate(selected):
        ax = axes[idx]
        # Show first repeat
        img = spins[param_idx, 0, :, :]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=q-1, interpolation='nearest')
        ax.set_title(f'T={temps[param_idx]:.3f}\nf0={fracs[param_idx]:.3f}', fontsize=9)
        ax.axis('off')
    
    fig.suptitle(f'Temperature Variation at f0≈{fixed_frac} (q={q})', fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[potts_visualize_data] Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Potts simulation data from potts_gen output"
    )
    parser.add_argument(
        "--h5", type=str, default="",
        help="Input HDF5 file. If empty, finds latest under --data_dir"
    )
    parser.add_argument(
        "--data_dir", type=str, default="potts_data",
        help="Directory containing potts_gen output"
    )
    parser.add_argument(
        "--outdir", type=str, default="potts_figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI"
    )
    parser.add_argument(
        "--n_sample_structures", type=int, default=9,
        help="Number of sample microstructures to show"
    )
    parser.add_argument(
        "--n_repeat_params", type=int, default=4,
        help="Number of parameter points for repeat uniqueness check"
    )
    parser.add_argument(
        "--n_repeats_show", type=int, default=6,
        help="Number of repeats to show per parameter"
    )
    
    args = parser.parse_args()
    
    # Find input file
    if args.h5.strip():
        in_h5 = Path(args.h5).expanduser().resolve()
    else:
        data_root = Path(args.data_dir)
        in_h5 = find_latest_h5_under(data_root)
    
    print(f"[potts_visualize_data] Loading: {in_h5}")
    
    # Load data
    with h5py.File(in_h5, "r") as f:
        temps = np.array(f["parameters/temperature"])
        fracs = np.array(f["parameters/fraction_initial"])
        spins = np.array(f["states/final_spins"])
        q = int(f.attrs.get("q", 3))
    
    N, R, H, W = spins.shape
    print(f"[potts_visualize_data] Loaded: N={N} parameters, R={R} repeats, grid={H}x{W}, q={q}")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate figures
    print("[potts_visualize_data] Generating figures...")
    
    # 1. Parameter space coverage
    plot_parameter_space(
        temps, fracs,
        outdir / f"parameter_space_{timestamp}.png",
        dpi=args.dpi
    )
    
    # 2. Sample microstructures
    plot_sample_microstructures(
        spins, temps, fracs, q,
        outdir / f"sample_microstructures_{timestamp}.png",
        n_samples=args.n_sample_structures,
        dpi=args.dpi
    )
    
    # 3. Repeat uniqueness check
    plot_repeat_uniqueness(
        spins, temps, fracs, q,
        outdir / f"repeat_uniqueness_{timestamp}.png",
        n_params=args.n_repeat_params,
        n_repeats_show=args.n_repeats_show,
        dpi=args.dpi
    )
    
    # 4. Phase fractions
    plot_phase_fractions(
        spins, temps, fracs, q,
        outdir / f"phase_fractions_{timestamp}.png",
        dpi=args.dpi
    )
    
    # 5. Temperature comparison at fixed fraction
    plot_temperature_comparison(
        spins, temps, fracs, q,
        outdir / f"temperature_comparison_{timestamp}.png",
        fixed_frac=0.5,
        n_temps=6,
        dpi=args.dpi
    )
    
    print(f"[potts_visualize_data] Done! Figures saved to: {outdir}")


if __name__ == "__main__":
    main()
