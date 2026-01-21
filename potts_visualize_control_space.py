#!/usr/bin/env python3
"""
Sanity-check visualization for Potts (q=3) controls + descriptor snapshots.

Runs 9 simulations on a 3x3 grid over:
    f0 in [f0_min, f0_max]
    T  in [T_min,  T_max]

Visualizations:
    - One figure that tiles the 3x3 grid (rows=f0, cols=T) of final spins.
    - (Optional, default on) Descriptor figures for `radial1d` and `corr2d` using
        100 repeats per grid point (matches potts_gen default n_repeats) to show
        mean and standard deviation across repeats.

Uses fixed seed:
    - Deterministic RNG for init and dynamics so the grid is reproducible

Saves to: potts_figures/ (default).
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from potts_gen import (
    Config,
    create_initial_states,
    potts_step,
)
from injectivity_analysis_helpers import potts_build_Y_from_spins


@torch.no_grad()
def tile_3x3(block: np.ndarray, pad: int = 2) -> np.ndarray:
    """
    block: (3,3,H,W)
    returns: (3H+2pad, 3W+2pad)
    """
    H, W = block.shape[2], block.shape[3]
    canvas = np.zeros((3 * H + 2 * pad, 3 * W + 2 * pad), dtype=block.dtype)
    for i in range(3):
        for j in range(3):
            y0 = i * (H + pad)
            x0 = j * (W + pad)
            canvas[y0:y0 + H, x0:x0 + W] = block[i, j]
    return canvas


@torch.no_grad()
def simulate_potts_with_progress(
    spins: torch.Tensor,
    temperatures: torch.Tensor,  # (B,)
    steps: int,
    q: int,
    gen: torch.Generator,
    periodic: bool,
    remove_spurious: bool,
    progress_every: int,
):
    beta = 1.0 / temperatures
    pe = max(1, int(progress_every))
    for t in range(int(steps)):
        spins = potts_step(spins, beta, q=q, periodic=periodic, remove_spurious=remove_spurious)
        if (t + 1) % pe == 0 or (t + 1) == int(steps):
            print(f"step {t+1}/{steps}")
    return spins


@torch.no_grad()
def simulate_repeats_for_control(
    f0: float,
    T: float,
    repeats: int,
    grid_size: int,
    steps: int,
    q: int,
    device: torch.device,
    seed_repeat_base: int,
    periodic: bool,
    remove_spurious: bool,
):
    """Simulate `repeats` independent runs for a single (f0, T) control."""
    f0_all = torch.full((int(repeats),), float(f0), device=device, dtype=torch.float32)
    T_all = torch.full((int(repeats),), float(T), device=device, dtype=torch.float32)

    spins0 = create_initial_states(
        batch_size=int(repeats),
        grid_size=grid_size,
        fractions=f0_all,
        q=q,
        seeds=torch.randint(0, 2**32 - 1, (int(repeats),), device=device, dtype=torch.int64),
        device=device,
    )

    beta = 1.0 / T_all
    spins = spins0
    for _ in range(int(steps)):
        spins = potts_step(spins, beta, q=q, periodic=periodic, remove_spurious=remove_spurious)
    
    return spins[:, 0].detach().to("cpu", dtype=torch.int8).numpy()  # (R,H,W)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--grid-size", type=int, default=Config.grid_size)
    p.add_argument("--steps", type=int, default=Config.nsteps)

    p.add_argument("--f0-min", type=float, default=Config.fraction_range[0])
    p.add_argument("--f0-max", type=float, default=Config.fraction_range[1])
    p.add_argument("--T-min", type=float, default=Config.temp_range[0])
    p.add_argument("--T-max", type=float, default=Config.temp_range[1])

    p.add_argument("--q", type=int, default=Config.q)
    p.add_argument("--periodic", action="store_true")
    p.add_argument("--remove-spurious", action="store_true")

    p.add_argument("--seed", type=int, default=Config.seed_params)
    p.add_argument("--seed-repeat-base", type=int, default=Config.seed_params)
    p.add_argument("--outdir", type=str, default="potts_figures")
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--progress-every", type=int, default=50)

    p.add_argument("--descriptor-figs", action="store_true", dest="descriptor_figs")
    p.add_argument("--no-descriptor-figs", action="store_false", dest="descriptor_figs")
    p.set_defaults(descriptor_figs=True)
    p.add_argument("--desc-repeats", type=int, default=Config.n_repeats)
    p.add_argument("--nbins", type=int, default=64)
    p.add_argument("--corr2d-downsample", type=int, default=1)
    p.add_argument("--corr2d-weight-power", type=float, default=2.0)

    args = p.parse_args()

    device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    g = int(args.grid_size)
    q = int(args.q)
    ds = int(max(1, args.corr2d_downsample))
    if (g % ds) != 0:
        raise RuntimeError(f"corr2d_downsample={ds} must divide grid-size={g}")

    # 3-point grids (inclusive endpoints)
    f0s = torch.linspace(float(args.f0_min), float(args.f0_max), 3, device=device, dtype=torch.float32)
    Ts  = torch.linspace(float(args.T_min),  float(args.T_max),  3, device=device, dtype=torch.float32)

    # Combinations in f0-major order, then T cols
    f0_list, T_list = [], []
    for i in range(3):
        for j in range(3):
            f0_list.append(f0s[i])
            T_list.append(Ts[j])

    f0_all = torch.stack(f0_list)  # (9,)
    T_all = torch.stack(T_list)    # (9,)
    B = f0_all.shape[0]

    gen = torch.Generator(device=device).manual_seed(int(args.seed))

    spins0 = create_initial_states(
        batch_size=B,
        grid_size=g,
        fractions=f0_all,
        q=q,
        seeds=torch.randint(0, 2**32 - 1, (B,), device=device, dtype=torch.int64),
        device=device,
    )

    spinsF = simulate_potts_with_progress(
        spins0,
        temperatures=T_all,
        steps=int(args.steps),
        q=q,
        gen=gen,
        periodic=bool(args.periodic),
        remove_spurious=bool(args.remove_spurious),
        progress_every=int(args.progress_every),
    )

    # Arrange outputs: (f0, T, H, W)
    spins_cpu = spinsF[:, 0].detach().to("cpu", dtype=torch.int8).numpy()  # (B,H,W)
    imgs = spins_cpu.reshape(3, 3, g, g)

    mosaic = tile_3x3(imgs, pad=int(args.pad))

    cmap = ListedColormap([
        (0.05, 0.05, 0.05),
        (0.20, 0.60, 1.00),
        (1.00, 0.40, 0.10),
    ])

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0), constrained_layout=True)
    im = ax.imshow(mosaic, cmap=cmap, vmin=0, vmax=max(q - 1, 2), interpolation="nearest")
    ax.set_axis_off()

    f0_vals = f0s.detach().cpu().numpy()
    T_vals  = Ts.detach().cpu().numpy()

    # Column labels (T)
    for j in range(3):
        ax.text((j + 0.5) / 3.0, 1.02, f"T={T_vals[j]:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)

    # Row labels (f0)
    for i in range(3):
        ax.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"f0={f0_vals[i]:.2f}", transform=ax.transAxes, ha="right", va="center", fontsize=8)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"potts_control_grid_{ts}.png"

    fig.suptitle(
        f"Potts sanity grid (seed={args.seed}, steps={int(args.steps)}, grid={g}, q={q}, periodic={bool(args.periodic)})",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"[OK] saved: {outpath}")

    # Descriptor summaries (mean/std over repeats) per control point
    if not args.descriptor_figs:
        return

    desc_repeats = int(args.desc_repeats)
    pair_labels = ["00", "01", "02", "11", "12", "22"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

    radial_results = []
    corr_mean_maps = []
    corr_std_maps = []

    for idx, (f0_val, T_val) in enumerate(zip(f0_all, T_all)):
        spins_rep = simulate_repeats_for_control(
            f0=float(f0_val),
            T=float(T_val),
            repeats=desc_repeats,
            grid_size=g,
            steps=int(args.steps),
            q=q,
            device=device,
            seed_repeat_base=int(args.seed_repeat_base + idx),
            periodic=bool(args.periodic),
            remove_spurious=bool(args.remove_spurious),
        )  # (R,H,W)

        # radial1d descriptor stats
        rad = potts_build_Y_from_spins(
            spins_rep,
            kind="radial1d",
            nbins=int(args.nbins),
            device=str(device),
        )  # (R, 3 + 6*nbins)
        rad_feats = rad[:, 3:].reshape(desc_repeats, 6, int(args.nbins))
        radial_mean = rad_feats.mean(axis=0)
        radial_std = rad_feats.std(axis=0)
        radial_results.append((float(f0_val), float(T_val), radial_mean, radial_std))

        # corr2d descriptor stats (flattened surfaces)
        corr = potts_build_Y_from_spins(
            spins_rep,
            kind="corr2d",
            nbins=int(args.nbins),
            corr2d_downsample=ds,
            corr2d_weight_power=float(args.corr2d_weight_power),
            device=str(device),
        )
        H2 = g // ds
        W2 = g // ds
        corr_feats = corr[:, 3:].reshape(desc_repeats, 6, H2, W2)
        corr_mean = corr_feats.mean(axis=0)
        corr_std = corr_feats.std(axis=0)
        corr_mean_maps.append(corr_mean)
        corr_std_maps.append(corr_std)

    # Radial1d figure: 3x3 grid of profiles with std bands
    r_axis = np.linspace(0.0, g / 2.0, int(args.nbins), dtype=np.float32)
    fig_rad, axes_rad = plt.subplots(3, 3, figsize=(13.5, 13.5), sharex=True, sharey=True, constrained_layout=True)
    for idx, (f0_val, T_val, mu, sd) in enumerate(radial_results):
        i = idx // 3
        j = idx % 3
        ax = axes_rad[i, j]
        for k, (lab, col) in enumerate(zip(pair_labels, colors)):
            ax.plot(r_axis, mu[k], color=col, lw=1.0, label=lab)
            ax.fill_between(r_axis, mu[k] - sd[k], mu[k] + sd[k], color=col, alpha=0.2, linewidth=0)
        ax.set_title(f"f0={f0_val:.3f}, T={T_val:.3f}", fontsize=9)
        ax.grid(True, alpha=0.2)
    axes_rad[-1, 1].set_xlabel("radius (pixels)", fontsize=10)
    axes_rad[1, 0].set_ylabel("corr amplitude", fontsize=10)
    fig_rad.suptitle(
        f"Potts radial1d descriptor (mean±std over {desc_repeats} repeats, seed={args.seed_repeat_base})",
        fontsize=12,
    )
    # Single legend outside plot
    handles, labels = axes_rad[0, 0].get_legend_handles_labels()
    fig_rad.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    out_rad = outdir / f"potts_descriptor_radial1d_{ts}.png"
    fig_rad.savefig(out_rad, dpi=200)
    plt.close(fig_rad)

    # Corr2d figure: aggregate mean/std across the 6 pair channels (RMS std across channels)
    mean_maps = np.stack([m.mean(axis=0) for m in corr_mean_maps], axis=0).reshape(3, 3, H2, W2)
    std_maps = np.stack([np.sqrt((s * s).mean(axis=0)) for s in corr_std_maps], axis=0).reshape(3, 3, H2, W2)

    mosaic_mean = tile_3x3(mean_maps, pad=int(args.pad))
    mosaic_std = tile_3x3(std_maps, pad=int(args.pad))

    vmax_mean = float(np.max(np.abs(mosaic_mean))) if np.isfinite(mosaic_mean).any() else 1.0
    vmax_std = float(np.max(mosaic_std)) if np.isfinite(mosaic_std).any() else 1.0

    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    im0 = axes_corr[0].imshow(mosaic_mean, cmap="coolwarm", vmin=-vmax_mean, vmax=vmax_mean, interpolation="nearest")
    axes_corr[0].set_title("corr2d mean (channel-avg)")
    axes_corr[0].set_axis_off()
    fig_corr.colorbar(im0, ax=axes_corr[0], fraction=0.046, pad=0.02)

    im1 = axes_corr[1].imshow(mosaic_std, cmap="magma", vmin=0.0, vmax=vmax_std, interpolation="nearest")
    axes_corr[1].set_title("corr2d std (RMS over channels)")
    axes_corr[1].set_axis_off()
    fig_corr.colorbar(im1, ax=axes_corr[1], fraction=0.046, pad=0.02)

    fig_corr.suptitle(
        f"Potts corr2d descriptor (mean/std over {desc_repeats} repeats, seed={args.seed_repeat_base})",
        fontsize=12,
    )
    out_corr = outdir / f"potts_descriptor_corr2d_{ts}.png"
    fig_corr.savefig(out_corr, dpi=200)
    plt.close(fig_corr)

    print(f"[OK] saved descriptors: {out_rad}, {out_corr}")


if __name__ == "__main__":
    main()
