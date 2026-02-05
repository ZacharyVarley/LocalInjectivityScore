#!/usr/bin/env python3
"""
2D Correlation surface descriptor visualization for CH (alpha, beta, c0) controls.

Runs 27 simulations on a 3x3x3 grid over:
  alpha in [alpha_min, alpha_max]
  beta  in [beta_min,  beta_max]
  c0    in [c0_min,    c0_max]

Visualization:
  - Top row: microstructures (single deterministic run)
  - Middle row: autocorrelation surfaces (single run)
  - Bottom row: averaged autocorrelation surfaces from 100 repeated runs per configuration
  - Each row has 3 subplots (one per c0 slice)
  - Each subplot is a 3x3 grid of heatmaps (rows=alpha, cols=beta)
  - Spatial-lag autocorrelation computed via FFT, publication-ready styling

Uses fixed seed for reproducibility:
  - Initial deterministic noise field for single-run microstructure
  - 100 repeats per configuration with different random noise seeds (seed + repeat_idx + 1)
  - c initial = c0 + noise_amp * noise_field, clamped to [0,1]

Saves to: ch_ab_figures/ (default) with consistent filenames (no timestamps).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def prepare_spectral(grid: int, L: float, device: torch.device):
    dx = float(L) / float(grid)
    kx = torch.fft.fftfreq(grid, d=dx, device=device) * 2.0 * math.pi
    Kx, Ky = torch.meshgrid(kx, kx, indexing="ij")
    K2 = (Kx * Kx + Ky * Ky).to(torch.float32)
    kcut = kx.abs().max() * 2.0 / 3.0
    dealias = ((Kx.abs() < kcut) & (Ky.abs() < kcut)).to(torch.complex64)
    return K2, dealias


@torch.no_grad()
def dfdc(c: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return 2.0 * W * (c * (1.0 - c) ** 2 - (1.0 - c) * c**2)


@torch.no_grad()
def ch_step(
    c: torch.Tensor,                 # (B,g,g)
    c_hat: torch.Tensor,             # (B,g,g) complex
    K2: torch.Tensor,                # (g,g)
    dealias: torch.Tensor,           # (g,g) complex
    W: torch.Tensor,                 # (B,1,1)
    kappa: torch.Tensor,             # (B,1,1)
    M: torch.Tensor,                 # (B,1,1)
    dt: torch.Tensor,                # (B,1,1)
):
    B, g, _ = c.shape
    K2e = K2.view(1, g, g)
    dfdc_hat = torch.fft.fftn(dfdc(c, W), dim=(-2, -1)) * dealias.view(1, g, g)

    num = c_hat - dt * K2e * M * dfdc_hat
    den = 1.0 + dt * M * kappa * (K2e**2)
    c_hat = num / den

    c = torch.fft.ifftn(c_hat, dim=(-2, -1)).real.clamp_(0.0, 1.0)
    return c, c_hat


@torch.no_grad()
def autocorr2d_fft(x: torch.Tensor) -> torch.Tensor:
    """
    Spatial-lag autocorrelation surface g(dx,dy), computed via FFT:
      z = x - mean(x)
      corr = ifft( fft(z) * conj(fft(z)) ) / (H*W)
    Returned in spatial lag domain after fftshift.
    x: (B,H,W) float32
    returns: (B,H,W) float32
    """
    B, H, W = x.shape
    z = x - x.mean(dim=(-2, -1), keepdim=True)
    f = torch.fft.rfft2(z)
    p = f * torch.conj(f)
    corr = torch.fft.irfft2(p, s=(H, W)) / float(H * W)
    corr = torch.fft.fftshift(corr, dim=(-2, -1))
    return corr.to(torch.float32)


def tile_3x3(block: np.ndarray, pad: int = 4) -> np.ndarray:
    """
    block: (3,3,H,W)
    returns: (3H+2pad, 3W+2pad)
    """
    H, W = block.shape[2], block.shape[3]
    pad_col = np.ones((H, pad), dtype=np.float32)
    pad_row = np.ones((pad, 3 * W + 2 * pad), dtype=np.float32)

    rows = []
    for i in range(3):
        row = np.concatenate(
            [block[i, 0], pad_col, block[i, 1], pad_col, block[i, 2]],
            axis=1,
        )
        rows.append(row)

    mosaic = np.concatenate([rows[0], pad_row, rows[1], pad_row, rows[2]], axis=0)
    return mosaic


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--L", type=float, default=128.0)
    p.add_argument("--steps", type=int, default=200)

    p.add_argument("--alpha-min", type=float, default=0.7)
    p.add_argument("--alpha-max", type=float, default=1.3)
    p.add_argument("--beta-min", type=float, default=0.7)
    p.add_argument("--beta-max", type=float, default=1.3)
    p.add_argument("--c0-min", type=float, default=0.4)
    p.add_argument("--c0-max", type=float, default=0.5)

    p.add_argument("--W-base", type=float, default=1.0)
    p.add_argument("--kappa-base", type=float, default=1.5)
    p.add_argument("--M-base", type=float, default=1.0)
    p.add_argument("--dt-base", type=float, default=0.5)

    p.add_argument("--noise-amp", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2024)

    p.add_argument("--outdir", type=str, default="ch_ab_figures")
    p.add_argument("--corr-downsample", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--progress-every", type=int, default=100)

    args = p.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    g = int(args.grid)
    K2, dealias = prepare_spectral(g, float(args.L), device)

    # 3-point grids (inclusive endpoints)
    alphas = torch.linspace(float(args.alpha_min), float(args.alpha_max), 3, device=device, dtype=torch.float32)
    betas  = torch.linspace(float(args.beta_min),  float(args.beta_max),  3, device=device, dtype=torch.float32)
    c0s    = torch.linspace(float(args.c0_min),    float(args.c0_max),    3, device=device, dtype=torch.float32)

    # Combinations in c0-major order, then alpha rows, beta cols:
    # idx = k*9 + i*3 + j
    alpha_list, beta_list, c0_list = [], [], []
    for k in range(3):
        for i in range(3):
            for j in range(3):
                c0_list.append(c0s[k])
                alpha_list.append(alphas[i])
                beta_list.append(betas[j])

    alpha_all = torch.stack(alpha_list)  # (27,)
    beta_all  = torch.stack(beta_list)   # (27,)
    c0_all    = torch.stack(c0_list)     # (27,)
    B = alpha_all.shape[0]

    # Parameters per sim
    kappa = (float(args.kappa_base) * (alpha_all**2) * (beta_all**2)).view(B, 1, 1)
    M     = (float(args.M_base)     * (alpha_all**2) / (beta_all**2).clamp_min(1e-12)).view(B, 1, 1)
    W     = torch.full((B, 1, 1), float(args.W_base), device=device, dtype=torch.float32)
    dt    = torch.full((B, 1, 1), float(args.dt_base), device=device, dtype=torch.float32)

    # Fixed noise field; c initial differs via c0
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    noise_field = torch.randn((1, g, g), device=device, generator=gen, dtype=torch.float32)

    c0b = c0_all.view(B, 1, 1)
    c = (c0b + noise_field.expand(B, g, g) * float(args.noise_amp)).clamp_(0.0, 1.0)
    c_hat = torch.fft.fftn(c, dim=(-2, -1))

    steps = int(args.steps)
    pe = max(1, int(args.progress_every))
    for t in range(steps):
        c, c_hat = ch_step(c, c_hat, K2, dealias, W, kappa, M, dt)
        if (t + 1) % pe == 0 or (t + 1) == steps:
            print(f"step {t+1}/{steps}")

    # Get final microstructures
    c_cpu = c.detach().cpu().numpy().astype(np.float32)
    imgs = c_cpu.reshape(3, 3, 3, g, g)

    # Compute 2D autocorrelation for each final field
    print("[CORR2D] computing spatial-lag autocorrelation for all 27 simulations...")
    corr2d_all = autocorr2d_fft(c)  # (27, g, g)

    # Downsample if requested
    ds = int(max(1, args.corr_downsample))
    if ds > 1:
        corr2d_all = corr2d_all[:, ::ds, ::ds]

    # Arrange outputs: (c0, alpha, beta, H, W)
    corr2d_cpu = corr2d_all.detach().cpu().numpy().astype(np.float32)
    H_corr, W_corr = corr2d_cpu.shape[-2:]
    corr2d_reshaped = corr2d_cpu.reshape(3, 3, 3, H_corr, W_corr)

    # Normalize autocorrelation to [0, 1] for full intensity range display
    corr_min = float(corr2d_cpu.min())
    corr_max = float(corr2d_cpu.max())
    corr2d_normalized = (corr2d_cpu - corr_min) / (corr_max - corr_min)
    corr2d_reshaped_normalized = corr2d_normalized.reshape(3, 3, 3, H_corr, W_corr)
    
    print(f"[INFO] Autocorrelation range before normalization: min={corr_min:.6f}, max={corr_max:.6f}")
    
    # Global normalization for consistent colormap across all subplots (legacy)
    corr_vmin = float(np.percentile(corr2d_cpu, 1))
    corr_vmax = float(np.percentile(corr2d_cpu, 99))
    corr_abs = float(np.max(np.abs(corr2d_cpu)))
    corr_vmin_sym = -corr_abs
    corr_vmax_sym = corr_abs

    # Compute averaged autocorrelations from 100 repeated runs per configuration
    print("[CORR2D_AVG] running 100 repeated simulations per configuration for averaged autocorrelations...")
    n_repeats = 100
    
    # For each repeat, run all 27 configurations with different noise
    corr2d_all_repeats = []
    
    for repeat_idx in range(n_repeats):
        # Different random seed for each repeat
        gen_repeat = torch.Generator(device=device).manual_seed(int(args.seed) + repeat_idx + 1)
        noise_repeat = torch.randn((B, g, g), device=device, generator=gen_repeat, dtype=torch.float32)
        
        c0b = c0_all.view(B, 1, 1)
        c_repeat = (c0b + noise_repeat * float(args.noise_amp)).clamp_(0.0, 1.0)
        c_hat_repeat = torch.fft.fftn(c_repeat, dim=(-2, -1))
        
        for t in range(steps):
            c_repeat, c_hat_repeat = ch_step(c_repeat, c_hat_repeat, K2, dealias, W, kappa, M, dt)
        
        # Compute autocorrelation for this repeat
        corr2d_repeat = autocorr2d_fft(c_repeat)  # (27, g, g)
        
        # Downsample if requested
        if ds > 1:
            corr2d_repeat = corr2d_repeat[:, ::ds, ::ds]
        
        corr2d_all_repeats.append(corr2d_repeat)
        
        if (repeat_idx + 1) % 10 == 0 or (repeat_idx + 1) == n_repeats:
            print(f"  completed {repeat_idx + 1}/{n_repeats} repeats")
    
    # Stack all repeats and average: (100, 27, H_corr, W_corr) -> (27, H_corr, W_corr)
    corr2d_stacked = torch.stack(corr2d_all_repeats, dim=0)  # (100, 27, H_corr, W_corr)
    corr2d_avg = corr2d_stacked.mean(dim=0)  # (27, H_corr, W_corr)
    
    corr2d_avg_all = corr2d_avg.detach().cpu().numpy().astype(np.float32)
    corr2d_avg_reshaped = corr2d_avg_all.reshape(3, 3, 3, H_corr, W_corr)
    
    # Normalize averaged autocorrelations
    corr_avg_min = float(corr2d_avg_all.min())
    corr_avg_max = float(corr2d_avg_all.max())
    corr2d_avg_normalized = (corr2d_avg_all - corr_avg_min) / (corr_avg_max - corr_avg_min)
    corr2d_avg_reshaped_normalized = corr2d_avg_normalized.reshape(3, 3, 3, H_corr, W_corr)
    
    print(f"[OK] Averaged autocorrelation range: min={corr_avg_min:.6f}, max={corr_avg_max:.6f}")

    alpha_vals = alphas.detach().cpu().numpy()
    beta_vals  = betas.detach().cpu().numpy()
    c0_vals    = c0s.detach().cpu().numpy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- SAVE MICROSTRUCTURE FIGURE ---
    fig_micro, axes_micro = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    for k, ax in enumerate(axes_micro):
        block = imgs[k]  # (3,3,g,g)
        mosaic = tile_3x3(block, pad=2)
        ax.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(f"$c_0 = {c0_vals[k]:.3f}$", fontsize=16)

        # Column labels (beta)
        for j in range(3):
            ax.text((j + 0.5) / 3.0, -0.05, f"$\\beta={beta_vals[j]:.2f}$",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10)

        # Row labels (alpha)
        for i in range(3):
            ax.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"$\\alpha={alpha_vals[i]:.2f}$",
                    transform=ax.transAxes, ha="right", va="center", fontsize=10)

    outpath_micro = outdir / "microstructures_3x3x3.png"
    fig_micro.savefig(outpath_micro, dpi=300, bbox_inches="tight")
    plt.close(fig_micro)
    print(f"[OK] saved microstructures: {outpath_micro}")

    # --- SAVE CORR2D FIGURE ---
    fig_corr, axes_corr = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for k, ax in enumerate(axes_corr):
        # Create a mosaic of 3x3 correlation surfaces for this c0 slice
        block = corr2d_reshaped[k]  # (3,3,H_corr,W_corr)
        mosaic = tile_3x3(block, pad=2)

        im = ax.imshow(mosaic, cmap="RdYlBu_r", vmin=corr_vmin, vmax=corr_vmax, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"$c_0 = {c0_vals[k]:.3f}$", fontsize=16, pad=10)

        # Column labels (beta)
        for j in range(3):
            ax.text((j + 0.5) / 3.0, -0.05, f"$\\beta={beta_vals[j]:.2f}$",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10)

        # Row labels (alpha)
        for i in range(3):
            ax.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"$\\alpha={alpha_vals[i]:.2f}$",
                    transform=ax.transAxes, ha="right", va="center", fontsize=10)

    # Add shared colorbar
    fig_corr.colorbar(im, ax=axes_corr, label="Spatial-lag autocorrelation", fraction=0.046, pad=0.04)

    outpath_corr = outdir / "corr2d_descriptors_3x3x3.png"
    fig_corr.savefig(outpath_corr, dpi=300, bbox_inches="tight")
    plt.close(fig_corr)
    print(f"[OK] saved corr2d descriptors: {outpath_corr}")

    # --- SAVE STACKED COMBINED FIGURE (with optional averaged autocorrelations) ---
    n_rows = 3 if corr2d_avg_all is not None else 2
    fig_stacked, axes_stacked = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows), constrained_layout=False)
    fig_stacked.subplots_adjust(hspace=0.05, wspace=0.1, left=0.08, right=0.92, top=0.95, bottom=0.08)

    for k in range(3):
        # Top row: microstructures
        ax_micro = axes_stacked[0, k]
        block_micro = imgs[k]
        mosaic_micro = tile_3x3(block_micro, pad=4)
        ax_micro.imshow(mosaic_micro, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_micro.set_axis_off()
        if k == 0:
            ax_micro.text(-0.25, 0.5, "Microstructure", transform=ax_micro.transAxes, 
                         fontsize=18, fontweight="bold", rotation=90, va="center", ha="right")
        ax_micro.set_title(f"$c_0 = {c0_vals[k]:.3f}$", fontsize=16, pad=10)

        # Only add labels on leftmost column (k=0)
        if k == 0:
            # Row labels (alpha) - only on left
            for i in range(3):
                ax_micro.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"$\\alpha={alpha_vals[i]:.2f}$",
                        transform=ax_micro.transAxes, ha="right", va="center", fontsize=16, rotation=45)

        # Middle row: corr2d descriptors (single run)
        ax_corr = axes_stacked[1, k]
        block_corr = corr2d_reshaped_normalized[k]
        mosaic_corr = tile_3x3(block_corr, pad=4)
        ax_corr.imshow(mosaic_corr, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax_corr.set_axis_off()
        if k == 0:
            ax_corr.text(-0.25, 0.5, "Autocorrelation", transform=ax_corr.transAxes, 
                        fontsize=18, fontweight="bold", rotation=90, va="center", ha="right")

        # Row labels (alpha) - only on left
        if k == 0:
            for i in range(3):
                ax_corr.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"$\\alpha={alpha_vals[i]:.2f}$",
                        transform=ax_corr.transAxes, ha="right", va="center", fontsize=16, rotation=45)

        # Bottom row: averaged autocorrelations (if available)
        if corr2d_avg_all is not None:
            ax_avg = axes_stacked[2, k]
            block_avg = corr2d_avg_reshaped_normalized[k]
            mosaic_avg = tile_3x3(block_avg, pad=4)
            ax_avg.imshow(mosaic_avg, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            ax_avg.set_axis_off()
            if k == 0:
                ax_avg.text(-0.25, 0.5, "Averaged Autocorrelations", transform=ax_avg.transAxes, 
                            fontsize=18, fontweight="bold", rotation=90, va="center", ha="right")

            # Column labels (beta) - only on bottom row
            for j in range(3):
                ax_avg.text((j + 0.5) / 3.0, -0.05, f"$\\beta={beta_vals[j]:.2f}$",
                        transform=ax_avg.transAxes, ha="center", va="top", fontsize=16, rotation=45)

            # Row labels (alpha) - only on left
            if k == 0:
                for i in range(3):
                    ax_avg.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"$\\alpha={alpha_vals[i]:.2f}$",
                            transform=ax_avg.transAxes, ha="right", va="center", fontsize=16, rotation=45)

    outpath_stacked = outdir / "ch_control_space.png"
    fig_stacked.savefig(outpath_stacked, dpi=300, bbox_inches="tight")
    # also save pdf for this one
    outpath_stacked_pdf = outdir / "ch_control_space.pdf"
    fig_stacked.savefig(outpath_stacked_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig_stacked)
    print(f"[OK] saved stacked combined figure: {outpath_stacked}")
    print(f"[OK] saved stacked combined figure: {outpath_stacked_pdf}")


if __name__ == "__main__":
    main()
