#!/usr/bin/env python3
"""
Sanity-check visualization for CH (alpha, beta, c0) controls.

Runs 27 simulations on a 3x3x3 grid over:
  alpha in [alpha_min, alpha_max]
  beta  in [beta_min,  beta_max]
  c0    in [c0_min,    c0_max]

Visualization:
  - One figure with 3 subplots (one per c0 slice)
  - Each subplot is a 3x3 image grid (rows=alpha, cols=beta) of final microstructures

Uses fixed seed:
  - One deterministic noise field added to each simulation
  - c initial = c0 + noise_amp * noise_field, clamped

Saves to: ch_ab_controls_figures/ (default).
"""

import argparse
import math
from datetime import datetime
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


def tile_3x3(block: np.ndarray, pad: int = 2) -> np.ndarray:
    """
    block: (3,3,H,W)
    returns: (3H+2pad, 3W+2pad)
    """
    H, W = block.shape[2], block.shape[3]
    canvas = np.zeros((3 * H + 2 * pad, 3 * W + 2 * pad), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            y0 = i * (H + pad)
            x0 = j * (W + pad)
            canvas[y0:y0 + H, x0:x0 + W] = block[i, j]
    return canvas


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
    p.add_argument("--pad", type=int, default=2)
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

    # Arrange outputs: (c0, alpha, beta, g, g)
    c_cpu = c.detach().cpu().numpy().astype(np.float32)
    imgs = c_cpu.reshape(3, 3, 3, g, g)

    # Plot: 3 subplots, each a 3x3 tiled grid
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    alpha_vals = alphas.detach().cpu().numpy()
    beta_vals  = betas.detach().cpu().numpy()
    c0_vals    = c0s.detach().cpu().numpy()

    for k, ax in enumerate(axes):
        block = imgs[k]  # (3,3,g,g)
        mosaic = tile_3x3(block, pad=int(args.pad))
        ax.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_axis_off()
        # ax.set_title(f"c0 = {c0_vals[k]:.3f}", fontsize=10)
        # use latex instead
        ax.set_title(f"$c_0 = {c0_vals[k]:.3f}$", fontsize=16)

        # Column labels (beta)
        for j in range(3):
            ax.text((j + 0.5) / 3.0, -0.05, f"$\\beta={beta_vals[j]:.2f}$",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

        # Row labels (alpha)
        for i in range(3):
            ax.text(-0.02, 1.0 - (i + 0.5) / 3.0, f"$\\alpha={alpha_vals[i]:.2f}$",
                    transform=ax.transAxes, ha="right", va="center", fontsize=10)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"sanity_abc0_3x3x3_{ts}.png"

    # fig.suptitle(f"CH sanity grid (seed={args.seed}, steps={steps}, grid={g}, L={float(args.L):g})", fontsize=11)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"[OK] saved: {outpath}")


if __name__ == "__main__":
    main()
