#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_control_lines_corr2d_mosaic_avg.py

Compact publication figure:

Two sweeps, each displayed as 2 rows × 6 cols (stacked => 4 × 6 total):
  A) f0 fixed at 0.30, T in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
  B) T fixed at 0.60, f0 in linspace(0.2, 0.8, 6)

Rows per sweep:
  1) Microstructure (single deterministic run)
  2) Corr2D averaged over n_repeats, shown as a 2×2 mosaic per column:
        [ 0-0   0-1 ]
        [ 1-1   1-2 ]

Rationale: phase 2 is statistically equivalent to phase 1 (only phase 0 is modulated),
so we omit (0-2) and (2-2).

Corr2D matches potts_descriptors.compute_correlations_2d:
  indicator fields -> mean-subtract -> FFT cross-power -> irfft -> fftshift.

Outputs:
  <outdir>/potts_control_lines_corr2d_avg_mosaic_00_01_11_12.(png|pdf)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ----------------------------- Potts dynamics (matches potts_gen.py) -----------------------------
@torch.no_grad()
def potts_step(
    spins: torch.Tensor,          # (B,1,H,W) int8
    beta: torch.Tensor,           # (B,) float
    q: int,
    periodic: bool = True,
    remove_spurious: bool = False,
) -> torch.Tensor:
    if periodic:
        up = torch.roll(spins, 1, dims=-2)
        down = torch.roll(spins, -1, dims=-2)
        left = torch.roll(spins, 1, dims=-1)
        right = torch.roll(spins, -1, dims=-1)
    else:
        spins_padded = F.pad(spins, (1, 1, 1, 1), mode="replicate")
        up = spins_padded[:, :, :-2, 1:-1]
        down = spins_padded[:, :, 2:, 1:-1]
        left = spins_padded[:, :, 1:-1, :-2]
        right = spins_padded[:, :, 1:-1, 2:]

    aligned = (
        (spins == up).byte()
        + (spins == down).byte()
        + (spins == left).byte()
        + (spins == right).byte()
    ).float()

    new_spins = torch.randint_like(spins, q, dtype=torch.int8)

    new_aligned = (
        (new_spins == up).byte()
        + (new_spins == down).byte()
        + (new_spins == left).byte()
        + (new_spins == right).byte()
    ).float()

    delta = new_aligned - aligned
    beta_expanded = beta.view(-1, 1, 1, 1)
    acceptance_probs = torch.exp(beta_expanded * delta)

    if remove_spurious:
        flips = (torch.rand_like(acceptance_probs) < acceptance_probs) & (new_aligned > 0)
    else:
        flips = torch.rand_like(acceptance_probs) < acceptance_probs

    return torch.where(flips, new_spins, spins)


@torch.no_grad()
def simulate_potts(
    spins: torch.Tensor,          # (B,1,H,W)
    temperatures: torch.Tensor,   # (B,)
    steps: int,
    q: int,
    periodic: bool,
    remove_spurious: bool,
) -> torch.Tensor:
    beta = 1.0 / temperatures.clamp_min(1e-12)
    for _ in range(int(steps)):
        spins = potts_step(spins, beta, q=q, periodic=periodic, remove_spurious=remove_spurious)
    return spins


# ----------------------------- initialization (distribution matches potts_gen.py) -----------------------------
@torch.no_grad()
def create_initial_states(
    batch_size: int,
    grid_size: int,
    fractions0: torch.Tensor,  # (B,) fraction of phase 0
    q: int,
    device: torch.device,
) -> torch.Tensor:
    B = int(batch_size)
    H = W = int(grid_size)
    fr = fractions0.view(B, 1, 1).clamp(0.0, 1.0)

    rand_grid = torch.rand((B, H, W), device=device, dtype=torch.float32)
    mask0 = rand_grid < fr

    spins = torch.empty((B, H, W), device=device, dtype=torch.int16)
    spins[mask0] = 0

    rem = ~mask0
    if rem.any():
        rem_rand = (rand_grid - fr) / (1.0 - fr).clamp_min(1e-12)
        other = 1 + torch.floor(rem_rand * float(q - 1)).to(torch.int16)
        other = other.clamp(1, q - 1)
        spins[rem] = other[rem]

    return spins.to(torch.int8).unsqueeze(1)  # (B,1,H,W)


# ----------------------------- corr2d (matches potts_descriptors.py) -----------------------------
def pair_labels(q: int) -> List[str]:
    return [f"{i}-{j}" for i in range(q) for j in range(i, q)]


def pair_to_index(q: int, pair: str) -> int:
    labels = pair_labels(q)
    if pair in labels:
        return labels.index(pair)
    raise ValueError(f"Unknown pair '{pair}'. Expected one of: {labels}")


@torch.no_grad()
def compute_correlations_2d(spins: torch.Tensor, q: int) -> torch.Tensor:
    # spins: (B,1,H,W) int8
    # returns: (B, n_pairs, H, W) float32
    B, _, H, W = spins.shape
    n_pairs = q * (q + 1) // 2
    out = torch.zeros((B, n_pairs, H, W), device=spins.device, dtype=torch.float32)

    pair_idx = 0
    for i in range(q):
        for j in range(i, q):
            ind_i = (spins[:, 0] == i).float()
            ind_j = (spins[:, 0] == j).float()

            ind_i -= ind_i.mean(dim=(-2, -1), keepdim=True)
            ind_j -= ind_j.mean(dim=(-2, -1), keepdim=True)

            fft_i = torch.fft.rfft2(ind_i)
            fft_j = torch.fft.rfft2(ind_j)
            cross_power = fft_i * torch.conj(fft_j)
            corr = torch.fft.irfft2(cross_power, s=(H, W)) / float(H * W)
            out[:, pair_idx] = torch.fft.fftshift(corr, dim=(-2, -1))
            pair_idx += 1

    return out


@torch.no_grad()
def downsample_2d(x: torch.Tensor, ds: int) -> torch.Tensor:
    # x: (B,P,H,W) or (B,H,W)
    ds = int(max(1, ds))
    if ds == 1:
        return x
    if x.ndim == 3:
        B, H, W = x.shape
        if (H % ds) != 0 or (W % ds) != 0:
            raise RuntimeError(f"corr_downsample={ds} must divide H,W (H={H}, W={W})")
        return F.avg_pool2d(x.unsqueeze(1), kernel_size=ds, stride=ds).squeeze(1)
    if x.ndim == 4:
        B, P, H, W = x.shape
        if (H % ds) != 0 or (W % ds) != 0:
            raise RuntimeError(f"corr_downsample={ds} must divide H,W (H={H}, W={W})")
        return F.avg_pool2d(x.view(B * P, 1, H, W), kernel_size=ds, stride=ds).view(B, P, H // ds, W // ds)
    raise ValueError("downsample_2d expects 3D or 4D tensor")


# ----------------------------- sweep runner (avg-only corr) -----------------------------
@torch.no_grad()
def run_sweep_avg(
    temperatures: np.ndarray,      # (B,)
    fractions0: np.ndarray,        # (B,)
    grid_size: int,
    steps: int,
    q: int,
    periodic: bool,
    remove_spurious: bool,
    n_repeats: int,
    seed_micro: int,
    seed_repeat_base: int,
    pair_indices: List[int],
    corr_downsample: int,
    device: torch.device,
    progress_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      micro_single: (B,H,W) in [0,1] (spin/(q-1))
      corr_avg:     (B,P,H',W') averaged over repeats
    """
    B = int(temperatures.shape[0])
    P = int(len(pair_indices))
    temps_t = torch.as_tensor(temperatures, device=device, dtype=torch.float32)
    fracs_t = torch.as_tensor(fractions0, device=device, dtype=torch.float32)

    # Microstructure: single deterministic run (for row 1)
    torch.manual_seed(int(seed_micro))
    spins0 = create_initial_states(B, grid_size, fracs_t, q=q, device=device)
    final_micro = simulate_potts(
        spins0,
        temperatures=temps_t,
        steps=int(steps),
        q=q,
        periodic=bool(periodic),
        remove_spurious=bool(remove_spurious),
    )
    micro_single = (final_micro[:, 0].to(torch.float32) / float(max(q - 1, 1))).clamp(0.0, 1.0)  # (B,H,W)

    # Corr2D averaged over repeats (for row 2)
    corr_sum = None
    for r in range(int(n_repeats)):
        torch.manual_seed(int(seed_repeat_base) + int(r))
        spins0_r = create_initial_states(B, grid_size, fracs_t, q=q, device=device)
        final_r = simulate_potts(
            spins0_r,
            temperatures=temps_t,
            steps=int(steps),
            q=q,
            periodic=bool(periodic),
            remove_spurious=bool(remove_spurious),
        )
        corr_all_r = compute_correlations_2d(final_r, q=q)[:, pair_indices]  # (B,P,H,W)
        corr_all_r = downsample_2d(corr_all_r, corr_downsample)              # (B,P,H',W')
        if corr_sum is None:
            corr_sum = corr_all_r.to(torch.float64)
        else:
            corr_sum += corr_all_r.to(torch.float64)

        if (r + 1) % max(1, int(progress_every)) == 0 or (r + 1) == int(n_repeats):
            print(f"  repeats: {r+1}/{n_repeats}")

    corr_avg = (corr_sum / float(n_repeats)).to(torch.float32)  # (B,P,H',W')

    return (
        micro_single.detach().cpu().numpy().astype(np.float32),
        corr_avg.detach().cpu().numpy().astype(np.float32),
    )


# ----------------------------- plotting helpers -----------------------------
def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    den = float(vmax - vmin)
    if den <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / den).astype(np.float32)


def make_mosaic_2x2(a00: np.ndarray, a01: np.ndarray, a11: np.ndarray, a12: np.ndarray, pad: int = 2) -> np.ndarray:
    """
    Inputs: (H,W) arrays, already normalized to [0,1]
    Output: (2H+pad, 2W+pad)
    """
    H, W = a00.shape
    pad_col = np.ones((H, pad), dtype=np.float32)
    pad_row = np.ones((pad, 2 * W + pad), dtype=np.float32)

    top = np.concatenate([a00, pad_col, a01], axis=1)
    bot = np.concatenate([a11, pad_col, a12], axis=1)
    return np.concatenate([top, pad_row, bot], axis=0)


def plot_compact(
    outdir: Path,
    temps_line: np.ndarray,
    fracs_line: np.ndarray,
    temps_frac: np.ndarray,
    fracs_frac: np.ndarray,
    micro_T: np.ndarray, corrT_avg: np.ndarray,   # micro: (6,H,W), corr: (6,4,H',W')
    micro_F: np.ndarray, corrF_avg: np.ndarray,
    pair_names: List[str],                         # ["0-0","0-1","1-1","1-2"]
    dpi: int,
) -> None:
    # Global per-channel normalization across both sweeps
    P = int(corrT_avg.shape[1])
    assert P == 4

    vmin = np.zeros((P,), dtype=np.float64)
    vmax = np.zeros((P,), dtype=np.float64)
    for p in range(P):
        allp = np.concatenate([corrT_avg[:, p], corrF_avg[:, p]], axis=0)
        vmin[p] = float(np.min(allp))
        vmax[p] = float(np.max(allp))

    # Build 2x2 mosaics per condition
    mosa_T = []
    mosa_F = []
    for c in range(6):
        cT = [normalize01(corrT_avg[c, p], vmin[p], vmax[p]) for p in range(P)]
        cF = [normalize01(corrF_avg[c, p], vmin[p], vmax[p]) for p in range(P)]
        mosa_T.append(make_mosaic_2x2(cT[0], cT[1], cT[2], cT[3], pad=2))
        mosa_F.append(make_mosaic_2x2(cF[0], cF[1], cF[2], cF[3], pad=2))

    mosa_T = np.stack(mosa_T, axis=0)  # (6, Hm, Wm)
    mosa_F = np.stack(mosa_F, axis=0)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.02, wspace=0.02, 
                           left=0.07, right=0.995, top=0.96, bottom=0.06,
                           height_ratios=[1, 1, 1, 1])
    # Add extra space between rows 1 and 2 (between A and B)
    gs.update(hspace=0.15)
    
    axes = np.empty((4, 6), dtype=object)
    for i in range(4):
        for j in range(6):
            axes[i, j] = fig.add_subplot(gs[i, j])

    # Row 0: micro (A)
    for c in range(6):
        axes[0, c].imshow(micro_T[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[0, c].set_axis_off()
        axes[0, c].set_title(f"$T={temps_line[c]:.1f}$  $f_0=0.2$", fontsize=12, pad=6)

    # Row 1: corr mosaic avg (A)
    for c in range(6):
        axes[1, c].imshow(mosa_T[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        axes[1, c].set_axis_off()

    # Row 2: micro (B)
    for c in range(6):
        axes[2, c].imshow(micro_F[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[2, c].set_axis_off()
        axes[2, c].set_title(f"$f_0={fracs_frac[c]:.1f}$  $T=0.6$", fontsize=12, pad=6)

    # Row 3: corr mosaic avg (B)
    for c in range(6):
        axes[3, c].imshow(mosa_F[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        axes[3, c].set_axis_off()

    # Left labels
    axes[0, 0].text(-0.23, 0.5, "Microstructure", transform=axes[0, 0].transAxes,
                    fontsize=13, fontweight="bold", rotation=90, va="center", ha="center")
    axes[1, 0].text(-0.23, 0.5, f"Corr2D avg mosaic\n({pair_names[0]},{pair_names[1]}; {pair_names[2]},{pair_names[3]})",
                    transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight="bold", rotation=90, va="center", ha="center")
    axes[2, 0].text(-0.23, 0.5, "Microstructure", transform=axes[2, 0].transAxes,
                    fontsize=13, fontweight="bold", rotation=90, va="center", ha="center")
    axes[3, 0].text(-0.23, 0.5, f"Corr2D avg mosaic\n({pair_names[0]},{pair_names[1]}; {pair_names[2]},{pair_names[3]})",
                    transform=axes[3, 0].transAxes,
                    fontsize=12, fontweight="bold", rotation=90, va="center", ha="center")

    # Panel labels
    axes[0, 0].text(-0.3, 1.05, "A", transform=axes[0, 0].transAxes, fontsize=26, fontweight="bold")
    axes[2, 0].text(-0.3, 1.05, "B", transform=axes[2, 0].transAxes, fontsize=26, fontweight="bold")

    out_png = outdir / "potts_control_lines_corr2d_avg_mosaic_00_01_11_12.png"
    out_pdf = outdir / "potts_control_lines_corr2d_avg_mosaic_00_01_11_12.pdf"
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_pdf, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")


# ----------------------------- CLI -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)

    ap.add_argument("--periodic", type=int, default=1)
    ap.add_argument("--remove_spurious", type=int, default=0)

    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--seed_micro_A", type=int, default=2024)
    ap.add_argument("--seed_micro_B", type=int, default=2041)
    ap.add_argument("--seed_repeat_base_A", type=int, default=9001)
    ap.add_argument("--seed_repeat_base_B", type=int, default=19001)

    ap.add_argument("--corr_downsample", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="potts_figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--progress_every", type=int, default=10)

    args = ap.parse_args()

    q = int(args.q)
    if q != 3:
        raise ValueError("This mosaic spec assumes q=3 with pairs 0-0,0-1,1-1,1-2.")

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(str(args.device) if use_cuda else "cpu")
    print(f"[potts_viz] device={device}")

    grid = int(args.grid)
    steps = int(args.steps)
    periodic = bool(int(args.periodic))
    remove_spurious = bool(int(args.remove_spurious))
    n_repeats = int(args.n_repeats)
    corr_downsample = int(max(1, args.corr_downsample))

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Sweeps
    temps_line = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float32)
    fracs_line = np.full((6,), 0.20, dtype=np.float32)

    temps_frac = np.full((6,), 0.60, dtype=np.float32)
    fracs_frac = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    # Pairs for 2×2 mosaic: [ [0-0, 0-1], [1-1, 1-2] ]
    pair_names = ["0-0", "0-1", "1-1", "1-2"]
    pair_indices = [pair_to_index(q, p) for p in pair_names]

    # Panel A
    print("[A] micro + corr2d(avg) sweep T with f0=0.30")
    micro_T, corrT_avg = run_sweep_avg(
        temperatures=temps_line,
        fractions0=fracs_line,
        grid_size=grid,
        steps=steps,
        q=q,
        periodic=periodic,
        remove_spurious=remove_spurious,
        n_repeats=n_repeats,
        seed_micro=int(args.seed_micro_A),
        seed_repeat_base=int(args.seed_repeat_base_A),
        pair_indices=pair_indices,
        corr_downsample=corr_downsample,
        device=device,
        progress_every=int(args.progress_every),
    )

    # Panel B
    print("[B] micro + corr2d(avg) sweep f0 with T=0.60")
    micro_F, corrF_avg = run_sweep_avg(
        temperatures=temps_frac,
        fractions0=fracs_frac,
        grid_size=grid,
        steps=steps,
        q=q,
        periodic=periodic,
        remove_spurious=remove_spurious,
        n_repeats=n_repeats,
        seed_micro=int(args.seed_micro_B),
        seed_repeat_base=int(args.seed_repeat_base_B),
        pair_indices=pair_indices,
        corr_downsample=corr_downsample,
        device=device,
        progress_every=int(args.progress_every),
    )

    plot_compact(
        outdir=outdir,
        temps_line=temps_line,
        fracs_line=fracs_line,
        temps_frac=temps_frac,
        fracs_frac=fracs_frac,
        micro_T=micro_T, corrT_avg=corrT_avg,
        micro_F=micro_F, corrF_avg=corrF_avg,
        pair_names=pair_names,
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
