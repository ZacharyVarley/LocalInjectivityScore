#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potts_control_lines_corr2d.py

Saves TWO figures:

1) Single-pair figure (default pair=0-0):
   <outdir>/potts_control_lines_corr2d_pair_0-0.(png|pdf)

2) Multi-pair figure including 0-0 plus cross-correlations 0-1 and 1-2:
   <outdir>/potts_control_lines_corr2d_pairs_0-0_0-1_1-2.(png|pdf)

Panels (each is rows x 6 cols):
  A) f0 fixed at 0.3, T in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
  B) T fixed at 0.6, f0 in linspace(0.2, 0.8, 6)

Rows per panel:
  - Microstructure (single deterministic run)
  - For each requested corr2d pair:
      Corr2D (single)
      Corr2D (avg over n_repeats)

Corr2D definition matches potts_descriptors.compute_correlations_2d.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    fractions: torch.Tensor,  # (B,) fraction of phase 0
    q: int,
    device: torch.device,
) -> torch.Tensor:
    B = int(batch_size)
    H = W = int(grid_size)
    fr = fractions.view(B, 1, 1).clamp(0.0, 1.0)

    rand_grid = torch.rand((B, H, W), device=device, dtype=torch.float32)
    mask0 = rand_grid < fr

    spins = torch.empty((B, H, W), device=device, dtype=torch.int16)
    spins[mask0] = 0

    rem = ~mask0
    if rem.any():
        rem_rand = (rand_grid - fr) / (1.0 - fr).clamp_min(1e-12)  # in [0,1)
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
def compute_correlations_2d(spins: torch.Tensor, q: int, whiten_eps: float = 0.0) -> torch.Tensor:
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

    return out + float(whiten_eps)


# ----------------------------- sweep runner (multi-pair) -----------------------------
def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    den = (vmax - vmin)
    if den <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / den).astype(np.float32)


@torch.no_grad()
def run_sweep_multi(
    temperatures: np.ndarray,      # (B,)
    fractions0: np.ndarray,        # (B,)
    grid_size: int,
    steps: int,
    q: int,
    periodic: bool,
    remove_spurious: bool,
    n_repeats: int,
    seed_single: int,
    seed_repeat_base: int,
    pair_indices: List[int],
    whiten_eps: float,
    device: torch.device,
    progress_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      micro_single: (B,H,W)
      corr_single:  (B,P,H,W)   for selected pairs
      corr_avg:     (B,P,H,W)   for selected pairs
    """
    B = int(temperatures.shape[0])
    P = int(len(pair_indices))
    temps_t = torch.as_tensor(temperatures, device=device, dtype=torch.float32)
    fracs_t = torch.as_tensor(fractions0, device=device, dtype=torch.float32)

    # Single deterministic run
    torch.manual_seed(int(seed_single))
    spins0 = create_initial_states(B, grid_size, fracs_t, q=q, device=device)
    final_single = simulate_potts(
        spins0,
        temperatures=temps_t,
        steps=int(steps),
        q=q,
        periodic=bool(periodic),
        remove_spurious=bool(remove_spurious),
    )
    corr_all_single = compute_correlations_2d(final_single, q=q, whiten_eps=float(whiten_eps))  # (B,NP,H,W)
    corr_single = corr_all_single[:, pair_indices]  # (B,P,H,W)

    micro_single = (final_single[:, 0].to(torch.float32) / float(max(q - 1, 1))).clamp(0.0, 1.0)  # (B,H,W)

    # Averaged corr2d over repeats
    corr_sum = torch.zeros((B, P, grid_size, grid_size), device=device, dtype=torch.float64)
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
        corr_all_r = compute_correlations_2d(final_r, q=q, whiten_eps=float(whiten_eps))
        corr_r = corr_all_r[:, pair_indices]  # (B,P,H,W)
        corr_sum += corr_r.to(torch.float64)

        if (r + 1) % max(1, int(progress_every)) == 0 or (r + 1) == int(n_repeats):
            print(f"  repeats: {r+1}/{n_repeats}")

    corr_avg = (corr_sum / float(n_repeats)).to(torch.float32)  # (B,P,H,W)

    return (
        micro_single.detach().cpu().numpy().astype(np.float32),
        corr_single.detach().cpu().numpy().astype(np.float32),
        corr_avg.detach().cpu().numpy().astype(np.float32),
    )


# ----------------------------- plotting (single + multi) -----------------------------
def plot_panels_singlepair(
    outdir: Path,
    temps_line: np.ndarray,
    fracs_line: np.ndarray,
    temps_frac: np.ndarray,
    fracs_frac: np.ndarray,
    micro_T: np.ndarray, corr_T: np.ndarray, corrT_avg: np.ndarray,   # corr_*: (B,H,W)
    micro_F: np.ndarray, corr_F: np.ndarray, corrF_avg: np.ndarray,
    pair: str,
    n_repeats: int,
    dpi: int,
) -> None:
    corr_all_single = np.concatenate([corr_T, corr_F], axis=0)
    corr_all_avg = np.concatenate([corrT_avg, corrF_avg], axis=0)

    vmin_s = float(np.min(corr_all_single))
    vmax_s = float(np.max(corr_all_single))
    vmin_a = float(np.min(corr_all_avg))
    vmax_a = float(np.max(corr_all_avg))

    corr_T_n = normalize01(corr_T, vmin_s, vmax_s)
    corr_F_n = normalize01(corr_F, vmin_s, vmax_s)
    corrT_avg_n = normalize01(corrT_avg, vmin_a, vmax_a)
    corrF_avg_n = normalize01(corrF_avg, vmin_a, vmax_a)

    fig, axes = plt.subplots(6, 6, figsize=(18, 18), constrained_layout=False)
    fig.subplots_adjust(hspace=0.02, wspace=0.02, left=0.06, right=0.995, top=0.96, bottom=0.05)

    row_names = [
        "Microstructure",
        f"Corr2D {pair} (single)",
        f"Corr2D {pair} (avg {n_repeats})",
    ]

    for c in range(6):
        axes[0, c].imshow(micro_T[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[1, c].imshow(corr_T_n[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        axes[2, c].imshow(corrT_avg_n[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")

        for r in range(3):
            axes[r, c].set_axis_off()

        axes[0, c].set_title(f"$T={temps_line[c]:.1f}$\n$f_0=0.30$", fontsize=12, pad=6)

    for c in range(6):
        axes[3, c].imshow(micro_F[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[4, c].imshow(corr_F_n[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        axes[5, c].imshow(corrF_avg_n[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")

        for r in range(3, 6):
            axes[r, c].set_axis_off()

        axes[3, c].set_title(f"$f_0={fracs_frac[c]:.2f}$\n$T=0.60$", fontsize=12, pad=6)

    for i, name in enumerate(row_names):
        axes[i, 0].text(
            -0.22, 0.5, name,
            transform=axes[i, 0].transAxes,
            fontsize=13, fontweight="bold",
            rotation=90, va="center", ha="center",
        )
        axes[i + 3, 0].text(
            -0.22, 0.5, name,
            transform=axes[i + 3, 0].transAxes,
            fontsize=13, fontweight="bold",
            rotation=90, va="center", ha="center",
        )

    axes[0, 0].text(-0.05, 1.12, "A", transform=axes[0, 0].transAxes, fontsize=18, fontweight="bold")
    axes[3, 0].text(-0.05, 1.12, "B", transform=axes[3, 0].transAxes, fontsize=18, fontweight="bold")

    out_png = outdir / f"potts_control_lines_corr2d_pair_{pair}.png"
    out_pdf = outdir / f"potts_control_lines_corr2d_pair_{pair}.pdf"
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_pdf, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")


def plot_panels_multipair(
    outdir: Path,
    temps_line: np.ndarray,
    fracs_line: np.ndarray,
    temps_frac: np.ndarray,
    fracs_frac: np.ndarray,
    micro_T: np.ndarray, corr_T: np.ndarray, corrT_avg: np.ndarray,   # corr_*: (B,P,H,W)
    micro_F: np.ndarray, corr_F: np.ndarray, corrF_avg: np.ndarray,
    pairs: List[str],
    n_repeats: int,
    dpi: int,
) -> None:
    P = int(len(pairs))
    rows_per_panel = 1 + 2 * P
    total_rows = 2 * rows_per_panel
    fig_h = max(18.0, 2.35 * float(total_rows))  # keep readable when rows increase
    fig, axes = plt.subplots(total_rows, 6, figsize=(18, fig_h), constrained_layout=False)
    fig.subplots_adjust(hspace=0.02, wspace=0.02, left=0.07, right=0.995, top=0.97, bottom=0.03)

    # Per-pair normalization (single and avg separately), using both panels.
    # This avoids cross-pair scale coupling while keeping within-pair consistency.
    vmin_s = np.zeros((P,), dtype=np.float64)
    vmax_s = np.zeros((P,), dtype=np.float64)
    vmin_a = np.zeros((P,), dtype=np.float64)
    vmax_a = np.zeros((P,), dtype=np.float64)
    for p in range(P):
        single_all = np.concatenate([corr_T[:, p], corr_F[:, p]], axis=0)
        avg_all = np.concatenate([corrT_avg[:, p], corrF_avg[:, p]], axis=0)
        vmin_s[p] = float(np.min(single_all))
        vmax_s[p] = float(np.max(single_all))
        vmin_a[p] = float(np.min(avg_all))
        vmax_a[p] = float(np.max(avg_all))

    # Helper to map row indices
    def r_micro(panel: int) -> int:
        return panel * rows_per_panel

    def r_single(panel: int, p: int) -> int:
        return panel * rows_per_panel + 1 + 2 * p

    def r_avg(panel: int, p: int) -> int:
        return panel * rows_per_panel + 1 + 2 * p + 1

    # Panel A (panel=0): T sweep at fixed f0
    for c in range(6):
        axes[r_micro(0), c].imshow(micro_T[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[r_micro(0), c].set_axis_off()
        axes[r_micro(0), c].set_title(f"$T={temps_line[c]:.1f}$\n$f_0=0.30$", fontsize=12, pad=6)

        for p in range(P):
            im_s = normalize01(corr_T[c, p], vmin_s[p], vmax_s[p])
            im_a = normalize01(corrT_avg[c, p], vmin_a[p], vmax_a[p])
            axes[r_single(0, p), c].imshow(im_s, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            axes[r_avg(0, p), c].imshow(im_a, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            axes[r_single(0, p), c].set_axis_off()
            axes[r_avg(0, p), c].set_axis_off()

    # Panel B (panel=1): f0 sweep at fixed T
    for c in range(6):
        axes[r_micro(1), c].imshow(micro_F[c], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[r_micro(1), c].set_axis_off()
        axes[r_micro(1), c].set_title(f"$f_0={fracs_frac[c]:.2f}$\n$T=0.60$", fontsize=12, pad=6)

        for p in range(P):
            im_s = normalize01(corr_F[c, p], vmin_s[p], vmax_s[p])
            im_a = normalize01(corrF_avg[c, p], vmin_a[p], vmax_a[p])
            axes[r_single(1, p), c].imshow(im_s, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            axes[r_avg(1, p), c].imshow(im_a, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            axes[r_single(1, p), c].set_axis_off()
            axes[r_avg(1, p), c].set_axis_off()

    # Left-side row labels (both panels)
    for panel in [0, 1]:
        rr0 = panel * rows_per_panel
        axes[rr0, 0].text(
            -0.23, 0.5, "Microstructure",
            transform=axes[rr0, 0].transAxes,
            fontsize=13, fontweight="bold",
            rotation=90, va="center", ha="center",
        )
        for p, lab in enumerate(pairs):
            rs = r_single(panel, p)
            ra = r_avg(panel, p)
            axes[rs, 0].text(
                -0.23, 0.5, f"Corr2D {lab} (single)",
                transform=axes[rs, 0].transAxes,
                fontsize=13, fontweight="bold",
                rotation=90, va="center", ha="center",
            )
            axes[ra, 0].text(
                -0.23, 0.5, f"Corr2D {lab} (avg {n_repeats})",
                transform=axes[ra, 0].transAxes,
                fontsize=13, fontweight="bold",
                rotation=90, va="center", ha="center",
            )

    # Panel labels A/B
    axes[r_micro(0), 0].text(-0.05, 1.12, "A", transform=axes[r_micro(0), 0].transAxes, fontsize=18, fontweight="bold")
    axes[r_micro(1), 0].text(-0.05, 1.12, "B", transform=axes[r_micro(1), 0].transAxes, fontsize=18, fontweight="bold")

    tag = "_".join(pairs)
    out_png = outdir / f"potts_control_lines_corr2d_pairs_{tag}.png"
    out_pdf = outdir / f"potts_control_lines_corr2d_pairs_{tag}.pdf"
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
    ap.add_argument("--seed_single", type=int, default=2024)
    ap.add_argument("--seed_repeat_base", type=int, default=9001)

    ap.add_argument("--pair", type=str, default="0-0")          # for the single-pair figure
    ap.add_argument("--whiten_eps", type=float, default=0.0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="potts_figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--progress_every", type=int, default=10)

    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(str(args.device) if use_cuda else "cpu")
    print(f"[potts_viz] device={device}")

    q = int(args.q)
    if q < 3:
        raise ValueError("Multi-pair figure requires q>=3 to include pairs 0-1 and 1-2.")

    grid = int(args.grid)
    steps = int(args.steps)
    periodic = bool(int(args.periodic))
    remove_spurious = bool(int(args.remove_spurious))
    n_repeats = int(args.n_repeats)

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Sweeps
    temps_line = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float32)
    fracs_line = np.full((6,), 0.30, dtype=np.float32)

    temps_frac = np.full((6,), 0.60, dtype=np.float32)
    fracs_frac = np.linspace(0.2, 0.8, 6, dtype=np.float32)

    # Pair sets
    single_pair = str(args.pair).strip()
    single_pair_idx = pair_to_index(q, single_pair)

    multi_pairs = ["0-0", "0-1", "1-2"]
    multi_pair_indices = [pair_to_index(q, p) for p in multi_pairs]

    # -------- Panel A --------
    print("[A] sweep T with f0=0.30")
    micro_T, corr_T_all, corrT_avg_all = run_sweep_multi(
        temperatures=temps_line,
        fractions0=fracs_line,
        grid_size=grid,
        steps=steps,
        q=q,
        periodic=periodic,
        remove_spurious=remove_spurious,
        n_repeats=n_repeats,
        seed_single=int(args.seed_single),
        seed_repeat_base=int(args.seed_repeat_base),
        pair_indices=multi_pair_indices + [single_pair_idx] if single_pair_idx not in multi_pair_indices else multi_pair_indices,
        whiten_eps=float(args.whiten_eps),
        device=device,
        progress_every=int(args.progress_every),
    )

    # Map indices within corr_T_all to the requested sets
    used_indices = (multi_pair_indices + [single_pair_idx]) if single_pair_idx not in multi_pair_indices else multi_pair_indices
    idx_map = {pi: k for k, pi in enumerate(used_indices)}

    # Extract single-pair slices
    corr_T_single = corr_T_all[:, idx_map[single_pair_idx]]
    corrT_avg_single = corrT_avg_all[:, idx_map[single_pair_idx]]

    # Extract multi-pair slices in order
    corr_T_multi = np.stack([corr_T_all[:, idx_map[pi]] for pi in multi_pair_indices], axis=1)        # (6,P,H,W)
    corrT_avg_multi = np.stack([corrT_avg_all[:, idx_map[pi]] for pi in multi_pair_indices], axis=1)  # (6,P,H,W)

    # -------- Panel B --------
    print("[B] sweep f0 with T=0.60")
    micro_F, corr_F_all, corrF_avg_all = run_sweep_multi(
        temperatures=temps_frac,
        fractions0=fracs_frac,
        grid_size=grid,
        steps=steps,
        q=q,
        periodic=periodic,
        remove_spurious=remove_spurious,
        n_repeats=n_repeats,
        seed_single=int(args.seed_single) + 17,
        seed_repeat_base=int(args.seed_repeat_base) + 10000,
        pair_indices=used_indices,
        whiten_eps=float(args.whiten_eps),
        device=device,
        progress_every=int(args.progress_every),
    )

    corr_F_single = corr_F_all[:, idx_map[single_pair_idx]]
    corrF_avg_single = corrF_avg_all[:, idx_map[single_pair_idx]]

    corr_F_multi = np.stack([corr_F_all[:, idx_map[pi]] for pi in multi_pair_indices], axis=1)
    corrF_avg_multi = np.stack([corrF_avg_all[:, idx_map[pi]] for pi in multi_pair_indices], axis=1)

    # -------- Save single-pair figure --------
    plot_panels_singlepair(
        outdir=outdir,
        temps_line=temps_line,
        fracs_line=fracs_line,
        temps_frac=temps_frac,
        fracs_frac=fracs_frac,
        micro_T=micro_T, corr_T=corr_T_single, corrT_avg=corrT_avg_single,
        micro_F=micro_F, corr_F=corr_F_single, corrF_avg=corrF_avg_single,
        pair=single_pair,
        n_repeats=n_repeats,
        dpi=int(args.dpi),
    )

    # -------- Save multi-pair figure (0-0, 0-1, 1-2) --------
    plot_panels_multipair(
        outdir=outdir,
        temps_line=temps_line,
        fracs_line=fracs_line,
        temps_frac=temps_frac,
        fracs_frac=fracs_frac,
        micro_T=micro_T, corr_T=corr_T_multi, corrT_avg=corrT_avg_multi,
        micro_F=micro_F, corr_F=corr_F_multi, corrF_avg=corrF_avg_multi,
        pairs=multi_pairs,
        n_repeats=n_repeats,
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
