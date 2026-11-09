#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_sampling.py

Sensitivity of the LOO local recoverability score e to neighborhood size k,
sample count N, and descriptor noise (reviewer response, PRE ES12621).

Part "toy": synthetic mappings with known failure regions. Ambiguous and
recoverable sets are defined from a dense, noise-free reference run
(N_ref=32000, k=15); for every (k, N, sigma) setting we report the median e on
each set over several sampling seeds. The identity map has no ambiguous set,
so any drop in e there is purely data-limited.

Part "potts": the three-state Potts dataset. k is swept and N is subsampled as
for the toys; descriptor noise is varied physically through the number of
repeat realizations R averaged into each descriptor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import potts_descriptors as pdesc
from toy_figures import (
    _heatmap_binned_core,
    fold_sheet,
    identity,
    knn_in_y_chunked,
    local_explainedcov_metrics_batched_LOO,
    pinch,
    ribbon,
)
from potts_analyze_explained_fraction_v13 import _smooth_nan, standardize_np

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SCENARIOS = {"identity": identity, "pinch": pinch, "fold": fold_sheet, "ribbon": ribbon}
COLORS = {"identity": "#2a78d6", "pinch": "#eb6834", "fold": "#1baf7a", "ribbon": "#eda100"}

BASE = dict(k=15, N=2000, sigma=0.0)
SWEEPS = {
    "k": [4, 6, 8, 10, 15, 20, 30, 50, 80],
    "N": [250, 500, 1000, 2000, 4000, 8000, 16000],
    "sigma": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
}
SIGMA0 = 0.01  # where sigma=0 is drawn on the log axis
SCORE_KW = dict(use_weights=False, eps_tau=1e-12, ridge_y=1e-3, ridge_x=1e-8, eps_trace=1e-18, batch_size=1024)


@torch.no_grad()
def loo_score(X: torch.Tensor, Y: torch.Tensor, k: int) -> np.ndarray:
    idx, d = knn_in_y_chunked(Y, k=k, chunk=1024)
    return local_explainedcov_metrics_batched_LOO(X=X, Y=Y, idxY=idx, dY=d, **SCORE_KW)["explained_frac"]


@torch.no_grad()
def nearest_rows(A: torch.Tensor, B: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest rows of B for each row of A."""
    out = []
    for s in range(0, A.shape[0], 2048):
        out.append(torch.topk(torch.cdist(A[s:s + 2048], B), k=k, largest=False).indices)
    return torch.cat(out, dim=0)


# ----------------------------- toy part -----------------------------

def toy_reference(scen: str, device: torch.device, n_ref: int = 32000, seed: int = 999):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.rand((n_ref, 2), device=device, generator=g) * 2 - 1
    e = torch.as_tensor(loo_score(X, SCENARIOS[scen](X, 1.0), k=15), device=device)
    # denoise the reference by averaging over the 15 nearest control-space neighbors
    e_s = e[nearest_rows(X, X, 15)].mean(dim=1)
    return X, e_s


def toy_sets(scen, X, ref):
    """Boolean (recoverable, ambiguous) sets for control points X."""
    if scen == "pinch":
        # the contraction is exactly degenerate only on x1=0, so a dense noise-free
        # reference resolves it; use the geometric band instead
        a = X[:, 0].abs().cpu().numpy()
        return a > 0.5, a < 0.1
    Xr, er = ref
    e_ref = er[nearest_rows(X, Xr, 1)[:, 0]].cpu().numpy()
    return e_ref > 0.9, e_ref < 0.5


def toy_run(scen, ref, k, N, sigma, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.rand((N, 2), device=device, generator=g) * 2 - 1
    Y = SCENARIOS[scen](X, 1.0)
    r_k = float(knn_in_y_chunked(Y, k=k, chunk=1024)[1][:, -1].mean())  # noise-free k-NN radius
    if sigma > 0:
        Y = Y + sigma * torch.randn(Y.shape, device=device, generator=g)
    e = loo_score(X, Y, k)
    rec, amb = toy_sets(scen, X, ref)
    return {
        "rec": float(np.median(e[rec])) if rec.any() else np.nan,
        "amb": float(np.median(e[amb])) if amb.any() else np.nan,
        "r_k": r_k,
    }


def run_toy(outdir: Path, device: torch.device, n_seeds: int) -> dict:
    refs = {s: toy_reference(s, device) for s in SCENARIOS}
    for s, (_, er) in refs.items():
        er = er.cpu().numpy()
        print(f"[ref] {s}: frac recoverable={np.mean(er > 0.9):.3f} ambiguous={np.mean(er < 0.5):.3f}")

    results = {}
    for param, values in SWEEPS.items():
        for scen in SCENARIOS:
            rows = []
            for v in values:
                cfg = dict(BASE, **{param: v})
                runs = [toy_run(scen, refs[scen], cfg["k"], cfg["N"], cfg["sigma"], 100 + s, device)
                        for s in range(n_seeds)]
                rows.append({key: [r[key] for r in runs] for key in ("rec", "amb", "r_k")})
            results[f"{param}/{scen}"] = rows
            print(f"[toy] {param} {scen}: rec={[round(np.mean(r['rec']), 2) for r in rows]} "
                  f"amb={[round(np.nanmean(r['amb']), 2) if not np.all(np.isnan(r['amb'])) else None for r in rows]}")

    for n_extra in (500, 8000):
        rows = []
        for v in SWEEPS["sigma"]:
            runs = [toy_run("identity", refs["identity"], BASE["k"], n_extra, v, 100 + s, device)
                    for s in range(n_seeds)]
            rows.append({key: [r[key] for r in runs] for key in ("rec", "amb", "r_k")})
        results[f"sigma_N{n_extra}/identity"] = rows

    xlabels = {
        "k": "neighborhood size $k$",
        "N": "number of samples $N$",
        "sigma": r"relative noise $\sigma/r_k$",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
    for ax, (param, values) in zip(axes, SWEEPS.items()):
        def xvals(rows):
            xv = np.asarray(values, dtype=float)
            if param == "sigma":
                xv = xv / np.mean([np.mean(r["r_k"]) for r in rows])
                xv[0] = SIGMA0
            return xv

        if param == "sigma":
            for n_extra, mk in ((500, "s"), (8000, "^")):
                rows = results[f"sigma_N{n_extra}/identity"]
                m = np.mean([r["rec"] for r in rows], axis=1)
                ax.plot(xvals(rows), m, ":", color=COLORS["identity"], lw=1.0, marker=mk, ms=3,
                        label=f"identity, $N={n_extra}$")
        for scen in SCENARIOS:
            rows = results[f"{param}/{scen}"]
            xv = xvals(rows)
            for key, ls in (("rec", "-"), ("amb", "--")):
                arr = np.asarray([r[key] for r in rows], dtype=float)
                if np.all(np.isnan(arr)):
                    continue
                m, sd = np.nanmean(arr, axis=1), np.nanstd(arr, axis=1)
                ax.plot(xv, m, ls, color=COLORS[scen], lw=1.5, marker="o", ms=3, zorder=4 if scen == "identity" else 3,
                        label=scen if key == "rec" else None)
                ax.fill_between(xv, m - sd, m + sd, color=COLORS[scen], alpha=0.15, lw=0)
        ax.set_xscale("log")
        if param != "sigma":
            ax.axvline(BASE[param], color="0.6", lw=0.8, ls=":")
        ax.set_xlabel(xlabels[param])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        if param == "sigma":
            ax.set_xticks([SIGMA0, 0.1, 1.0])
            ax.set_xticklabels(["0", "0.1", "1"])
    axes[0].set_ylabel("median score $e$")
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.02, 1.04, f"({lab})", transform=ax.transAxes, fontweight="bold", ha="right")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    h += h2[:2]
    l += l2[:2]
    h += [plt.Line2D([], [], color="0.3", ls="-"), plt.Line2D([], [], color="0.3", ls="--")]
    l += ["recoverable set", "ambiguous set"]
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(outdir / "sensitivity_toy.pdf")
    fig.savefig(outdir / "sensitivity_toy.png", dpi=300)
    plt.close(fig)
    return results


# ----------------------------- Potts part -----------------------------

@torch.no_grad()
def potts_descriptors_by_R(raw_h5: Path, R_list, device) -> tuple[np.ndarray, dict]:
    """Repeat-averaged (phase fractions || corr2d) descriptors using the first R repeats."""
    with h5py.File(raw_h5, "r") as f:
        X = np.stack([f["parameters/temperature"][:], f["parameters/fraction_initial"][:]], axis=1)
        spins = f["states/final_spins"]
        N, R_all, H, W = spins.shape
        q = 3
        Rmax = max(R_list)
        assert Rmax <= R_all
        out = {R: np.zeros((N, q + 6 * H * W), dtype=np.float32) for R in R_list}
        for i0 in range(0, N, 8):
            i1 = min(N, i0 + 8)
            s = torch.as_tensor(spins[i0:i1, :Rmax], device=device).reshape(-1, 1, H, W)
            ph = pdesc.compute_phase_fractions(s, q).reshape(i1 - i0, Rmax, q)
            c2 = pdesc.compute_correlations_2d(s, q).reshape(i1 - i0, Rmax, -1)
            ph_c, c2_c = ph.cumsum(dim=1), c2.cumsum(dim=1)
            for R in R_list:
                out[R][i0:i1] = torch.cat([ph_c[:, R - 1], c2_c[:, R - 1]], dim=1).div(R).cpu().numpy()
    return X.astype(np.float32), out


@torch.no_grad()
def global_pca(Y: np.ndarray, energy: float, device):
    """Global PCA basis keeping the leading components up to the given energy fraction."""
    Yt = torch.as_tensor(Y, device=device)
    mu = Yt.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Yt - mu, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    m = int(torch.searchsorted(cum, torch.tensor(energy, device=device)).item()) + 1
    return mu, Vh[:m], float(cum[m - 1])


@torch.no_grad()
def project(Y: np.ndarray, mu: torch.Tensor, V: torch.Tensor) -> np.ndarray:
    return ((torch.as_tensor(Y, device=mu.device) - mu) @ V.T).cpu().numpy()


def potts_score(X, Yp, k, device, y_scale=None):
    """Score on standardized controls and descriptors.

    y_scale: optional per-coordinate descriptor scale to use instead of Yp's own standard
    deviation, so that noisier descriptors are not rescaled back to unit variance.
    """
    Xt = torch.as_tensor(standardize_np(X.copy()), device=device)
    Ys = standardize_np(Yp.copy()) if y_scale is None else (Yp - Yp.mean(axis=0)) / y_scale
    return loo_score(Xt, torch.as_tensor(Ys, device=device, dtype=torch.float32), k)


def potts_map(X, e, bins=30, sigma_px=1.0):
    t, f = X[:, 0].astype(np.float64), X[:, 1].astype(np.float64)
    rng = [[0.4, 1.2], [0.1, 0.9]]
    s, _, _ = np.histogram2d(t, f, bins=bins, range=rng, weights=e)
    c, _, _ = np.histogram2d(t, f, bins=bins, range=rng)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = s / c
    img[c == 0] = np.nan
    return _smooth_nan(img, sigma_px)


def run_potts(raw_h5: Path, outdir: Path, device, energy: float) -> dict:
    R_list = [1, 10, 30, 100]
    X, Y_by_R = potts_descriptors_by_R(raw_h5, R_list, device)
    # One PCA basis, fit on the R=100 descriptors, is applied to every R so that the
    # representation is fixed and only the realization noise in the descriptor changes.
    mu, V, achieved = global_pca(Y_by_R[100], energy, device)
    print(f"[potts] PCA (R=100): m={V.shape[0]} energy={achieved:.3f}")
    proj = {R: project(Y_by_R[R], mu, V) for R in R_list}
    del Y_by_R
    Yp = proj[100]
    N = X.shape[0]

    panels = {"k": [], "N": [], "R": []}
    for k in [5, 15, 30, 50]:
        panels["k"].append((f"$k={k}$", X, potts_score(X, Yp, k, device)))
    rng = np.random.default_rng(0)
    for n in [125, 250, 500, N]:
        sub = np.sort(rng.choice(N, size=n, replace=False)) if n < N else np.arange(N)
        panels["N"].append((f"$N={n}$", X[sub], potts_score(X[sub], Yp[sub], 15, device)))
    for R in R_list:
        panels["R"].append((f"$R={R}$", X, potts_score(X, proj[R], 15, device, y_scale=Yp.std(axis=0))))

    # Agreement with the reference map (k=15, N=all, R=100), binned on a common grid.
    ref = potts_map(X, panels["k"][1][2])
    summary = {}
    for row, items in panels.items():
        for lab, Xs, e in items:
            img = potts_map(Xs, e)
            ok = np.isfinite(img) & np.isfinite(ref)
            r = float(np.corrcoef(img[ok], ref[ok])[0, 1])
            summary[f"{row}:{lab}"] = dict(pearson_vs_ref=r, median_e=float(np.median(e)),
                                           frac_low=float(np.mean(e < 0.5)))
            print(f"[potts] {row} {lab}: r={r:.3f} median e={np.median(e):.3f} frac(e<0.5)={np.mean(e < 0.5):.3f}")

    n_cols = max(len(v) for v in panels.values())
    fig, axes = plt.subplots(3, n_cols, figsize=(1.45 * n_cols + 0.6, 4.6), sharex=True, sharey=True,
                             constrained_layout=True)
    row_titles = {"k": "neighborhood size", "N": "number of samples", "R": "repeats per descriptor"}
    for i, (row, items) in enumerate(panels.items()):
        for j in range(n_cols):
            ax = axes[i, j]
            if j >= len(items):
                ax.set_axis_off()
                continue
            lab, Xs, e = items[j]
            im = ax.imshow(potts_map(Xs, e).T, origin="lower", extent=[0.4, 1.2, 0.1, 0.9], aspect="auto",
                           cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
            r = summary[f"{row}:{lab}"]["pearson_vs_ref"]
            ax.set_title(f"{lab}  ($r={r:.2f}$)", fontsize=8, pad=3)
            if i == 2:
                ax.set_xlabel("$T$")
            if j == 0:
                ax.set_ylabel(f"{row_titles[row]}\n$f_0$")
    fig.colorbar(im, ax=axes, shrink=0.6, label="score $e$")
    fig.savefig(outdir / "sensitivity_potts.pdf")
    fig.savefig(outdir / "sensitivity_potts.png", dpi=300)
    plt.close(fig)
    summary["pca"] = dict(m=int(V.shape[0]), energy=achieved)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["toy", "potts", "all"], default="all")
    ap.add_argument("--potts_raw_h5", type=str, default="", help="Raw spins HDF5 from potts_gen.py")
    ap.add_argument("--outdir", type=str, default="sensitivity_figures")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--pca_energy", type=float, default=0.95)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if args.part in ("toy", "all"):
        summary["toy"] = run_toy(outdir, device, args.n_seeds)
    if args.part in ("potts", "all"):
        summary["potts"] = run_potts(Path(args.potts_raw_h5), outdir, device, args.pca_energy)
    (outdir / f"sensitivity_summary_{args.part}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
