
#!/usr/bin/env python3
"""
Potts (q=3) analysis: explained-fraction heatmaps

Defaults:
  - data_dir: potts_data
  - reads all combinations: warp in {none, fold, ribbon, pinch} and mode in {fixedseed, repeated}
  - outputs to: potts_data/analysis_explained_fraction

Descriptor:
  - Default is radial correlation descriptor built from real-space correlation surfaces:
      (00,01,02,11,12,22) × nbins
  - For repeated mode, descriptors are averaged across repeats in descriptor space.
    (No averaging in Fourier domain: each repeat's correlation is converted to a real-space
     surface via IFFT and only then reduced / averaged.)

Outputs per run:
  - cache/{run_id}__Y_potts_radial.npz
  - figs/{run_id}__explained_fraction.png
  - figs/{run_id}__explained_fraction.npy
  - figs/{run_id}__explained_fraction_counts.npy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from tqdm import tqdm

from injectivity_analysis_helpers import (
    utcnow_iso,
    ensure_dir,
    potts_descriptor_radial,
    compute_explained_fraction,
    bin2d_mean,
    smooth_grid_nan,
    clip_quantiles,
)


WarpName = Literal["none", "fold", "ribbon", "pinch"]
SeedMode = Literal["fixedseed", "repeated"]


@dataclass(frozen=True)
class AnalysisConfig:
    data_dir: str = "potts_data"
    out_dir: str = ""  # default: {data_dir}/analysis_explained_fraction

    nbins: int = 64

    # explained fraction
    k: int = 15
    lambda0: float = 1e-3
    weights: str = "gaussian"

    # compute device
    device: str = "cuda"

    # heatmap
    heat_bins_x: int = 60
    heat_bins_y: int = 60
    smooth_sigma: float = 1.0
    clip_qlo: float = 0.01
    clip_qhi: float = 0.99

    # compute
    chunk_n: int = 8  # params per chunk when streaming repeated
    force_recompute: bool = False

    # plotting
    dpi: int = 250


def _default_outdir(data_dir: Path) -> Path:
    return data_dir / "analysis_explained_fraction"


def _run_id(warp: str, mode: str) -> str:
    return f"potts__{warp}__{mode}"


def _h5_path(data_dir: Path, warp: str, mode: str) -> Path:
    return data_dir / f"potts__{warp}__{mode}.h5"


def _cache_path(cache_dir: Path, run_id: str, nbins: int) -> Path:
    return cache_dir / f"{run_id}__Y_potts_radial__nb{nbins}.npz"


def _load_or_build_Y(
    h5p: Path,
    cachep: Path,
    nbins: int,
    chunk_n: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      ctrl2: (N,2) controls to plot (f0, T) original
      Y: (N,q) descriptor outcomes
    """
    if cachep.exists() and not force:
        z = np.load(cachep, allow_pickle=False)
        return z["ctrl2"], z["Y"]

    with h5py.File(str(h5p), "r") as f:
        ctrl2 = f["params_original"][...].astype(np.float32)  # (N,2) => (f0,T)
        spins = f["spins"]
        mode = "repeated" if spins.ndim == 4 else "fixedseed"
        N = int(spins.shape[0])

        rad_cache = {}

        if mode == "fixedseed":
            arr = spins[...].astype(np.uint8)  # (N,H,W)
            Y_list = []
            for i in tqdm(range(N), desc=f"Potts descriptors fixed (nbins={nbins})", total=N):
                Y_list.append(potts_descriptor_radial(arr[i], nbins=nbins, _rad_cache=rad_cache))
            Y = np.stack(Y_list, axis=0).astype(np.float32)

        else:
            R = int(spins.shape[1])
            test = potts_descriptor_radial(spins[0, 0, ...].astype(np.uint8), nbins=nbins, _rad_cache=rad_cache)
            q = int(test.size)
            Y = np.empty((N, q), dtype=np.float32)

            for i0 in tqdm(range(0, N, chunk_n), desc=f"Potts descriptors stream (nbins={nbins})", total=(N + chunk_n - 1)//chunk_n):
                i1 = min(N, i0 + chunk_n)
                blk = spins[i0:i1, :, :, :].astype(np.uint8)  # (B,R,H,W)
                B = blk.shape[0]
                out = np.zeros((B, q), dtype=np.float64)
                for b in range(B):
                    acc = np.zeros((q,), dtype=np.float64)
                    for r in range(R):
                        d = potts_descriptor_radial(blk[b, r], nbins=nbins, _rad_cache=rad_cache).astype(np.float64, copy=False)
                        acc += d
                    out[b] = acc / float(R)
                Y[i0:i1] = out.astype(np.float32)

    cachep.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cachep, ctrl2=ctrl2, Y=Y)
    return ctrl2, Y


def _plot_heatmap(
    out_png: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.2, 5.2), dpi=dpi)
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    im = plt.imshow(grid, origin="lower", extent=extent, aspect="auto")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(im, label="Explained fraction (e)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="potts_data")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--nbins", type=int, default=64)

    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--lambda0", type=float, default=1e-3)
    ap.add_argument("--weights", type=str, default="uniform", choices=["gaussian", "uniform"])

    ap.add_argument("--device", type=str, default="cuda", help="cuda (default) / cpu / cuda:1 / auto")

    ap.add_argument("--heat_bins_x", type=int, default=60)
    ap.add_argument("--heat_bins_y", type=int, default=60)
    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--clip_qlo", type=float, default=0.01)
    ap.add_argument("--clip_qhi", type=float, default=0.99)

    ap.add_argument("--chunk_n", type=int, default=8)
    ap.add_argument("--force_recompute", action="store_true")
    ap.add_argument("--dpi", type=int, default=250)

    ap.add_argument("--only_warp", type=str, default="", choices=["", "none", "fold", "ribbon", "pinch"])
    ap.add_argument("--only_mode", type=str, default="", choices=["", "fixedseed", "repeated"])

    args = ap.parse_args()
    cfg = AnalysisConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        nbins=args.nbins,
        k=args.k,
        lambda0=args.lambda0,
        weights=args.weights,
        device=args.device,
        heat_bins_x=args.heat_bins_x,
        heat_bins_y=args.heat_bins_y,
        smooth_sigma=args.smooth_sigma,
        clip_qlo=args.clip_qlo,
        clip_qhi=args.clip_qhi,
        chunk_n=args.chunk_n,
        force_recompute=args.force_recompute,
        dpi=args.dpi,
    )

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else _default_outdir(data_dir)
    cache_dir = ensure_dir(out_dir / "cache")
    figs_dir = ensure_dir(out_dir / "figs")
    meta_dir = ensure_dir(out_dir / "meta")

    warps = ["none", "fold", "ribbon", "pinch"]
    modes = ["fixedseed", "repeated"]
    if args.only_warp:
        warps = [args.only_warp]
    if args.only_mode:
        modes = [args.only_mode]

    session_meta = {
        "created_utc": utcnow_iso(),
        "analysis": "potts_explained_fraction",
        "config": asdict(cfg),
    }
    (meta_dir / "session.json").write_text(json.dumps(session_meta, indent=2))

    for warp in warps:
        for mode in modes:
            run_id = _run_id(warp, mode)
            h5p = _h5_path(data_dir, warp, mode)
            if not h5p.exists():
                print(f"[skip] missing {h5p}")
                continue

            cachep = _cache_path(cache_dir, run_id, cfg.nbins)
            ctrl2, Y = _load_or_build_Y(
                h5p=h5p,
                cachep=cachep,
                nbins=cfg.nbins,
                chunk_n=cfg.chunk_n,
                force=cfg.force_recompute,
            )

            # Controls for explained fraction (use original (f0,T))
            X = ctrl2.astype(np.float32)

            e = compute_explained_fraction(
                X=X,
                Y=Y,
                k=cfg.k,
                lambda0=cfg.lambda0,
                weights=cfg.weights,
                device=cfg.device,
                standardize=True,
                verbose=True,
            )

            # Heatmap on (f0,T)
            x = ctrl2[:, 0]
            y = ctrl2[:, 1]
            x_edges = np.linspace(float(x.min()), float(x.max()), cfg.heat_bins_x + 1, dtype=np.float32)
            y_edges = np.linspace(float(y.min()), float(y.max()), cfg.heat_bins_y + 1, dtype=np.float32)

            grid, counts = bin2d_mean(x, y, e, x_edges, y_edges)
            if cfg.smooth_sigma > 0:
                grid = smooth_grid_nan(grid, sigma=cfg.smooth_sigma)
            grid = clip_quantiles(grid, qlo=cfg.clip_qlo, qhi=cfg.clip_qhi)

            # np.save(figs_dir / f"{run_id}__explained_fraction.npy", grid)
            # np.save(figs_dir / f"{run_id}__explained_fraction_counts.npy", counts)
            # np.save(figs_dir / f"{run_id}__explained_fraction_points.npy", e)

            title = f"Potts explained fraction e | warp={warp} | mode={mode} | nbins={cfg.nbins}"
            out_png = figs_dir / f"{run_id}__explained_fraction.png"
            _plot_heatmap(
                out_png=out_png,
                x_edges=x_edges,
                y_edges=y_edges,
                grid=grid,
                xlabel="f0 (original)",
                ylabel="T (original)",
                title=title,
                dpi=cfg.dpi,
            )

            run_meta = {
                "created_utc": utcnow_iso(),
                "run_id": run_id,
                "h5": str(h5p),
                "descriptor": "potts_radial_corr",
                "nbins": cfg.nbins,
                "k": cfg.k,
                "lambda0": cfg.lambda0,
                "weights": cfg.weights,
                "device": cfg.device,
            }
            (meta_dir / f"{run_id}.json").write_text(json.dumps(run_meta, indent=2))

            print(f"[ok] {run_id} -> {out_png}")


if __name__ == "__main__":
    main()
