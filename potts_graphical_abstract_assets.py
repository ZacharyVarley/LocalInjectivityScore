#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

import potts_analyze_explained_fraction_v13 as analysis
import potts_publication_figures as publication
import potts_visualize_control_space_descriptors_v2 as control_viz


WORKSPACE_ROOT = Path(__file__).resolve().parent
PCA_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Potts graphical-abstract assets for PowerPoint assembly."
    )
    parser.add_argument(
        "--publication-metadata",
        type=str,
        default="potts_figures/publication/potts_publication_figures_metadata.json",
        help="Publication metadata JSON produced by potts_publication_figures.py",
    )
    parser.add_argument(
        "--analysis-metadata",
        type=str,
        default=None,
        help="Optional injectivity metadata JSON. If omitted, the newest matching run is used.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="potts_figures/publication/graphical_abstract_assets",
    )
    parser.add_argument("--micro-count", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20260512)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--periodic", type=int, default=1)
    parser.add_argument("--remove-spurious", type=int, default=0)
    parser.add_argument(
        "--sim-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--transparent-points", type=int, default=1)
    return parser.parse_args()


def _resolve_path(raw_path: str | None, *, base_dir: Path | None = None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [WORKSPACE_ROOT / path]
    if base_dir is not None:
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _build_analysis_config(desc_h5: Path, metadata: Dict[str, Any] | None) -> analysis.Config:
    cfg_data = {} if metadata is None else dict(metadata.get("config", {}))
    cfg_data["h5"] = str(desc_h5)
    allowed = set(analysis.Config.__dataclass_fields__.keys())
    cfg_data = {key: value for key, value in cfg_data.items() if key in allowed}
    return analysis.Config(**cfg_data)


def _find_matching_analysis_metadata(desc_h5: Path, analysis_root: Path) -> Path | None:
    matches: list[tuple[str, Path]] = []
    for meta_path in analysis_root.glob("*/**/metadata_local_explainedcov_injectivity.json"):
        try:
            meta = _load_json(meta_path)
        except Exception:
            continue
        raw_h5 = (
            meta.get("summary", {}).get("input_descriptor_h5")
            or meta.get("config", {}).get("h5")
        )
        resolved_h5 = _resolve_path(raw_h5, base_dir=meta_path.parent)
        if resolved_h5 is None:
            continue
        if resolved_h5 == desc_h5.resolve():
            stamp = str(meta.get("created_utc", meta_path.parent.parent.parent.name))
            matches.append((stamp, meta_path.resolve()))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _load_explained_fraction_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=np.float64)
    if table.size == 0:
        raise ValueError(f"No rows found in explained-fraction table: {csv_path}")
    table = np.atleast_1d(table)
    temp = np.asarray(table["temperature"], dtype=np.float32)
    frac = np.asarray(table["fraction_initial"], dtype=np.float32)
    explained = np.asarray(table["explained_frac"], dtype=np.float32)
    return temp, frac, explained


def _load_cached_projection(
    desc_h5: Path,
    cfg: analysis.Config,
    analysis_meta_path: Path,
    analysis_meta: Dict[str, Any],
) -> np.ndarray | None:
    selected_components = int(
        analysis_meta.get("summary", {})
        .get("global_pca", {})
        .get("selected_components", 0)
    )
    pca_rel = analysis_meta.get("files", {}).get("pca_components")
    pca_path = _resolve_path(pca_rel, base_dir=analysis_meta_path.parent)
    if selected_components <= 0 or pca_path is None or not pca_path.exists():
        return None

    dtype = PCA_DTYPES[str(cfg.pca_dtype)]
    device = publication.resolve_device(str(cfg.pca_device))
    mu = analysis.compute_global_mean_Y(desc_h5, cfg, batch_size=int(cfg.pca_batch_size))
    Q = torch.load(str(pca_path), map_location="cpu")
    if not isinstance(Q, torch.Tensor) or Q.ndim != 2 or Q.shape[1] < selected_components:
        return None

    return analysis.project_Y_with_Q(
        desc_h5=desc_h5,
        cfg=cfg,
        mu=mu,
        Qm=Q[:, :selected_components].detach().clone(),
        batch_size=int(cfg.pca_batch_size),
        device=device,
        dtype=dtype,
    )


def _load_or_compute_projection(
    desc_h5: Path,
    cfg: analysis.Config,
    analysis_meta_path: Path | None,
    analysis_meta: Dict[str, Any] | None,
) -> np.ndarray:
    if analysis_meta_path is not None and analysis_meta is not None:
        Yp = _load_cached_projection(desc_h5, cfg, analysis_meta_path, analysis_meta)
        if Yp is not None:
            return Yp
    Yp, _ = publication.compute_projected_Y(desc_h5, cfg)
    return Yp


def _normalize_pcs(Yp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return pc1, pc2, pc3


def _select_sample_indices(
    temp: np.ndarray,
    frac: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("micro-count must be positive")
    if count > temp.shape[0]:
        raise ValueError("micro-count cannot exceed the number of controls")

    coords = np.column_stack([temp, frac]).astype(np.float64)
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / max(np.ptp(coords[:, 0]), 1e-12)
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / max(np.ptp(coords[:, 1]), 1e-12)

    rng = np.random.default_rng(seed)
    order = rng.permutation(coords.shape[0])
    selected: list[int] = []
    min_dist = 0.22
    for idx in order:
        if not selected:
            selected.append(int(idx))
        else:
            dists = np.linalg.norm(coords[idx] - coords[selected], axis=1)
            if np.all(dists >= min_dist):
                selected.append(int(idx))
        if len(selected) == count:
            return np.asarray(selected, dtype=np.int64)

    selected_set = set(selected)
    for idx in order:
        idx = int(idx)
        if idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)
        if len(selected) == count:
            break
    return np.asarray(selected[:count], dtype=np.int64)


def _simulate_microstructures(
    temp: np.ndarray,
    frac: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    device = publication.resolve_device(args.sim_device)
    temp_t = torch.as_tensor(temp, device=device, dtype=torch.float32)
    frac_t = torch.as_tensor(frac, device=device, dtype=torch.float32)

    torch.manual_seed(int(seed))
    spins0 = control_viz.create_initial_states(
        batch_size=temp.shape[0],
        grid_size=int(args.grid),
        fractions0=frac_t,
        q=int(args.q),
        device=device,
    )
    final_micro = control_viz.simulate_potts(
        spins=spins0,
        temperatures=temp_t,
        steps=int(args.steps),
        q=int(args.q),
        periodic=bool(int(args.periodic)),
        remove_spurious=bool(int(args.remove_spurious)),
    )
    return (
        final_micro[:, 0]
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
        / float(max(int(args.q) - 1, 1))
    )


def _save_recoverability_heatmap(
    outbase: Path,
    temp: np.ndarray,
    frac: np.ndarray,
    explained_frac: np.ndarray,
    cfg: analysis.Config,
    dpi: int,
) -> None:
    img_s, tx, fx = publication._build_binned_image(
        temp=temp,
        frac=frac,
        values=explained_frac,
        bins_t=int(cfg.hm_bins_temp),
        bins_f=int(cfg.hm_bins_frac),
        sigma_px=float(cfg.hm_sigma_px),
    )

    fig, ax = plt.subplots(figsize=(5.3, 4.6), dpi=dpi)
    im = ax.imshow(
        img_s.T,
        origin="lower",
        extent=[tx[0], tx[-1], fx[0], fx[-1]],
        aspect="auto",
        cmap="viridis",
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Explained Fraction", fontsize=10)
    ax.set_xlabel(r"$T$", fontsize=11)
    ax.set_ylabel(r"$f_0$", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(outbase) + ".png", bbox_inches="tight", dpi=dpi)
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _save_processing_space_samples(
    outbase: Path,
    temp: np.ndarray,
    frac: np.ndarray,
    sample_idx: np.ndarray | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.4), dpi=dpi)
    ax.scatter(temp, frac, s=18, c="#d2d2d2", alpha=0.8, edgecolors="none")

    if sample_idx is not None:
        colors = plt.get_cmap("tab10")(np.arange(sample_idx.shape[0]) % 10)
        ax.scatter(
            temp[sample_idx],
            frac[sample_idx],
            s=92,
            c=colors,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        frac_span = max(float(np.ptp(frac)), 1e-6)
        for rank, idx in enumerate(sample_idx, start=1):
            ax.text(
                float(temp[idx]),
                float(frac[idx]) + 0.025 * frac_span,
                str(rank),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel(r"$T$", fontsize=11)
    ax.set_ylabel(r"$f_0$", fontsize=11)
    ax.set_xlim(float(temp.min()) - 0.03, float(temp.max()) + 0.03)
    ax.set_ylim(float(frac.min()) - 0.03, float(frac.max()) + 0.03)
    fig.tight_layout()
    fig.savefig(str(outbase) + ".png", bbox_inches="tight", dpi=dpi)
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _save_microstructure_tiles(
    outdir: Path,
    microstructures: np.ndarray,
    temp: np.ndarray,
    frac: np.ndarray,
) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for rank, (field, temp_i, frac_i) in enumerate(zip(microstructures, temp, frac), start=1):
        stem = f"microstructure_{rank:02d}_T{temp_i:.3f}_f0_{frac_i:.3f}"
        path = outdir / f"{stem}.png"
        plt.imsave(path, field, cmap="gray", vmin=0.0, vmax=1.0)
        written.append(str(path))
    return written


def _style_3d_points_only(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    elev: float,
    azim: float,
) -> None:
    for setter, values in ((ax.set_xlim, x), (ax.set_ylim, y), (ax.set_zlim, z)):
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = max(hi - lo, 1e-6)
        pad = 0.08 * span
        setter(lo - pad, hi + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.15, 1.1, 0.9), zoom=1.45)
    except TypeError:
        ax.set_box_aspect((1.15, 1.1, 0.9))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()


def _save_synthetic_descriptor_cloud(outbase: Path, dpi: int, transparent: bool) -> None:
    np.random.seed(11)
    M = 600
    u = np.random.uniform(-1, 1, M)
    v = np.random.uniform(-1, 1, M)

    width_at_v = 1.0 - 0.35 * (v + 1.0) / 2.0
    X_main = u * width_at_v + 0.04 * np.random.randn(M)
    Y_main = v + 0.04 * np.random.randn(M)
    Z_main = 0.10 * u * v + 0.04 * np.random.randn(M)

    Mf = 140
    uf = np.random.uniform(-0.95, -0.55, Mf)
    Xf = 0.40 + 0.10 * np.random.randn(Mf)
    Yf = -0.45 + 0.12 * np.random.randn(Mf)
    Zf = 0.30 + 0.04 * np.random.randn(Mf)

    fig = plt.figure(figsize=(6.0, 4.7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X_main,
        Y_main,
        Z_main,
        c=u,
        cmap="Spectral_r",
        s=14,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.2,
        vmin=-1,
        vmax=1,
    )
    ax.scatter(
        Xf,
        Yf,
        Zf,
        c=uf,
        cmap="Spectral_r",
        s=14,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.2,
        vmin=-1,
        vmax=1,
    )
    x_all = np.concatenate([X_main, Xf])
    y_all = np.concatenate([Y_main, Yf])
    z_all = np.concatenate([Z_main, Zf])
    _style_3d_points_only(ax, x=x_all, y=y_all, z=z_all, elev=22, azim=-55)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(
        str(outbase) + ".png",
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        str(outbase) + ".pdf",
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _save_pca_descriptor_cloud(
    outbase: Path,
    Yp: np.ndarray,
    explained_frac: np.ndarray,
    dpi: int,
    transparent: bool,
) -> None:
    pc1, pc2, pc3 = _normalize_pcs(Yp)
    fig = plt.figure(figsize=(6.0, 4.7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pc1,
        pc2,
        pc3,
        c=explained_frac,
        cmap="viridis",
        s=22,
        alpha=0.65,
        linewidth=0.0,
        vmin=0.0,
        vmax=1.0,
    )
    _style_3d_points_only(ax, x=pc1, y=pc2, z=pc3, elev=20, azim=45)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(
        str(outbase) + ".png",
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        str(outbase) + ".pdf",
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    publication_meta_path = _resolve_path(args.publication_metadata)
    if publication_meta_path is None or not publication_meta_path.exists():
        raise FileNotFoundError("Publication metadata JSON was not found.")
    publication_meta = _load_json(publication_meta_path)

    desc_h5 = _resolve_path(str(publication_meta["descriptor_h5"]))
    if desc_h5 is None or not desc_h5.exists():
        raise FileNotFoundError("Descriptor HDF5 referenced by publication metadata was not found.")

    analysis_meta_path = _resolve_path(args.analysis_metadata)
    if analysis_meta_path is None:
        analysis_meta_path = _find_matching_analysis_metadata(
            desc_h5=desc_h5,
            analysis_root=WORKSPACE_ROOT / "potts_analysis",
        )
    analysis_meta = _load_json(analysis_meta_path) if analysis_meta_path is not None else None

    cfg = _build_analysis_config(desc_h5, analysis_meta)
    outdir = _resolve_path(args.outdir)
    if outdir is None:
        raise RuntimeError("Unable to resolve output directory.")
    outdir.mkdir(parents=True, exist_ok=True)

    temp: np.ndarray
    frac: np.ndarray
    explained_frac: np.ndarray
    if analysis_meta_path is not None and analysis_meta is not None:
        csv_rel = analysis_meta.get("files", {}).get("csv")
        csv_path = _resolve_path(csv_rel, base_dir=analysis_meta_path.parent)
        if csv_path is None or not csv_path.exists():
            raise FileNotFoundError("Cached explained-fraction CSV referenced in metadata was not found.")
        temp, frac, explained_frac = _load_explained_fraction_table(csv_path)
    else:
        X, Yp_tmp, metrics, _ = publication.compute_explained_fraction_metrics(desc_h5, cfg)
        temp = X[:, 0].astype(np.float32, copy=False)
        frac = X[:, 1].astype(np.float32, copy=False)
        explained_frac = metrics["explained_frac"].astype(np.float32, copy=False)
        Yp = Yp_tmp

    if "Yp" not in locals():
        Yp = _load_or_compute_projection(desc_h5, cfg, analysis_meta_path, analysis_meta)

    sample_idx = _select_sample_indices(temp, frac, int(args.micro_count), int(args.sample_seed))
    sample_temp = temp[sample_idx]
    sample_frac = frac[sample_idx]
    sample_explained = explained_frac[sample_idx]
    microstructures = _simulate_microstructures(sample_temp, sample_frac, args, int(args.sample_seed) + 17)

    _save_recoverability_heatmap(
        outbase=outdir / "recoverability_heatmap",
        temp=temp,
        frac=frac,
        explained_frac=explained_frac,
        cfg=cfg,
        dpi=int(args.dpi),
    )
    _save_processing_space_samples(
        outbase=outdir / "processing_space_random_samples",
        temp=temp,
        frac=frac,
        sample_idx=sample_idx,
        dpi=int(args.dpi),
    )
    _save_processing_space_samples(
        outbase=outdir / "processing_space_background_only",
        temp=temp,
        frac=frac,
        sample_idx=None,
        dpi=int(args.dpi),
    )
    micro_paths = _save_microstructure_tiles(
        outdir=outdir / "microstructures",
        microstructures=microstructures,
        temp=sample_temp,
        frac=sample_frac,
    )
    transparent = bool(int(args.transparent_points))
    _save_synthetic_descriptor_cloud(
        outbase=outdir / "descriptor_space_option1_synthetic_cloud",
        dpi=int(args.dpi),
        transparent=transparent,
    )
    _save_pca_descriptor_cloud(
        outbase=outdir / "descriptor_space_option2_pca_points",
        Yp=Yp,
        explained_frac=explained_frac,
        dpi=int(args.dpi),
        transparent=transparent,
    )

    metadata = {
        "created_utc": analysis._utc_now_z(),
        "publication_metadata": str(publication_meta_path),
        "analysis_metadata": str(analysis_meta_path) if analysis_meta_path is not None else None,
        "descriptor_h5": str(desc_h5),
        "outdir": str(outdir),
        "sample_seed": int(args.sample_seed),
        "sample_indices": sample_idx.tolist(),
        "files": {
            "recoverability_heatmap_png": str(outdir / "recoverability_heatmap.png"),
            "recoverability_heatmap_pdf": str(outdir / "recoverability_heatmap.pdf"),
            "processing_space_random_samples_png": str(outdir / "processing_space_random_samples.png"),
            "processing_space_random_samples_pdf": str(outdir / "processing_space_random_samples.pdf"),
            "processing_space_background_only_png": str(outdir / "processing_space_background_only.png"),
            "processing_space_background_only_pdf": str(outdir / "processing_space_background_only.pdf"),
            "descriptor_space_option1_png": str(outdir / "descriptor_space_option1_synthetic_cloud.png"),
            "descriptor_space_option1_pdf": str(outdir / "descriptor_space_option1_synthetic_cloud.pdf"),
            "descriptor_space_option2_png": str(outdir / "descriptor_space_option2_pca_points.png"),
            "descriptor_space_option2_pdf": str(outdir / "descriptor_space_option2_pca_points.pdf"),
        },
        "samples": [
            {
                "rank": rank,
                "index": int(idx),
                "temperature": float(temp[idx]),
                "fraction_initial": float(frac[idx]),
                "explained_fraction": float(explained_frac[idx]),
                "microstructure_png": path,
            }
            for rank, (idx, path) in enumerate(zip(sample_idx, micro_paths), start=1)
        ],
    }
    (outdir / "graphical_abstract_assets_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved graphical abstract assets to {outdir}")


if __name__ == "__main__":
    main()