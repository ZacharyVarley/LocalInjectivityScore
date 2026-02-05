#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pca_3d_interactive.py

Interactive 3D visualization of PCA-projected data using Plotly.
Self-contained script that loads analysis outputs and projects data in real-time.

Usage:
  python plot_pca_3d_interactive.py --analysis_dir <path_to_analysis_session> --h5 <descriptor_h5>
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from tqdm import tqdm


def find_latest_analysis_dir(base_dir: str = "potts_analysis") -> Path:
    """
    Find the most recent analysis directory.
    Expected structure: potts_analysis/YYYYMMDD_HHMMSSZ/stem/descriptor/
    Returns the descriptor directory (lowest level).
    """
    base = Path(base_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base analysis directory not found: {base}")
    
    # Find all descriptor directories (4 levels deep)
    descriptor_dirs = sorted(
        base.glob("*/*/*/metadata_local_explainedcov_injectivity.json"),
        reverse=True
    )
    
    if not descriptor_dirs:
        raise FileNotFoundError(
            f"No analysis directories found in {base}\n"
            f"Expected structure: {base}/YYYYMMDD_HHMMSSZ/stem/descriptor/"
        )
    
    latest = descriptor_dirs[0].parent
    print(f"Found latest analysis: {latest}")
    return latest


def iter_Y_batches_simple(
    desc_h5: Path,
    prepend_phases: bool = True,
    batch_size: int = 256,
):
    """
    Stream Y batches from H5 file.
    Yields (N, Y_batch) where Y_batch is (B, y_dim)
    """
    with h5py.File(str(desc_h5), "r") as f:
        descriptor = str(f.attrs.get("descriptor", "corr2d"))
        N = int(f.attrs.get("n_parameters"))
        
        meanph_ds = f["phases/final_fraction_mean"]
        mean2d_ds = f["correlations/correlations_2d_mean"]
        mean1d_ds = f["correlations/correlations_radial_mean"]
        
        if descriptor == "radial1d":
            feat_dim = int(np.prod(mean1d_ds.shape[1:]))
        else:
            feat_dim = int(np.prod(mean2d_ds.shape[1:]))
        
        ph_dim = int(meanph_ds.shape[1]) if len(meanph_ds.shape) == 2 else int(meanph_ds.shape[-1])
        
        for i0 in range(0, N, int(batch_size)):
            i1 = min(N, i0 + int(batch_size))
            B = i1 - i0
            
            meanph = np.array(meanph_ds[i0:i1], dtype=np.float32).reshape(B, -1)
            
            if descriptor == "radial1d":
                feat = np.array(mean1d_ds[i0:i1], dtype=np.float32).reshape(B, -1)
            else:
                feat = np.array(mean2d_ds[i0:i1], dtype=np.float32).reshape(B, -1)
            
            if prepend_phases:
                Yb = np.concatenate([meanph, feat], axis=1)
            else:
                Yb = feat
            
            yield i0, i1, Yb


def project_data_with_components(
    desc_h5: Path,
    pca_components_path: Path,
    pca_mean_path: Optional[Path] = None,
    prepend_phases: bool = True,
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """
    Load data from H5, apply PCA mean centering, and project onto PCA components.
    Returns: (N, m) projected data
    """
    # Load PCA components
    Q = torch.load(pca_components_path, map_location="cpu")
    Q = Q.to(device=device, dtype=torch.float32)
    
    # Load PCA mean if available
    if pca_mean_path and pca_mean_path.exists():
        mu = np.load(pca_mean_path)
    else:
        # Compute mean from data
        with h5py.File(str(desc_h5), "r") as f:
            N = int(f.attrs.get("n_parameters"))
        
        mu = np.zeros((0,), dtype=np.float32)
        n = 0
        
        for _, _, Yb in tqdm(iter_Y_batches_simple(desc_h5, prepend_phases, batch_size),
                             desc="Computing mean", unit="batch"):
            if mu.size == 0:
                mu = np.zeros(Yb.shape[1], dtype=np.float64)
            B = Yb.shape[0]
            n_new = n + B
            batch_mean = Yb.mean(axis=0, dtype=np.float64)
            mu += (B / float(n_new)) * (batch_mean - mu)
            n = n_new
        
        mu = mu.astype(np.float32)
    
    mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32)
    
    # Get total count
    with h5py.File(str(desc_h5), "r") as f:
        N = int(f.attrs.get("n_parameters"))
    
    m = int(Q.shape[1])
    Yp = np.empty((N, m), dtype=np.float32)
    
    # Project batches
    for i0, i1, Yb in tqdm(iter_Y_batches_simple(desc_h5, prepend_phases, batch_size),
                           total=int(np.ceil(N / batch_size)),
                           desc="Projecting PCA", unit="batch"):
        Yb_t = torch.as_tensor(Yb, device=device, dtype=torch.float32)
        Yb_c = Yb_t - mu_t
        proj = Yb_c @ Q
        Yp[i0:i1, :] = proj.detach().to("cpu").numpy()
    
    return Yp


def create_interactive_3d_plot(
    temps: np.ndarray,
    fracs: np.ndarray,
    Yp: np.ndarray,
    expl_frac: np.ndarray,
    outpath: Path,
    title: str = "3D PCA Projection - Potts Injectivity Analysis",
) -> None:
    """
    Create interactive 3D scatter plot using Plotly.
    
    Parameters:
      temps: (N,) temperature
      fracs: (N,) fraction_initial
      Yp: (N, m) PCA-projected data (use first 3 dims)
      expl_frac: (N,) explained fraction metric (for coloring)
      outpath: output HTML file path
      title: plot title
    """
    N = len(temps)
    
    # Extract first 3 PCA dimensions
    if Yp.shape[1] >= 3:
        pc1 = Yp[:, 0]
        pc2 = Yp[:, 1]
        pc3 = Yp[:, 2]
    elif Yp.shape[1] == 2:
        pc1 = Yp[:, 0]
        pc2 = Yp[:, 1]
        pc3 = np.zeros(N)
    else:
        pc1 = Yp[:, 0]
        pc2 = np.zeros(N)
        pc3 = np.zeros(N)
    
    # Normalize for better visualization
    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    pc2 = (pc2 - pc2.mean()) / (pc2.std() + 1e-8)
    pc3 = (pc3 - pc3.mean()) / (pc3.std() + 1e-8)
    
    # Create hover text with parameter values
    hover_text = [
        f"<b>Index {i}</b><br>" +
        f"T = {temps[i]:.3f}<br>" +
        f"f₀ = {fracs[i]:.3f}<br>" +
        f"Explained Frac = {expl_frac[i]:.4f}<br>" +
        f"PC1 = {pc1[i]:.3f}<br>" +
        f"PC2 = {pc2[i]:.3f}<br>" +
        f"PC3 = {pc3[i]:.3f}"
        for i in range(N)
    ]
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=pc1,
        y=pc2,
        z=pc3,
        mode='markers',
        marker=dict(
            size=4,
            color=expl_frac,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Explained<br>Fraction",
                thickness=15,
                len=0.7,
            ),
            line=dict(width=0.5, color='white'),
            opacity=0.8,
        ),
        text=hover_text,
        hoverinfo='text',
        name='Data points',
    )])
    
    # Update layout for publication-ready appearance
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=14)),
        scene=dict(
            xaxis=dict(title='PC1', backgroundcolor="rgb(230,230,230)", gridcolor="white"),
            yaxis=dict(title='PC2', backgroundcolor="rgb(230,230,230)", gridcolor="white"),
            zaxis=dict(title='PC3', backgroundcolor="rgb(230,230,230)", gridcolor="white"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
            ),
        ),
        width=1000,
        height=800,
        hovermode='closest',
        font=dict(family="Arial, sans-serif", size=11),
    )
    
    # Save to HTML
    fig.write_html(str(outpath))
    print(f"✓ Interactive plot saved to: {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create interactive 3D PCA visualization from Potts analysis outputs"
    )
    ap.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Path to analysis session directory (default: auto-find latest)"
    )
    ap.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Path to descriptor H5 file (default: read from analysis metadata)"
    )
    ap.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Output HTML file path (default: <analysis_dir>/pca_3d_interactive.html)"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for data streaming"
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for projection (cuda or cpu)"
    )
    
    args = ap.parse_args()
    
    # Determine analysis directory
    if args.analysis_dir is None:
        analysis_dir = find_latest_analysis_dir()
        print(f"Auto-selected analysis directory")
    else:
        analysis_dir = Path(args.analysis_dir).resolve()
    
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    
    # Load metadata to get H5 path if not provided
    meta_path = analysis_dir / "metadata_local_explainedcov_injectivity.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Determine H5 path
    if args.h5 is None:
        h5_path = Path(meta["summary"]["input_descriptor_h5"]).resolve()
        print(f"Using H5 from metadata: {h5_path}")
    else:
        h5_path = Path(args.h5).resolve()
    
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    
    # Determine output path
    if args.outpath is None:
        outpath = analysis_dir / "pca_3d_interactive.html"
    else:
        outpath = Path(args.outpath).resolve()
    
    print(f"Loading analysis from: {analysis_dir}")
    print(f"Loading data from: {h5_path}")
    
    csv_path = analysis_dir / "potts_local_explainedcov_injectivity.csv"
    df = pd.read_csv(csv_path)
    temps = df["temperature"].values.astype(np.float32)
    fracs = df["fraction_initial"].values.astype(np.float32)
    expl_frac = df["explained_frac"].values.astype(np.float32)
    
    print(f"Loaded {len(temps)} samples")
    print(f"Explained fraction range: [{expl_frac.min():.4f}, {expl_frac.max():.4f}]")
    
    # Load PCA components and project data
    pca_components_path = analysis_dir / "pca_components.pt"
    if not pca_components_path.exists():
        raise FileNotFoundError(f"PCA components not found: {pca_components_path}")
    
    pca_mean_path = analysis_dir / "pca_mean.npy"
    if not pca_mean_path.exists():
        pca_mean_path = None
    
    print(f"\nProjecting data onto PCA components...")
    Yp = project_data_with_components(
        desc_h5=h5_path,
        pca_components_path=pca_components_path,
        pca_mean_path=pca_mean_path,
        prepend_phases=meta.get("summary", {}).get("prepend_phases", True),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    
    print(f"Projected data shape: {Yp.shape}")
    
    # Create interactive plot
    descriptor = meta["load_metadata"].get("descriptor", "unknown")
    title = f"3D PCA Projection - {descriptor.upper()} Injectivity Analysis"
    
    print(f"\nCreating interactive visualization...")
    create_interactive_3d_plot(
        temps=temps,
        fracs=fracs,
        Yp=Yp,
        expl_frac=expl_frac,
        outpath=outpath,
        title=title,
    )
    
    print(f"\n✓ Done! Open in browser: {outpath}")


if __name__ == "__main__":
    main()

