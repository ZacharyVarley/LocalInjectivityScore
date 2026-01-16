#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch_local_identifiability.py

CH local identifiability diagnostics via local explained covariance (LOO-scored dual ridge),
INVERSE direction only: predict centered controls X from centered outcomes Y using Y-neighborhoods.

CRITICAL CONVENTION (fixed here):
  - X is the original 2D sheet coordinates: X = h5["sheet_u2"]  (N,2) in [-1,1]^2
  - Y is outcomes: Y = h5["fields"]       (N,H,W) or (N,R,H,W)

Descriptors:
  - fixed mode (fields: (N,H,W)):
      raw_flat: flattened (optionally downsampled) final field

  - repeats mode (fields: (N,R,H,W)):
      radial1d:   radial profile g(r) from repeat-mean 2D autocorr g(dx,dy)
      corr2d:     flattened repeat-mean 2D autocorr g(dx,dy) (downsampled), weighted by 1/r^power

Notes:
  - 2D autocorrelation is computed via FFT and returned in spatial-lag domain after fftshift.
  - Weighting is in spatial-lag coordinates: w(dx,dy) = 1 / max(r,1)^power

Outputs:
  out_root/<timestamp>/<warp>_<mode>/<descriptor>/inv/
    metrics.csv
    meta.json
    plots (png/pdf)

Dependencies: numpy, torch, h5py, tqdm, ch_plot_utils.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ch_plot_utils import scatter_2d, heatmap_binned_2d


# ---------------------- defaults ----------------------
CFG: Dict[str, Any] = dict(
    ch_data_root="ch_ab_data",
    out_root="ch_ab_analysis",

    # X MUST be the original 2D sheet coordinates in [-1,1]^2
    x_key="sheet_u2",   # (N,2)
    y_key="fields",     # (N,H,W) or (N,R,H,W)

    standardize_X=False,
    standardize_Y=False,

    downsample_fixed=1,

    descriptor="all",  # radial1d | corr2d | all (repeats mode only)
    n_radial_bins=64,

    corr2d_downsample=1,
    corr2d_weight_power=2.0,   # w = 1/max(r,1)^power

    kY=20,
    knn_chunk=512,

    use_weights=False,
    eps_tau=1e-12,

    ridge_inv=1e-3,
    ridge_x=1e-8,
    eps_trace=1e-18,

    y_feature_chunk=4096,
    batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=1337,

    xdim0=0,
    xdim1=1,
    dpi=300,
    hm_bins0=60,
    hm_bins1=60,
    hm_sigma_px=1.0,
)

# ---------------------- misc utils ----------------------
def standardize_np(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (A - mu) / sd

def json_sanitize(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    if isinstance(x, dict): return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_sanitize(v) for v in x]
    return x

def read_h5_attrs(f: h5py.File) -> Dict[str, Any]:
    out = {}
    for k, v in f.attrs.items():
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8")
            except Exception:
                v = str(v)
        if k == "config" and isinstance(v, str):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                out[k] = {"raw_config_str": v}
                continue
        out[k] = v
    return out


# ---------------------- timestamp discovery ----------------------
def find_most_recent_timestamp(ch_root: Path) -> Optional[Path]:
    if not ch_root.exists():
        return None
    timestamp_dirs = []
    for d in ch_root.iterdir():
        if d.is_dir() and len(d.name) == 15 and "_" in d.name:
            parts = d.name.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                timestamp_dirs.append(d)
    if not timestamp_dirs:
        return None
    timestamp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return timestamp_dirs[0]

def discover_h5_files_in_timestamp(timestamp_dir: Path) -> List[Tuple[str, str, Path]]:
    files = []
    for h5_path in timestamp_dir.glob("*.h5"):
        name = h5_path.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            warp, mode = parts
            files.append((warp, mode, h5_path))
    files.sort(key=lambda t: (t[0], t[1]))
    return files


# ---------------------- kNN (chunked cdist) ----------------------
@torch.no_grad()
def knn_chunked(A: torch.Tensor, k: int, chunk: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    device = A.device
    N = A.shape[0]
    if k >= N:
        k = N - 1
    idx_out = torch.empty((N, k), device=device, dtype=torch.int64)
    d_out = torch.empty((N, k), device=device, dtype=torch.float32)

    for s in tqdm(range(0, N, chunk), desc="kNN(cdist) chunks", leave=False):
        e = min(N, s + chunk)
        Dc = torch.cdist(A[s:e], A, p=2.0)

        r = torch.arange(e - s, device=device)
        c = torch.arange(s, e, device=device)
        Dc[r, c] = float("inf")

        vals, idx = torch.topk(Dc, k=k, largest=False, sorted=False)
        ord_local = torch.argsort(vals, dim=1)
        vals_s = torch.gather(vals, dim=1, index=ord_local)
        idx_s = torch.gather(idx, dim=1, index=ord_local)

        idx_out[s:e] = idx_s
        d_out[s:e] = vals_s

    return idx_out, d_out


# ---------------------- autocorrelation + weighting ----------------------
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

@torch.no_grad()
def make_corr2d_weight(H: int, W: int, power: float, device: torch.device) -> torch.Tensor:
    """
    Weight matrix w(dx,dy) = 1 / max(r,1)^power in fftshifted spatial-lag coordinates.
    """
    cy = H // 2
    cx = W // 2
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - float(cx)) ** 2 + (y - float(cy)) ** 2)
    r = torch.clamp(r, min=1.0)
    w = 1.0 / (r ** float(power))
    return w.to(torch.float32)


# ---------------------- radial averaging ----------------------
@torch.no_grad()
def radial_average_scalar(corr2d: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    corr2d: (B,H,W) fftshifted
    returns: (B,n_bins)
    """
    B, H, W = corr2d.shape
    cy = H // 2
    cx = W // 2
    max_r = min(cy, cx)

    y, x = torch.meshgrid(
        torch.arange(H, device=corr2d.device, dtype=torch.float32),
        torch.arange(W, device=corr2d.device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - float(cx)) ** 2 + (y - float(cy)) ** 2).flatten()

    edges = torch.linspace(0.0, float(max_r), int(n_bins) + 1, device=corr2d.device)
    b = torch.bucketize(r, edges, right=False) - 1
    b = b.clamp(0, int(n_bins) - 1)

    flat_val = corr2d.reshape(B, H * W)
    sums = torch.zeros((B, int(n_bins)), device=corr2d.device, dtype=torch.float32)
    cnts = torch.zeros((B, int(n_bins)), device=corr2d.device, dtype=torch.float32)

    idx = b.unsqueeze(0).expand(B, -1)
    sums.scatter_add_(1, idx, flat_val.to(torch.float32))
    cnts.scatter_add_(1, idx, torch.ones_like(flat_val, dtype=torch.float32))

    return torch.where(cnts > 0, sums / cnts, torch.zeros_like(sums))


# ---------------------- Y builders ----------------------
@torch.no_grad()
def build_Y_fixed_raw(fields: np.ndarray, downsample: int, device_pool: torch.device) -> np.ndarray:
    N, H, W = fields.shape
    ds = int(max(1, downsample))
    if ds > 1 and ((H % ds) != 0 or (W % ds) != 0):
        raise RuntimeError(f"downsample_fixed={ds} must divide H,W exactly (H={H}, W={W})")

    out = []
    chunk = 128
    for s in tqdm(range(0, N, chunk), desc="build_Y_fixed_raw chunks", leave=False):
        e = min(N, s + chunk)
        t = torch.from_numpy(fields[s:e]).to(device_pool, dtype=torch.float32)
        if ds > 1:
            t = F.avg_pool2d(t.unsqueeze(1), kernel_size=ds, stride=ds).squeeze(1)
        out.append(t.reshape(t.shape[0], -1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)

@torch.no_grad()
def build_Y_repeats_descriptor(
    fields: np.ndarray,                  # (N,R,H,W)
    descriptor: str,
    n_radial_bins: int,
    corr2d_downsample: int,
    corr2d_weight_power: float,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    N, R, H, W = fields.shape
    meta: Dict[str, Any] = dict(descriptor=descriptor, N=int(N), R=int(R), H=int(H), W=int(W))

    if descriptor == "radial1d":
        nb = int(n_radial_bins)
        meta["n_radial_bins"] = nb
        Y = np.empty((N, nb), dtype=np.float32)

        for i in tqdm(range(N), desc="build radial1d", leave=False):
            x = torch.from_numpy(fields[i]).to(device=device, dtype=torch.float32)  # (R,H,W)
            c2 = autocorr2d_fft(x)             # (R,H,W)
            c2m = c2.mean(dim=0, keepdim=True) # (1,H,W)
            rad = radial_average_scalar(c2m, n_bins=nb)  # (1,nb)
            Y[i] = rad[0].detach().cpu().numpy().astype(np.float32)

        # prepend to the radial descriptor the mean field value
        mean_field = fields.mean(axis=(1, 2, 3))  # (N,)
        Y = np.hstack([mean_field[:, None].astype(np.float32), Y])

        meta["Y_shape"] = [int(N), int(nb) + 1]  # +1 for mean field
        return Y, meta

    elif descriptor == "corr2d":
        ds = int(max(1, corr2d_downsample))
        if ds > 1 and ((H % ds) != 0 or (W % ds) != 0):
            raise RuntimeError(f"corr2d_downsample={ds} must divide H,W exactly (H={H}, W={W})")

        H2, W2 = (H // ds, W // ds) if ds > 1 else (H, W)
        meta["corr2d_downsample"] = ds
        meta["corr2d_shape"] = [int(H2), int(W2)]
        meta["corr2d_weight_power"] = float(corr2d_weight_power)

        w = make_corr2d_weight(H2, W2, power=float(corr2d_weight_power), device=device)  # (H2,W2)
        corr_flat = np.empty((N, H2 * W2), dtype=np.float32)

        for i in tqdm(range(N), desc="build corr2d surfaces", leave=False):
            x = torch.from_numpy(fields[i]).to(device=device, dtype=torch.float32)  # (R,H,W)
            c2 = autocorr2d_fft(x)     # (R,H,W)
            cmean = c2.mean(dim=0)     # (H,W)

            if ds > 1:
                cmean = F.avg_pool2d(cmean.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds)[0, 0]  # (H2,W2)

            cmean_w = cmean * w
            corr_flat[i] = cmean_w.reshape(-1).detach().cpu().numpy().astype(np.float32)

        
        # prepend to the correlation descriptor the mean field value
        mean_field = fields.mean(axis=(1, 2, 3))  # (N,)
        corr_flat = np.hstack([mean_field[:, None].astype(np.float32), corr_flat])

        if descriptor == "corr2d":
            meta["Y_shape"] = [int(N), int(H2 * W2) + 1] # +1 for mean field
            return corr_flat, meta
        
    else:
        raise KeyError(f"Unknown descriptor '{descriptor}' (expected radial1d|corr2d)")


# ---------------------- inverse LOO ridge metrics ----------------------
@torch.no_grad()
def local_inverse_metrics_LOO_chunkY(
    X: torch.Tensor,           # (N,p)
    Y: torch.Tensor,           # (N,q)
    idxY: torch.Tensor,        # (N,kY)
    dY: torch.Tensor,          # (N,kY)
    use_weights: bool,
    eps_tau: float,
    ridge_inv: float,
    ridge_x: float,
    eps_trace: float,
    y_feature_chunk: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    device = X.device
    N, p = X.shape
    kY = idxY.shape[1]
    k = kY + 1

    unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    expl = torch.empty((N,), device=device, dtype=torch.float32)
    trX = torch.empty((N,), device=device, dtype=torch.float32)
    trR = torch.empty((N,), device=device, dtype=torch.float32)
    worst_unexpl = torch.empty((N,), device=device, dtype=torch.float32)
    worst_ret = torch.empty((N,), device=device, dtype=torch.float32)
    unexpl_coord_max = torch.empty((N,), device=device, dtype=torch.float32)
    unexpl_coords = torch.empty((N, p), device=device, dtype=torch.float32)
    avg_dy = torch.empty((N,), device=device, dtype=torch.float32)

    I_k = torch.eye(k, device=device, dtype=torch.float32)
    I_p = torch.eye(p, device=device, dtype=torch.float32)

    q = Y.shape[1]
    qc = int(max(256, y_feature_chunk))

    for i0 in tqdm(range(0, N, batch_size), desc="inverse LOO batches", leave=False):
        i1 = min(N, i0 + batch_size)
        B = i1 - i0

        centers = torch.arange(i0, i1, device=device, dtype=torch.int64)
        neigh = torch.cat([centers[:, None], idxY[i0:i1]], dim=1)  # (B,k)

        dn = torch.cat(
            [torch.zeros((B, 1), device=device, dtype=torch.float32), dY[i0:i1].to(torch.float32)],
            dim=1,
        )
        avg_dy[i0:i1] = dn[:, 1:].mean(dim=1)

        Xn = X[neigh]

        if use_weights:
            tau = dn.max(dim=1).values.clamp_min(eps_tau)
            w = torch.exp(-0.5 * (dn / tau[:, None]).pow(2)).clamp_min(1e-12)
        else:
            w = torch.ones((B, k), device=device, dtype=torch.float32)

        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-18)
        sw = torch.sqrt(w).to(torch.float32)

        muX = (w[:, :, None] * Xn).sum(dim=1)
        Xc = Xn.to(torch.float32) - muX[:, None, :]
        Xs = Xc * sw[:, :, None]

        Kmat = torch.zeros((B, k, k), device=device, dtype=torch.float32)
        for q0 in range(0, q, qc):
            q1 = min(q, q0 + qc)
            Yn_blk = Y[neigh, q0:q1]
            muY_blk = (w[:, :, None] * Yn_blk).sum(dim=1, keepdim=True)
            Yc_blk = Yn_blk.to(torch.float32) - muY_blk
            Ys_blk = Yc_blk * sw[:, :, None]
            Kmat += torch.bmm(Ys_blk, Ys_blk.transpose(1, 2))

        trK = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
        lam = (ridge_inv * trK / float(k)).to(torch.float32)
        Kreg = Kmat + lam[:, None, None] * I_k[None, :, :]

        Hinv = torch.linalg.solve(Kreg, I_k[None, :, :].expand(B, k, k))
        alpha = torch.bmm(Hinv, Xs)

        hdiag = Hinv.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
        Rloo = alpha / hdiag[:, :, None]

        trX_b = (Xs * Xs).sum(dim=(1, 2)).clamp_min(0.0)
        trR_b = (Rloo * Rloo).sum(dim=(1, 2)).clamp_min(0.0)

        u = (trR_b / (trX_b + eps_trace)).clamp(0.0, 1.0)
        e = (1.0 - u).clamp(0.0, 1.0)

        trX[i0:i1] = trX_b
        trR[i0:i1] = trR_b
        unexpl[i0:i1] = u
        expl[i0:i1] = e

        varX = (Xs * Xs).sum(dim=1)
        varR = (Rloo * Rloo).sum(dim=1)
        ucoord = (varR / (varX + eps_trace)).clamp(0.0, 1.0)
        unexpl_coords[i0:i1, :] = ucoord
        unexpl_coord_max[i0:i1] = ucoord.max(dim=1).values

        SigmaX = torch.bmm(Xs.transpose(1, 2), Xs)
        SigmaR = torch.bmm(Rloo.transpose(1, 2), Rloo)
        gam = (ridge_x * trX_b / float(max(p, 1))).to(torch.float32)
        SigmaXr = SigmaX + gam[:, None, None] * I_p[None, :, :]

        L = torch.linalg.cholesky(SigmaXr)
        Z = torch.linalg.solve_triangular(L, SigmaR, upper=False)
        M = torch.linalg.solve_triangular(L.transpose(1, 2), Z, upper=True)
        M = 0.5 * (M + M.transpose(1, 2))
        ev = torch.linalg.eigvalsh(M)
        wmax = ev[:, -1].clamp(0.0, 1.0)

        worst_unexpl[i0:i1] = wmax
        worst_ret[i0:i1] = (1.0 - wmax).clamp(0.0, 1.0)

    return dict(
        inv_unexplained_frac=unexpl.detach().cpu().numpy(),
        inv_explained_frac=expl.detach().cpu().numpy(),
        inv_trX=trX.detach().cpu().numpy(),
        inv_trR=trR.detach().cpu().numpy(),
        inv_worst_unexplained_ratio=worst_unexpl.detach().cpu().numpy(),
        inv_worst_retention=worst_ret.detach().cpu().numpy(),
        inv_unexplained_coord_max=unexpl_coord_max.detach().cpu().numpy(),
        inv_unexplained_coords=unexpl_coords.detach().cpu().numpy(),
        inv_avg_dY=avg_dy.detach().cpu().numpy(),
    )


# ---------------------- output helpers ----------------------
def write_csv(path: Path, header_cols: List[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(header_cols), comments="")

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_sanitize(obj), f, indent=2)

def plot_explained_maps(
    outdir: Path,
    title_base: str,
    X: np.ndarray,
    z: np.ndarray,
    xdim0: int,
    xdim1: int,
    dpi: int,
    hm_bins0: int,
    hm_bins1: int,
    hm_sigma_px: float,
    prefix: str,
) -> None:
    p = X.shape[1]
    if not (0 <= xdim0 < p and 0 <= xdim1 < p):
        raise ValueError(f"xdim0/xdim1 must be in [0,{p-1}] (got {xdim0},{xdim1})")

    x0 = X[:, xdim0]
    x1 = X[:, xdim1]
    xlabel = f"X_dim{xdim0}"
    ylabel = f"X_dim{xdim1}"

    z = z.astype(np.float64)
    zmin = float(np.nanmin(z)) if np.isfinite(np.nanmin(z)) else 0.0
    zmax = float(np.nanmax(z)) if np.isfinite(np.nanmax(z)) else 1.0
    if zmin == zmax:
        zmax = zmin + 1e-6

    tasks = [
        ("scatter_adaptive", lambda: scatter_2d(x0, x1, z, xlabel, ylabel, f"{title_base} (adaptive)",
                                               outdir / f"{prefix}_scatter_adaptive", dpi=dpi, vmin=zmin, vmax=zmax)),
        ("heatmap_adaptive", lambda: heatmap_binned_2d(x0, x1, z, xlabel, ylabel, f"{title_base} heatmap (adaptive)",
                                                      outdir / f"{prefix}_heatmap_adaptive",
                                                      bins0=hm_bins0, bins1=hm_bins1, sigma_px=hm_sigma_px, dpi=dpi,
                                                      vmin=zmin, vmax=zmax)),
        ("scatter_fixed01",  lambda: scatter_2d(x0, x1, z, xlabel, ylabel, f"{title_base} (fixed [0,1])",
                                               outdir / f"{prefix}_scatter_fixed01", dpi=dpi, vmin=0.0, vmax=1.0)),
        ("heatmap_fixed01",  lambda: heatmap_binned_2d(x0, x1, z, xlabel, ylabel, f"{title_base} heatmap (fixed [0,1])",
                                                      outdir / f"{prefix}_heatmap_fixed01",
                                                      bins0=hm_bins0, bins1=hm_bins1, sigma_px=hm_sigma_px, dpi=dpi,
                                                      vmin=0.0, vmax=1.0)),
    ]
    for name, fn in tqdm(tasks, desc="plotting", leave=False):
        print(f"[PLOT] {outdir.parent.parent.name}/{outdir.parent.name}/{outdir.name}: {prefix} -> {name}")
        fn()


# ---------------------- pipeline ----------------------
def run_descriptor_block(
    h5_path: Path,
    warp: str,
    mode: str,
    timestamp: str,
    cfg: Dict[str, Any],
    desc_key: str,
    X: np.ndarray,
    Y: np.ndarray,
    attrs: Dict[str, Any],
    y_meta: Dict[str, Any],
) -> None:
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    device = torch.device(str(cfg["device"]))

    N, p = X.shape
    q = Y.shape[1]

    inv_dir = Path(cfg["out_root"]) / timestamp / f"{warp}_{mode}" / desc_key / "inv"
    inv_dir.mkdir(parents=True, exist_ok=True)

    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    Y_t = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

    print(f"[KNN] {warp}_{mode} | {desc_key} | Y-neighborhoods (N={N}, q={q})")
    idxY_t, dY_t = knn_chunked(Y_t, k=int(cfg["kY"]), chunk=int(cfg["knn_chunk"]))

    print(f"[METRICS] {warp}_{mode} | {desc_key} | inverse LOO ridge")
    inv = local_inverse_metrics_LOO_chunkY(
        X=X_t,
        Y=Y_t,
        idxY=idxY_t,
        dY=dY_t,
        use_weights=bool(cfg["use_weights"]),
        eps_tau=float(cfg["eps_tau"]),
        ridge_inv=float(cfg["ridge_inv"]),
        ridge_x=float(cfg["ridge_x"]),
        eps_trace=float(cfg["eps_trace"]),
        y_feature_chunk=int(cfg["y_feature_chunk"]),
        batch_size=int(cfg["batch_size"]),
    )

    x_cols = [f"x_dim{j}" for j in range(p)]
    cols = x_cols + [
        "inv_unexplained_frac",
        "inv_explained_frac",
        "inv_worst_unexplained_ratio",
        "inv_worst_retention",
        "inv_unexplained_coord_max",
        "inv_trX",
        "inv_trR",
        "inv_avg_dY",
    ] + [f"inv_unexplained_coord{j}" for j in range(p)]

    M = np.concatenate(
        [
            X,
            inv["inv_unexplained_frac"][:, None],
            inv["inv_explained_frac"][:, None],
            inv["inv_worst_unexplained_ratio"][:, None],
            inv["inv_worst_retention"][:, None],
            inv["inv_unexplained_coord_max"][:, None],
            inv["inv_trX"][:, None],
            inv["inv_trR"][:, None],
            inv["inv_avg_dY"][:, None],
            inv["inv_unexplained_coords"],
        ],
        axis=1,
    )
    write_csv(inv_dir / "metrics.csv", cols, M)

    xdim0 = int(cfg["xdim0"])
    xdim1 = int(cfg["xdim1"])
    plot_explained_maps(
        outdir=inv_dir,
        title_base=f"Inverse locality e_inv (Y-neigh -> X) | {warp} | {mode} | Y={desc_key} | q={q}",
        X=X,
        z=inv["inv_explained_frac"],
        xdim0=xdim0,
        xdim1=xdim1,
        dpi=int(cfg["dpi"]),
        hm_bins0=int(cfg["hm_bins0"]),
        hm_bins1=int(cfg["hm_bins1"]),
        hm_sigma_px=float(cfg["hm_sigma_px"]),
        prefix="e_inv",
    )

    summary = dict(
        input_file=str(h5_path.resolve()),
        timestamp=str(timestamp),
        warp=str(warp),
        mode=str(mode),
        descriptor=str(desc_key),
        N=int(N),
        p=int(p),
        y_dim=int(q),
        device=str(device),
        x_key=str(cfg["x_key"]),
        y_key=str(cfg["y_key"]),
        kY=int(cfg["kY"]),
        use_weights=bool(cfg["use_weights"]),
        ridge_inv=float(cfg["ridge_inv"]),
        ridge_x=float(cfg["ridge_x"]),
        eps_trace=float(cfg["eps_trace"]),
        y_feature_chunk=int(cfg["y_feature_chunk"]),
        plot_dims=dict(xdim0=int(xdim0), xdim1=int(xdim1)),
        stats=dict(
            inv_expl_median=float(np.median(inv["inv_explained_frac"])),
            inv_expl_q10=float(np.quantile(inv["inv_explained_frac"], 0.10)),
            inv_expl_q90=float(np.quantile(inv["inv_explained_frac"], 0.90)),
            inv_worst_ret_median=float(np.median(inv["inv_worst_retention"])),
        ),
    )
    meta_out = dict(config=cfg, summary=summary, h5_attrs=attrs, y_meta=y_meta)
    write_json(inv_dir / "meta.json", meta_out)

    print(f"[OK] {warp}_{mode} | {desc_key} -> {inv_dir}")


def run_one_h5(h5_path: Path, warp: str, mode: str, timestamp: str, cfg: Dict[str, Any]) -> None:
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    device = torch.device(str(cfg["device"]))
    pool_device = device if device.type == "cuda" else torch.device("cpu")

    with h5py.File(str(h5_path.resolve()), "r") as f:
        if cfg["x_key"] not in f:
            raise KeyError(f"X key '{cfg['x_key']}' not found in {h5_path}")
        if cfg["y_key"] not in f:
            raise KeyError(f"Y key '{cfg['y_key']}' not found in {h5_path}")

        X0 = np.array(f[cfg["x_key"]], dtype=np.float32)
        Y0 = np.array(f[cfg["y_key"]], dtype=np.float32)
        attrs = read_h5_attrs(f)

    # Enforce X=sheet_u2 (2D sheet coords).
    if cfg["x_key"] != "sheet_u2":
        raise RuntimeError(f"x_key must be 'sheet_u2' for this analysis script; got '{cfg['x_key']}'")

    if X0.ndim != 2 or X0.shape[1] != 2:
        raise RuntimeError(f"Expected X='{cfg['x_key']}' to be (N,2); got {X0.shape}")
    if Y0.ndim not in (3, 4):
        raise RuntimeError(f"Expected Y='{cfg['y_key']}' to be (N,H,W) or (N,R,H,W); got {Y0.shape}")
    if int(X0.shape[0]) != int(Y0.shape[0]):
        raise RuntimeError(f"X/Y mismatch: X {X0.shape} vs Y {Y0.shape} in {h5_path}")

    X = X0
    Yraw = Y0

    if bool(cfg["standardize_X"]):
        X = standardize_np(X)

    detected_mode = "fixed" if Yraw.ndim == 3 else "repeats"

    if detected_mode == "fixed":
        print(f"[BUILD] {warp}_{mode} | raw_flat (fixed)")
        Yemb = build_Y_fixed_raw(Yraw, downsample=int(cfg["downsample_fixed"]), device_pool=pool_device)
        y_meta = dict(
            mode="fixed",
            Y_raw_shape=list(Yraw.shape),
            Y_shape=list(Yemb.shape),
            downsample_fixed=int(cfg["downsample_fixed"]),
        )
        if bool(cfg["standardize_Y"]):
            Yemb = standardize_np(Yemb)

        run_descriptor_block(
            h5_path=h5_path,
            warp=warp,
            mode=mode,
            timestamp=timestamp,
            cfg=cfg,
            desc_key="raw_flat",
            X=X,
            Y=Yemb,
            attrs=attrs,
            y_meta=y_meta,
        )
        return

    desc_req = str(cfg["descriptor"]).strip().lower()
    if desc_req == "all":
        desc_list = ["radial1d", "corr2d"]
    else:
        desc_list = [desc_req]

    for desc in desc_list:
        print(f"[BUILD] {warp}_{mode} | {desc} (repeats)")
        Yemb, y_meta = build_Y_repeats_descriptor(
            fields=Yraw,
            descriptor=desc,
            n_radial_bins=int(cfg["n_radial_bins"]),
            corr2d_downsample=int(cfg["corr2d_downsample"]),
            corr2d_weight_power=float(cfg["corr2d_weight_power"]),
            device=device,
        )
        y_meta["mode"] = "repeats"
        y_meta["Y_raw_shape"] = list(Yraw.shape)

        if bool(cfg["standardize_Y"]):
            Yemb = standardize_np(Yemb)

        run_descriptor_block(
            h5_path=h5_path,
            warp=warp,
            mode=mode,
            timestamp=timestamp,
            cfg=cfg,
            desc_key=desc,
            X=X,
            Y=Yemb,
            attrs=attrs,
            y_meta=y_meta,
        )


# ---------------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Local identifiability (inverse-only) with LOO dual ridge in Y-neighborhoods.")
    p.add_argument("--ch_data_root", type=str, default=CFG["ch_data_root"])
    p.add_argument("--out_root", type=str, default=CFG["out_root"])
    p.add_argument("--h5", type=str, default=None, help="Optional explicit H5 path. If set, timestamp='manual'.")
    p.add_argument("--device", type=str, default=CFG["device"])

    p.add_argument("--x_key", type=str, default=CFG["x_key"])
    p.add_argument("--y_key", type=str, default=CFG["y_key"])

    p.add_argument("--descriptor", type=str, default=CFG["descriptor"])
    p.add_argument("--downsample_fixed", type=int, default=CFG["downsample_fixed"])

    p.add_argument("--n_radial_bins", type=int, default=CFG["n_radial_bins"])
    p.add_argument("--corr2d_downsample", type=int, default=CFG["corr2d_downsample"])
    p.add_argument("--corr2d_weight_power", type=float, default=CFG["corr2d_weight_power"])

    p.add_argument("--kY", type=int, default=CFG["kY"])
    p.add_argument("--knn_chunk", type=int, default=CFG["knn_chunk"])
    p.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    p.add_argument("--use_weights", type=int, default=int(CFG["use_weights"]))

    p.add_argument("--ridge_inv", type=float, default=CFG["ridge_inv"])
    p.add_argument("--ridge_x", type=float, default=CFG["ridge_x"])
    p.add_argument("--y_feature_chunk", type=int, default=CFG["y_feature_chunk"])

    p.add_argument("--standardize_X", type=int, default=int(CFG["standardize_X"]))
    p.add_argument("--standardize_Y", type=int, default=int(CFG["standardize_Y"]))

    p.add_argument("--xdim0", type=int, default=CFG["xdim0"])
    p.add_argument("--xdim1", type=int, default=CFG["xdim1"])
    p.add_argument("--dpi", type=int, default=CFG["dpi"])
    p.add_argument("--hm_bins0", type=int, default=CFG["hm_bins0"])
    p.add_argument("--hm_bins1", type=int, default=CFG["hm_bins1"])
    p.add_argument("--hm_sigma_px", type=float, default=CFG["hm_sigma_px"])

    p.add_argument("--seed", type=int, default=CFG["seed"])
    return p.parse_args()

def main():
    args = parse_args()
    cfg = dict(CFG)

    cfg["ch_data_root"] = args.ch_data_root
    cfg["out_root"] = args.out_root
    cfg["device"] = args.device

    cfg["x_key"] = str(args.x_key)
    cfg["y_key"] = str(args.y_key)

    cfg["descriptor"] = str(args.descriptor).strip().lower()
    cfg["downsample_fixed"] = int(args.downsample_fixed)

    cfg["n_radial_bins"] = int(args.n_radial_bins)
    cfg["corr2d_downsample"] = int(args.corr2d_downsample)
    cfg["corr2d_weight_power"] = float(args.corr2d_weight_power)

    cfg["kY"] = int(args.kY)
    cfg["knn_chunk"] = int(args.knn_chunk)
    cfg["batch_size"] = int(args.batch_size)
    cfg["use_weights"] = bool(int(args.use_weights))

    cfg["ridge_inv"] = float(args.ridge_inv)
    cfg["ridge_x"] = float(args.ridge_x)
    cfg["y_feature_chunk"] = int(args.y_feature_chunk)

    cfg["standardize_X"] = bool(int(args.standardize_X))
    cfg["standardize_Y"] = bool(int(args.standardize_Y))

    cfg["xdim0"] = int(args.xdim0)
    cfg["xdim1"] = int(args.xdim1)
    cfg["dpi"] = int(args.dpi)
    cfg["hm_bins0"] = int(args.hm_bins0)
    cfg["hm_bins1"] = int(args.hm_bins1)
    cfg["hm_sigma_px"] = float(args.hm_sigma_px)

    cfg["seed"] = int(args.seed)

    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    if args.h5 is not None:
        h5_path = Path(args.h5).expanduser().resolve()
        name = h5_path.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            warp, mode = parts
        else:
            warp, mode = "unknown", "unknown"
        timestamp = "manual"
        print(f"[RUN] manual | {warp}_{mode} | {h5_path}")
        run_one_h5(h5_path, warp=warp, mode=mode, timestamp=timestamp, cfg=cfg)
        return

    ch_root = Path(cfg["ch_data_root"]).expanduser().resolve()
    timestamp_dir = find_most_recent_timestamp(ch_root)
    if timestamp_dir is None:
        raise FileNotFoundError(f"No timestamp directories found under {ch_root}")

    timestamp = timestamp_dir.name
    h5_files = discover_h5_files_in_timestamp(timestamp_dir)
    if not h5_files:
        raise FileNotFoundError(f"No *.h5 files found in {timestamp_dir}")

    print(f"[RUN] timestamp={timestamp} | files={len(h5_files)} | root={timestamp_dir}")
    for warp, mode, h5_path in tqdm(h5_files, desc="H5 files"):
        print(f"[FILE] {warp}_{mode} | {h5_path.name}")
        run_one_h5(h5_path, warp=warp, mode=mode, timestamp=timestamp, cfg=cfg)

if __name__ == "__main__":
    main()
