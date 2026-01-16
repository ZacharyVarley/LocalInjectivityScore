
#!/usr/bin/env python3
"""
Shared helpers for injectivity analysis:
- Descriptor builders (CH + Potts)
- Local CCA-inspired explained-fraction metric (dual ridge + LOO)
- Heatmap binning and plotting utilities

Design goals:
- Streaming-friendly for large repeated datasets (HDF5)
- No averaging in Fourier domain: compute correlation surfaces in real space then average
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import h5py
import numpy as np

import torch

# Optional (but expected) deps
from tqdm import tqdm

try:
    from scipy.ndimage import gaussian_filter
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    gaussian_filter = None
    ndi = None


Array = np.ndarray


# ----------------------------- Device helpers -----------------------------

def default_device(requested: str = "cuda") -> torch.device:
    """Resolve a torch.device.

    requested:
      - "cuda" (default): use CUDA if available else CPU
      - "cpu": force CPU
      - "auto": same as "cuda"
      - "cuda:1" etc supported
    """
    if requested is None:
        requested = "cuda"
    req = str(requested).lower()
    if req == "auto":
        req = "cuda"
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def torch_zscore(x: Array, device: torch.device, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score along axis=0 on the target device."""
    xt = torch.as_tensor(x, device=device, dtype=torch.float32)
    mu = xt.mean(dim=0)
    sd = xt.std(dim=0, unbiased=False)
    sd = torch.where(sd < eps, torch.ones_like(sd), sd)
    return (xt - mu) / sd, mu, sd


# ----------------------------- I/O helpers -----------------------------

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_h5(path: Path, mode: str = "r") -> h5py.File:
    return h5py.File(str(path), mode)


# ----------------------------- Standardization -----------------------------

def zscore(x: Array, eps: float = 1e-12) -> Tuple[Array, Array, Array]:
    """
    Z-score features across axis=0.
    Returns: xz, mean, std
    """
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd, mu, sd


# ----------------------------- Correlations (real-space averaged, FFT used per sample only) -----------------------------

@dataclass(frozen=True)
class RadialBinner:
    h: int
    w: int
    nbins: int = 64

    def __post_init__(self):
        if self.h != self.w:
            raise ValueError("RadialBinner currently assumes square grids for simplicity.")

    def precompute(self) -> Tuple[Array, Array]:
        H = self.h
        c = (H // 2, H // 2)
        yy, xx = np.indices((H, H))
        rr = np.sqrt((yy - c[0]) ** 2 + (xx - c[1]) ** 2)
        rmax = (H / 2.0)
        bins = np.linspace(0.0, rmax, self.nbins + 1, dtype=np.float32)
        # bin index in [0, nbins-1]
        idx = np.clip(np.digitize(rr.ravel(), bins) - 1, 0, self.nbins - 1).astype(np.int32)
        counts = np.bincount(idx, minlength=self.nbins).astype(np.float32)
        return idx, counts


def corr2d_autocorr(field: Array) -> Array:
    """
    2D autocorrelation surface (g-like), computed per-sample using FFT but returned in real space.
    The surface is fftshift'ed so zero-separation is at the center.
    Normalization is by number of pixels.
    """
    f = field.astype(np.float32, copy=False)
    f = f - f.mean(dtype=np.float64)
    F = np.fft.fft2(f)
    C = np.fft.ifft2(F * np.conj(F)).real
    C /= f.size
    C = np.fft.fftshift(C)
    return C.astype(np.float32, copy=False)


def corr2d_crosscorr(a: Array, b: Array) -> Array:
    """
    2D cross-correlation surface between two scalar fields a and b, both centered by mean.
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a = a - a.mean(dtype=np.float64)
    b = b - b.mean(dtype=np.float64)
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    C = np.fft.ifft2(A * np.conj(B)).real
    C /= a.size
    C = np.fft.fftshift(C)
    return C.astype(np.float32, copy=False)


def radial_average(surface: Array, binner: RadialBinner, _cache: dict) -> Array:
    key = (binner.h, binner.nbins)
    if key not in _cache:
        idx, counts = binner.precompute()
        _cache[key] = (idx, counts)
    idx, counts = _cache[key]
    vals = surface.ravel().astype(np.float64)
    sums = np.bincount(idx, weights=vals, minlength=binner.nbins).astype(np.float64)
    prof = (sums / np.maximum(counts, 1.0)).astype(np.float32)
    return prof


# ----------------------------- Euler characteristic curve (optional, spatial-domain, not Fourier) -----------------------------

def euler_characteristic_curve(
    field: Array,
    n_levels: int = 64,
    connectivity: Literal[4, 8] = 4,
    value_range: Optional[Tuple[float, float]] = None,
) -> Array:
    """
    Euler characteristic curve EC(t) over threshold levels, using connected-component counting.
    EC = (#components of mask) - (#holes), where holes are components of the inverse mask not connected to border.

    This is more expensive than correlations; keep optional.

    Requires scipy.ndimage.
    """
    if ndi is None:
        raise ImportError("scipy is required for Euler characteristic curves (scipy.ndimage).")

    f = field.astype(np.float32, copy=False)
    if value_range is None:
        lo = float(np.min(f))
        hi = float(np.max(f))
    else:
        lo, hi = value_range

    levels = np.linspace(lo, hi, n_levels, dtype=np.float32)
    out = np.empty((n_levels,), dtype=np.float32)

    if connectivity == 4:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int8)
    elif connectivity == 8:
        structure = np.ones((3,3), dtype=np.int8)
    else:
        raise ValueError("connectivity must be 4 or 8")

    for i, t in enumerate(levels):
        mask = f > t
        lbl, ncomp = ndi.label(mask, structure=structure)

        inv = ~mask
        inv_lbl, ninv = ndi.label(inv, structure=structure)
        # find labels that touch border
        border = np.concatenate([
            inv_lbl[0, :], inv_lbl[-1, :], inv_lbl[:, 0], inv_lbl[:, -1]
        ])
        border_labels = np.unique(border)
        # holes are inverse components not touching border and not label 0
        holes = int(ninv - (len(border_labels) - (1 if 0 in border_labels else 0)))
        out[i] = float(ncomp - holes)

    return out


# ----------------------------- Descriptor builders -----------------------------

def ch_descriptor(
    field: Array,
    kind: Literal["raw", "corr2d", "radial", "euler", "concat"] = "raw",
    nbins: int = 64,
    euler_levels: int = 64,
    _rad_cache: Optional[dict] = None,
) -> Array:
    if _rad_cache is None:
        _rad_cache = {}
    H, W = field.shape
    if kind == "raw":
        return field.astype(np.float32, copy=False).ravel()
    if kind == "corr2d":
        return corr2d_autocorr(field).ravel()
    if kind == "radial":
        surf = corr2d_autocorr(field)
        binner = RadialBinner(H, W, nbins=nbins)
        return radial_average(surf, binner, _rad_cache)
    if kind == "euler":
        return euler_characteristic_curve(field, n_levels=euler_levels)
    if kind == "concat":
        surf = corr2d_autocorr(field)
        binner = RadialBinner(H, W, nbins=nbins)
        rad = radial_average(surf, binner, _rad_cache)
        ec = euler_characteristic_curve(field, n_levels=euler_levels)
        return np.concatenate([rad, ec], axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown CH descriptor kind: {kind}")


def potts_descriptor_radial(
    spins: Array,  # (H,W) uint8 in {0,1,2}
    nbins: int = 64,
    _rad_cache: Optional[dict] = None,
) -> Array:
    """
    Pairwise (i<=j) centered indicator cross-correlations, radially averaged.
    Returns vector length 6*nbins in order: (00,01,02,11,12,22).
    """
    if _rad_cache is None:
        _rad_cache = {}
    H, W = spins.shape
    binner = RadialBinner(H, W, nbins=nbins)

    # indicators
    out = []
    inds = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    for (i, j) in inds:
        a = (spins == i).astype(np.float32)
        b = (spins == j).astype(np.float32)
        surf = corr2d_crosscorr(a, b)  # centered inside corr2d_crosscorr
        prof = radial_average(surf, binner, _rad_cache)
        out.append(prof)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


# ----------------------------- Explained fraction metric (CCA-inspired local inverse test) -----------------------------

def compute_explained_fraction(
    X: Array,  # (N,p) standardized or not (see standardize)
    Y: Array,  # (N,q) standardized or not (see standardize)
    k: int = 15,
    lambda0: float = 1e-3,
    weights: Literal["gaussian", "uniform"] = "gaussian",
    eps: float = 1e-12,
    device: str = "cuda",
    metric: Literal["l2", "l1"] = "l2",
    standardize: bool = False,
    verbose: bool = True,
) -> Array:
    """
    Compute pointwise explained fraction e_i in [0,1] as:
      e_i = 1 - ||R_i||^2 / (||X_i||^2 + eps)
    where R_i are LOO residuals from local kernel ridge regression (dual), using neighborhoods in outcome space Y.

    Notes:
    - Neighborhoods are in Y (inverse question).
    - Weighted centering is applied per neighborhood.
    - Kernel is linear in Y after centering: K = Ys Ys^T (dual of linear ridge in Y).
    """
    N = X.shape[0]
    if N != Y.shape[0]:
        raise ValueError("X and Y must have same N")

    dev = default_device(device)
    # Optionally standardize on-device (recommended for large Y)
    if standardize:
        Xt, _, _ = torch_zscore(X, device=dev, eps=eps)
        Yt, _, _ = torch_zscore(Y, device=dev, eps=eps)
    else:
        Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
        Yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)

    # kNN in Y (exact, dense) on device
    pval = 2.0 if metric == "l2" else 1.0
    D = torch.cdist(Yt, Yt, p=pval)
    D.fill_diagonal_(float("inf"))
    neigh_d, neigh_idx = torch.topk(D, k=k, dim=1, largest=False, sorted=True)  # (N,k)

    e_out = torch.empty((N,), device=dev, dtype=torch.float32)
    it = range(N)
    if verbose:
        it = tqdm(it, desc=f"explained_fraction[{dev}]", total=N)

    I = torch.eye(k + 1, device=dev, dtype=torch.float32)

    for i in it:
        nidx = neigh_idx[i]  # (k,)
        J = torch.cat([torch.tensor([i], device=dev, dtype=torch.long), nidx], dim=0)  # (k+1,)

        Xb = Xt.index_select(0, J)  # (k+1,p)
        Yb = Yt.index_select(0, J)  # (k+1,q)

        # distances to anchor for weights: include 0 for self
        di = torch.cat([torch.zeros((1,), device=dev), neigh_d[i]], dim=0)  # (k+1,)
        if weights == "uniform":
            w = torch.full((k + 1,), 1.0 / float(k + 1), device=dev, dtype=torch.float32)
        else:
            tau = torch.max(di)
            if float(tau) < eps:
                w = torch.full((k + 1,), 1.0 / float(k + 1), device=dev, dtype=torch.float32)
            else:
                ww = torch.exp(-0.5 * (di / tau) ** 2)
                ww = ww / torch.sum(ww)
                w = ww
        sw = torch.sqrt(w).unsqueeze(1)  # (k+1,1)

        mux = torch.sum(w.unsqueeze(1) * Xb, dim=0)
        muy = torch.sum(w.unsqueeze(1) * Yb, dim=0)

        Xs = (Xb - mux) * sw
        Ys = (Yb - muy) * sw

        K = Ys @ Ys.t()  # (k+1,k+1)
        lam = float(lambda0) * (torch.trace(K) / float(k + 1))
        H = K + lam * I

        # alpha = H^{-1} Xs
        alpha = torch.linalg.solve(H, Xs)
        # G = H^{-1} for LOO diagonal
        G = torch.linalg.inv(H)
        diagG = torch.diagonal(G)
        R = alpha / diagG.unsqueeze(1)

        num = torch.sum(R * R)
        den = torch.sum(Xs * Xs) + eps
        ei = 1.0 - (num / den)
        e_out[i] = torch.clamp(ei, 0.0, 1.0)

    return e_out.detach().cpu().numpy().astype(np.float32)


# ----------------------------- Heatmap binning + smoothing -----------------------------

def bin2d_mean(
    x: Array,
    y: Array,
    v: Array,
    x_edges: Array,
    y_edges: Array,
) -> Tuple[Array, Array]:
    """
    Bin scattered (x,y,v) to a grid via mean per bin.
    Returns (grid, counts) where empty bins are NaN in grid.
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    sums = np.zeros((ny, nx), dtype=np.float64)
    cnts = np.zeros((ny, nx), dtype=np.int64)

    for a, b, val in zip(xi, yi, v.astype(np.float64)):
        sums[b, a] += val
        cnts[b, a] += 1

    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    mask = cnts > 0
    grid[mask] = (sums[mask] / cnts[mask]).astype(np.float32)
    return grid, cnts.astype(np.int32)


def smooth_grid_nan(grid: Array, sigma: float = 1.0) -> Array:
    """
    Gaussian smooth a grid with NaNs by smoothing numerator and denominator separately.
    Requires scipy.ndimage.gaussian_filter.
    """
    if gaussian_filter is None:
        raise ImportError("scipy is required for smoothing (scipy.ndimage.gaussian_filter).")

    g = grid.astype(np.float64)
    m = np.isfinite(g).astype(np.float64)
    g0 = np.where(np.isfinite(g), g, 0.0)
    num = gaussian_filter(g0, sigma=sigma)
    den = gaussian_filter(m, sigma=sigma)
    out = np.full_like(g, np.nan, dtype=np.float64)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out.astype(np.float32)


def clip_quantiles(grid: Array, qlo: float = 0.01, qhi: float = 0.99) -> Array:
    vals = grid[np.isfinite(grid)]
    if vals.size == 0:
        return grid
    lo = float(np.quantile(vals, qlo))
    hi = float(np.quantile(vals, qhi))
    return np.clip(grid, lo, hi).astype(np.float32, copy=False)
