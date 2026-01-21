
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

import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Literal

import h5py
import numpy as np
import torch

from tqdm import tqdm

from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi

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

@torch.no_grad()
def make_corr2d_weight_torch(H: int, W: int, power: float, device: torch.device) -> torch.Tensor:
    """
    Weight matrix w(dx,dy) = 1 / max(r,1)^power in fftshifted spatial-lag coordinates.
    (Matches CH convention.) :contentReference[oaicite:3]{index=3}
    """
    cy = H // 2
    cx = W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    r = torch.clamp(r, min=1.0)
    return (1.0 / (r ** float(power))).to(torch.float32)


@torch.no_grad()
def corr2d_fftshifted_crosscorr_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cross-correlation surface in spatial-lag domain (fftshifted).
    a,b: (...,H,W) float32
    Returns: (...,H,W) float32

    Centering is internal: subtract mean over H,W from each.
    """
    *prefix, H, W = a.shape
    a0 = a - a.mean(dim=(-2, -1), keepdim=True)
    b0 = b - b.mean(dim=(-2, -1), keepdim=True)

    A = torch.fft.rfft2(a0, dim=(-2, -1))
    B = torch.fft.rfft2(b0, dim=(-2, -1))
    C = torch.fft.irfft2(A * torch.conj(B), s=(H, W), dim=(-2, -1)).real
    C = C / float(H * W)
    C = torch.fft.fftshift(C, dim=(-2, -1))
    return C.to(torch.float32)


# ----------------------------- Torch radial averaging (cached binning) -----------------------------

@dataclass(frozen=True)
class _RadialCacheKey:
    H: int
    W: int
    nbins: int

@torch.no_grad()
def _radial_bins_torch(H: int, W: int, nbins: int, device: torch.device, cache: Dict[Any, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (H, W, nbins, device.type, device.index)
    if key in cache:
        return cache[key]

    cy = H // 2
    cx = W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2).flatten()
    max_r = float(min(cy, cx))
    edges = torch.linspace(0.0, max_r, int(nbins) + 1, device=device)
    b = torch.bucketize(r, edges, right=False) - 1
    b = b.clamp(0, int(nbins) - 1).to(torch.int64)

    ones = torch.ones_like(r, dtype=torch.float32)
    counts = torch.zeros((int(nbins),), device=device, dtype=torch.float32)
    counts.scatter_add_(0, b, ones)

    cache[key] = (b, counts)
    return b, counts


@torch.no_grad()
def radial_average_torch(surfaces: torch.Tensor, nbins: int, cache: Dict[Any, Any]) -> torch.Tensor:
    """
    surfaces: (B,S,H,W) fftshifted
    returns:  (B,S,nbins)
    """
    B, S, H, W = surfaces.shape
    b, counts = _radial_bins_torch(H, W, nbins, surfaces.device, cache)  # (H*W,), (nbins,)
    vals = surfaces.reshape(B * S, H * W).to(torch.float32)

    idx = b.unsqueeze(0).expand(B * S, -1)  # (B*S, H*W)
    sums = torch.zeros((B * S, int(nbins)), device=surfaces.device, dtype=torch.float32)
    sums.scatter_add_(1, idx, vals)
    prof = sums / torch.clamp(counts.unsqueeze(0), min=1.0)
    return prof.reshape(B, S, int(nbins))


# ----------------------------- Potts descriptors: corr2d and radial1d -----------------------------

_PottsDescKind = Literal["corr2d", "radial1d"]

@torch.no_grad()
def potts_build_Y_from_spins(
    spins_np: np.ndarray,                 # (B,H,W) or (B,R,H,W) uint8
    kind: _PottsDescKind,
    nbins: int = 64,
    corr2d_downsample: int = 1,
    corr2d_weight_power: float = 2.0,
    device: str = "cuda",
    _rad_cache: Optional[Dict[Any, Any]] = None,
) -> np.ndarray:
    """
    Potts descriptor builder.

    Fractions convention:
      prepend [phi0, phi1, phi2] where phi_s = mean(1[spin==s]) averaged across repeats if present.

    corr2d convention:
      build 6 centered indicator cross-correlations:
        (00,01,02,11,12,22), each as fftshifted spatial-lag surface g(dx,dy),
      then downsample (avg-pool), apply radial weight w=1/max(r,1)^power, flatten, concatenate.

    Output sizes:
      - corr2d:   (B, 3 + 6*H2*W2)
      - radial1d: (B, 3 + 6*nbins)
    """
    if _rad_cache is None:
        _rad_cache = {}

    dev = default_device(device)
    t = torch.from_numpy(spins_np)
    if t.ndim == 3:
        t = t.unsqueeze(1)  # (B,1,H,W)
    if t.ndim != 4:
        raise ValueError(f"spins must be (B,H,W) or (B,R,H,W); got {spins_np.shape}")

    t = t.to(device=dev, dtype=torch.int64)
    B, R, H, W = t.shape

    # masks: (B,R,H,W) float32
    m0 = (t == 0).to(torch.float32)
    m1 = (t == 1).to(torch.float32)
    m2 = (t == 2).to(torch.float32)

    # fractions per repeat: (B,R); then mean over repeats -> (B,)
    f0 = m0.mean(dim=(-2, -1))
    f1 = m1.mean(dim=(-2, -1))
    f2 = m2.mean(dim=(-2, -1))
    fracs = torch.stack([f0.mean(dim=1), f1.mean(dim=1), f2.mean(dim=1)], dim=1)  # (B,3)

    # centered indicators per repeat (so corr is covariance-like, consistent with CH mean-prepend convention :contentReference[oaicite:4]{index=4})
    m0c = m0 - f0.unsqueeze(-1).unsqueeze(-1)
    m1c = m1 - f1.unsqueeze(-1).unsqueeze(-1)
    m2c = m2 - f2.unsqueeze(-1).unsqueeze(-1)

    # FFTs: only 3 forward FFTs for 6 pair surfaces
    F0 = torch.fft.rfft2(m0c, dim=(-2, -1))
    F1 = torch.fft.rfft2(m1c, dim=(-2, -1))
    F2 = torch.fft.rfft2(m2c, dim=(-2, -1))

    def _ifft_prod(P: torch.Tensor) -> torch.Tensor:
        C = torch.fft.irfft2(P, s=(H, W), dim=(-2, -1)) / float(H * W)
        return torch.fft.fftshift(C, dim=(-2, -1)).to(torch.float32)

    C00 = _ifft_prod(F0 * torch.conj(F0))
    C01 = _ifft_prod(F0 * torch.conj(F1))
    C02 = _ifft_prod(F0 * torch.conj(F2))
    C11 = _ifft_prod(F1 * torch.conj(F1))
    C12 = _ifft_prod(F1 * torch.conj(F2))
    C22 = _ifft_prod(F2 * torch.conj(F2))

    # stack as (B,R,6,H,W) then mean over repeats -> (B,6,H,W)
    S = torch.stack([C00, C01, C02, C11, C12, C22], dim=2).mean(dim=1)

    if kind == "radial1d":
        prof = radial_average_torch(S, nbins=int(nbins), cache=_rad_cache)  # (B,6,nbins)
        y = torch.cat([fracs, prof.reshape(B, -1)], dim=1)
        return y.detach().cpu().numpy().astype(np.float32)

    # corr2d flattened
    ds = int(max(1, corr2d_downsample))
    if ds > 1 and ((H % ds) != 0 or (W % ds) != 0):
        raise RuntimeError(f"corr2d_downsample={ds} must divide H,W exactly (H={H}, W={W})")

    if ds > 1:
        # pool each surface independently: (B*6,1,H,W) -> (B*6,1,H2,W2) -> (B,6,H2,W2)
        S2 = S.reshape(B * 6, 1, H, W)
        S2 = F.avg_pool2d(S2, kernel_size=ds, stride=ds)
        H2, W2 = S2.shape[-2], S2.shape[-1]
        S = S2.reshape(B, 6, H2, W2)
    else:
        H2, W2 = H, W

    if float(corr2d_weight_power) > 0.0:
        w = make_corr2d_weight_torch(int(H2), int(W2), power=float(corr2d_weight_power), device=dev)
        S = S * w.unsqueeze(0).unsqueeze(0)

    y = torch.cat([fracs, S.reshape(B, -1)], dim=1)
    # y = S.reshape(B, -1)
    return y.detach().cpu().numpy().astype(np.float32)


# ----------------------------- Scalable explained fraction (inverse direction, chunked) -----------------------------

@torch.no_grad()
def knn_chunked_torch(A: torch.Tensor, k: int, chunk: int = 512, p: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact kNN via chunked torch.cdist to cap peak memory.
    Returns (idx, dist): (N,k), (N,k).
    """
    device = A.device
    N = A.shape[0]
    if k >= N:
        k = N - 1
    idx_out = torch.empty((N, k), device=device, dtype=torch.int64)
    d_out = torch.empty((N, k), device=device, dtype=torch.float32)

    for s in tqdm(range(0, N, int(chunk)), desc="kNN(cdist) chunks", leave=False):
        e = min(N, s + int(chunk))
        Dc = torch.cdist(A[s:e], A, p=float(p))
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


# ---------------------- inverse LOO ridge metrics ----------------------
@torch.no_grad()
def local_inverse_metrics_LOO_chunkY(
    X: torch.Tensor,           # (N,p)
    Y: torch.Tensor,           # (N,q)
    idxY: torch.Tensor,        # (N,kY)
    dY: torch.Tensor,          # (N,kY)
    use_weights: bool,
    eps_tau: float,
    ridge_inv: float = 1e-3,
    ridge_x: float = 1e-8,
    eps_trace: float = 1e-18,
    y_feature_chunk: int = 4096,
    batch_size: int = 256,
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


def compute_explained_fraction(
    X: Array,
    Y: Array,
    k: int = 15,
    lambda0: float = 1e-3,
    use_weights: bool = False,
    eps: float = 1e-12,
    device: str = "cuda",
    metric: Literal["l2", "l1"] = "l2",
    standardize: bool = True,
    verbose: bool = True,
    knn_chunk: int = 512,
    y_feature_chunk: int = 4096,
    batch_size: int = 256,
) -> Array:
    """
    Drop-in replacement API for your prior compute_explained_fraction,
    but implemented with:
      - chunked exact kNN in Y
      - chunked-Y K construction for inverse LOO dual ridge
    """
    N = X.shape[0]
    if N != Y.shape[0]:
        raise ValueError("X and Y must have same N")

    dev = default_device(device)

    if standardize:
        Xt, _, _ = torch_zscore(X, device=dev, eps=eps)
        Yt, _, _ = torch_zscore(Y, device=dev, eps=eps)
    else:
        Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
        Yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)

    pval = 2.0 if metric == "l2" else 1.0

    if verbose:
        idx, dist = knn_chunked_torch(Yt, k=int(k), chunk=int(knn_chunk), p=float(pval))
    else:
        # still run chunked; tqdm suppression handled by tqdm(leave=False) upstream
        idx, dist = knn_chunked_torch(Yt, k=int(k), chunk=int(knn_chunk), p=float(pval))

    e = local_inverse_metrics_LOO_chunkY(
        X=Xt,
        Y=Yt,
        idxY=idx,
        dY=dist,
        use_weights=use_weights,
        eps_tau=float(eps)
    )

    return e["inv_explained_frac"]
