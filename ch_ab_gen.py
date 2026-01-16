#!/usr/bin/env python3
"""
Cahn–Hilliard dataset generator (clean bookkeeping, same warps).

What is saved (per sample):
  1) sheet_u2        (N,2)  in [-1,1]^2      # original 2D sheet samples BEFORE warping (sampled X,Y)
  2) warped_u3       (N,3)  in [-1,1]^3      # warped coords; z=0 for nowarp/fold/pinch
  3) controls_warped (N,3)  in physical ranges (alpha,beta,c0)  # DENORM(warped_u3)
       - For nowarp/fold/pinch: c0 is midpoint since z=0
       - For ribbon: c0 varies via ribbon z

Outputs:
  outdir/<timestamp>/<warp_tag>_<mode_tag>.h5
  outdir/<timestamp>/<warp_tag>_<mode_tag>.json

Modes:
  - fixedseed: deterministic noise field shared across samples
  - repeated:  R repeats; per repeat r seed = seed_repeat_base + r

HDF5 datasets:
  sheet_u2, warped_u3, controls_warped, fields
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch


# -------------------- warps in normalized [-1,1] domain (UNCHANGED behavior) --------------------
@torch.no_grad()
def warp_ribbon(u2: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    # u2: (N,2) in [-1,1]
    y_prime = t * (0.75 * math.pi) * (u2[:, 1] + 1.0) + math.pi / 4
    curl = math.sin(t * math.pi / 2)
    x = u2[:, 0]
    y = t * u2[:, 1] * torch.cos(y_prime * curl) + (1 - t) * u2[:, 1]
    z = u2[:, 1] * torch.sin(y_prime * curl) + 0.5 * t**2
    return torch.stack((x, y, z), dim=1)

@torch.no_grad()
def warp_fold_sheet(u2: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    # PRESERVED exactly from original
    x, y = u2[:, 0], u2[:, 1]
    mask = y > -x + 1
    x_new, y_new = x.clone(), y.clone()
    z_new = torch.zeros_like(x)

    if torch.any(mask):
        px, py = x[mask], y[mask]
        dx, dy = px - 0.5, py - 0.5
        theta = math.pi * t
        axis = torch.tensor([1.0, -1.0, 0.0], device=u2.device)
        axis = axis / torch.norm(axis)

        pts = torch.stack((dx, dy, torch.zeros_like(dx)), dim=1)
        k = axis
        cos_t = torch.cos(torch.tensor(theta, device=u2.device))
        sin_t = torch.sin(torch.tensor(theta, device=u2.device))
        rot = (
            pts * cos_t
            + torch.cross(k.expand_as(pts), pts, dim=1) * sin_t
            + k * torch.sum(pts * k, dim=1, keepdim=True) * (1 - cos_t)
        )
        x_new[mask] = rot[:, 0] + 0.5
        y_new[mask] = rot[:, 1] + 0.5

    return torch.stack((x_new, y_new, z_new), dim=1)

@torch.no_grad()
def warp_pinch(u2: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    # PRESERVED exactly from original
    x, y = u2[:, 0], u2[:, 1]
    return torch.stack(
        (x, y * torch.abs(x).clamp_min(1e-12) ** (2.0 * float(t)), torch.zeros_like(x)),
        dim=1,
    )

@torch.no_grad()
def apply_warp_2d_to_3d(u2: torch.Tensor, warp: str, t: float) -> torch.Tensor:
    if warp == "fold":
        return warp_fold_sheet(u2, t)
    if warp == "pinch":
        return warp_pinch(u2, t)
    if warp == "ribbon":
        return warp_ribbon(u2, t)
    raise ValueError(warp)


# -------------------- normalized -> physical mapping (done inline, no pm1/unpm1 helpers) --------------------
@torch.no_grad()
def denorm_u3_to_controls(
    u3: torch.Tensor,
    alpha_min: float, alpha_max: float,
    beta_min: float, beta_max: float,
    c0_min: float, c0_max: float,
) -> torch.Tensor:
    # u in [-1,1] -> x in [lo,hi] via lo + (u+1)/2*(hi-lo)
    a = alpha_min + 0.5 * (u3[:, 0:1] + 1.0) * (alpha_max - alpha_min)
    b = beta_min  + 0.5 * (u3[:, 1:2] + 1.0) * (beta_max - beta_min)
    c = c0_min    + 0.5 * (u3[:, 2:3] + 1.0) * (c0_max - c0_min)
    return torch.cat([a, b, c], dim=1)


# -------------------- CH physics (semi-implicit spectral, periodic) --------------------
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


# -------------------- dataset generation --------------------
def run_one(warp: str, mode: str, timestamp: str, args: argparse.Namespace):
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    warp_tag = "nowarp" if warp == "none" else warp
    mode_tag = "fixed" if mode == "fixedseed" else "repeats"

    outdir = Path(args.outdir) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = outdir / f"{warp_tag}_{mode_tag}.h5"
    js_path = outdir / f"{warp_tag}_{mode_tag}.json"

    N = int(args.n_draws)
    g = int(args.grid)
    R = int(args.n_repeats) if mode == "repeated" else 1

    # 1) sheet_u2 in [-1,1]^2: original 2D samples before warping
    torch.manual_seed(int(args.seed_sheet))
    sheet_u2 = (2.0 * torch.rand((N, 2), device=device, dtype=torch.float32) - 1.0).clamp(-1.0, 1.0)

    # 2) warped_u3 in [-1,1]^3
    if warp == "none":
        warped_u3 = torch.cat([sheet_u2, torch.zeros((N, 1), device=device, dtype=torch.float32)], dim=1)
    else:
        warped_u3 = apply_warp_2d_to_3d(sheet_u2, warp=warp, t=float(args.warp_strength))
    warped_u3 = warped_u3.clamp(-1.0, 1.0)

    # 3) physical controls used for physics: controls_warped = denorm(warped_u3)
    controls_warped = denorm_u3_to_controls(
        warped_u3,
        alpha_min=float(args.alpha_min), alpha_max=float(args.alpha_max),
        beta_min=float(args.beta_min), beta_max=float(args.beta_max),
        c0_min=float(args.c0_min), c0_max=float(args.c0_max),
    ).to(device=device, dtype=torch.float32)

    # Physics per sample
    alpha = controls_warped[:, 0].clamp_min(1e-12)
    beta  = controls_warped[:, 1].clamp_min(1e-12)
    c0    = controls_warped[:, 2]

    kappa_all = (float(args.kappa_base) * (alpha**2) * (beta**2)).view(N, 1, 1)
    M_all     = (float(args.M_base) * (alpha**2) / (beta**2).clamp_min(1e-12)).view(N, 1, 1)
    W_all     = torch.full((N, 1, 1), float(args.W_base), device=device, dtype=torch.float32)
    dt_all    = torch.full((N, 1, 1), float(args.dt_base), device=device, dtype=torch.float32)

    # Spectral operators (shared)
    K2, dealias = prepare_spectral(g, float(args.L), device)

    # Deterministic noise field for fixedseed mode
    noise_fixed = None
    if mode == "fixedseed":
        gen0 = torch.Generator(device=device).manual_seed(int(args.seed_fixed_init))
        noise_fixed = torch.randn((1, g, g), device=device, generator=gen0, dtype=torch.float32)

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["created"] = timestamp
        h5.attrs["warp"] = warp
        h5.attrs["mode"] = mode_tag
        h5.attrs["ranges_alpha"] = np.array([float(args.alpha_min), float(args.alpha_max)], dtype=np.float32)
        h5.attrs["ranges_beta"]  = np.array([float(args.beta_min), float(args.beta_max)], dtype=np.float32)
        h5.attrs["ranges_c0"]    = np.array([float(args.c0_min), float(args.c0_max)], dtype=np.float32)
        h5.attrs["warp_strength"] = float(args.warp_strength)

        h5.create_dataset("sheet_u2", data=sheet_u2.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("warped_u3", data=warped_u3.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("controls_warped", data=controls_warped.detach().cpu().numpy().astype(np.float32))

        if mode == "fixedseed":
            fields = h5.create_dataset(
                "fields",
                shape=(N, g, g),
                dtype=np.float32,
                chunks=(min(int(args.chunk_sims), N), g, g),
            )
        else:
            fields = h5.create_dataset(
                "fields",
                shape=(N, R, g, g),
                dtype=np.float32,
                chunks=(min(int(args.chunk_sims), N), 1, g, g),
            )

        bs = int(args.batch_size)
        n_runs = R if mode == "repeated" else 1
        total_batches = n_runs * ((N + bs - 1) // bs)
        done = 0

        for r in range(n_runs):
            gen_r = None
            if mode == "repeated":
                gen_r = torch.Generator(device=device).manual_seed(int(args.seed_repeat_base) + int(r))

            for s in range(0, N, bs):
                e = min(N, s + bs)
                B = e - s

                Wb     = W_all[s:e]
                kappab = kappa_all[s:e]
                Mb     = M_all[s:e]
                dtb    = dt_all[s:e]
                c0b    = c0[s:e].view(B, 1, 1)

                if mode == "fixedseed":
                    c = (c0b + noise_fixed.expand(B, g, g) * float(args.noise_amp)).clamp_(0.0, 1.0)
                else:
                    noise = torch.randn((B, g, g), device=device, generator=gen_r, dtype=torch.float32)
                    c = (c0b + noise * float(args.noise_amp)).clamp_(0.0, 1.0)

                c_hat = torch.fft.fftn(c, dim=(-2, -1))
                for _ in range(int(args.steps)):
                    c, c_hat = ch_step(c, c_hat, K2, dealias, Wb, kappab, Mb, dtb)

                out = c.detach().to("cpu", dtype=torch.float32).numpy()
                if mode == "fixedseed":
                    fields[s:e, :, :] = out
                else:
                    fields[s:e, r, :, :] = out

                done += 1
                if (done % 10) == 0 or done == total_batches:
                    print(f"[{warp_tag}_{mode_tag}] {done}/{total_batches} batches")

    meta = {
        "created": timestamp,
        "warp": warp,
        "mode": mode_tag,
        "outdir": str(outdir),
        "h5": str(h5_path),
        "args": vars(args),
        "shapes": {
            "sheet_u2": [N, 2],
            "warped_u3": [N, 3],
            "controls_warped": [N, 3],
            "fields": ([N, g, g] if mode == "fixedseed" else [N, R, g, g]),
        },
        "semantics": {
            "sheet_u2": "2D sheet samples in [-1,1]^2 before warping",
            "warped_u3": "Warped coordinates in [-1,1]^3; z=0 for nowarp/fold/pinch",
            "controls_warped": "Physical (alpha,beta,c0) = denorm(warped_u3) used in physics",
        },
        "physics": {
            "W": "constant W_base",
            "kappa": "kappa_base * alpha^2 * beta^2",
            "M": "M_base * alpha^2 / beta^2",
            "dt": "dt_base (constant)",
        },
        "ic": {
            "c_initial": "c0 + noise_amp * N(0,1), clamped to [0,1]",
            "fixedseed": "noise field deterministic; shared across samples",
            "repeats": "per repeat r, RNG seed = seed_repeat_base + r",
        },
    }
    js_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] {h5_path}")


def main():
    p = argparse.ArgumentParser()

    # output + selection
    p.add_argument("--outdir", default="ch_ab_data")
    p.add_argument("--only-warp", default="", choices=["", "none", "fold", "ribbon", "pinch"])
    p.add_argument("--only-mode", default="", choices=["", "fixedseed", "repeated"])

    # dataset sizing
    p.add_argument("--n-draws", dest="n_draws", type=int, default=2000)
    p.add_argument("--n-repeats", dest="n_repeats", type=int, default=100)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--chunk-sims", dest="chunk_sims", type=int, default=16)

    # grid + time
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--L", type=float, default=128.0)
    p.add_argument("--steps", type=int, default=200)

    # physical ranges
    p.add_argument("--alpha-min", dest="alpha_min", type=float, default=0.7)
    p.add_argument("--alpha-max", dest="alpha_max", type=float, default=1.3)
    p.add_argument("--beta-min",  dest="beta_min",  type=float, default=0.7)
    p.add_argument("--beta-max",  dest="beta_max",  type=float, default=1.3)
    p.add_argument("--c0-min", dest="c0_min", type=float, default=0.45)
    p.add_argument("--c0-max", dest="c0_max", type=float, default=0.50)

    # warp strength
    p.add_argument("--warp-strength", dest="warp_strength", type=float, default=1.0)

    # IC noise amplitude
    p.add_argument("--noise-amp", dest="noise_amp", type=float, default=0.05)

    # CH physics base params
    p.add_argument("--dt-base", dest="dt_base", type=float, default=0.5)
    p.add_argument("--W-base", dest="W_base", type=float, default=1.0)
    p.add_argument("--kappa-base", dest="kappa_base", type=float, default=1.5)
    p.add_argument("--M-base", dest="M_base", type=float, default=1.0)

    # seeds + device
    p.add_argument("--seed-sheet", dest="seed_sheet", type=int, default=1337)
    p.add_argument("--seed-fixed-init", dest="seed_fixed_init", type=int, default=2024)
    p.add_argument("--seed-repeat-base", dest="seed_repeat_base", type=int, default=9001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    warps = ["none", "fold", "ribbon", "pinch"]
    modes = ["fixedseed", "repeated"]
    if args.only_warp:
        warps = [args.only_warp]
    if args.only_mode:
        modes = [args.only_mode]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting run with timestamp: {timestamp}")

    for w in warps:
        for m in modes:
            run_one(w, m, timestamp, args)


if __name__ == "__main__":
    main()
