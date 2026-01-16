#!/usr/bin/env python3
"""
Compact Cahn–Hilliard temperature-profile dataset generator (warps + repeats)

- Warps: none, fold, ribbon, pinch
- Modes:
  - fixed: one deterministic initial condition broadcast to all samples
  - repeats: R repeats; per repeat r, RNG seed = seed_repeat_base + r; per sample random IC

Outputs (auto subfolders under outdir):
  ch_tprofiles_data/
    nowarp_fixed/   dataset_YYYYmmdd_HHMMSS.h5 + .json
    nowarp_repeats/ ...
    fold_fixed/     ...
    fold_repeats/   ...
    ribbon_fixed/   ...
    ribbon_repeats/ ...
    pinch_fixed/    ...
    pinch_repeats/  ...

HDF5 datasets:
  controlling_temperatures_original  (N,3) float32   # unwarped
  controlling_temperatures           (N,3) float32   # warped + used
  temperature_profiles               (N,steps) float32
  fields                             (N,H,W) or (N,R,H,W) float32  # final c

Critical control convention:
  - Segment 0 (T0) is NON-controlled for all warps except ribbon.
  - For ribbon, warped z maps into T0 (i.e., T0 <- x, T1 <- y, T2 <- z).
"""

import argparse, json, math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch


# -------------------- toy warps in normalized (-1,1) domain --------------------

@torch.no_grad()
def warp_ribbon(u2, t=1.0):
    # u2: (N,2) in [-1,1]
    y_prime = t * (0.75 * math.pi) * (u2[:, 1] + 1.0) + math.pi / 4
    curl = math.sin(t * math.pi / 2)
    x = u2[:, 0]
    y = t * u2[:, 1] * torch.cos(y_prime * curl) + (1 - t) * u2[:, 1]
    z = u2[:, 1] * torch.sin(y_prime * curl) + 0.5 * t**2
    return torch.stack((x, y, z), dim=1)

@torch.no_grad()
def warp_fold_sheet(u2, t=1.0):
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
def warp_pinch(u2, t=1.0):
    x, y = u2[:, 0], u2[:, 1]
    return torch.stack((x, y * torch.abs(x) ** (2 * t), torch.zeros_like(x)), dim=1)

def apply_warp_2d_to_3d(u2, warp, t):
    if warp == "fold": return warp_fold_sheet(u2, t)
    if warp == "pinch": return warp_pinch(u2, t)
    if warp == "ribbon": return warp_ribbon(u2, t)
    raise ValueError(warp)


# -------------------- sampling controls (physical) with control convention --------------------

def pm1(x, lo, hi):     # physical -> [-1,1]
    return ((x - lo) / (hi - lo)) * 2.0 - 1.0

def unpm1(u, lo, hi):   # [-1,1] -> physical
    return lo + ((u + 1.0) * 0.5) * (hi - lo)

@torch.no_grad()
def sample_controls(N, warp, t_strength, temp_min, temp_max, device):
    lo, hi = float(temp_min), float(temp_max)
    mid = 0.5 * (lo + hi)

    if warp == "none":
        # T0 non-controlled
        T1T2 = lo + (hi - lo) * torch.rand((N, 2), device=device)
        ctrl_orig = torch.cat([torch.full((N, 1), mid, device=device), T1T2], dim=1)
        u2_prewarp = pm1(T1T2, lo, hi)  # save normalized prewarp coords
        return ctrl_orig, ctrl_orig.clone(), u2_prewarp

    # For fold/pinch/ribbon: start from a 2D sheet over the controlled segments (T1,T2).
    T1T2 = lo + (hi - lo) * torch.rand((N, 2), device=device)
    ctrl_orig = torch.cat([torch.full((N, 1), mid, device=device), T1T2], dim=1)

    u2 = pm1(T1T2, lo, hi)
    u3 = apply_warp_2d_to_3d(u2, warp=warp, t=float(t_strength)).clamp(-1.0, 1.0)

    if warp in ("fold", "pinch"):
        # Warp only affects (T1,T2); T0 remains non-controlled (mid)
        T1T2_w = unpm1(u3[:, :2], lo, hi)
        ctrl_warp = torch.cat([torch.full((N, 1), mid, device=device), T1T2_w], dim=1)
        return ctrl_orig, ctrl_warp, u2

    # ribbon: use all three coords; z maps into T0
    xyz = unpm1(u3, lo, hi)           # (x,y,z) in physical units
    ctrl_warp = torch.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2]], dim=1)  # (T0,T1,T2) = (x,y,z)
    return ctrl_orig, ctrl_warp, u2

@torch.no_grad()
def controls_to_profiles(ctrl, steps):
    # ctrl: (N,3) -> (N,steps) piecewise-constant, equal-length segments
    N, segs = ctrl.shape
    sp = max(1, steps // segs)
    prof = ctrl.repeat_interleave(sp, dim=1)
    if prof.shape[1] < steps:
        prof = torch.cat([prof, prof[:, -1:].repeat(1, steps - prof.shape[1])], dim=1)
    return prof[:, :steps]


# -------------------- NEW CH physics driver (spectral periodic) --------------------

@torch.no_grad()
def prepare_spectral(grid, L, device):
    dx = float(L) / float(grid)
    kx = torch.fft.fftfreq(grid, d=dx, device=device) * 2.0 * math.pi
    Kx, Ky = torch.meshgrid(kx, kx, indexing="ij")
    K2 = Kx * Kx + Ky * Ky
    kcut = kx.abs().max() * 2.0 / 3.0
    dealias = ((Kx.abs() < kcut) & (Ky.abs() < kcut)).to(torch.complex64)
    return K2, dealias

@torch.no_grad()
def dfdc(c, W):
    # matches CHSimulator._dfdc (algebraically same as 2W*c*(1-c)*(1-2c))
    return 2.0 * W * (c * (1.0 - c) ** 2 - (1.0 - c) * c**2)

@torch.no_grad()
def temp_scaling(T, T_ref, La_base, La_temp_coeff, R, Q_mob):
    sW = ((La_base - La_temp_coeff * T) / (La_base - La_temp_coeff * T_ref)) * (T_ref / T)
    sK = T / T_ref
    sM = torch.exp(-Q_mob / R * (1.0 / T - 1.0 / T_ref)) * (T_ref / T)
    return sW, sK, sM

@torch.no_grad()
def ch_step(c, c_hat, K2, dealias, W, kappa, M, dt):
    # Vectorized semi-implicit spectral update (periodic)
    g = c.shape[-1]
    dfdc_hat = torch.fft.fftn(dfdc(c, W), dim=(-2, -1)) * dealias
    K2e = K2.view(1, 1, g, g)
    dte = dt.view(-1, 1, 1, 1)
    Me = M.view(-1, 1, 1, 1)
    num = c_hat - dte * K2e * Me * dfdc_hat
    den = 1.0 + dte * Me * kappa * (K2e**2)
    c_hat = num / den
    c = torch.fft.ifftn(c_hat, dim=(-2, -1)).real.clamp_(0.0, 1.0)
    return c, c_hat


# -------------------- dataset generation --------------------

def run_one(warp, mode, timestamp, args):
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    if args.segments != 3:
        raise SystemExit("This generator is fixed to 3 segments (T0,T1,T2) per the warp semantics.")

    warp_tag = "nowarp" if warp == "none" else warp
    mode_tag = "fixed" if mode == "fixedseed" else "repeats"
    outdir = Path(args.outdir) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    h5_path = outdir / f"{warp_tag}_{mode_tag}.h5"
    js_path = outdir / f"{warp_tag}_{mode_tag}.json"

    torch.manual_seed(int(args.seed_controls))
    ctrl_orig, ctrl_used, u2_prewarp = sample_controls(
        args.n_draws, warp, args.warp_strength, args.temp_min, args.temp_max, device
    )
    profiles = controls_to_profiles(ctrl_used, args.steps)  # (N,steps)

    N, g = int(args.n_draws), int(args.grid)
    R = int(args.n_repeats) if mode == "repeated" else 1

    # physics constants (from new driver)
    K2, dealias = prepare_spectral(g, args.L, device)
    W0 = float(args.W_base)
    kappa0 = float(args.kappa_base) * (float(args.alpha) ** 2) * (float(args.beta) ** 2)
    M0 = float(args.M_base) * (float(args.alpha) ** 2) / (float(args.beta) ** 2)

    # fixed IC if needed
    c0_fixed = None
    if mode == "fixedseed":
        gen = torch.Generator(device=device).manual_seed(int(args.seed_fixed_init))
        c0_fixed = (torch.full((1, 1, g, g), float(args.c_init), device=device)
                    + torch.randn((1, 1, g, g), device=device, generator=gen) * float(args.noise_amp)).clamp_(0.0, 1.0)

    # write
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["created"] = timestamp
        h5.attrs["warp"] = warp
        h5.attrs["mode"] = mode_tag
        h5.create_dataset("controlling_temperatures_original",
                          data=ctrl_orig.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("controlling_temperatures",
                          data=ctrl_used.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("prewarp_coordinates_normalized",
                          data=u2_prewarp.detach().cpu().numpy().astype(np.float32))
        h5.create_dataset("temperature_profiles",
                          data=profiles.detach().cpu().numpy().astype(np.float32))

        if mode == "fixedseed":
            fields = h5.create_dataset("fields", shape=(N, g, g), dtype=np.float32,
                                       chunks=(min(args.chunk_sims, N), g, g))
        else:
            fields = h5.create_dataset("fields", shape=(N, R, g, g), dtype=np.float32,
                                       chunks=(min(args.chunk_sims, N), 1, g, g))

        # move profiles once
        profiles_dev = profiles.to(device=device, dtype=torch.float32, non_blocking=True)

        n_runs = R if mode == "repeated" else 1
        bs = int(args.batch_size)
        total_batches = n_runs * ((N + bs - 1) // bs)
        done = 0

        for r in range(n_runs):
            for s in range(0, N, bs):
                e = min(N, s + bs)
                B = e - s
                Tp = profiles_dev[s:e]  # (B,steps)

                if mode == "fixedseed":
                    c = c0_fixed.expand(B, 1, g, g).contiguous()
                else:
                    c = torch.full((B, 1, g, g), float(args.c_init), device=device)
                    c = (c + torch.randn_like(c) * float(args.noise_amp)).clamp_(0.0, 1.0)

                c_hat = torch.fft.fftn(c, dim=(-2, -1))

                dt = torch.full((B,), float(args.dt_base), device=device)
                for t in range(int(args.steps)):
                    Tt = Tp[:, t]
                    sW, sK, sM = temp_scaling(
                        Tt,
                        float(args.T_ref),
                        float(args.La_base),
                        float(args.La_temp_coeff),
                        float(args.R_gas),
                        float(args.Q_mob),
                    )
                    W = torch.tensor(W0, device=device).view(1, 1, 1, 1) * sW.view(-1, 1, 1, 1)
                    kappa = torch.tensor(kappa0, device=device).view(1, 1, 1, 1) * sK.view(-1, 1, 1, 1)
                    M = torch.tensor(M0, device=device) * sM
                    c, c_hat = ch_step(c, c_hat, K2, dealias, W, kappa, M, dt)

                out = c[:, 0].detach().to("cpu", dtype=torch.float32).numpy()
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
            "controls": [N, 3],
            "prewarp_coords": [N, 2],
            "profiles": [N, int(args.steps)],
            "fields": ([N, g, g] if mode == "fixedseed" else [N, R, g, g]),
        },
        "control_convention": {
            "T0_noncontrolled_except_ribbon": True,
            "ribbon_mapping": "T0<-z, T1<-x, T2<-y",
        },
    }
    js_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] {h5_path}")


def main():
    p = argparse.ArgumentParser()

    # output + selection
    p.add_argument("--outdir", default="ch_tprofiles_data")
    p.add_argument("--only-warp", default="", choices=["", "none", "fold", "ribbon", "pinch"])
    p.add_argument("--only-mode", default="", choices=["", "fixedseed", "repeated"])

    # dataset sizing
    p.add_argument("--n-draws", dest="n_draws", type=int, default=1000)
    p.add_argument("--n-repeats", dest="n_repeats", type=int, default=200)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--chunk-sims", dest="chunk_sims", type=int, default=16)

    # grid + time
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--L", type=float, default=128.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--segments", type=int, default=3)  # fixed to 3 by design

    # temperature range + warps
    p.add_argument("--temp-min", dest="temp_min", type=float, default=400.0)
    p.add_argument("--temp-max", dest="temp_max", type=float, default=650.0)
    p.add_argument("--warp-strength", dest="warp_strength", type=float, default=1.0)

    # IC
    p.add_argument("--c-init", dest="c_init", type=float, default=0.5)
    p.add_argument("--noise-amp", dest="noise_amp", type=float, default=1e-3)

    # new CH physics params (from CHSimulator)
    p.add_argument("--dt-base", dest="dt_base", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--W-base", dest="W_base", type=float, default=1.0)
    p.add_argument("--kappa-base", dest="kappa_base", type=float, default=1.0)
    p.add_argument("--M-base", dest="M_base", type=float, default=1.0)
    p.add_argument("--T-ref", dest="T_ref", type=float, default=673.0)
    p.add_argument("--La-base", dest="La_base", type=float, default=20000.0)
    p.add_argument("--La-temp-coeff", dest="La_temp_coeff", type=float, default=9.0)
    p.add_argument("--R-gas", dest="R_gas", type=float, default=8.314)
    p.add_argument("--Q-mob", dest="Q_mob", type=float, default=1e4)

    # seeds + device
    p.add_argument("--seed-controls", dest="seed_controls", type=int, default=1337)
    p.add_argument("--seed-fixed-init", dest="seed_fixed_init", type=int, default=2024)
    p.add_argument("--seed-repeat-base", dest="seed_repeat_base", type=int, default=9001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    warps = ["none", "fold", "ribbon", "pinch"]
    modes = ["fixedseed", "repeated"]
    if args.only_warp: warps = [args.only_warp]
    if args.only_mode: modes = [args.only_mode]

    # Create single timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting run with timestamp: {timestamp}")

    for w in warps:
        for m in modes:
            run_one(w, m, timestamp, args)


if __name__ == "__main__":
    main()
