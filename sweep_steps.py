#!/usr/bin/env python3
"""
sweep_steps.py — CSGM BraTS hyperparameter grid search for the 1-step / 10-step goal.

GOAL:  find (t_start, n_steps_each, step_lr, mse) such that
           1-step  PSNR  >  ZF (FBP) PSNR
       and 10-step PSNR  >  1-step PSNR

DEFINITIONS (per the user):
    "1 step"  = --n_jumps 1  at T=0   -> timesteps [0]           -> 1  level  x n_steps_each updates
    "10 step" = --n_jumps 10          -> equispaced t_start -> 0 -> 10 levels x n_steps_each updates

    NOTE: timesteps are rounded to ints and de-duplicated, so `n_jumps=10` yields a full
    10 levels only for t_start >= 9 (t_start=7 -> 8 levels, 5 -> 6, 3 -> 4, 1 -> 2).
    Every row records the ACTUAL n_levels and total_updates, so nothing is silently assumed.

DATA: 5 fixed mid-slices from 5 fixed volumes (VOL_INDICES, seed=42 draw reused from
      sweep_csgm_brats.py), multicoil 8-coil HR (120x120x78), 4x. init_mode=zf for both arms.

FAITHFULNESS: imports reconstruct_csgm_brats and mirrors its main() mid-slice path exactly
      (same brats_vol_to_csgm_2d 6-tuple, same FBP, same optimizer construction with
      scaling_factor=1.0, same kno_* metrics). Only the optimizer config differs per grid point;
      the model / SDE / prior_trafo / 5 slices are loaded ONCE and reused.

Resumable: rows already in --out (matched on exact config dict) are skipped.

Usage:
    python sweep_steps.py --out sweep.jsonl                      # both arms, full grid
    python sweep_steps.py --out sweep.jsonl --arm 1step          # one arm only
    python sweep_steps.py --out sweep.jsonl --limit 20           # chunk / smoke
"""
import os
import sys
import json
import time
import logging
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# --- locate reconstruct_csgm_brats.py (override with --csgm_dir on another machine) -------
DEFAULT_CSGM_DIR = "/work/10471/peterwg/vista/r/3dmri_new/mri3d/src/ram/csgm-mri-langevin"

# --- defaults (Vista paths; override on RTX) ----------------------------------------------
DEFAULT_LMDB = "/scratch/10471/peterwg/brats2021_lmdb/brats_val_120x120x78_4x_8coil_lmdb"

# 5 fixed volumes (seed=42 draw, reused verbatim from sweep_csgm_brats.py so results are
# comparable with the earlier BraTS CSGM sweeps). Mid-slice of each (slice_idx=None).
VOL_INDICES = [6, 28, 70, 163, 189]

# ===========================================================================
# GRID
# ===========================================================================
GRID = {
    # 10-step arm only; the 1-step arm is pinned at T=0 so t_start does not apply to it.
    "t_start":      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # Extended to 50/100: the first (n_steps<=20) sweep put the best 1-step config on the
    # grid EDGE (n_steps=20, lr=5e-4), so the optimum is likely beyond it.
    "n_steps_each": [1, 2, 3, 5, 10, 20, 50, 100],
    # 1e-6 dropped: measured, it never moves off ZF (27.53 dB). Extended up to 1e-2.
    "step_lr":      [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    # 0.1 and 500 added: mse (the DC weight) traded off strongly against lr in the first sweep.
    "mse":          [0.1, 1.0, 5.0, 25.0, 100.0, 500.0],
}
# -> 1-step arm  :  8 x 8 x 6            =   384 configs
#    10-step arm : 10 x 8 x 8 x 6        =  3840 configs
#    total 4224. Measured cost ~ 0.051 + 0.136 * total_updates sec/config (5 slices, GH200)
#    => ~22.5 h. Resumable, so a tmux session can be stopped/restarted freely.


def iter_configs(arm):
    """1-step arm is pinned at T=0 (t_start=0, n_jumps=1) so t_start is NOT swept for it."""
    if arm == "1step":
        for n_steps_each, step_lr, mse in itertools.product(
            GRID["n_steps_each"], GRID["step_lr"], GRID["mse"]
        ):
            yield {"arm": "1step", "t_start": 0, "n_jumps": 1,
                   "n_steps_each": n_steps_each, "step_lr": step_lr, "mse": mse}
    else:
        for t_start, n_steps_each, step_lr, mse in itertools.product(
            GRID["t_start"], GRID["n_steps_each"], GRID["step_lr"], GRID["mse"]
        ):
            yield {"arm": "10step", "t_start": t_start, "n_jumps": 10,
                   "n_steps_each": n_steps_each, "step_lr": step_lr, "mse": mse}


def config_key(cfg):
    return json.dumps(cfg, sort_keys=True)


def load_done_keys(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(config_key(json.loads(line)["config"]))
    return done


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--csgm_dir", default=DEFAULT_CSGM_DIR)
    p.add_argument("--lmdb", default=DEFAULT_LMDB)
    p.add_argument("--ema_ckpt", default=None)
    p.add_argument("--train_cfg", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--readout_axis", type=int, default=0)
    p.add_argument("--arm", choices=["1step", "10step", "both"], default="both")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prior_input_scale", type=float, default=None)
    p.add_argument("--slice_cache", default=None,
                   help="Path to a .pt cache of the 5 preloaded mid-slices + ZF baseline. "
                        "Loaded if it exists, otherwise built from --lmdb and saved. Lets the "
                        "sweep run without the 38 GB LMDB (e.g. on the RTX box).")
    p.add_argument("--make_cache_only", action="store_true",
                   help="Build --slice_cache from --lmdb, then exit (no sweep).")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.csgm_dir))
    import reconstruct_csgm_brats as R  # noqa: E402

    class StridedAnyT(R.LangevinOptimizerStrided):
        """LangevinOptimizerStrided that also supports t_start=0 (a single level at T=0),
        which the 1-step arm needs.

        Upstream limitation (NOT modified — reconstruct_csgm_brats.py is left untouched):
        LangevinOptimizer.__init__ does `abar = ...squeeze()` then `self.sigmas[-1]`. For
        t_start=0 the timestep list is [0], so abar squeezes to 0-dim and sigmas[-1] raises
        IndexError. LangevinOptimizerStrided already carries the `if abar.dim()==0: unsqueeze`
        guard, but the parent runs first. So: construct with a safe t_start (>=1) and then
        rewrite timesteps/sigmas to [0] using the subclass's own formula. For t_start>=1
        (the whole 10-step arm) this class is a no-op passthrough."""

        def __init__(self, config, device, score, sde, prior_trafo, sf,
                     prior_input_scale=1.0):
            want = int(config["t_start"])
            safe = dict(config)
            safe["t_start"] = max(want, 1)
            super().__init__(safe, device, score, sde, prior_trafo, sf,
                             prior_input_scale=prior_input_scale)
            if want == 0:
                self.timesteps = [0]
                with torch.no_grad():
                    abar = sde._compute_alpha_cumprod(
                        torch.tensor(self.timesteps, device=device)).squeeze()
                    if abar.dim() == 0:
                        abar = abar.unsqueeze(0)
                    self.sigmas = (1.0 - abar).sqrt().detach()
                self.sigma_min = float(self.sigmas[-1].item())

    ema_ckpt  = args.ema_ckpt  or str(R.EMA_CKPT)
    train_cfg = args.train_cfg or str(R.TRAIN_CFG)
    device    = args.device
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- model / SDE / prior_trafo, ONCE -----------------------------------
    score       = R.load_score_model(ema_ckpt, train_cfg, device)
    sde         = R.DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)
    prior_trafo = R.load_prior_trafo(train_cfg, scaling_factor=1.0)

    # prior_input_scale: resolve exactly as main() does (fail-loud, never silent 1.0).
    if args.prior_input_scale is not None:
        prior_input_scale = float(args.prior_input_scale)
    else:
        _tcfg = OmegaConf.load(train_cfg)
        _path = "problem_trafos.dataset_trafo.target_scaling_factor"
        _val  = OmegaConf.select(_tcfg, _path, default=None)
        if _val is None:
            raise ValueError(f"Could not resolve '{_path}' in {train_cfg}. "
                             f"Pass --prior_input_scale explicitly.")
        prior_input_scale = float(_val)
    logging.info(f"prior_input_scale = {prior_input_scale}")

    # ---- preload the 5 fixed mid-slices + their ZF/FBP baseline, ONCE ------
    # Cached so the sweep can run anywhere without the 38 GB LMDB (RTX bundle).
    if args.slice_cache and Path(args.slice_cache).exists() and not args.make_cache_only:
        blob = torch.load(args.slice_cache, map_location=device)
        vol_data = [(d["vol_idx"], d["ref"], None, d["maps_b"], d["mask_b"],
                     d["gt_2d"], d["H"], d["W"]) for d in blob["slices"]]
        ZF = blob["ZF"]
        logging.info(f"loaded {len(vol_data)} cached mid-slices from {args.slice_cache}")
    else:
        vol_data, zf_p, zf_s, zf_n, cache_slices = [], [], [], [], []
        for vol_idx in VOL_INDICES:
            ref, mvue, maps_b, mask_b, gt_2d, kspace_scale = R.brats_vol_to_csgm_2d(
                args.lmdb, vol_idx, device, args.readout_axis, slice_idx=None
            )
            H, W = ref.shape[-2], ref.shape[-1]
            with torch.no_grad():                   # FBP == adjoint-SENSE zero-filled
                coils_img = R._ifft(ref)
                fbp_hw    = (torch.conj(maps_b) * coils_img).sum(dim=1)
                fbp_sf_2d = torch.view_as_real(fbp_hw.squeeze(0))
            gt_cpu = gt_2d.cpu()
            zf_p.append(R.kno_psnr(fbp_sf_2d.cpu(), gt_cpu).item())
            zf_s.append(R.kno_ssim(fbp_sf_2d.cpu(), gt_cpu).item())
            zf_n.append(R.kno_nmse(fbp_sf_2d.cpu(), gt_cpu).item())
            vol_data.append((vol_idx, ref, mvue, maps_b, mask_b, gt_2d, H, W))
            cache_slices.append({"vol_idx": vol_idx, "ref": ref.cpu(), "maps_b": maps_b.cpu(),
                                 "mask_b": mask_b.cpu(), "gt_2d": gt_2d.cpu(), "H": H, "W": W})
            logging.info(f"[vol {vol_idx}] mid-slice ({H},{W})  ZF PSNR={zf_p[-1]:.2f} dB "
                         f"SSIM={zf_s[-1]:.4f} NMSE={zf_n[-1]:.6f}")

        ZF = {"mean_psnr": float(np.mean(zf_p)), "mean_ssim": float(np.mean(zf_s)),
              "mean_nmse": float(np.mean(zf_n)),
              "std_psnr": float(np.std(zf_p)), "per_volume_psnr": zf_p}

        if args.slice_cache:
            Path(args.slice_cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"slices": cache_slices, "ZF": ZF,
                        "vol_indices": VOL_INDICES, "lmdb": args.lmdb},
                       args.slice_cache)
            logging.info(f"wrote slice cache -> {args.slice_cache}")
        if args.make_cache_only:
            logging.info("--make_cache_only set; exiting before the sweep.")
            return
    logging.info(f"\n{'='*64}\n  ZF (FBP) baseline over {len(VOL_INDICES)} mid-slices: "
                 f"PSNR={ZF['mean_psnr']:.2f} +/- {ZF['std_psnr']:.2f} dB  "
                 f"SSIM={ZF['mean_ssim']:.4f}  NMSE={ZF['mean_nmse']:.6f}\n{'='*64}")
    with open(out_path.parent / "zf_baseline.json", "w") as f:
        json.dump(ZF, f, indent=2)

    # ---- grid --------------------------------------------------------------
    arms     = ["1step", "10step"] if args.arm == "both" else [args.arm]
    all_cfgs = [c for a in arms for c in iter_configs(a)]
    done     = load_done_keys(out_path)
    todo     = [c for c in all_cfgs if config_key(c) not in done]

    # Cheapest-first. Runtime is ~linear in total_updates (= n_levels x n_steps_each), so
    # ordering by it means a long tmux run that gets stopped early has still covered the
    # WHOLE grid coarsely, instead of finishing one expensive corner of it.
    def est_updates(c):
        if c["arm"] == "1step":
            return c["n_steps_each"]
        n_lvl = len(sorted(set(int(round(t))
                               for t in np.linspace(c["t_start"], 0, c["n_jumps"]))))
        return n_lvl * c["n_steps_each"]
    todo.sort(key=est_updates)

    if args.limit is not None:
        todo = todo[:args.limit]
    est_h = sum(0.0510 + 0.13577 * est_updates(c) for c in todo) / 3600
    logging.info(f"arms={arms}  total={len(all_cfgs)}  done={len(done)}  todo={len(todo)}  "
                 f"(cheapest-first; est ~{est_h:.1f} h at GH200 speed)")

    t_wall = time.time()
    with open(out_path, "a") as fout:
        for j, cfg in enumerate(todo):
            t0 = time.time()
            psnrs, ssims, nmses, per_vol = [], [], [], []
            n_levels = None

            for vol_idx, ref, mvue, maps_b, mask_b, gt_2d, H, W in vol_data:
                oc = {
                    "device": device, "image_size": (H, W),
                    "t_start": cfg["t_start"], "n_steps_each": cfg["n_steps_each"],
                    "step_lr": cfg["step_lr"], "mse": cfg["mse"],
                    "init_mode": "zf", "track": False,
                    "gt_2d": gt_2d.to(device), "no_prior": False,
                    "n_jumps": cfg["n_jumps"],
                }
                optim = StridedAnyT(
                    oc, device, score, sde, prior_trafo, 1.0,
                    prior_input_scale=prior_input_scale
                ).to(device)
                n_levels = len(optim.timesteps)

                with torch.no_grad():
                    samples = optim.sample((ref, mvue, maps_b, mask_b))
                rec_sf_2d = samples.squeeze(0).permute(1, 2, 0).contiguous()

                gt_cpu = gt_2d.cpu()
                pv = R.kno_psnr(rec_sf_2d.cpu(), gt_cpu).item()
                sv = R.kno_ssim(rec_sf_2d.cpu(), gt_cpu).item()
                nv = R.kno_nmse(rec_sf_2d.cpu(), gt_cpu).item()
                psnrs.append(pv); ssims.append(sv); nmses.append(nv)
                per_vol.append({"vol_idx": vol_idx, "psnr": pv, "ssim": sv, "nmse": nv})

            row = {
                "config": cfg,
                "n_levels": n_levels,
                "total_updates": n_levels * cfg["n_steps_each"],
                "mean_psnr": float(np.mean(psnrs)), "std_psnr": float(np.std(psnrs)),
                "mean_ssim": float(np.mean(ssims)), "mean_nmse": float(np.mean(nmses)),
                "zf_mean_psnr": ZF["mean_psnr"],
                "beats_zf": bool(np.mean(psnrs) > ZF["mean_psnr"]),
                "sec": round(time.time() - t0, 2),
                "per_volume": per_vol,
            }
            fout.write(json.dumps(row) + "\n"); fout.flush(); os.fsync(fout.fileno())

            if (j + 1) % 10 == 0 or j == 0:
                rate = (time.time() - t_wall) / (j + 1)
                logging.info(f"[{j+1}/{len(todo)}] {cfg['arm']} t={cfg['t_start']} "
                             f"ns={cfg['n_steps_each']} lr={cfg['step_lr']:g} mse={cfg['mse']:g} "
                             f"-> {row['mean_psnr']:.2f} dB (ZF {ZF['mean_psnr']:.2f}) "
                             f"[{n_levels} lvl, {row['total_updates']} upd, {row['sec']}s] "
                             f"| eta {(len(todo)-j-1)*rate/60:.0f} min")

    logging.info(f"DONE  {len(todo)} configs in {(time.time()-t_wall)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
