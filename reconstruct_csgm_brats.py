#!/usr/bin/env python3
"""
reconstruct_csgm_brats.py

Faithful CSGM (Jalal et al. 2021) reconstruction adapted to BraTS val data.

Built from reconstruct_csgm_kno.py (SKM-TEA version) by changing ONLY:
  1. Dataset: BraTS LMDB sub-dbs (kspace/, maps/, target/, shapes/, masks/) replace
     LMDBVolumeDataset. BraTS kspace is already masked with no mask bug.
  2. Geometry: readout_axis=0 (X is readout, fully sampled) → 2D slice is (Y,Z).
     Mask is 2D (Y,Z) from the Gaussian3DMaskFunc; MulticoilForwardMRI uses
     orientation='2d' to trigger the existing mask.ndim==3 broadcast branch.
  3. Target: precomputed clean magnitude image in [0,1] from LMDB target sub-db.
  4. Prior trafo: loaded from BraTS train_cfg (CroppedMagnitudeImagePriorTrafo
     with swap_channels=True, scaling_factor=1.0 — same net effect as SKM-TEA
     move_axis=[-1,1] but with no amplitude scaling).

Everything else is byte-for-byte reconstruct_csgm_kno.py:
  - annealed Langevin loop, normalize/unnormalize, per-step gradient direction
    normalization, Langevin update
  - DDPM schedule, KNO-style PSNR/SSIM metrics
  - sf scaling: sqrt(H*W*2) / ||ref||

Key scaling-factor note (see CLAUDE.md gotcha #14 and #15):
  dataset.target_scaling_factor=5050 is the TRAINING dataset transform scale.
  prior_trafo.scaling_factor=1.0 is what the trafo itself applies. These are
  independent. For CSGM reconstruction use prior_input_scale=5050 (loaded from
  train_cfg.dataset.target_scaling_factor) in _predict_eps so the score model
  receives inputs at its training distribution scale O(5050).

Three fixes vs the original broken version:
  Fix 1: ref recomputed via _fft(target_2d) — stored kspace had 1D-IFFT artifact.
  Fix 2: MVUE via _ifft (ortho) not sp.ifft (backward) — consistent with _fft ref.
  Fix 3: prior_in * prior_input_scale in _predict_eps — score model training scale.

Run from baselines/resolution_robust_3d_mri/ with .venv active:
    python ../csgm-mri-langevin/reconstruct_csgm_brats.py --num_volumes 1 --no_wandb
"""

import sys
import os
import json
import math
import logging
import argparse
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.fft as torch_fft
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: res_rob first (src.diffmodels etc.), then KNO venv
# (torchmetrics 0.11.4 needed by meddlr), then KNO src (fastmri).
# ---------------------------------------------------------------------------
# Use abspath (not resolve) so symlinks are NOT followed — the script lives at
# csgm-mri-langevin/reconstruct_csgm_brats.py → scratch/..., and resolve()
# would follow the symlink, making parents[3] == "/" instead of mri3d/.
_HERE    = Path(os.path.abspath(__file__))
RES_ROB  = _HERE.parents[1] / "resolution_robust_3d_mri"
KNO_SRC  = _HERE.parents[2]
KNO_VENV = _HERE.parents[3] / ".venv/lib/python3.10/site-packages"
sys.path.insert(0, str(KNO_VENV))
sys.path.append(str(KNO_SRC))
sys.path.insert(0, str(RES_ROB))

if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore[attr-defined]

import meddlr.metrics.functional as _meddlr_metrics

# datasets.skmtea imports `from fastmri.subsample import MaskFunc` (KNO-local).
# The installed fastmri exposes it at fastmri.data.subsample — alias.
import fastmri.data.subsample as _fmri_sub
sys.modules.setdefault("fastmri.subsample", _fmri_sub)

from src.diffmodels.diffmodels_resolver import create_dense_model
from src.diffmodels.ema import ExponentialMovingAverage
from src.diffmodels.sde import DDPM
from src.problem_trafos.trafo_resolver import get_prior_trafo


# ===========================================================================
# Default paths
# ===========================================================================
BRATS_VAL_LMDB = Path("/scratch/10471/peterwg/brats2021_lmdb/brats_val_60x60x39_4x_singlecoil_lmdb")
EMA_CKPT       = Path("/scratch/10471/peterwg/brats_runs/bs4/ema_model_26.pt")
TRAIN_CFG      = Path("/scratch/10471/peterwg/brats_runs/bs4/outputs/brats/train_dense/"
                      "2026-06-04T20:18:40.347833Z_bs_4/.hydra/config.yaml")


# ===========================================================================
# Forward operator (verbatim from csgm-mri-langevin/utils.py — unchanged)
# ===========================================================================
def _ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x


def _fft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x


class MulticoilForwardMRI(torch.nn.Module):
    """Verbatim port of csgm-mri-langevin/utils.py MulticoilForwardMRI."""

    def __init__(self, orientation):
        super().__init__()
        self.orientation = orientation

    def forward(self, image, maps, mask):
        coils = image[:, None] * maps                    # (B, C, H, W) complex
        ksp_coils = _fft(coils)
        if self.orientation == "vertical":
            ksp_coils = ksp_coils * mask[:, None, None, :]
        elif self.orientation == "horizontal":
            ksp_coils = ksp_coils * mask[:, None, :, None]
        else:
            if mask.ndim == 3:
                # BraTS: 2D mask (B, H, W) → broadcast over coils
                ksp_coils = ksp_coils * mask[:, None, :, :]
            else:
                raise NotImplementedError("mask orientation not supported")
        return ksp_coils


# ===========================================================================
# CSGM normalize / unnormalize via 99th-percentile MVUE
# (verbatim from csgm-mri-langevin/main.py — unchanged)
# ===========================================================================
def get_mvue(kspace_np, smaps_np):
    """MVUE estimate from coil k-space + sensitivity maps (numpy)."""
    import sigpy as sp
    return (
        np.sum(sp.ifft(kspace_np, axes=(-1, -2)) * np.conj(smaps_np), axis=1)
        / np.sqrt(np.sum(np.square(np.abs(smaps_np)), axis=1))
    )


def normalize(gen_img, estimated_mvue):
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling


def unnormalize(gen_img, estimated_mvue):
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling


# ===========================================================================
# KNO metrics — same as reconstruct_csgm_kno.py.
# ===========================================================================
def _kno_psnr_fn(pred_mag, target_mag):
    B, C = pred_mag.shape[:2]
    pred_flat   = pred_mag.view(B, C, -1).float()
    target_flat = target_mag.view(B, C, -1).float()
    rmse    = (pred_flat - target_flat).pow(2).mean(dim=-1).sqrt()
    max_val = target_flat.abs().amax(dim=-1)
    return (20.0 * torch.log10(max_val / (rmse + 1e-8))).mean()


def _to_mag(t):
    return torch.view_as_complex(t.contiguous()).abs() if t.shape[-1] == 2 else t.abs()


def kno_psnr(rec_hw2, gt_hw2):
    """rec, gt: (H, W, 2). Returns scalar KNO PSNR (dB)."""
    rec_mag = _to_mag(rec_hw2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_hw2).unsqueeze(0).unsqueeze(0)
    return _kno_psnr_fn(rec_mag, gt_mag).squeeze()


def kno_ssim(rec_hw2, gt_hw2):
    rec_mag = _to_mag(rec_hw2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_hw2).unsqueeze(0).unsqueeze(0)
    return _meddlr_metrics.ssim(rec_mag, gt_mag).mean()


# ===========================================================================
# Score model loader — same as reconstruct_csgm_kno.py.
# ===========================================================================
def load_score_model(ckpt_path, train_cfg_path, device):
    train_cfg = OmegaConf.load(train_cfg_path)
    try:
        arch_cfg = train_cfg.arch
    except Exception:
        arch_cfg = train_cfg.diffmodels.arch
    if "name" not in arch_cfg:
        param_dict = {"name": "dense", "params": dict(arch_cfg)}
    else:
        param_dict = dict(arch_cfg)
    score = create_dense_model(**param_dict.get("params", param_dict)).to(device)
    ema   = ExponentialMovingAverage(score.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(ckpt_path, map_location=device))
    ema.copy_to(score.parameters())
    score.eval()
    logging.info(f"Loaded EMA score model from {ckpt_path}")
    return score


def load_prior_trafo(train_cfg_path, scaling_factor=1.0):
    """BraTS prior_trafo: load from train_cfg (CroppedMagnitudeImagePriorTrafo).
    swap_channels=True converts (B,H,W,2) → (B,2,H,W), same net effect as
    SKM-TEA move_axis=[-1,1] but no amplitude scaling."""
    tcfg = OmegaConf.load(train_cfg_path)
    pt   = tcfg.problem_trafos.prior_trafo
    kwargs = {k: v for k, v in OmegaConf.to_container(pt, resolve=True).items()
              if k not in ("name", "defaults")}
    kwargs["center_crop_enabled"] = False
    kwargs["crop_size"]           = None
    kwargs["scaling_factor"]      = scaling_factor
    return get_prior_trafo(name=pt.name, **kwargs)


# ===========================================================================
# LangevinOptimizer — BraTS variant.
# Fix 2: MVUE computed via _ifft (consistent with _fft-built ref).
# Fix 3: prior_in scaled by prior_input_scale before score model (training scale).
# ===========================================================================
class LangevinOptimizer(torch.nn.Module):
    def __init__(self, config, device, score, sde, prior_trafo, scaling_factor,
                 prior_input_scale=1.0):
        super().__init__()
        self.config            = config
        self.device            = device
        self.score             = score
        self.sde               = sde
        self.prior_trafo       = prior_trafo
        self.scaling_factor    = scaling_factor
        self.prior_input_scale = prior_input_scale  # Fix 3: match training distribution scale

        t_start = min(int(config["t_start"]), sde.num_steps - 1)
        self.timesteps = list(range(t_start, -1, -1))
        with torch.no_grad():
            abar = sde._compute_alpha_cumprod(
                torch.tensor(self.timesteps, device=device)).squeeze()
            self.sigmas = (1.0 - abar).sqrt().detach()
        self.sigma_L = float(self.sigmas[0].item())

    @torch.no_grad()
    def _predict_eps(self, samples_b2hw, t_idx):
        x_hw2    = samples_b2hw.permute(0, 2, 3, 1).contiguous()
        prior_in = self.prior_trafo(x_hw2)
        t_vec    = torch.full((samples_b2hw.shape[0],), t_idx,
                              device=self.device, dtype=torch.long)
        # Fix 3: scale to match score model training distribution O(target_scaling_factor)
        eps_prior = self.score(prior_in * self.prior_input_scale, t_vec)
        eps_hw2   = self.prior_trafo.trafo_inv(eps_prior)
        return eps_hw2.permute(0, 3, 1, 2).contiguous()

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y

        # Fix 2: MVUE via _ifft (ortho norm) — ref was built with _fft (ortho norm).
        # sp.ifft uses backward norm (÷H×W) → would underestimate scaling by sqrt(H×W).
        coil_imgs      = _ifft(ref)
        denom          = (maps.abs().pow(2).sum(1, keepdim=True) + 1e-8).sqrt()
        estimated_mvue = (coil_imgs * maps.conj()).sum(1) / denom.squeeze(1)

        logging.info(f"Running {len(self.timesteps)} DDPM timesteps × "
                     f"{self.config['n_steps_each']} inner Langevin steps "
                     f"(total {len(self.timesteps)*self.config['n_steps_each']})")

        # BraTS mask is 2D (B,H,W) → orientation='2d' → mask.ndim==3 branch
        forward_operator = lambda x: MulticoilForwardMRI("2d")(
            torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask
        )

        B = ref.shape[0]
        H, W = self.config["image_size"]
        samples = torch.rand(B, 2, H, W, device=self.device)

        step_lr = float(self.config["step_lr"])
        pbar    = tqdm(self.timesteps, desc="anneal", leave=False)
        pbar_labels = ["t", "step_size", "error", "mean", "max"]

        with torch.no_grad():
            for k, t_idx in enumerate(pbar):
                sigma        = float(self.sigmas[k].item())
                step_size    = step_lr * (sigma / self.sigma_L) ** 2
                n_steps_each = int(self.config["n_steps_each"])

                for _ in range(n_steps_each):
                    noise  = torch.randn_like(samples) * math.sqrt(step_size * 2)
                    eps    = self._predict_eps(samples, t_idx)
                    p_grad = -eps / (sigma + 1e-8)

                    meas      = forward_operator(normalize(samples, estimated_mvue))
                    meas_grad = torch.view_as_real(
                        torch.sum(_ifft(meas - ref) * torch.conj(maps), axis=1)
                    ).permute(0, 3, 1, 2)
                    meas_grad = unnormalize(meas_grad, estimated_mvue)
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor) \
                                if samples.is_cuda else meas_grad.float()
                    meas_grad /= torch.norm(meas_grad)
                    meas_grad *= torch.norm(p_grad)
                    meas_grad *= self.config["mse"]

                    samples = samples + step_size * (p_grad - meas_grad) + noise

                    err = (meas - ref).norm()
                    metrics = [t_idx, step_size, err.item(),
                               (p_grad - meas_grad).abs().mean().item(),
                               (p_grad - meas_grad).abs().max().item()]
                    pbar.set_description("; ".join(
                        f"{lbl}: {m:.6g}" for lbl, m in zip(pbar_labels, metrics)
                    ))

                    if torch.isnan(err):
                        logging.warning(f"NaN at t={t_idx} — early stop.")
                        return normalize(samples, estimated_mvue)

        return normalize(samples, estimated_mvue)

    def sample(self, y):
        return self._sample(y)


# ===========================================================================
# BraTS LMDB helpers.
# The BraTS LMDB is organized as separate LMDB directories per sub-db:
#   lmdb_path/shapes/   — JSON shape records, key = str(vol_idx)
#   lmdb_path/kspace/   — complex64 bytes
#   lmdb_path/maps/     — complex64 bytes
#   lmdb_path/target/   — complex64 bytes (shape[0] is channel dim, always 1)
#   lmdb_path/masks/    — complex64 bytes (note: "masks" not "mask")
# ===========================================================================
_ENV_CACHE: dict = {}


def _open_lmdb(path):
    key = str(Path(path).resolve())
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = lmdb.open(key, readonly=True, lock=False, readahead=False)
    return _ENV_CACHE[key]


def _lmdb_read(env, vk, shape, dtype=np.complex64):
    with env.begin() as txn:
        buf = txn.get(vk.encode())
    if buf is None:
        raise KeyError(f"key {vk!r} missing")
    return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()


def load_brats_volume(lmdb_path, vol_idx):
    """Load one BraTS val volume. Returns (kspace, maps, target, mask) all complex64.
    target is (X,Y,Z) — leading channel dim stripped. mask is (1,X,Y,Z)."""
    lmdb_path = Path(lmdb_path)
    vk = str(vol_idx)
    env_shapes = _open_lmdb(lmdb_path / "shapes")
    with env_shapes.begin() as txn:
        shp = json.loads(txn.get(vk.encode()).decode())
    kspace = _lmdb_read(_open_lmdb(lmdb_path / "kspace"), vk, tuple(shp["kspace"]))
    maps   = _lmdb_read(_open_lmdb(lmdb_path / "maps"),   vk, tuple(shp["maps"]))
    target = _lmdb_read(_open_lmdb(lmdb_path / "target"), vk, tuple(shp["target"]))
    mask   = _lmdb_read(_open_lmdb(lmdb_path / "masks"),  vk, tuple(shp["mask"]))
    return kspace, maps, target[0], mask   # target[0]: strip leading dim → (X,Y,Z)


def count_brats_volumes(lmdb_path):
    env = _open_lmdb(Path(lmdb_path) / "shapes")
    with env.begin() as txn:
        return txn.stat()["entries"]


# ===========================================================================
# Per-volume preprocessing — BraTS LMDB → CSGM-shaped tensors.
#   ref:   (1, C, H, W) complex   masked k-space (Fix 1: via _fft from target)
#   mvue:  (1, H, W) complex      for logging (not used inside _sample)
#   maps:  (1, C, H, W) complex   sensitivity maps
#   mask:  (1, H, W) float        2D phase-encode mask
#   gt_2d: (H, W, 2) float        ground-truth target at gt scale
# ===========================================================================
def brats_vol_to_csgm_2d(lmdb_path, vol_idx, device, readout_axis=0):
    kspace, maps, target, mask = load_brats_volume(lmdb_path, vol_idx)
    mid     = target.shape[readout_axis] // 2
    ax_czyz = 1 + readout_axis  # coil axis in (C, X, Y, Z)

    target_2d = np.take(target, mid, axis=readout_axis).real.astype(np.float32)  # (H,W)
    maps_2d   = np.take(maps,   mid, axis=ax_czyz).astype(np.complex64)          # (C,H,W)
    # mask is uniform along readout → any readout index gives same 2D mask
    mask_2d   = np.take(mask[0].real, 0, axis=readout_axis).astype(np.float32)   # (H,W)

    target_t = torch.from_numpy(target_2d).to(device)
    maps_t   = torch.from_numpy(maps_2d).to(device)
    mask_t   = torch.from_numpy(mask_2d).to(device)

    # Fix 1: recompute ref from clean target via _fft — stored kspace had
    # near-zero amplitude from 1D-IFFT approach, giving sf ≈ 1.7e9 (catastrophic).
    target_c  = torch.complex(target_t, torch.zeros_like(target_t))
    coil_imgs = target_c.unsqueeze(0) * maps_t    # (C,H,W) complex  [broadcast target]
    ksp_2d_t  = _fft(coil_imgs)                   # (C,H,W) complex
    ref_2d_t  = ksp_2d_t * mask_t.unsqueeze(0)    # (C,H,W) masked

    # MVUE for logging (uses get_mvue/sp.ifft — not used in _sample after Fix 2)
    mvue_np = get_mvue(ref_2d_t.cpu().numpy()[np.newaxis], maps_2d[np.newaxis])
    mvue_t  = torch.from_numpy(mvue_np).to(device)  # (1,H,W) complex

    ref    = ref_2d_t.unsqueeze(0)   # (1,C,H,W) complex
    maps_b = maps_t.unsqueeze(0)     # (1,C,H,W) complex
    mask_b = mask_t.unsqueeze(0)     # (1,H,W) float
    gt_2d  = torch.view_as_real(target_c.contiguous())  # (H,W,2) float
    return ref, mvue_t, maps_b, mask_b, gt_2d


# ===========================================================================
# Main
# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb",       type=str,   default=str(BRATS_VAL_LMDB))
    parser.add_argument("--ema_ckpt",   type=str,   default=str(EMA_CKPT))
    parser.add_argument("--train_cfg",  type=str,   default=str(TRAIN_CFG))
    parser.add_argument("--num_volumes", type=int,  default=1)
    parser.add_argument("--vol_start",  type=int,   default=0)
    parser.add_argument("--device",     type=str,   default="cuda:0")
    parser.add_argument("--no_wandb",   action="store_true")
    parser.add_argument("--readout_axis", type=int, default=0,
                        help="Fully-sampled readout axis in (X,Y,Z). 2D slice is the other two.")
    # CSGM hyperparameters (defaults from upstream brain config)
    parser.add_argument("--t_start",      type=int,   default=399,
                        help="Highest DDPM noise step (replaces NCSNv2 L=232).")
    parser.add_argument("--n_steps_each", type=int,   default=3,
                        help="Inner Langevin steps per noise level.")
    parser.add_argument("--step_lr",      type=float, default=5e-5,
                        help="Base Langevin step size.")
    parser.add_argument("--mse",          type=float, default=5.0,
                        help="DC weight λ on the meas-grad direction.")
    parser.add_argument("--prior_scaling_factor", type=float, default=1.0,
                        help="prior_trafo.scaling_factor (1.0 = no amplitude change).")
    args = parser.parse_args()
    device = args.device

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "brats_csgm"),
        name=os.environ.get("WANDB_NAME", None),
        mode="disabled" if args.no_wandb else "online",
        config=vars(args),
    )

    # ---- Score model + SDE + prior_trafo ------------------------------------
    score       = load_score_model(args.ema_ckpt, args.train_cfg, device)
    sde         = DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)
    prior_trafo = load_prior_trafo(args.train_cfg,
                                   scaling_factor=args.prior_scaling_factor)

    # Fix 3: load the training dataset scale so the score model receives
    # inputs at its training distribution O(target_scaling_factor).
    # This is distinct from prior_scaling_factor (the trafo's own amplitude scale).
    _tcfg = OmegaConf.load(args.train_cfg)
    prior_input_scale = float(
        OmegaConf.select(_tcfg, "dataset.target_scaling_factor", default=1.0)
    )
    logging.info(f"prior_input_scale (train_cfg.dataset.target_scaling_factor) "
                 f"= {prior_input_scale}")

    # ---- Volume loop --------------------------------------------------------
    n_total     = count_brats_volumes(args.lmdb)
    num_volumes = min(args.num_volumes, n_total - args.vol_start)
    logging.info(f"BraTS LMDB: {n_total} volumes at {args.lmdb}")

    fbp_psnrs, rec_psnrs = [], []
    fbp_ssims, rec_ssims = [], []

    for i in tqdm(range(num_volumes), desc="volumes"):
        vol_idx = args.vol_start + i
        ref, mvue, maps, mask_b, gt_2d = brats_vol_to_csgm_2d(
            args.lmdb, vol_idx, device, args.readout_axis
        )

        H, W = ref.shape[-2], ref.shape[-1]

        # Scale ref to unit-ish coordinate system (same formula as kno.py)
        ref_real_norm  = torch.view_as_real(ref.squeeze(0)).norm().item()
        scaling_factor = math.sqrt(float(H * W * 2)) / ref_real_norm

        ref_sf  = ref  * scaling_factor
        mvue_sf = mvue * scaling_factor

        if i == 0:
            logging.info(
                f"[vol {vol_idx}] ref={tuple(ref.shape)}  maps={tuple(maps.shape)}  "
                f"mask={tuple(mask_b.shape)}  gt={tuple(gt_2d.shape)}  "
                f"|ref|max={ref.abs().max():.4f}  |gt|max={gt_2d.abs().max():.4f}  "
                f"sf={scaling_factor:.6f}"
            )

        # ---- FBP (zero-filled SENSE) at sf scale, then unscale ---------------
        with torch.no_grad():
            coils_img = _ifft(ref_sf)                             # (1,C,H,W) complex
            fbp_hw    = (torch.conj(maps) * coils_img).sum(dim=1) # (1,H,W) complex
            fbp_sf_2d = torch.view_as_real(fbp_hw.squeeze(0))     # (H,W,2) at sf scale
            fbp_2d    = fbp_sf_2d / scaling_factor                # (H,W,2) at gt scale

        fbp_p = kno_psnr(fbp_2d.cpu(), gt_2d.cpu())
        fbp_s = kno_ssim(fbp_2d.cpu(), gt_2d.cpu())
        fbp_psnrs.append(fbp_p.item()); fbp_ssims.append(fbp_s.item())
        logging.info(f"[vol {vol_idx}] FBP  PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}")
        wandb.log({"fbp_psnr": fbp_p.item(), "fbp_ssim": fbp_s.item(),
                   "global_step": i})

        # ---- CSGM Langevin sampling ------------------------------------------
        config = {
            "device":       device,
            "image_size":   (H, W),
            "t_start":      args.t_start,
            "n_steps_each": args.n_steps_each,
            "step_lr":      args.step_lr,
            "mse":          args.mse,
        }
        optim = LangevinOptimizer(
            config, device, score, sde, prior_trafo,
            scaling_factor, prior_input_scale=prior_input_scale
        ).to(device)

        samples = optim.sample((ref_sf, mvue_sf, maps, mask_b))   # (1,2,H,W) at sf scale

        # ---- KNO metrics — unscale to gt scale --------------------------------
        rec_sf_2d = samples.squeeze(0).permute(1, 2, 0).contiguous()  # (H,W,2) at sf
        rec_2d    = (rec_sf_2d / scaling_factor).cpu()                 # (H,W,2) at gt
        rec_p     = kno_psnr(rec_2d, gt_2d.cpu())
        rec_s     = kno_ssim(rec_2d, gt_2d.cpu())
        rec_psnrs.append(rec_p.item()); rec_ssims.append(rec_s.item())
        logging.info(f"[vol {vol_idx}] CSGM PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}")
        wandb.log({
            "rec_psnr":      rec_p.item(),
            "rec_ssim":      rec_s.item(),
            "rec_psnr_mean": float(np.mean(rec_psnrs)),
            "rec_ssim_mean": float(np.mean(rec_ssims)),
            "global_step":   i,
        })

    logging.info(
        f"\n{'='*60}\n"
        f"  FBP  PSNR: {np.mean(fbp_psnrs):.2f} ± {np.std(fbp_psnrs):.2f} dB\n"
        f"  FBP  SSIM: {np.mean(fbp_ssims):.4f} ± {np.std(fbp_ssims):.4f}\n"
        f"  CSGM PSNR: {np.mean(rec_psnrs):.2f} ± {np.std(rec_psnrs):.2f} dB\n"
        f"  CSGM SSIM: {np.mean(rec_ssims):.4f} ± {np.std(rec_ssims):.4f}\n"
        f"{'='*60}"
    )

    if wandb.run is not None:
        wandb.run.summary["fbp_psnr_mean"] = float(np.mean(fbp_psnrs))
        wandb.run.summary["fbp_ssim_mean"] = float(np.mean(fbp_ssims))
        wandb.run.summary["rec_psnr_mean"] = float(np.mean(rec_psnrs))
        wandb.run.summary["rec_ssim_mean"] = float(np.mean(rec_ssims))

    wandb.finish()


if __name__ == "__main__":
    main()
