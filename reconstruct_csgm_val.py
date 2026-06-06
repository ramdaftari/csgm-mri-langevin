#!/usr/bin/env python3
"""
reconstruct_csgm_val.py

CSGM (Jalal et al. 2021, "Robust Compressed Sensing MRI with Deep Generative Priors")
reconstruction on the middle Z-slice of KNO LMDB val volumes.

Algorithm — Annealed Langevin dynamics, per 2D slice (Jalal et al. Algorithm 1):
  x_T ~ N(0, I)
  for l = t_start, t_start-1, ..., 0:
    step_size = step_lr * (sigma_l / sigma_L)^2
    for s = 1, ..., n_steps_each:
      eps   = score_theta(x, l)            # noise prediction via DDPM score model
      p_grad = -eps / sigma_l              # prior (score) gradient
      meas_grad = A^T(A(x) - y)           # data consistency gradient
      meas_grad = meas_grad / ||meas_grad|| * ||p_grad|| * dc_weight   # CSGM norm
      z = N(0, I)
      x = x + step_size*(p_grad - meas_grad) + sqrt(2*step_size)*z

Preprocessing (LMDB loading, (1-1j)/2 correction, sensitivity maps, mask) and
metrics (PSNR, SSIM) are identical to reconstruct_kno_val.py.
The 3D Gaussian-masked k-space from LMDBVolumeDataset is used; the middle
Z-slice is extracted after volume loading for all 2D operations.

Run from baselines/resolution_robust_3d_mri/:
    .venv/bin/python reconstruct_csgm_val.py --num_volumes 1 --no_wandb
    .venv/bin/python reconstruct_csgm_val.py --num_volumes 5 --t_start 399 --dc_weight 5.0
"""

import sys
import math
import logging
import argparse
from pathlib import Path
from itertools import islice

import numpy as np
import torch
import wandb
import fastmri
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: KNO venv first (for torchmetrics 0.11.4), KNO src last
# ---------------------------------------------------------------------------
KNO_SRC  = Path(__file__).resolve().parents[2]
KNO_VENV = Path(__file__).resolve().parents[3] / ".venv/lib/python3.10/site-packages"
sys.path.insert(0, str(KNO_VENV))
sys.path.append(str(KNO_SRC))

# meddlr (imported transitively) uses deprecated np.complex alias
if not hasattr(np, 'complex'):
    np.complex = complex  # type: ignore[attr-defined]

import meddlr.metrics.functional as _meddlr_metrics

import fastmri.data.subsample as _fmri_sub
sys.modules.setdefault("fastmri.subsample", _fmri_sub)

# ---------------------------------------------------------------------------
# res_rob imports — only prior_trafo needed (fwd model implemented inline 2D)
# ---------------------------------------------------------------------------
from src.problem_trafos.trafo_resolver import get_prior_trafo
from src.diffmodels.diffmodels_resolver import create_dense_model
from src.diffmodels.ema import ExponentialMovingAverage
from src.diffmodels.sde import DDPM

# ---------------------------------------------------------------------------
# KNO dataset
# ---------------------------------------------------------------------------
from datasets.skmtea import LMDBVolumeDataset


# ---------------------------------------------------------------------------
# Constants — identical to reconstruct_kno_val.py
# ---------------------------------------------------------------------------
VAL_LMDB   = Path("/scratch/10846/armeet/datasets/skmtea_val_0.5first2_4x_lmdb")
EMA_CKPT   = Path("recon_workdir/ema_model_110.pt")
TRAIN_CFG  = Path("recon_workdir/.hydra/config.yaml")
HYDRA_ROOT = Path("hydra")


# ---------------------------------------------------------------------------
# KNO-style PSNR/SSIM — identical formula to reconstruct_kno_val.py.
# Works on any spatial shape because _kno_psnr_fn flattens spatial dims.
# ---------------------------------------------------------------------------
def _kno_psnr_fn(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    B, C = pred_mag.shape[:2]
    pred_flat   = pred_mag.view(B, C, -1).float()
    target_flat = target_mag.view(B, C, -1).float()
    rmse    = (pred_flat - target_flat).pow(2).mean(dim=-1).sqrt()
    max_val = target_flat.abs().amax(dim=-1)
    return (20.0 * torch.log10(max_val / (rmse + 1e-8))).mean()


def _to_mag(t: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(t.contiguous()).abs() if t.shape[-1] == 2 else t.abs()


def slice_psnr(rec: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """rec, gt: (X, Y, 2). Returns scalar KNO PSNR (dB)."""
    rec_mag = _to_mag(rec).unsqueeze(0).unsqueeze(0)   # (1, 1, X, Y)
    gt_mag  = _to_mag(gt).unsqueeze(0).unsqueeze(0)
    return _kno_psnr_fn(rec_mag, gt_mag).squeeze()


def slice_ssim(rec: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """rec, gt: (X, Y, 2). Returns scalar SSIM (meddlr, same as KNO)."""
    rec_mag = _to_mag(rec).unsqueeze(0).unsqueeze(0)   # (1, 1, X, Y)
    gt_mag  = _to_mag(gt).unsqueeze(0).unsqueeze(0)
    return _meddlr_metrics.ssim(rec_mag, gt_mag).mean()


# ---------------------------------------------------------------------------
# Score model loader — identical to reconstruct_kno_val.py
# ---------------------------------------------------------------------------
def load_score_model(ckpt_path: Path, train_cfg_path: Path, device: str) -> torch.nn.Module:
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
    logging.info(f"Loaded EMA score model from {ckpt_path}")
    return score


# ---------------------------------------------------------------------------
# Prior trafo — identical config to reconstruct_kno_val.py.
# Converts (1, X, Y, 2) ↔ (1, 2, X, Y) for the score model:
#   forward:  x * 2.0  then  moveaxis(-1, 1)
#   inverse:  moveaxis(1, -1)  then  / 2.0
# ---------------------------------------------------------------------------
def _load_prior_trafo():
    prior_cfg = OmegaConf.load(HYDRA_ROOT / "problem_trafos/prior_trafo/crop_mag.yaml")
    OmegaConf.update(prior_cfg, "scaling_factor", 2.0)
    OmegaConf.update(prior_cfg, "move_axis", [-1, 1])
    _SKIP = {"name", "defaults"}
    prior_kwargs = {k: v for k, v in OmegaConf.to_container(prior_cfg).items()
                   if k not in _SKIP}
    return get_prior_trafo(name=prior_cfg.name, **prior_kwargs)


# ---------------------------------------------------------------------------
# DDPM helpers
# ---------------------------------------------------------------------------
def _abar(sde: DDPM, t: int, device: str) -> torch.Tensor:
    """ᾱ_t = cumprod(1-β) up to step t. Returns scalar tensor."""
    return sde._compute_alpha_cumprod(torch.tensor([t], device=device)).squeeze()


# ---------------------------------------------------------------------------
# 2D SENSE forward / adjoint (inline, for per-slice CSGM)
#
# Forward  A : (X, Y, 2) → (C, X, Y, 2)
#   x_c      = view_as_complex(x)                  # (X, Y) complex
#   coil_imgs = sens_2d * x_c                       # (C, X, Y) complex  [broadcast]
#   ksp       = fft2c(view_as_real(coil_imgs))      # (C, X, Y, 2)
#   return ksp * mask_2d                            # mask_2d: (Y,1) → broadcasts
#
# Adjoint  A^H : (C, X, Y, 2) → (X, Y, 2)
#   coil_imgs_c = view_as_complex(ifft2c(y * mask_2d))   # (C, X, Y) complex
#   x_c         = sum_c  conj(S_c) * coil_imgs_c          # (X, Y) complex
#   return view_as_real(x_c)                              # (X, Y, 2)
#
# mask_2d shape (Y, 1) broadcasts correctly over (C, X, Y, 2):
#   PyTorch pads on the left → (1, 1, Y, 1), then broadcasts to (C, X, Y, 2).
# ---------------------------------------------------------------------------
def sense_forward_2d(
    x: torch.Tensor,          # (X, Y, 2)  real-as-complex
    sens_2d: torch.Tensor,    # (C, X, Y)  complex
    mask_2d: torch.Tensor,    # (Y, 1)     float binary
) -> torch.Tensor:            # (C, X, Y, 2)
    x_c       = torch.view_as_complex(x.contiguous())           # (X, Y)
    coil_imgs = sens_2d * x_c.unsqueeze(0)                      # (C, X, Y)
    ksp       = fastmri.fft2c(torch.view_as_real(coil_imgs))    # (C, X, Y, 2)
    return ksp * mask_2d                                         # broadcast (Y,1)


def sense_adjoint_2d(
    y: torch.Tensor,          # (C, X, Y, 2)  real-as-complex
    sens_2d: torch.Tensor,    # (C, X, Y)     complex
    mask_2d: torch.Tensor,    # (Y, 1)        float binary
) -> torch.Tensor:            # (X, Y, 2)
    coil_imgs_c = torch.view_as_complex(
        fastmri.ifft2c(y * mask_2d).contiguous()
    )                                                            # (C, X, Y)
    x_c = (sens_2d.conj() * coil_imgs_c).sum(dim=0)            # (X, Y)
    return torch.view_as_real(x_c.contiguous())                 # (X, Y, 2)


# ---------------------------------------------------------------------------
# Noise prediction for a single 2D slice via the score model.
#
# prior_trafo maps (1, X, Y, 2) → (1, 2, X, Y) for the score model:
#   forward:  x * 2  then  moveaxis(-1, 1)
#   inverse:  moveaxis(1, -1)  then  / 2
# The returned eps is in the same representation space as x_t.
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_noise_2d(
    x_t: torch.Tensor,          # (X, Y, 2)
    t_idx: int,
    score: torch.nn.Module,
    prior_trafo,
    device: str,
) -> torch.Tensor:              # (X, Y, 2)
    t_vec      = torch.tensor([t_idx], device=device)
    prior_in   = prior_trafo(x_t.unsqueeze(0))        # (1, 2, X, Y)
    eps_prior  = score(prior_in, t_vec)               # (1, 2, X, Y)
    eps_repr   = prior_trafo.trafo_inv(eps_prior)     # (1, X, Y, 2)
    return eps_repr.squeeze(0)                        # (X, Y, 2)


# ---------------------------------------------------------------------------
# Annealed Langevin step — Jalal et al. 2021 (CSGM), Algorithm 1 exactly.
#
# From the CSGM reference implementation (main.py lines 131-151):
#   noise      = randn * sqrt(2 * step_size)
#   p_grad     = score(x, sigma_l)                  # ≈ -eps / sigma_l (DDPM)
#   meas       = A(normalize(x, mvue))              # A(x / ||mvue||)
#   meas_grad  = A^T(meas - y_norm)                 # adjoint of residual
#   meas_grad  = unnormalize(meas_grad, mvue)        # * ||mvue||
#   meas_grad /= ||meas_grad||
#   meas_grad *= ||p_grad||
#   meas_grad *= dc_weight
#   x_new = x + step_size * (p_grad - meas_grad) + noise
#
# Note: normalize(x, mvue) = x / ||mvue|| and then unnormalize(grad) = grad * ||mvue||
# cancel identically in the gradient computation:
#   A^T(A(x/||mvue||) - y/||mvue||) * ||mvue||  =  A^T(A(x) - y)
# So we compute  meas_grad = A^T(A(x) - y)  directly, which is algebraically
# equivalent. The subsequent per-step normalization (/ ||meas_grad||) is what
# the CSGM paper specifies regardless of the MVUE normalization.
#
# step_size = step_lr * (sigma_l / sigma_L)^2   (CSGM step size schedule)
# ---------------------------------------------------------------------------
def csgm_langevin_step_2d(
    x_t: torch.Tensor,         # (X, Y, 2) in representation space
    eps: torch.Tensor,         # (X, Y, 2) noise pred in representation space
    t_idx: int,
    sde: DDPM,
    sens_2d: torch.Tensor,     # (C, X, Y) complex
    mask_2d: torch.Tensor,     # (Y, 1) binary
    obs_2d: torch.Tensor,      # (C, X, Y, 2) observed k-space (unscaled)
    scaling_factor: float,     # same formula as reconstruct_kno_val.py
    dc_weight: float,          # λ in CSGM (mse_weight), default 5.0
    step_lr: float,            # base step size, default 5e-5
    sigma_L: float,            # sigma at t_start (= sqrt(1 - ᾱ_{t_start}))
    device: str,
):
    abar_t  = _abar(sde, t_idx, device)
    sigma_t = float((1.0 - abar_t).sqrt())

    # ------------------------------------------------------------------
    # 1. Prior gradient: score ≈ -eps / sigma_t  (DDPM score approximation)
    #    Equivalent to s_θ(x, sigma_l) in the CSGM paper.
    # ------------------------------------------------------------------
    p_grad = -eps / (sigma_t + 1e-8)                    # (X, Y, 2)

    # ------------------------------------------------------------------
    # 2. Step size: α_l = step_lr * (σ_l / σ_L)^2   (CSGM schedule)
    # ------------------------------------------------------------------
    step_size = step_lr * (sigma_t / (sigma_L + 1e-8)) ** 2

    # ------------------------------------------------------------------
    # 3. Data consistency gradient (see algebraic equivalence note above)
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred_ksp  = sense_forward_2d(x_t, sens_2d, mask_2d)       # (C, X, Y, 2) at sf scale
        residual  = pred_ksp - obs_2d * scaling_factor             # both at sf scale
        meas_grad = sense_adjoint_2d(residual, sens_2d, mask_2d)   # (X, Y, 2) at sf scale

    res_norm = residual.norm().item()

    # CSGM per-step gradient normalization (Jalal et al. main.py lines 146-149):
    #   meas_grad /= ||meas_grad||  →  unit direction
    #   meas_grad *= ||p_grad||     →  rescale to prior gradient magnitude
    #   meas_grad *= dc_weight      →  apply measurement weight λ
    meas_grad = (meas_grad / (meas_grad.norm() + 1e-8)) * p_grad.norm() * dc_weight

    # ------------------------------------------------------------------
    # 4. Langevin update (CSGM line 151):
    #    noise  = randn * sqrt(2 * step_size)
    #    x_new  = x + step_size * (p_grad - meas_grad) + noise
    # ------------------------------------------------------------------
    noise  = torch.randn_like(x_t) * math.sqrt(max(2.0 * step_size, 0.0))
    x_next = x_t + step_size * (p_grad - meas_grad) + noise

    return x_next, res_norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_volumes",  type=int,   default=1)
    parser.add_argument("--vol_start",   type=int,   default=0)
    parser.add_argument("--device",      type=str,   default="cuda:0")
    parser.add_argument("--no_wandb",    action="store_true")
    parser.add_argument("--t_start",     type=int,   default=399,
                        help="Highest noise level. Score model trained with "
                             "steps_scaler=0.4 → valid range [1, 399].")
    parser.add_argument("--n_steps_each", type=int,  default=3,
                        help="Inner Langevin steps per noise level (CSGM default: 3).")
    parser.add_argument("--step_lr",     type=float, default=5e-5,
                        help="Base step size multiplier (CSGM default: 5e-5).")
    parser.add_argument("--dc_weight",   type=float, default=5.0,
                        help="Measurement consistency weight λ (CSGM default: 5.0).")
    args   = parser.parse_args()
    device = args.device

    wandb.init(project="skmtea_csgm_recon", name="csgm_val_2d_midslice",
               mode="disabled" if args.no_wandb else "online",
               config=vars(args))

    # ---- Prior trafo (identical config to reconstruct_kno_val.py) -----------
    prior_trafo = _load_prior_trafo()

    # ---- Score model + SDE (same checkpoint as reconstruct_kno_val.py) ------
    score = load_score_model(EMA_CKPT, TRAIN_CFG, device)
    score.eval()
    sde = DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)

    t_start   = min(args.t_start, 399)
    timesteps = list(range(t_start, -1, -1))          # [t_start, ..., 0]
    # σ_L = sqrt(1 - ᾱ_{t_start}): noise std at the highest level (CSGM σ_L)
    sigma_L   = float((1.0 - _abar(sde, t_start, device)).sqrt())
    logging.info(f"CSGM: t_start={t_start}, sigma_L={sigma_L:.4f}, "
                 f"n_steps_each={args.n_steps_each}, "
                 f"step_lr={args.step_lr}, dc_weight={args.dc_weight}, "
                 f"total inner steps={len(timesteps) * args.n_steps_each}")

    # ---- Dataset — identical to reconstruct_kno_val.py ----------------------
    dataset = LMDBVolumeDataset(root_dir=VAL_LMDB, norm_constant=3e7)
    logging.info(f"Val LMDB: {len(dataset)} volumes at {VAL_LMDB}")
    num_volumes = min(args.num_volumes, len(dataset) - args.vol_start)

    fbp_psnrs, rec_psnrs = [], []
    fbp_ssims, rec_ssims = [], []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(
        tqdm(islice(loader, args.vol_start, args.vol_start + num_volumes),
             total=num_volumes, desc="volumes")
    ):
        # ------------------------------------------------------------------
        # Preprocessing — identical to reconstruct_kno_val.py
        # kspace stored as m*(1+j)*k; correction m*(1+j)*k*(1-1j)/2 = m*k
        # ------------------------------------------------------------------
        kspace = batch.kspace.squeeze(0)    # (C, X, Y, Z) complex64
        maps   = batch.maps.squeeze(0)      # (C, X, Y, Z) complex64
        target = batch.target.squeeze(0)    # (1, X, Y, Z) complex64
        mask   = batch.mask.squeeze(0)      # (1, X, Y, Z) complex64

        kspace_clean = kspace * torch.tensor((1 - 1j) / 2, dtype=torch.complex64)

        dtype = torch.get_default_dtype()

        # Full-volume observation (C, X, Y, Z, 2) and ground truth (X, Y, Z, 2)
        observation  = torch.view_as_real(kspace_clean).to(dtype=dtype, device=device)
        ground_truth = torch.view_as_real(target.squeeze(0)).to(dtype=dtype, device=device)

        if i == 0:
            logging.info(f"kspace shape:  {kspace.shape}")
            logging.info(f"kspace stored   |max|={kspace.abs().max():.4f}")
            logging.info(f"kspace cleaned  |max|={kspace_clean.abs().max():.4f}")
            logging.info(f"target          |max|={target.abs().max():.4f}")

        # Binary mask (Y, Z, 1) — same extraction as reconstruct_kno_val.py.
        # mask.real[0, 0] takes the first-X row (mask is constant along X).
        mask_binary = mask.real[0, 0, :, :].unsqueeze(-1)    # (Y, Z, 1)

        # ------------------------------------------------------------------
        # Extract middle Z slice — CSGM operates per 2D slice.
        # All subsequent operations are 2D.
        # ------------------------------------------------------------------
        Z     = kspace_clean.shape[-1]
        z_mid = Z // 2

        # Observed k-space for the middle slice: (C, X, Y, 2)
        obs_2d  = observation[:, :, :, z_mid, :]              # on device
        # Ground truth for the middle slice: (X, Y, 2)
        gt_2d   = ground_truth[:, :, z_mid, :]                # on device
        # Sensitivity maps for the middle slice: (C, X, Y) complex
        sens_2d = maps[:, :, :, z_mid].to(device)             # complex
        # 1D phase-encode mask for this slice: (Y, 1)
        mask_2d = mask_binary[:, z_mid, :].to(device)         # (Y, 1)

        # Scaling factor — identical formula to reconstruct_kno_val.py, applied to
        # the 2D rep shape.  Brings filtbackproj * sf to the same scale as gt_2d,
        # compensating for the 2D vs 3D IFFT normalisation difference.
        X, Y = obs_2d.shape[1], obs_2d.shape[2]
        scaling_factor = (
            math.sqrt(float(X * Y * 2))
            / obs_2d.detach().cpu().norm().item()
        )

        if i == 0:
            logging.info(f"Middle slice z={z_mid}  "
                         f"obs_2d={tuple(obs_2d.shape)}  gt_2d={tuple(gt_2d.shape)}  "
                         f"scaling_factor={scaling_factor:.6f}")

        # ------------------------------------------------------------------
        # Zero-filled SENSE reconstruction (filtbackproj) for this slice.
        # Scaled by scaling_factor so it lives in the same space as x_t.
        # FBP metrics unscale back: filtbackproj / sf ≈ gt_2d scale.
        # ------------------------------------------------------------------
        with torch.no_grad():
            filtbackproj_2d = sense_adjoint_2d(obs_2d, sens_2d, mask_2d)  # (X, Y, 2)

        fbp_p = slice_psnr((filtbackproj_2d * scaling_factor).cpu(), gt_2d.cpu())
        fbp_s = slice_ssim((filtbackproj_2d * scaling_factor).cpu(), gt_2d.cpu())
        fbp_psnrs.append(fbp_p.item())
        fbp_ssims.append(fbp_s.item())
        logging.info(f"[vol {i}] FBP  PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}")
        wandb.log({"fbp_psnr_kno": fbp_p.item(), "fbp_ssim_kno": fbp_s.item(),
                   "global_step": i, "step": i})

        # ------------------------------------------------------------------
        # Initialisation: x_T ~ N(0, I) in sf-scaled space — CSGM random init.
        # filtbackproj * sf has unit-ish norm per element, so N(0,I) is at the
        # right DDPM noise scale (sigma_L ≈ 0.89 at t_start=399).
        # ------------------------------------------------------------------
        x_t = torch.randn(X, Y, 2, device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # Annealed Langevin loop — Jalal et al. 2021, Algorithm 1 exactly.
        # Outer loop: noise levels from t_start down to 0.
        # Inner loop: n_steps_each Langevin steps per noise level.
        # ------------------------------------------------------------------
        nan_detected = False
        pbar        = tqdm(timesteps, desc=f"vol {i} CSGM", leave=False)
        inner_step  = 0

        for t_idx in pbar:
            for s in range(args.n_steps_each):
                # 1. Noise prediction (prior gradient) via score model
                eps = predict_noise_2d(x_t, t_idx, score, prior_trafo, device)

                # 2. Annealed Langevin step (CSGM exactly)
                x_t, res_norm = csgm_langevin_step_2d(
                    x_t            = x_t,
                    eps            = eps,
                    t_idx          = t_idx,
                    sde            = sde,
                    sens_2d        = sens_2d,
                    mask_2d        = mask_2d,
                    obs_2d         = obs_2d,
                    scaling_factor = scaling_factor,
                    dc_weight      = args.dc_weight,
                    step_lr        = args.step_lr,
                    sigma_L        = sigma_L,
                    device         = device,
                )

                if math.isnan(res_norm):
                    logging.warning(f"NaN at t={t_idx}, s={s} — stopping early.")
                    nan_detected = True
                    break

                if inner_step % 50 == 0:
                    with torch.no_grad():
                        p = slice_psnr((x_t / scaling_factor).cpu(), gt_2d.cpu())
                    wandb.log({"rec_psnr_kno": p.item(), "global_step": inner_step})
                    pbar.set_postfix_str(f"t={t_idx} res={res_norm:.3f} psnr={p:.1f}dB")

                inner_step += 1

            if nan_detected:
                break

        # ------------------------------------------------------------------
        # Final reconstruction metrics — unscale x_t back to image space.
        # ------------------------------------------------------------------
        with torch.no_grad():
            x_final = x_t / scaling_factor
            rec_p = slice_psnr(x_final.cpu(), gt_2d.cpu())
            rec_s = slice_ssim(x_final.cpu(), gt_2d.cpu())

        rec_psnrs.append(rec_p.item())
        rec_ssims.append(rec_s.item())
        logging.info(f"[vol {i}] REC  PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}")
        wandb.log({
            "rec_psnr_kno_final": rec_p.item(),
            "rec_ssim_kno_final": rec_s.item(),
            "rec_psnrs_kno_mean": float(np.mean(rec_psnrs)),
            "rec_ssims_kno_mean": float(np.mean(rec_ssims)),
            "global_step": i,
        })

    # ---- Summary -------------------------------------------------------------
    logging.info(
        f"\n{'='*60}\n"
        f"  FBP  PSNR: {np.mean(fbp_psnrs):.2f} ± {np.std(fbp_psnrs):.2f} dB\n"
        f"  FBP  SSIM: {np.mean(fbp_ssims):.4f} ± {np.std(fbp_ssims):.4f}\n"
        f"  CSGM PSNR: {np.mean(rec_psnrs):.2f} ± {np.std(rec_psnrs):.2f} dB\n"
        f"  CSGM SSIM: {np.mean(rec_ssims):.4f} ± {np.std(rec_ssims):.4f}\n"
        f"{'='*60}"
    )

    if wandb.run is not None:
        wandb.run.summary["fbp_psnrs_mean"] = float(np.mean(fbp_psnrs))
        wandb.run.summary["fbp_ssims_mean"] = float(np.mean(fbp_ssims))
        wandb.run.summary["rec_psnrs_mean"] = float(np.mean(rec_psnrs))
        wandb.run.summary["rec_ssims_mean"] = float(np.mean(rec_ssims))

    wandb.finish()


if __name__ == "__main__":
    main()
