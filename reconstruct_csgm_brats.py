#!/usr/bin/env python3
"""
reconstruct_csgm_brats.py

Faithful CSGM (Jalal et al. 2021) reconstruction adapted to BraTS val data.

Built from reconstruct_csgm_skmtea.py (SKM-TEA version) by changing ONLY:
  1. Dataset: BraTS LMDB sub-dbs (kspace/, maps/, target/, shapes/, masks/) replace
     LMDBVolumeDataset. BraTS kspace is already masked with no mask bug.
  2. Geometry: readout_axis=0 (X is readout, fully sampled) → 2D slice is (Y,Z).
     Mask is 2D (Y,Z) from the Gaussian3DMaskFunc; MulticoilForwardMRI uses
     orientation='2d' to trigger the existing mask.ndim==3 broadcast branch.
  3. Target: precomputed clean magnitude image in [0,1] from LMDB target sub-db.
  4. Prior trafo: loaded from BraTS train_cfg (CroppedMagnitudeImagePriorTrafo
     with swap_channels=True, scaling_factor=1.0 — same net effect as SKM-TEA
     move_axis=[-1,1] but with no amplitude scaling).

Shares with reconstruct_csgm_skmtea.py:
  - annealed Langevin loop, per-step gradient direction normalization, Langevin update
  - DDPM schedule, KNO-project PSNR/SSIM metrics

Normalization (training-faithful, differs from the SKM-TEA quantile scheme):
  Each slice is scaled by the WHOLE-VOLUME factor sqrt(prod(vol_shape)) / ||target_3D||_F
  in brats_vol_to_csgm_2d (mirrors brats_slice_dataset + reconstruct_modified.py), so ref/gt
  live in the score model's training GT domain. There is NO 99th-percentile MVUE
  normalize/unnormalize (that was a leftover from the original NCSNv2 port and pushed the
  score input off its training basis). _predict_eps then multiplies by prior_input_scale (2.0)
  right before the score, reproducing the exact training input target_slice·sqrt(prod)/||·||·2.0.

Key scaling-factor note (score-model input scale = prior_input_scale, Fix 3):
  The current prior is the 3axis_downsample model (EMA_CKPT below), trained with
  target_scaling_factor=2.0. The score model must be fed inputs at that same
  training-distribution scale, so `_predict_eps` multiplies prior_in by
  prior_input_scale. When --prior_input_scale is not passed on the CLI, main()
  resolves it from the train cfg at
      problem_trafos.dataset_trafo.target_scaling_factor   (= 2.0 here)
  and FAILS LOUD if that key is missing (it never silently defaults to 1.0).
  This is distinct from --prior_scaling_factor, which is the prior_trafo's own
  amplitude scale (default 1.0). NOTE: the previous bs4 model used
  target_scaling_factor=5050 under a different cfg path — that model, and the
  old 5050 constant, are superseded (see EMA_CKPT comment). CLAUDE.md gotchas
  #14/#15 describe that old 5050 CSGM path.

Key fixes vs the original broken version:
  Fix 1: ref recomputed via _fft(target_2d) — stored kspace had 1D-IFFT artifact.
  Fix 3: samples * prior_input_scale in _predict_eps — score model training scale.
  (An earlier MVUE-normalization fix is obsolete: the 99th-pctile MVUE normalize/unnormalize
   was removed with the volume-RMS normalization change, so MVUE is no longer computed.)

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
# BraTS diffusion prior: 3axis_halfres (trained on 120x120x78 half-res slices,
# target_scaling_factor=2.0). Epoch 160 = latest. Closer to QR/half-res BraTS
# than 3axis_downsample (which was trained on full-res 240x240x155).
EMA_CKPT       = Path("/scratch/10471/peterwg/clean_brats_diff_models/outputs/3axis_halfres/"
                      "ema_model_160.pt")
TRAIN_CFG      = Path("/scratch/10471/peterwg/clean_brats_diff_models/outputs/3axis_halfres/"
                      ".hydra/config.yaml")


# ===========================================================================
# Forward operator (verbatim from csgm-mri-langevin/utils.py — unchanged)
# ===========================================================================
def _ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x


def _fft(x):
    # Centered fft2c: ifftshift -> fft2 -> fftshift. Must mirror _ifft so the pair is a
    # true forward/adjoint on ALL sizes. The upstream csgm order (fftshift->fft2->ifftshift)
    # is only correct for even N; on odd axes (e.g. BraTS Z=39) it is NOT the inverse of
    # _ifft and lands DC at ceil(N/2) instead of floor(N/2)=shape//2, mis-registering the
    # centered Gaussian mask and corrupting the data-consistency gradient along that axis.
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.fftshift(x, dim=(-2, -1))
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


# ===========================================================================
# KNO metrics — same as reconstruct_csgm_skmtea.py.
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


def kno_nmse(rec_hw2, gt_hw2):
    """rec, gt: (H, W, 2). NMSE = ||rec_mag - gt_mag||^2 / ||gt_mag||^2 — same
    formula as reconstruct_modified.py's _vol_nmse (canonical, fastmri convention)."""
    rec_mag = _to_mag(rec_hw2).float()
    gt_mag  = _to_mag(gt_hw2).float()
    return (rec_mag - gt_mag).pow(2).sum() / gt_mag.pow(2).sum()


# ===========================================================================
# Score model loader — same as reconstruct_csgm_skmtea.py.
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
# samples are fed to the score at their training scale (× prior_input_scale) in _predict_eps;
# ref/gt are pre-scaled to the volume-RMS training domain in brats_vol_to_csgm_2d.
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

        # Per-DDPM-level tracking (opt-in). When track=True, after each noise level
        # we record (t, sigma, step_size, psnr, ssim, nmse, |p_grad|, |meas_grad_raw|,
        # |meas_grad_used|, |p_grad-meas_grad|, err) into self.trace. gt_2d is the
        # volume-RMS-scaled (H,W,2) GT slice on `device` (metrics are scale-invariant).
        self.track = bool(config.get("track", False))
        self.gt_2d = config.get("gt_2d", None)
        self.trace = []

        # Ablation: --no_prior drops the prior's directional term from the Langevin
        # update ONLY. Default 1.0 → self.prior_scale*p_grad is byte-identical to p_grad.
        self.prior_scale = 0.0 if config.get("no_prior", False) else 1.0

        t_start = min(int(config["t_start"]), sde.num_steps - 1)
        self.timesteps = list(range(t_start, -1, -1))
        with torch.no_grad():
            abar = sde._compute_alpha_cumprod(
                torch.tensor(self.timesteps, device=device)).squeeze()
            self.sigmas = (1.0 - abar).sqrt().detach()
        # FIX (per reconstruct_csgm_skmtea.py "critical bug, now fixed"): divide by
        # sigmas[-1] (SMALLEST sigma, t≈0) not sigmas[0] (largest) — gives step
        # sizes LARGE at high noise / step_lr at lowest noise (correct annealed
        # Langevin behavior; sigmas[0] made steps ~8000x too small at high noise).
        self.sigma_min = float(self.sigmas[-1].item())

    @torch.no_grad()
    def _predict_eps(self, samples_b2hw, t_idx):
        # The BraTS prior_trafo is a pure channel-swap (swap_channels=True, no crop/
        # magnitude/scale), so the old permute→prior_trafo→...→trafo_inv→permute round-trip
        # was an identity. Just feed the score at its training distribution scale (×2.0).
        t_vec = torch.full((samples_b2hw.shape[0],), t_idx,
                           device=self.device, dtype=torch.long)
        return self.score(samples_b2hw * self.prior_input_scale, t_vec)

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y

        # ref/gt already live in the training GT domain (volume-RMS scaled in
        # brats_vol_to_csgm_2d), so samples stay in that same domain — no quantile
        # normalize/unnormalize. coil_imgs is the adjoint k-space, reused for the zf init.
        coil_imgs = _ifft(ref)

        logging.info(f"Running {len(self.timesteps)} DDPM timesteps × "
                     f"{self.config['n_steps_each']} inner Langevin steps "
                     f"(total {len(self.timesteps)*self.config['n_steps_each']})")

        # BraTS mask is 2D (B,H,W) → orientation='2d' → mask.ndim==3 branch
        forward_operator = lambda x: MulticoilForwardMRI("2d")(
            torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask
        )

        B = ref.shape[0]
        H, W = self.config["image_size"]
        if self.config.get("init_mode", "random") == "zf":
            # Adjoint-SENSE ZF reconstruction — same as SenseModel.backward() in the 3D project.
            zf_complex = (coil_imgs * maps.conj()).sum(dim=1)                          # (B,H,W) complex
            samples    = torch.view_as_real(zf_complex.contiguous()).permute(0, 3, 1, 2)  # (B,2,H,W)
        else:
            samples = torch.rand(B, 2, H, W, device=self.device)

        step_lr = float(self.config["step_lr"])
        pbar    = tqdm(self.timesteps, desc="anneal", leave=False)
        pbar_labels = ["t", "step_size", "error", "mean", "max"]

        with torch.no_grad():
            for k, t_idx in enumerate(pbar):
                sigma        = float(self.sigmas[k].item())
                step_size    = step_lr * (sigma / self.sigma_min) ** 2
                # brain_T1.yaml: n_steps_each=3 throughout (both c<=1800 and c>1800 branches
                # give 3 for brain anatomy). No dynamic split needed.
                n_steps_each = int(self.config["n_steps_each"])

                # Gradient-norm snapshot from the LAST inner Langevin step of this level.
                pgrad = dgrad_raw = dgrad_used = total_grad = float("nan")
                for _ in range(n_steps_each):
                    noise  = torch.randn_like(samples) * math.sqrt(step_size * 2)
                    eps    = self._predict_eps(samples, t_idx)
                    p_grad = -eps / (sigma + 1e-8)

                    meas      = forward_operator(samples)
                    meas_grad = torch.view_as_real(
                        torch.sum(_ifft(meas - ref) * torch.conj(maps), axis=1)
                    ).permute(0, 3, 1, 2)
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor) \
                                if samples.is_cuda else meas_grad.float()
                    dgrad_raw = float(torch.norm(meas_grad))   # DC gradient A^H(Ax-ref), pre-renorm
                    meas_grad /= torch.norm(meas_grad)
                    meas_grad *= torch.norm(p_grad)
                    meas_grad *= self.config["mse"]

                    samples = samples + step_size * (self.prior_scale * p_grad - meas_grad) + noise

                    pgrad      = float(torch.norm(p_grad))                 # prior/score gradient
                    dgrad_used = float(torch.norm(meas_grad))              # DC gradient used (= mse*|p_grad|)
                    total_grad = float(torch.norm(p_grad - meas_grad))     # net negative-loss drift

                    err = (meas - ref).norm()
                    metrics = [t_idx, step_size, err.item(),
                               (p_grad - meas_grad).abs().mean().item(),
                               (p_grad - meas_grad).abs().max().item()]
                    pbar.set_description("; ".join(
                        f"{lbl}: {m:.6g}" for lbl, m in zip(pbar_labels, metrics)
                    ))

                    if torch.isnan(err):
                        logging.warning(f"NaN at t={t_idx} — early stop.")
                        if self.track:
                            self._record_level(t_idx, sigma, step_size, samples,
                                                pgrad, dgrad_raw, dgrad_used, total_grad,
                                                float(err))
                        return samples

                if self.track:
                    self._record_level(t_idx, sigma, step_size, samples,
                                        pgrad, dgrad_raw, dgrad_used, total_grad,
                                        float(err))

        return samples

    # Column order for self.trace rows / saved npz (see main() TRACE_COLS).
    def _record_level(self, t_idx, sigma, step_size, samples,
                      pgrad, dgrad_raw, dgrad_used, total_grad, err):
        """Append one per-DDPM-level record. Metrics compare the running iterate
        (volume-RMS domain) to gt_2d; PSNR/SSIM/NMSE are scale-invariant."""
        rec_hw2 = samples.squeeze(0).permute(1, 2, 0)  # (H,W,2), on device
        if self.gt_2d is not None:
            psnr = float(kno_psnr(rec_hw2, self.gt_2d))
            ssim = float(kno_ssim(rec_hw2, self.gt_2d))
            nmse = float(kno_nmse(rec_hw2, self.gt_2d))
        else:
            psnr = ssim = nmse = float("nan")
        self.trace.append((int(t_idx), float(sigma), float(step_size),
                           psnr, ssim, nmse,
                           pgrad, dgrad_raw, dgrad_used, total_grad, err))

    def sample(self, y):
        return self._sample(y)


# ===========================================================================
# LangevinOptimizerStrided — "multi_step" sampler (n_jumps uniformly-spaced
# DDPM timesteps). Faithful port of reconstruct_csgm_skmtea.py's subclass: the
# ONLY override is __init__ (replaces the dense [t_start..0] timestep list with
# n_jumps uniformly spaced timesteps and recomputes sigmas/sigma_min). _sample/
# _predict_eps/sample are inherited UNCHANGED, so all BraTS-specific behavior
# (2D mask orientation='2d', prior_input_scale, volume-RMS scaling, metrics)
# carries over automatically.
# ===========================================================================
class LangevinOptimizerStrided(LangevinOptimizer):
    def __init__(self, config, device, score, sde, prior_trafo, scaling_factor,
                 prior_input_scale=1.0):
        super().__init__(config, device, score, sde, prior_trafo, scaling_factor,
                         prior_input_scale=prior_input_scale)
        n_jumps = int(config["n_jumps"])
        t_max   = self.timesteps[0]
        strided = sorted(
            set(int(round(t)) for t in np.linspace(t_max, 0, n_jumps)),
            reverse=True,
        )
        self.timesteps = strided
        with torch.no_grad():
            abar = sde._compute_alpha_cumprod(
                torch.tensor(self.timesteps, device=device)).squeeze()
            if abar.dim() == 0:
                abar = abar.unsqueeze(0)
            self.sigmas = (1.0 - abar).sqrt().detach()
        self.sigma_min = float(self.sigmas[-1].item())


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
#   mvue:  None                   (obsolete; kept for tuple/call-site stability)
#   maps:  (1, C, H, W) complex   sensitivity maps
#   mask:  (1, H, W) float        2D phase-encode mask
#   gt_2d: (H, W, 2) float        ground-truth target at gt scale
# ===========================================================================
def brats_vol_to_csgm_2d(lmdb_path, vol_idx, device, readout_axis=0, slice_idx=None):
    kspace, maps, target, mask = load_brats_volume(lmdb_path, vol_idx)
    mid     = slice_idx if slice_idx is not None else target.shape[readout_axis] // 2
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
    ksp_2d_t  = _fft(coil_imgs)                   # (C,H,W) complex  [UNMASKED full k-space]

    # Training-faithful normalization (mirror brats_slice_dataset + reconstruct_modified.py):
    # scale every slice by the WHOLE-VOLUME factor sqrt(prod(vol_shape)) / ||target_3D||_F,
    # NOT a per-slice norm. Training fed the model `target_slice * sqrt(prod(target_vol_shape))
    # / ||target_3D||_F * target_scaling_factor`; a per-slice RMS-1 norm would flatten the
    # inter-slice brightness the model learned and push the score input off its training basis.
    # Use the magnitude-image element count (no ×2) and the volume norm so the ×2.0 applied in
    # _predict_eps reproduces the exact training input scale. Parseval keeps ref consistent:
    # ref = A(gt_2d) after both are scaled by the same kspace_scale.
    gt_real      = torch.view_as_real(target_c.contiguous())   # (H,W,2) UNSCALED gt slice
    vol_norm     = float(np.linalg.norm(target.real))          # ||target_3D||_F (whole volume)
    kspace_scale = math.sqrt(float(target.size)) / max(vol_norm, 1e-8)

    ref_2d_t  = ksp_2d_t * mask_t.unsqueeze(0) * kspace_scale   # (C,H,W) masked + scaled
    gt_2d     = gt_real * kspace_scale                          # (H,W,2) scaled to match

    # MVUE is no longer consumed by _sample (the quantile normalize/unnormalize that used it
    # was removed). Skip the per-slice sigpy FFT — it also did 0/0 → NaN on background slices.
    # Kept in the return tuple as None for call-site/signature stability.
    mvue_t = None

    ref    = ref_2d_t.unsqueeze(0)   # (1,C,H,W) complex
    maps_b = maps_t.unsqueeze(0)     # (1,C,H,W) complex
    mask_b = mask_t.unsqueeze(0)     # (1,H,W) float
    return ref, mvue_t, maps_b, mask_b, gt_2d, kspace_scale


def _save_trace_npz(save_dir, vol_idx, vol_trace, cols):
    """Atomically (re)write the consolidated per-slice trace npz. Called after every
    slice so a manually-stopped / interrupted run keeps all completed slices."""
    out = {"columns": np.array(cols)}
    for s_key, arr in vol_trace.items():
        out[f"slice_{s_key}"] = arr
    tmp   = os.path.join(save_dir, f"vol{vol_idx}_trace.tmp.npz")   # ends in .npz → no auto-append
    final = os.path.join(save_dir, f"vol{vol_idx}_trace.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, final)


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
    parser.add_argument("--n_jumps",      type=int,   default=None,
                        help="If set, use LangevinOptimizerStrided with this many "
                             "uniformly-spaced timesteps (the 'multi_step' sampler). "
                             "If omitted, use the standard LangevinOptimizer ('t1').")
    parser.add_argument("--prior_scaling_factor", type=float, default=1.0,
                        help="INERT for BraTS: _predict_eps no longer routes through prior_trafo, "
                             "so this has no effect (must stay 1.0). The score-model amplitude is "
                             "set by --prior_input_scale instead. Passing !=1.0 fails loud.")
    parser.add_argument("--prior_input_scale", type=float, default=None,
                        help="Override for the score-model input scale. If omitted, resolved from "
                             "train_cfg problem_trafos.dataset_trafo.target_scaling_factor (=2.0 for "
                             "the 3axis prior); fails loud if that key is missing. See CLAUDE.md #15.")
    parser.add_argument("--init_mode", type=str, default="zf",
                        choices=["random", "zf"],
                        help="'zf': initialize from zero-filled SENSE reconstruction (default). "
                             "'random': uniform [0,1) init (original paper default).")
    parser.add_argument("--all_slices", action="store_true",
                        help="Reconstruct ALL readout slices per volume and report 3D "
                             "volume PSNR/SSIM/NMSE (comparable to KNO/diffusion tables). "
                             "~60× slower than the default mid-slice mode.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="If set, save per-volume GT/recon/FBP magnitude volumes (.npy) "
                             "and the per-slice/per-DDPM-level metric trace (.npz) here.")
    parser.add_argument("--track", action="store_true",
                        help="Record per-DDPM-level psnr/ssim/nmse + gradient norms "
                             "(pgrad, dgrad_raw, dgrad_used, total) for every slice. "
                             "Written to --save_dir; wandb gets per-slice + volume summaries.")
    parser.add_argument("--no_prior", action="store_true",
                        help="Ablation: drop the diffusion-prior term (p_grad) from the "
                             "Langevin update (data-consistency + noise only). Everything else "
                             "identical. Default OFF → recon is byte-identical to the paper run.")
    args = parser.parse_args()
    device = args.device

    # prior_trafo is no longer consumed by _predict_eps, so --prior_scaling_factor is inert.
    # Fail loud instead of silently ignoring a non-default value (would be a no-op footgun).
    if args.prior_scaling_factor != 1.0:
        raise ValueError(
            "--prior_scaling_factor is inert for BraTS (prior amplitude is handled by "
            "--prior_input_scale); leave it at 1.0. Got "
            f"{args.prior_scaling_factor}."
        )

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
    if args.prior_input_scale is not None:
        prior_input_scale = float(args.prior_input_scale)
        logging.info(f"prior_input_scale = {prior_input_scale}  (CLI override)")
    else:
        # The training target scale lives at problem_trafos.dataset_trafo.target_scaling_factor
        # (2.0 for 3axis_downsample, 5050.0 for the old bs4 model) — NOT under the top-level
        # `dataset:` block (which holds only name/paths/readout_axes). The previous path
        # "dataset.target_scaling_factor" resolved to that wrong block, so OmegaConf.select
        # silently returned default=1.0 → the score model was fed inputs target_scaling_factor×
        # too small and the prior went inert. Read the real path and FAIL LOUD on a miss so it
        # can never silently fall back to 1.0 again.
        _tcfg = OmegaConf.load(args.train_cfg)
        _path = "problem_trafos.dataset_trafo.target_scaling_factor"
        _val  = OmegaConf.select(_tcfg, _path, default=None)
        if _val is None:
            raise ValueError(
                f"Could not resolve '{_path}' in {args.train_cfg}. "
                f"Pass --prior_input_scale explicitly; do NOT let it silently default to 1.0."
            )
        prior_input_scale = float(_val)
        logging.info(f"prior_input_scale ({_path}) = {prior_input_scale}")

    # ---- Volume loop --------------------------------------------------------
    n_total     = count_brats_volumes(args.lmdb)
    num_volumes = min(args.num_volumes, n_total - args.vol_start)
    logging.info(f"BraTS LMDB: {n_total} volumes at {args.lmdb}")

    fbp_psnrs, rec_psnrs = [], []
    fbp_ssims, rec_ssims = [], []
    rec_nmses = []

    # Per-slice/per-DDPM-level trace columns (see LangevinOptimizer._record_level).
    TRACE_COLS = ["t", "sigma", "step_size", "psnr", "ssim", "nmse",
                  "pgrad", "dgrad_raw", "dgrad_used", "total", "err"]
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        logging.info(f"Saving GT/recon/FBP volumes + per-slice traces to {args.save_dir}")

    for i in tqdm(range(num_volumes), desc="volumes"):
        vol_idx = args.vol_start + i

        # Determine which slices to reconstruct.
        # all_slices=True  → loop over every readout slice, assemble 3D volume, compute 3D metrics.
        # all_slices=False → reconstruct mid-slice only, compute 2D metrics (fast default).
        if args.all_slices:
            _, _, target_vol, _ = load_brats_volume(args.lmdb, vol_idx)
            target_3d  = torch.from_numpy(target_vol.real).float()  # (X,Y,Z)
            slices_iter = list(range(target_3d.shape[args.readout_axis]))
        else:
            target_3d  = None
            slices_iter = [None]   # None → mid-slice in brats_vol_to_csgm_2d

        rec_mags, fbp_mags = [], []   # (H,W) magnitude tensors, unscaled — filled for all_slices
        vol_trace = {}                # slice_idx (or -1 for mid) → (n_levels, len(TRACE_COLS)) array

        for s_idx in tqdm(slices_iter, desc="slices", leave=False, disable=not args.all_slices):
            ref, mvue, maps, mask_b, gt_2d, kspace_scale = brats_vol_to_csgm_2d(
                args.lmdb, vol_idx, device, args.readout_axis, slice_idx=s_idx
            )
            H, W = ref.shape[-2], ref.shape[-1]

            # Skip fully-background readout slices (target≈0). Reconstructing one only injects
            # Langevin noise, and its zero DC residual makes `meas_grad /= ||meas_grad||` == 0/0
            # = NaN, which would poison the whole assembled volume's 3D metrics. Zero-fill instead.
            # (Detect via gt content: the old `isfinite(ref)` check relied on the per-slice norm
            # blowing kspace_scale to inf, but kspace_scale is now a finite volume constant, so
            # ref stays finite on background slices and that guard never fired.)
            if args.all_slices and float(gt_2d.abs().max()) < 1e-8:
                fbp_mags.append(torch.zeros(H, W)); rec_mags.append(torch.zeros(H, W))
                continue

            if i == 0 and s_idx in (None, 0):
                logging.info(
                    f"[vol {vol_idx}] ref={tuple(ref.shape)}  maps={tuple(maps.shape)}  "
                    f"mask={tuple(mask_b.shape)}  gt={tuple(gt_2d.shape)}  "
                    f"|ref|max={ref.abs().max():.4f}  |gt|max={gt_2d.abs().max():.4f}"
                )

            # ---- FBP (zero-filled SENSE) ----------------------------------------
            with torch.no_grad():
                coils_img = _ifft(ref)                              # (1,C,H,W) complex
                fbp_hw    = (torch.conj(maps) * coils_img).sum(dim=1)  # (1,H,W) complex
                fbp_sf_2d = torch.view_as_real(fbp_hw.squeeze(0))  # (H,W,2)

            # ---- CSGM Langevin sampling -----------------------------------------
            config = {
                "device":       device,
                "image_size":   (H, W),
                "t_start":      args.t_start,
                "n_steps_each": args.n_steps_each,
                "step_lr":      args.step_lr,
                "mse":          args.mse,
                "init_mode":    args.init_mode,
                "track":        args.track,
                "gt_2d":        gt_2d.to(device),
                "no_prior":     args.no_prior,
            }
            if args.n_jumps is not None:
                config["n_jumps"] = args.n_jumps
                Cls = LangevinOptimizerStrided
            else:
                Cls = LangevinOptimizer
            optim = Cls(config, device, score, sde, prior_trafo,
                        1.0, prior_input_scale=prior_input_scale).to(device)

            samples   = optim.sample((ref, mvue, maps, mask_b))         # (1,2,H,W)
            rec_sf_2d = samples.squeeze(0).permute(1, 2, 0).contiguous()  # (H,W,2)

            # ---- collect per-slice trace + per-slice wandb summary --------------
            if args.track and len(optim.trace):
                key = s_idx if s_idx is not None else -1
                vol_trace[key] = np.asarray(optim.trace, dtype=np.float32)
                if args.save_dir is not None:
                    _save_trace_npz(args.save_dir, vol_idx, vol_trace, TRACE_COLS)  # incremental
                final = optim.trace[-1]   # last DDPM level = converged slice metrics
                wandb.log({"slice_idx":  float(key),
                           "slice_psnr": final[3], "slice_ssim": final[4],
                           "slice_nmse": final[5], "slice_pgrad": final[6],
                           "slice_dgrad_raw": final[7], "slice_dgrad_used": final[8],
                           "slice_total_grad": final[9]})

            if args.all_slices:
                # Unscale to original [0,1] target domain for 3D assembly.
                # kspace_scale = sqrt(prod(vol_shape)) / ||target_3D||_F (volume constant); dividing undoes it.
                fbp_mags.append((_to_mag(fbp_sf_2d.cpu()) / kspace_scale))  # (H,W)
                rec_mags.append((_to_mag(rec_sf_2d.cpu()) / kspace_scale))  # (H,W)
            else:
                # 2D mid-slice metrics (not comparable to 3D baselines, but fast).
                fbp_p = kno_psnr(fbp_sf_2d.cpu(), gt_2d.cpu())
                fbp_s = kno_ssim(fbp_sf_2d.cpu(), gt_2d.cpu())
                rec_p = kno_psnr(rec_sf_2d.cpu(), gt_2d.cpu())
                rec_s = kno_ssim(rec_sf_2d.cpu(), gt_2d.cpu())
                rec_n = kno_nmse(rec_sf_2d.cpu(), gt_2d.cpu())

                fbp_psnrs.append(fbp_p.item()); fbp_ssims.append(fbp_s.item())
                rec_psnrs.append(rec_p.item()); rec_ssims.append(rec_s.item())
                rec_nmses.append(rec_n.item())

                logging.info(f"[vol {vol_idx}] FBP  PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}")
                logging.info(f"[vol {vol_idx}] CSGM PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}  NMSE={rec_n:.6f}")
                wandb.log({"fbp_psnr": fbp_p.item(), "fbp_ssim": fbp_s.item(),
                           "rec_psnr": rec_p.item(), "rec_ssim": rec_s.item(),
                           "rec_nmse": rec_n.item(),
                           "rec_psnr_mean": float(np.mean(rec_psnrs)),
                           "rec_ssim_mean": float(np.mean(rec_ssims)),
                           "rec_nmse_mean": float(np.mean(rec_nmses)),
                           "global_step": i})

                if args.save_dir is not None:
                    # mid-slice 2D magnitudes (symmetric with the all_slices save below)
                    np.save(os.path.join(args.save_dir, f"vol{vol_idx}_gt2d.npy"),
                            _to_mag(gt_2d.cpu()).numpy())
                    np.save(os.path.join(args.save_dir, f"vol{vol_idx}_recon2d.npy"),
                            _to_mag(rec_sf_2d.cpu()).numpy())
                    np.save(os.path.join(args.save_dir, f"vol{vol_idx}_fbp2d.npy"),
                            _to_mag(fbp_sf_2d.cpu()).numpy())

        if args.all_slices:
            # Assemble 3D volume and compute 3D metrics comparable to KNO/diffusion tables.
            # Per-slice (H,W) mags are the two non-readout axes, so stack them back along
            # readout_axis to recover (X,Y,Z) aligned with gt_3d (works for any readout_axis).
            rec_3d = torch.stack(rec_mags, dim=args.readout_axis).unsqueeze(0).unsqueeze(0)  # (1,1,X,Y,Z)
            fbp_3d = torch.stack(fbp_mags, dim=args.readout_axis).unsqueeze(0).unsqueeze(0)  # (1,1,X,Y,Z)
            gt_3d  = target_3d.abs().unsqueeze(0).unsqueeze(0)               # (1,1,X,Y,Z)

            fbp_p = _meddlr_metrics.psnr(fbp_3d, gt_3d).mean()
            fbp_s = _meddlr_metrics.ssim(fbp_3d, gt_3d).mean()
            rec_p = _meddlr_metrics.psnr(rec_3d, gt_3d).mean()
            rec_s = _meddlr_metrics.ssim(rec_3d, gt_3d).mean()
            rec_n = ((rec_3d - gt_3d).pow(2).sum() / gt_3d.pow(2).sum())

            fbp_psnrs.append(fbp_p.item()); fbp_ssims.append(fbp_s.item())
            rec_psnrs.append(rec_p.item()); rec_ssims.append(rec_s.item())
            rec_nmses.append(rec_n.item())

            logging.info(f"[vol {vol_idx}] FBP  3D PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}")
            logging.info(f"[vol {vol_idx}] CSGM 3D PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}  NMSE={rec_n:.6f}")
            wandb.log({"fbp_psnr": fbp_p.item(), "fbp_ssim": fbp_s.item(),
                       "rec_psnr": rec_p.item(), "rec_ssim": rec_s.item(),
                       "rec_nmse": rec_n.item(),
                       "rec_psnr_mean": float(np.mean(rec_psnrs)),
                       "rec_ssim_mean": float(np.mean(rec_ssims)),
                       "rec_nmse_mean": float(np.mean(rec_nmses)),
                       "global_step": i})

            if args.save_dir is not None:
                np.save(os.path.join(args.save_dir, f"vol{vol_idx}_gt.npy"),
                        gt_3d.squeeze().cpu().numpy())
                np.save(os.path.join(args.save_dir, f"vol{vol_idx}_recon.npy"),
                        rec_3d.squeeze().cpu().numpy())
                np.save(os.path.join(args.save_dir, f"vol{vol_idx}_fbp.npy"),
                        fbp_3d.squeeze().cpu().numpy())
                logging.info(f"[vol {vol_idx}] saved gt/recon/fbp volumes to {args.save_dir}")

        # ---- final trace consolidation (already written incrementally per slice) --
        if args.track and args.save_dir is not None and vol_trace:
            _save_trace_npz(args.save_dir, vol_idx, vol_trace, TRACE_COLS)
            logging.info(f"[vol {vol_idx}] saved trace for {len(vol_trace)} slices; "
                         f"columns={TRACE_COLS}")

    logging.info(
        f"\n{'='*60}\n"
        f"  FBP  PSNR: {np.mean(fbp_psnrs):.2f} ± {np.std(fbp_psnrs):.2f} dB\n"
        f"  FBP  SSIM: {np.mean(fbp_ssims):.4f} ± {np.std(fbp_ssims):.4f}\n"
        f"  CSGM PSNR: {np.mean(rec_psnrs):.2f} ± {np.std(rec_psnrs):.2f} dB\n"
        f"  CSGM SSIM: {np.mean(rec_ssims):.4f} ± {np.std(rec_ssims):.4f}\n"
        f"  CSGM NMSE: {np.mean(rec_nmses):.6f} ± {np.std(rec_nmses):.6f}\n"
        f"{'='*60}"
    )

    if wandb.run is not None:
        wandb.run.summary["fbp_psnr_mean"] = float(np.mean(fbp_psnrs))
        wandb.run.summary["fbp_ssim_mean"] = float(np.mean(fbp_ssims))
        wandb.run.summary["rec_psnr_mean"] = float(np.mean(rec_psnrs))
        wandb.run.summary["rec_ssim_mean"] = float(np.mean(rec_ssims))
        wandb.run.summary["rec_nmse_mean"] = float(np.mean(rec_nmses))

    wandb.finish()


if __name__ == "__main__":
    main()
