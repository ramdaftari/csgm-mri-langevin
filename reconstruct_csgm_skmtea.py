#!/usr/bin/env python3
"""
reconstruct_csgm_skmtea.py

Faithful CSGM (Jalal et al. 2021, "Robust Compressed Sensing MRI with Deep
Generative Priors") reconstruction adapted to SKM-TEA val data.
(Formerly reconstruct_csgm_kno.py — renamed: nothing here is KNO-specific, it is
plain CSGM on the SKM-TEA LMDB val set. This is the production version imported by
recon_volume.py / sweep_csgm_kno.py; run entrypoints live in resolution_robust_3d_mri/.)

Built by starting from csgm-mri-langevin/main.py and changing ONLY:
  1. Diffusion prior: SKM-TEA-trained DDPM (res_rob create_dense_model + EMA)
     replaces brain-trained NCSNv2. The NCSNv2 sigma sweep becomes a DDPM
     timestep sweep with sigma_t = sqrt(1 - abar_t) and p_grad = -eps / sigma_t.
  2. Dataset + mask: LMDBVolumeDataset (3D Gaussian mask, m*(1+j)*k storage with
     the (1-1j)/2 correction) replaces MVU_Estimator_Brain. CSGM is a 2D method
     so we extract the middle Z-slice; the slice's (Y,) mask matches CSGM's
     'vertical' MulticoilForwardMRI orientation.
  3. Metrics: the KNO-project PSNR (20*log10(max|gt| / RMSE)) + meddlr SSIM,
     computed on the magnitude of the final reconstruction.

The annealed Langevin loop, normalize/unnormalize via the 99th-percentile MVUE,
the per-step gradient direction normalization
    meas_grad /= ||meas_grad||;  meas_grad *= ||p_grad||;  meas_grad *= mse
and the update
    x_new = x + step_size * (p_grad - meas_grad) + sqrt(2*step_size)*noise
are byte-for-byte the upstream CSGM algorithm.

Run from baselines/resolution_robust_3d_mri/:
    # One volume (quick test):
    python reconstruct_csgm_skmtea.py --dataset half_4x --num_volumes 1 --no_wandb
    # Entire dataset:
    python reconstruct_csgm_skmtea.py --dataset half_4x --out_json results_half_4x.json
"""

import sys
import json
import math
import logging
import argparse
from pathlib import Path
from itertools import islice

import numpy as np
import torch
import torch.fft as torch_fft
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: KNO venv first (for torchmetrics 0.11.4 needed by meddlr),
# KNO src last (for datasets.skmtea, src.fastmri).
# ---------------------------------------------------------------------------
def _find_kno_root(start: Path) -> Path:
    """Walk up to the mri3d root (the dir holding src/datasets/skmtea.py).

    Replaces a fixed `parents[2]`/`parents[3]` depth, which silently resolves to the
    wrong directory whenever this file is run from anywhere other than
    <mri3d>/src/ram/csgm-mri-langevin/ — e.g. from a git worktree, where it picked up
    `.claude/` as KNO_SRC and then failed with `ModuleNotFoundError: meddlr`.
    Resolution is identical to the old code at the canonical location.
    """
    for p in start.parents:
        if (p / "src" / "datasets" / "skmtea.py").exists():
            return p
    return start.parents[3]


_KNO_ROOT = _find_kno_root(Path(__file__).resolve())
KNO_SRC  = _KNO_ROOT / "src"
KNO_VENV = _KNO_ROOT / ".venv/lib/python3.10/site-packages"

# `import src.diffmodels` (res_rob) cannot be relied on via the venv's editable
# install: that .pth points at src/baselines/resolution_robust_3d_mri/src, which does
# not exist — the real tree is under src/ram/. This is a PRE-EXISTING breakage (the
# script could not `import src` before these changes either), so resolve res_rob
# explicitly from whichever location actually holds it.
for _rr_cand in (_KNO_ROOT / "src/ram/resolution_robust_3d_mri",
                 _KNO_ROOT / "src/baselines/resolution_robust_3d_mri"):
    if (_rr_cand / "src" / "diffmodels").is_dir():
        sys.path.insert(0, str(_rr_cand))
        break
sys.path.insert(0, str(KNO_VENV))
sys.path.append(str(KNO_SRC))

# meddlr (imported transitively) uses the deprecated np.complex alias.
if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore[attr-defined]

import meddlr.metrics.functional as _meddlr_metrics

# datasets/skmtea.py does `from fastmri.subsample import MaskFunc, Gaussian3DMaskFunc`.
# The installed fastmri doesn't expose fastmri.subsample; the KNO-local copy at
# /home/ram/fastmri/subsample.py does — and it only depends on numpy/torch/sigpy.
# Inject it directly into sys.modules so res_rob's fastmri.data.transforms keeps working.
import importlib.util as _ilu
_sub_spec = _ilu.spec_from_file_location("fastmri.subsample", str(KNO_SRC / "fastmri/subsample.py"))
_sub_mod  = _ilu.module_from_spec(_sub_spec)
sys.modules["fastmri.subsample"] = _sub_mod
_sub_spec.loader.exec_module(_sub_mod)

# ---- res_rob diffusion prior ------------------------------------------------
from src.diffmodels.diffmodels_resolver import create_dense_model
from src.diffmodels.ema import ExponentialMovingAverage
from src.diffmodels.sde import DDPM
from src.problem_trafos.trafo_resolver import get_prior_trafo

# ---- KNO LMDB dataset (optional — only needed if --lmdb mode) ---------------
try:
    from datasets.skmtea import LMDBVolumeDataset
    _HAS_LMDB = True
except ImportError:
    _HAS_LMDB = False


# ===========================================================================
# Constants
# ===========================================================================
SKMTEA_ROOT = Path("/home/ram/skmtea")

# All available LMDB validation datasets.
# Naming: {resolution}_{acceleration}
#   half (256x256x160) = 0.5× base spatial dims (512×512×160)
#   qr   (128x128x80)  = 0.25× base spatial dims
LMDB_DATASETS = {
    "half_4x": SKMTEA_ROOT / "skmtea_val_0.5first2_4x_lmdb",
    "half_8x": SKMTEA_ROOT / "skmtea_val_256x256x160_8x_lmdb",
    "qr_4x":   SKMTEA_ROOT / "skmtea_val_128x128x80_4x_lmdb",
    "qr_8x":   SKMTEA_ROOT / "skmtea_val_128x128x80_8x_lmdb",
}

# Acceleration factor per dataset — used to correct sf to match training (calibrated at R=4).
_DATASET_ACCEL = {"half_4x": 4, "half_8x": 8, "qr_4x": 4, "qr_8x": 8}

H5_PATH    = Path("/home/ram/skmtea_sample/MTR_001.h5")   # H5 fallback
EMA_CKPT   = Path("recon_workdir/ema_model_110.pt")
TRAIN_CFG  = Path("recon_workdir/.hydra/config.yaml")
HYDRA_ROOT = Path("hydra")


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


def _q99_per_sample(estimated_mvue):
    """Per-sample 99th percentile of |mvue|, as a (B,) tensor.

    Must be per sample once slices are batched: a global quantile over a (B,X,Y)
    stack would let one slice's intensity set another slice's normalization.
    At B=1 this is exactly the original `torch.quantile(estimated_mvue.abs(), 0.99)`.
    """
    return torch.quantile(estimated_mvue.abs().flatten(1), 0.99, dim=1)


def normalize(gen_img, estimated_mvue):
    scaling = _q99_per_sample(estimated_mvue).view(-1, *([1] * (gen_img.dim() - 1)))
    return gen_img * scaling


def unnormalize(gen_img, estimated_mvue):
    scaling = _q99_per_sample(estimated_mvue).view(-1, *([1] * (gen_img.dim() - 1)))
    return gen_img / scaling


# ===========================================================================
# KNO-project metrics — replaces CSGM's RSS / MVUE PSNR/SSIM at evaluation time.
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


def _bnorm(t):
    """Per-sample L2 norm of a (B, ...) tensor, shaped (B,1,..,1) for broadcasting.

    Slice batching requires the Langevin gradient renormalization to be PER SAMPLE:
    a global `torch.norm(t)` reduces over the batch dim too, which would make each
    slice's data-consistency step depend on the slices stacked beside it.
    For B=1 this returns exactly `torch.norm(t)`, so --slice_batch 1 (the default)
    stays byte-identical to the original per-slice code path.
    """
    return t.flatten(1).norm(dim=1).view(-1, *([1] * (t.dim() - 1)))


def kno_psnr(rec_xy2, gt_xy2):
    """rec, gt: (X, Y, 2). Returns scalar KNO PSNR (dB)."""
    rec_mag = _to_mag(rec_xy2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_xy2).unsqueeze(0).unsqueeze(0)
    return _kno_psnr_fn(rec_mag, gt_mag).squeeze()


def kno_ssim(rec_xy2, gt_xy2):
    rec_mag = _to_mag(rec_xy2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_xy2).unsqueeze(0).unsqueeze(0)
    return _meddlr_metrics.ssim(rec_mag, gt_mag).mean()


def kno_nmse(rec_xy2, gt_xy2):
    """fastmri/evaluate.py convention: ||gt-pred||^2 / ||gt||^2 on magnitudes."""
    rec_mag = _to_mag(rec_xy2)
    gt_mag  = _to_mag(gt_xy2)
    return (rec_mag - gt_mag).pow(2).sum() / gt_mag.pow(2).sum()


def volume_metrics(rec_3d_mag, gt_3d_mag):
    """3D volume PSNR/SSIM/NMSE, comparable to the KIR-NO / res_rob tables.

    Both inputs are (X, Y, Z) magnitude tensors. meddlr selects its 3D SSIM kernel
    from the tensor rank, so the (1,1,X,Y,Z) shaping here is what makes this a true
    3D SSIM rather than the 2D one used by the mid-slice path.
    """
    rec = rec_3d_mag.unsqueeze(0).unsqueeze(0)
    gt  = gt_3d_mag.unsqueeze(0).unsqueeze(0)
    psnr = _meddlr_metrics.psnr(rec, gt).mean()
    ssim = _meddlr_metrics.ssim(rec, gt).mean()
    nmse = (rec - gt).pow(2).sum() / gt.pow(2).sum()
    return psnr, ssim, nmse


# ===========================================================================
# Score model loader — SKM-TEA DDPM (res_rob create_dense_model + EMA).
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


def load_prior_trafo():
    """SKM-TEA training prior_trafo: scale by 1.0 (matches training config), move (X,Y,2)→(2,X,Y).
    Training used scaling_factor=1.0; target_scaling_factor=2.0 is in the dataset trafo,
    already baked into the sf-scaling via sqrt(N)/||ref||. Using 2.0 here doubles the
    score model input to ~18 when training expects ~8.9."""
    prior_cfg = OmegaConf.load(HYDRA_ROOT / "problem_trafos/prior_trafo/crop_mag.yaml")
    OmegaConf.update(prior_cfg, "scaling_factor", 1.0)
    OmegaConf.update(prior_cfg, "move_axis", [-1, 1])
    skip = {"name", "defaults"}
    kwargs = {k: v for k, v in OmegaConf.to_container(prior_cfg).items() if k not in skip}
    return get_prior_trafo(name=prior_cfg.name, **kwargs)


# ===========================================================================
# LangevinOptimizer — same name and shape as csgm-mri-langevin/main.py, but
# the score model is DDPM and the outer sweep iterates timesteps t_start..0
# instead of NCSNv2 sigma indices L..0.
# ===========================================================================
class LangevinOptimizer(torch.nn.Module):
    def __init__(self, config, device, score, sde, prior_trafo, scaling_factor):
        super().__init__()
        self.config         = config
        self.device         = device
        self.score          = score
        self.sde            = sde
        self.prior_trafo    = prior_trafo
        self.scaling_factor = scaling_factor

        # Discrete DDPM timesteps to anneal over — replaces self.sigmas in CSGM.
        t_start = min(int(config["t_start"]), sde.num_steps - 1)
        self.timesteps = list(range(t_start, -1, -1))            # [t_start..0]
        with torch.no_grad():
            abar = sde._compute_alpha_cumprod(
                torch.tensor(self.timesteps, device=device)).squeeze()
            self.sigmas = (1.0 - abar).sqrt().detach()           # (T,)
        # σ_min = sigma at t=0 ≈ sqrt(beta_0) ≈ 0.01.
        # Original CSGM: step_size = step_lr * (σ / sigmas[-1])²
        # sigmas[-1] is the SMALLEST sigma (minimum noise level), identical to
        # the role of σ_L in Song & Ermon (2019) annealed Langevin, which makes
        # step sizes LARGE at high noise and step_lr at the lowest noise level.
        self.sigma_min = float(self.sigmas[-1].item())           # CSGM σ_L

    # -- DDPM noise prediction in (B, 2, X, Y) format the score model uses ----
    @torch.no_grad()
    def _predict_eps(self, samples_b2xy, t_idx):
        # samples_b2xy: (B, 2, X, Y) in image scale
        # prior_trafo expects (B, X, Y, 2): move (2 from dim 1 → dim -1), then *2
        x_xy2  = samples_b2xy.permute(0, 2, 3, 1).contiguous()       # (B, X, Y, 2)
        prior_in = self.prior_trafo(x_xy2)                           # (B, 2, X, Y)
        t_vec  = torch.full((samples_b2xy.shape[0],), t_idx,
                            device=self.device, dtype=torch.long)
        eps_prior = self.score(prior_in, t_vec)                      # (B, 2, X, Y)
        eps_xy2   = self.prior_trafo.trafo_inv(eps_prior)            # (B, X, Y, 2)
        return eps_xy2.permute(0, 3, 1, 2).contiguous()              # (B, 2, X, Y)

    # -- Annealed Langevin sample loop — CSGM main.py _sample(), DDPM-flavoured
    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(), maps.cpu().numpy()),
            device=ref.device,
        )
        # In our sf-scaling framework, normalize(x) = x * Q99(|mvue|) must be ~identity
        # so that A(normalize(x_gt)) ≈ ref at the true solution. Divide out Q99 so it → 1.
        _q99 = _q99_per_sample(estimated_mvue).clamp(min=1e-6).view(-1, 1, 1)
        estimated_mvue = estimated_mvue / _q99
        logging.info(f"Running {len(self.timesteps)} DDPM timesteps × "
                     f"{self.config['n_steps_each']} inner Langevin steps "
                     f"(total {len(self.timesteps)*self.config['n_steps_each']})")

        forward_operator = lambda x: MulticoilForwardMRI(self.config["orientation"])(
            torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask
        )

        B = ref.shape[0]
        H, W = self.config["image_size"]

        if self.config.get("init_mode", "random") == "zf":
            # Adjoint-SENSE zero-filled init (as reconstruct_csgm_brats.py does).
            # With a short --n_jumps budget the chain cannot travel far from its
            # starting point, so a uniform-noise init leaves the reconstruction far
            # below zero-filling; starting AT zero-filling makes few-step numbers
            # meaningful. Default stays "random" = original CSGM behavior.
            zf_complex = (_ifft(ref) * maps.conj()).sum(dim=1)        # (B, H, W) complex
            samples = torch.view_as_real(
                zf_complex.contiguous()
            ).permute(0, 3, 1, 2).float().contiguous()                # (B, 2, H, W)
        else:
            samples = torch.rand(B, 2, H, W, device=self.device)     # CSGM init (uniform [0,1])

        step_lr = float(self.config["step_lr"])
        pbar    = tqdm(self.timesteps, desc="anneal", leave=False)
        pbar_labels = ["t", "step_size", "error", "mean", "max"]

        with torch.no_grad():
            for k, t_idx in enumerate(pbar):
                sigma     = float(self.sigmas[k].item())
                step_size = step_lr * (sigma / self.sigma_min) ** 2
                n_steps_each = int(self.config["n_steps_each"])

                for _ in range(n_steps_each):
                    noise  = torch.randn_like(samples) * math.sqrt(step_size * 2)

                    # DDPM score:  s(x, σ_t) = -eps(x, t) / σ_t
                    eps    = self._predict_eps(samples, t_idx)
                    p_grad = -eps / (sigma + 1e-8)

                    # ---- Data consistency gradient — verbatim CSGM ----
                    meas       = forward_operator(normalize(samples, estimated_mvue))
                    meas_grad  = torch.view_as_real(
                        torch.sum(_ifft(meas - ref) * torch.conj(maps), axis=1)
                    ).permute(0, 3, 1, 2)
                    meas_grad  = unnormalize(meas_grad, estimated_mvue)
                    meas_grad  = meas_grad.type(torch.cuda.FloatTensor) \
                                if samples.is_cuda else meas_grad.float()
                    # PER-SAMPLE norms (_bnorm): with --slice_batch > 1 a global
                    # torch.norm would reduce over the batch dim too, making each
                    # slice's DC step depend on the slices stacked beside it.
                    # At B=1, _bnorm(t) == torch.norm(t), so this is byte-identical.
                    meas_grad = meas_grad / _bnorm(meas_grad).clamp(min=1e-12)
                    meas_grad = meas_grad * _bnorm(p_grad)
                    meas_grad = meas_grad * self.config["mse"]

                    # CSGM Langevin update
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
# Strided variant — identical to LangevinOptimizer but replaces the full
# t_start..0 timestep list with n_jumps uniformly spaced timesteps.
# E.g., t_start=299, n_jumps=11 → [299,269,239,...,30,0], 11×n_steps_each total steps.
# _sample() and everything else are inherited unchanged.
# ===========================================================================
class LangevinOptimizerStrided(LangevinOptimizer):
    def __init__(self, config, device, score, sde, prior_trafo, scaling_factor):
        super().__init__(config, device, score, sde, prior_trafo, scaling_factor)
        n_jumps = int(config["n_jumps"])
        t_max   = self.timesteps[0]   # t_start already clamped by parent __init__
        # n_jumps integers uniformly spaced from t_max down to 0, both ends inclusive.
        strided = sorted(
            set(int(round(t)) for t in np.linspace(t_max, 0, n_jumps)),
            reverse=True,
        )
        self.timesteps = strided
        # Recompute sigmas for the new (shorter) timestep list.
        with torch.no_grad():
            abar = sde._compute_alpha_cumprod(
                torch.tensor(self.timesteps, device=device)).squeeze()
            if abar.dim() == 0:          # guard for n_jumps == 1
                abar = abar.unsqueeze(0)
            self.sigmas = (1.0 - abar).sqrt().detach()
        self.sigma_min = float(self.sigmas[-1].item())


# ===========================================================================
# H5 fallback — loads directly from raw SKM-TEA .h5 when LMDB is unavailable.
# ===========================================================================
def _gaussian_1d_mask(num_cols, accel, center_frac=0.08, seed=0):
    """1D Gaussian-weighted vertical mask (matches LMDB 3D Gaussian per-slice column)."""
    num_center = round(num_cols * center_frac)
    num_total  = round(num_cols / accel)
    num_rand   = max(0, num_total - num_center)
    mask = np.zeros(num_cols, dtype=np.float32)
    c = num_cols // 2
    mask[c - num_center // 2 : c + num_center // 2] = 1.0
    sigma = num_cols / 4.0
    probs = np.exp(-0.5 * ((np.arange(num_cols) - c) / sigma) ** 2)
    probs[mask > 0] = 0.0
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    mask[rng.choice(num_cols, num_rand, replace=False, p=probs)] = 1.0
    return mask


def h5_to_csgm_2d(h5_path, device, x_mid=None, echo=0,
                   accel=4, center_frac=0.08, norm_constant=3e7):
    """
    Load a raw SKM-TEA H5 file and extract a CSGM-ready X-slice at x_mid.

    H5 kspace layout: (X, Y, Z, echo, coil) — already 1D-IFFT'd along readout X.
    An X-slice kspace[x, :, :, echo, coil] IS the 2D (Y-kspace, Z-kspace) directly
    — no FFT/IFFT needed.

    The score model was trained on SKM-TEA X-slices of shape (Y=512, Z=160), so
    this matches the model's training distribution. Z-slices (512×512) do not.

    A 1D Gaussian mask along Y (accel×) is applied; use orientation='horizontal'
    in the CSGM config so MulticoilForwardMRI masks along H=Y.
    """
    import h5py
    import sigpy as sp

    with h5py.File(str(h5_path), "r") as f:
        ksp_np  = np.asarray(f["kspace"][:, :, :, echo, :], dtype=np.complex64)  # (X,Y,Z,C)
        maps_np = np.asarray(f["maps"][:, :, :, :, 0],      dtype=np.complex64)  # (X,Y,Z,C)
        tgt_np  = np.asarray(f["target"][:, :, :, echo, 0], dtype=np.complex64)  # (X,Y,Z)

    ksp_np /= norm_constant
    tgt_np /= norm_constant

    X     = ksp_np.shape[0]
    x_mid = x_mid if x_mid is not None else X // 2

    # X-slice: kspace[x_mid, :, :, :] = (Y, Z, C) — already proper 2D kspace
    ksp_2d  = torch.from_numpy(ksp_np[x_mid]).permute(2, 0, 1).to(device)   # (C, Y, Z)
    maps_2d = torch.from_numpy(maps_np[x_mid]).permute(2, 0, 1).to(device)  # (C, Y, Z)
    gt_2d   = torch.view_as_real(
        torch.from_numpy(tgt_np[x_mid]).to(device)
    )                                                                          # (Y, Z, 2)

    # 1D Gaussian mask along Y (H dimension for orientation='horizontal')
    Y      = ksp_2d.shape[1]
    mask_y = torch.from_numpy(_gaussian_1d_mask(Y, accel, center_frac)).to(device)  # (Y,)
    ref    = ksp_2d * mask_y[None, :, None]                                   # (C, Y, Z) masked

    # MVUE
    mvue_np = (
        np.sum(sp.ifft(ref.cpu().numpy(), axes=(-1, -2))
               * np.conj(maps_2d.cpu().numpy()), axis=0)
        / np.sqrt(np.sum(np.square(np.abs(maps_2d.cpu().numpy())), axis=0))
    )
    mvue_2d = torch.from_numpy(mvue_np).to(device)  # (Y, Z) complex

    return (
        ref.unsqueeze(0),       # (1, C, Y, Z)
        mvue_2d.unsqueeze(0),   # (1, Y, Z)
        maps_2d.unsqueeze(0),   # (1, C, Y, Z)
        mask_y.unsqueeze(0),    # (1, Y)  — horizontal orientation
        gt_2d,                  # (Y, Z, 2)
    )


# ===========================================================================
# Per-volume preprocessing — KNO LMDB → CSGM-shaped (ref, mvue, maps, mask).
#   ref:  (B=1, C, H=X, W=Y) complex   masked k-space (with (1-1j)/2 correction)
#   mvue: (B=1, H, W) complex          for diagnostic only
#   maps: (B=1, C, H, W) complex       sensitivity maps
#   mask: (B=1, W) float               1D phase-encode mask
#   gt:   (X, Y, 2) real-as-complex    ground-truth target (for KNO metrics)
# ===========================================================================
def lmdb_volume_to_csgm_2d(batch, device, z_slice=-1):
    kspace = batch.kspace.squeeze(0)        # (C, X, Y, Z) complex64  full 3D kspace
    maps   = batch.maps.squeeze(0)          # (C, X, Y, Z) complex64
    target = batch.target.squeeze(0)        # (1, X, Y, Z) complex64  3D image
    mask   = batch.mask.squeeze(0)          # (1, X, Y, Z) complex64  3D kspace mask

    # KNO mask bug correction: m*(1+j)*k → m*k
    kspace_clean = kspace * torch.tensor((1 - 1j) / 2, dtype=torch.complex64)

    # ------------------------------------------------------------------
    # The LMDB stores FULL 3D k-space (after skmtea.py applies fftnc along X).
    # `target` is in 3D image domain. So kspace[:, :, :, z_mid] is NOT the
    # 2D FFT of target[:, :, z_mid] — it's a phase-mixed Z-projection.
    #
    # Fix: 1D-IFFT along Z first → hybrid space (Z image-domain, X/Y kspace).
    # Then each Z-slice has a proper 2D ksp ↔ 2D image relationship:
    #     hybrid_ksp[:, :, :, z]  is the masked 2D kspace for slice z
    #     2D-IFFT_xy → coil images at slice z → SENSE adjoint → target[:, :, z]
    # 1D IFFT is unitary, so magnitudes are preserved correctly.
    # ------------------------------------------------------------------
    ksp_z_hybrid = torch_fft.fftshift(
        torch_fft.ifft(
            torch_fft.ifftshift(kspace_clean, dim=-1),
            dim=-1, norm="ortho",
        ),
        dim=-1,
    )                                                         # (C, X, Y, Z)

    Z = ksp_z_hybrid.shape[-1]
    z_idx = Z // 2 if z_slice < 0 else int(z_slice)
    z_idx = max(0, min(z_idx, Z - 1))

    ksp_2d  = ksp_z_hybrid[:, :, :, z_idx].to(device)        # (C, X, Y) complex
    maps_2d = maps[:, :, :, z_idx].to(device)                # (C, X, Y) complex
    gt_2d   = torch.view_as_real(target.squeeze(0)[:, :, z_idx].to(device))  # (X,Y,2)

    # Phase-encode mask is constant along X. Take a single X-row, real part.
    # Shape: (Y,) — broadcasts as 'vertical' in MulticoilForwardMRI: mask[:, None, None, :]
    mask_y = mask.real[0, 0, :, z_idx].to(device).float()     # (Y,)

    # MVUE (diagnostic, also the scaling reference used by CSGM normalize)
    import sigpy as sp
    mvue_np = (
        np.sum(sp.ifft(ksp_2d.cpu().numpy(), axes=(-1, -2)) * np.conj(maps_2d.cpu().numpy()), axis=0)
        / np.sqrt(np.sum(np.square(np.abs(maps_2d.cpu().numpy())), axis=0))
    )
    mvue_2d = torch.from_numpy(mvue_np).to(device)            # (X, Y) complex

    # Add batch dim for CSGM
    ref  = ksp_2d.unsqueeze(0)                                # (1, C, X, Y) complex
    mvue = mvue_2d.unsqueeze(0)                               # (1, X, Y) complex
    maps_b = maps_2d.unsqueeze(0)                             # (1, C, X, Y) complex
    mask_b = mask_y.unsqueeze(0)                              # (1, Y) float

    return ref, mvue, maps_b, mask_b, gt_2d


# ===========================================================================
# Whole-volume prep + batched Z-slice gather (used by --all_slices).
#
# Mirrors reconstruct_csgm_brats.py's all-slices design, with one SKM-TEA-specific
# difference: the hybrid 1D-IFFT along Z is a (C,X,Y,Z) transform, so it is done
# ONCE per volume here instead of once per slice (lmdb_volume_to_csgm_2d recomputes
# it every call, which would dominate runtime over 160-320 slices).
# ===========================================================================
def prepare_volume(batch, accel):
    """One-time per-volume prep shared by every slice batch.

    Returns a dict with the hybrid k-space and a VOLUME-CONSTANT scaling factor.
    The scale must be constant across slices: the single-slice path derives it from
    that slice's own k-space norm, but doing that per slice would give each slice a
    different magnitude scale and corrupt the assembled 3D volume. Using the
    root-mean-square per-slice norm (||full|| / sqrt(Z)) reproduces the single-slice
    formula for a typical slice while staying constant over z.
    """
    kspace = batch.kspace.squeeze(0)        # (C, X, Y, Z) complex64
    maps   = batch.maps.squeeze(0)          # (C, X, Y, Z)
    target = batch.target.squeeze(0)        # (1, X, Y, Z)
    mask   = batch.mask.squeeze(0)          # (1, X, Y, Z)

    # KNO mask bug correction: m*(1+j)*k → m*k   (same as the single-slice path)
    kspace_clean = kspace * torch.tensor((1 - 1j) / 2, dtype=torch.complex64)

    ksp_z_hybrid = torch_fft.fftshift(
        torch_fft.ifft(
            torch_fft.ifftshift(kspace_clean, dim=-1),
            dim=-1, norm="ortho",
        ),
        dim=-1,
    )                                                         # (C, X, Y, Z)

    C, X, Y, Z = ksp_z_hybrid.shape
    vol_norm  = torch.view_as_real(ksp_z_hybrid).norm().item()
    rms_slice = vol_norm / math.sqrt(Z)                       # typical per-slice norm
    raw_sf    = math.sqrt(float(X * Y * 2)) / max(rms_slice, 1e-12)
    scaling_factor = raw_sf * (2.0 / math.sqrt(float(accel)))

    return {
        "ksp": ksp_z_hybrid, "maps": maps, "target": target, "mask": mask,
        "Z": Z, "X": X, "Y": Y, "scaling_factor": scaling_factor,
    }


def slice_batch_from_prep(prep, device, z_indices):
    """Gather a batch of Z-slices as CSGM-shaped tensors.

    ref/maps: (B, C, X, Y) complex,  mask: (B, Y) float,  gt: (B, X, Y, 2).
    For a single index this is identical to lmdb_volume_to_csgm_2d's output.
    """
    zs = torch.as_tensor(list(z_indices), dtype=torch.long)

    ref    = prep["ksp"][:, :, :, zs].permute(3, 0, 1, 2).contiguous().to(device)
    maps_b = prep["maps"][:, :, :, zs].permute(3, 0, 1, 2).contiguous().to(device)
    gt_b   = torch.view_as_real(
        prep["target"].squeeze(0)[:, :, zs].permute(2, 0, 1).contiguous().to(device)
    )                                                          # (B, X, Y, 2)
    # Phase-encode mask is constant along X → take one X-row per slice.
    mask_b = prep["mask"].real[0, 0, :, zs].permute(1, 0).contiguous().to(device).float()

    # MVUE is diagnostic only (the optimizer recomputes it from ref via get_mvue),
    # but keep it correct and batched for parity with the single-slice path.
    mvue_b = torch.from_numpy(
        get_mvue(ref.cpu().numpy(), maps_b.cpu().numpy())
    ).to(device)                                               # (B, X, Y) complex

    return ref, mvue_b, maps_b, mask_b, gt_b


# ===========================================================================
# Main
# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    # Dataset selection
    parser.add_argument("--dataset",      type=str,   default="half_4x",
                        choices=list(LMDB_DATASETS.keys()),
                        help="LMDB dataset to reconstruct. "
                             "half=256x256x160, qr=128x128x80; accel 4x or 8x.")
    parser.add_argument("--lmdb_path",    type=str,   default=None,
                        help="Override LMDB path directly (ignores --dataset).")
    parser.add_argument("--num_volumes",  type=int,   default=None,
                        help="Number of volumes to reconstruct (default: entire dataset).")
    parser.add_argument("--vol_start",    type=int,   default=0,
                        help="Index of first volume to reconstruct.")
    parser.add_argument("--out_json",     type=str,   default=None,
                        help="Save per-volume metrics here (updated after each volume).")
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--no_wandb",     action="store_true")
    # CSGM hyperparameters:
    parser.add_argument("--t_start",      type=int,   default=299,
                        help="Highest DDPM noise step (sweep-best: 299).")
    parser.add_argument("--n_steps_each", type=int,   default=5,
                        help="Inner Langevin steps per noise level (sweep-best: 5).")
    parser.add_argument("--step_lr",      type=float, default=6e-9,
                        help="Base Langevin step size. step_size = step_lr*(σ/σ_min)^2 "
                             "so initial step ≈ step_lr*8000 at t_start=399. "
                             "6e-9 ≈ equivalent of 5e-5 under the old (wrong) formula.")
    parser.add_argument("--mse",          type=float, default=25.0,
                        help="DC weight λ on the meas-grad direction (sweep-best: 25).")
    parser.add_argument("--orientation",  type=str,   default="vertical",
                        choices=["vertical", "horizontal"])
    parser.add_argument("--n_jumps",      type=int,   default=None,
                        help="If set, use LangevinOptimizerStrided with n_jumps uniformly spaced "
                             "noise levels instead of all t_start+1 levels. "
                             "E.g., --n_jumps 11 with t_start=299, n_steps_each=5 → 55 total steps.")
    parser.add_argument("--init_mode",    type=str, default="random",
                        choices=["random", "zf"],
                        help="Langevin init. 'random' (default) = uniform [0,1), the "
                             "original CSGM behavior. 'zf' = adjoint-SENSE zero-filled "
                             "init, which is what makes short --n_jumps budgets "
                             "meaningful (reconstruct_csgm_brats.py defaults to zf).")
    parser.add_argument("--all_slices",   action="store_true",
                        help="Reconstruct ALL Z-slices per volume, assemble the 3D "
                             "volume and report 3D PSNR/SSIM/NMSE (comparable to the "
                             "KIR-NO / res_rob tables). Default off = mid-slice 2D "
                             "metrics, which are NOT comparable. LMDB mode only.")
    parser.add_argument("--slice_batch",  type=int,   default=1,
                        help="Number of Z-slices reconstructed per Langevin run when "
                             "--all_slices is set. >1 batches slices on the GPU. "
                             "Default 1 is byte-identical to serial per-slice.")
    parser.add_argument("--z_slice",      type=int,   default=-1,
                        help="Z-slice index to reconstruct (-1 = middle).")
    args = parser.parse_args()
    device = args.device

    run_name = f"csgm_kno_{args.dataset}"
    wandb.init(project="skmtea_csgm_kno", name=run_name,
               mode="disabled" if args.no_wandb else "online",
               config=vars(args))

    # ---- Score model + SDE + prior_trafo ----------------------------------
    score       = load_score_model(EMA_CKPT, TRAIN_CFG, device)
    sde         = DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)
    prior_trafo = load_prior_trafo()

    # ---- Dataset — LMDB if available, else H5 fallback ---------------------
    lmdb_path = Path(args.lmdb_path) if args.lmdb_path else LMDB_DATASETS[args.dataset]
    use_lmdb = lmdb_path.exists() and _HAS_LMDB
    if use_lmdb:
        dataset = LMDBVolumeDataset(root_dir=lmdb_path, norm_constant=3e7)
        n_available = len(dataset) - args.vol_start
        num_volumes = (min(args.num_volumes, n_available)
                       if args.num_volumes is not None else n_available)
        logging.info(f"LMDB mode: {len(dataset)} volumes total, "
                     f"reconstructing {num_volumes} starting at vol {args.vol_start} "
                     f"({lmdb_path.name})")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        volume_iter = (
            (args.vol_start + i, lmdb_volume_to_csgm_2d(batch, device,
                                                         z_slice=args.z_slice))
            for i, batch in enumerate(
                tqdm(islice(loader, args.vol_start, args.vol_start + num_volumes),
                     total=num_volumes, desc="volumes")
            )
        )
    else:
        logging.info(f"H5 mode (X-slice): {H5_PATH}  [LMDB not found at {lmdb_path}]")
        if args.orientation == "vertical":
            # X-slices need horizontal mask (along Y=H); override silently
            args.orientation = "horizontal"
            logging.info("H5 mode: switched orientation to 'horizontal' (Y-mask on X-slices)")
        num_volumes = 1
        volume_iter = (
            (0, h5_to_csgm_2d(H5_PATH, device)),
        )

    # =======================================================================
    # --all_slices: reconstruct every Z-slice, assemble the 3D volume, and report
    # 3D PSNR/SSIM/NMSE — the same protocol reconstruct_csgm_brats.py uses, so the
    # numbers are comparable to the KIR-NO / res_rob tables (the default mid-slice
    # path below reports 2D metrics, which are NOT comparable).
    # Slices are reconstructed in GPU batches of --slice_batch.
    # =======================================================================
    if args.all_slices:
        if not use_lmdb:
            raise SystemExit("--all_slices requires LMDB mode (the H5 fallback is single-slice).")

        _R = _DATASET_ACCEL.get(args.dataset, 4)
        vol_rows = []
        loader_all = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, batch in enumerate(
            tqdm(islice(loader_all, args.vol_start, args.vol_start + num_volumes),
                 total=num_volumes, desc="volumes")
        ):
            vol_idx = args.vol_start + i
            prep = prepare_volume(batch, _R)
            sf, Z = prep["scaling_factor"], prep["Z"]
            zs = list(range(Z))
            chunks = [zs[j:j + args.slice_batch]
                      for j in range(0, len(zs), args.slice_batch)]

            rec_by_z, fbp_by_z = {}, {}

            for chunk in tqdm(chunks, desc="slice-batches", leave=False):
                ref, mvue, maps_b, mask_b, gt_b = slice_batch_from_prep(prep, device, chunk)
                X, Y = ref.shape[-2], ref.shape[-1]

                # Zero-fill fully-background slices instead of reconstructing them:
                # their zero DC residual makes meas_grad/||meas_grad|| = 0/0 = NaN, which
                # would poison the assembled volume's 3D metrics. Filtered OUT of the
                # batch so one empty slice cannot NaN the others beside it.
                keep = [b for b in range(len(chunk)) if float(gt_b[b].abs().max()) >= 1e-8]
                for b in range(len(chunk)):
                    if b not in keep:
                        fbp_by_z[chunk[b]] = torch.zeros(X, Y)
                        rec_by_z[chunk[b]] = torch.zeros(X, Y)
                if not keep:
                    continue
                if len(keep) < len(chunk):
                    sel = torch.as_tensor(keep, device=ref.device)
                    ref, mvue, maps_b, mask_b, gt_b = (
                        ref[sel], mvue[sel], maps_b[sel], mask_b[sel], gt_b[sel])
                keys = [chunk[b] for b in keep]

                ref_sf, mvue_sf = ref * sf, mvue * sf

                if i == 0 and keys[0] == 0:
                    logging.info(
                        f"[vol {vol_idx}] all_slices Z={Z} batch={len(keys)} "
                        f"ref={tuple(ref.shape)} maps={tuple(maps_b.shape)} "
                        f"mask={tuple(mask_b.shape)} gt={tuple(gt_b.shape)} sf={sf:.6f}")

                with torch.no_grad():
                    coils_img = _ifft(ref_sf)                                 # (B,C,X,Y)
                    fbp_xy    = (torch.conj(maps_b) * coils_img).sum(dim=1)   # (B,X,Y)
                    fbp_sf_b  = torch.view_as_real(fbp_xy.contiguous())       # (B,X,Y,2)

                config = {
                    "device":       device,
                    "image_size":   (X, Y),
                    "orientation":  args.orientation,
                    "t_start":      args.t_start,
                    "n_steps_each": args.n_steps_each,
                    "step_lr":      args.step_lr,
                    "mse":          args.mse,
                    "init_mode":    args.init_mode,
                }
                if args.n_jumps is not None:
                    config["n_jumps"] = args.n_jumps
                    Cls = LangevinOptimizerStrided
                else:
                    Cls = LangevinOptimizer
                optim = Cls(config, device, score, sde, prior_trafo, sf).to(device)

                samples  = optim.sample((ref_sf, mvue_sf, maps_b, mask_b))    # (B,2,X,Y)
                rec_sf_b = samples.permute(0, 2, 3, 1).contiguous()           # (B,X,Y,2)

                for bi, k in enumerate(keys):
                    fbp_by_z[k] = _to_mag(fbp_sf_b[bi].cpu()) / sf
                    rec_by_z[k] = _to_mag(rec_sf_b[bi].cpu()) / sf

            # ---- assemble 3D volume and score it ---------------------------
            rec_3d = torch.stack([rec_by_z[z] for z in zs], dim=-1)   # (X,Y,Z)
            fbp_3d = torch.stack([fbp_by_z[z] for z in zs], dim=-1)
            gt_3d  = prep["target"].squeeze(0).abs().cpu()            # (X,Y,Z)

            fp, fs, fn = volume_metrics(fbp_3d, gt_3d)
            rp, rs, rn = volume_metrics(rec_3d, gt_3d)

            logging.info(f"[vol {vol_idx}] 3D FBP  PSNR={fp:.2f} dB  SSIM={fs:.4f}  NMSE={fn:.6f}")
            logging.info(f"[vol {vol_idx}] 3D CSGM PSNR={rp:.2f} dB  SSIM={rs:.4f}  NMSE={rn:.6f}")
            wandb.log({"vol_fbp_psnr_3d": fp.item(), "vol_fbp_ssim_3d": fs.item(),
                       "vol_rec_psnr_3d": rp.item(), "vol_rec_ssim_3d": rs.item(),
                       "vol_rec_nmse_3d": rn.item(), "global_step": i})

            vol_rows.append({"vol_idx": vol_idx,
                             "fbp_psnr": fp.item(), "fbp_ssim": fs.item(), "fbp_nmse": fn.item(),
                             "rec_psnr": rp.item(), "rec_ssim": rs.item(), "rec_nmse": rn.item()})

        if vol_rows:
            def _m(k):
                return float(np.mean([r[k] for r in vol_rows]))
            logging.info("=" * 72)
            logging.info(f"3D VOLUME METRICS over n={len(vol_rows)} volumes "
                         f"({args.dataset}, slice_batch={args.slice_batch}, "
                         f"n_jumps={args.n_jumps})")
            logging.info(f"  FBP   PSNR={_m('fbp_psnr'):.2f}  SSIM={_m('fbp_ssim'):.4f}  "
                         f"NMSE={_m('fbp_nmse'):.6f}")
            logging.info(f"  CSGM  PSNR={_m('rec_psnr'):.2f}  SSIM={_m('rec_ssim'):.4f}  "
                         f"NMSE={_m('rec_nmse'):.6f}")
            logging.info("=" * 72)
        if args.out_json:
            with open(args.out_json, "w") as _f:
                json.dump(vol_rows, _f, indent=2)
            logging.info(f"wrote {args.out_json}")
        return

    fbp_psnrs, rec_psnrs = [], []
    fbp_ssims, rec_ssims = [], []
    per_vol_results = []

    for i, (ref, mvue, maps, mask_b, gt_2d) in volume_iter:

        H, W = ref.shape[-2], ref.shape[-1]

        # ------------------------------------------------------------------
        # Bring ref into a unit-ish coordinate system so CSGM's [0,1]-init
        # samples are at the same scale as the underlying image and the
        # score model sees inputs in its training distribution.
        # ------------------------------------------------------------------
        ref_real_norm = torch.view_as_real(ref.squeeze(0)).norm().item()
        # Training sf = 2*sqrt(H*W*2)/||kspace_full||. Masked kspace norm ≈ ||full||/sqrt(R).
        # So raw inference sf = sqrt(H*W*2)*sqrt(R)/||full||.
        # For R=4 this equals training sf exactly. For R=8 it's √2× too large.
        # Correction: multiply by 2/sqrt(R) so inference always matches R=4 training calibration.
        raw_sf         = math.sqrt(float(H * W * 2)) / ref_real_norm
        _R             = _DATASET_ACCEL.get(args.dataset, 4)
        scaling_factor = raw_sf * (2.0 / math.sqrt(_R))

        ref_sf  = ref  * scaling_factor
        mvue_sf = mvue * scaling_factor

        if i == args.vol_start:
            logging.info(f"[vol {i}] ref={tuple(ref.shape)}  maps={tuple(maps.shape)}  "
                         f"mask={tuple(mask_b.shape)}  gt={tuple(gt_2d.shape)}  "
                         f"|ref|max={ref.abs().max():.4f}  |gt|max={gt_2d.abs().max():.4f}  "
                         f"sf={scaling_factor:.6f}")

        # ---- FBP (zero-filled SENSE) — at sf-scale, then unscale for metric
        with torch.no_grad():
            coils_img = _ifft(ref_sf)                                    # (1, C, X, Y)
            fbp_xy    = (torch.conj(maps) * coils_img).sum(dim=1)        # (1, X, Y) complex
            fbp_sf_2d = torch.view_as_real(fbp_xy.squeeze(0))            # (X, Y, 2) at sf scale
            fbp_2d    = fbp_sf_2d / scaling_factor                       # (X, Y, 2) at gt scale

        fbp_p = kno_psnr(fbp_2d.cpu(), gt_2d.cpu())
        fbp_s = kno_ssim(fbp_2d.cpu(), gt_2d.cpu())
        fbp_psnrs.append(fbp_p.item()); fbp_ssims.append(fbp_s.item())
        logging.info(f"[vol {i}] FBP  PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}")
        wandb.log({"fbp_psnr_kno": fbp_p.item(), "fbp_ssim_kno": fbp_s.item(),
                   "global_step": i})

        # ---- CSGM Langevin sampling (run loop in sf-scale) -----------------
        config = {
            "device":       device,
            "image_size":   (H, W),
            "orientation":  args.orientation,
            "t_start":      args.t_start,
            "n_steps_each": args.n_steps_each,
            "step_lr":      args.step_lr,
            "mse":          args.mse,
                    "init_mode":    args.init_mode,
        }
        if args.n_jumps is not None:
            config["n_jumps"] = args.n_jumps
            optim = LangevinOptimizerStrided(config, device, score, sde, prior_trafo,
                                             scaling_factor).to(device)
        else:
            optim = LangevinOptimizer(config, device, score, sde, prior_trafo,
                                      scaling_factor).to(device)

        samples = optim.sample((ref_sf, mvue_sf, maps, mask_b))          # (1, 2, X, Y) at sf scale

        # ---- KNO metrics — unscale samples back to gt scale ----------------
        rec_sf_2d = samples.squeeze(0).permute(1, 2, 0).contiguous()     # (X, Y, 2) at sf
        rec_2d    = (rec_sf_2d / scaling_factor).cpu()                   # (X, Y, 2) at gt scale
        rec_p     = kno_psnr(rec_2d, gt_2d.cpu())
        rec_s     = kno_ssim(rec_2d, gt_2d.cpu())
        rec_psnrs.append(rec_p.item()); rec_ssims.append(rec_s.item())
        logging.info(f"[vol {i}] CSGM PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}")
        wandb.log({
            "rec_psnr_kno_final": rec_p.item(),
            "rec_ssim_kno_final": rec_s.item(),
            "rec_psnrs_kno_mean": float(np.mean(rec_psnrs)),
            "rec_ssims_kno_mean": float(np.mean(rec_ssims)),
            "global_step": i,
        })

        per_vol_results.append({
            "vol_idx":  i,
            "z_slice":  args.z_slice,
            "fbp_psnr": fbp_p.item(),
            "fbp_ssim": fbp_s.item(),
            "rec_psnr": rec_p.item(),
            "rec_ssim": rec_s.item(),
        })
        if args.out_json:
            with open(args.out_json, "w") as _f:
                json.dump(per_vol_results, _f, indent=2)

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
