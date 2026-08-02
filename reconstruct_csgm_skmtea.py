#!/usr/bin/env python3
"""
reconstruct_csgm_skmtea.py

Faithful CSGM (Jalal et al. 2021, "Robust Compressed Sensing MRI with Deep
Generative Priors") reconstruction adapted to the SKM-TEA LMDB val sets.

This is the SKM-TEA sibling of reconstruct_csgm_brats.py and is deliberately kept
structurally identical to it: same LangevinOptimizer / LangevinOptimizerStrided,
same centered FFT pair, same volume-norm scaling scheme, same prior_input_scale
handling, same --all_slices 3D metrics, same --slice_batch GPU batching. The ONLY
SKM-TEA-specific parts are the LMDB reader and the score model / prior it loads.

Geometry (verified against the data, not assumed)
-------------------------------------------------
SKM-TEA LMDB stores FULL 3D k-space (C, X, Y, Z): `datasets/skmtea.py` applies
`fftnc(kspace, 0)` to the raw h5, which already ships 1D-FFT'd along the readout X.
The undersampling mask comes from `Gaussian3DMaskFunc`, which builds a 2D Gaussian
pattern over (Y, Z) and `expand()`s it along X -- i.e.

    X is the fully-sampled READOUT axis; (Y, Z) are the phase-encode axes.

Because the mask is constant along kx, an inverse FFT along kx commutes with it and
each readout slice x carries an exact, self-contained 2D problem:

    ifft_kx( M(ky,kz) * fft3c(S_c * img) )[:, x] == M(ky,kz) * fft2c( (S_c*img)[:, x] )

Measured on real volumes (both qr_4x and half_4x), relative error of that identity:
    axis X: 1.1e-07   <- exact (float32 storage precision)
    axis Y: 3.9e-01   BROKEN
    axis Z: 2.9e-01   BROKEN
So the 2D slice is (Y, Z), the mask is 2D over (Y, Z), and MulticoilForwardMRI runs
in its `orientation='2d'` branch -- exactly the BraTS readout_axis=0 configuration.

This also matches the prior: the production SKM-TEA score model (recon_workdir/
ema_model_110.pt, res_rob runs of Apr 9-11 and Apr 26-28 2026) was trained by the
then-current SkmteaSliceDataset, which sliced axis 0 only -> (Y, Z) slices. (The
three-plane "diverse" slicing landed in res_rob commit 34e09d1 on 2026-05-28, after
both training runs, so it does NOT apply to this checkpoint.)

Normalization (training-faithful, mask-independent)
---------------------------------------------------
res_rob's FastMRI2DDataTransform fed the score model
    target_slice * sqrt(prod(target_vol_shape)) / ||kspace_vol|| * target_scaling_factor
(`scale_target_by_kspacenorm=true`, `target_scaling_factor=2.0`). We reproduce that
with the whole-VOLUME factor

    kspace_scale = sqrt(X*Y*Z) / ||target_3D||_F

(the image-domain form used by reconstruct_modified.py; Parseval-equal to the
k-space norm for RSS-normalized maps -- measured sum_c|S_c|^2 = 0.993 on support),
and the remaining `target_scaling_factor` is applied as `prior_input_scale` inside
`_predict_eps`, exactly as in reconstruct_csgm_brats.py.

`ref` is recomputed as M * fft2c(S_c * target_slice) rather than read from the
stored k-space. That is not an approximation: the stored k-space (after the
documented (1-1j)/2 mask-bug correction) equals the recomputed one to 1.6e-07
relative error, verified on both resolutions. Recomputing keeps the forward
operator self-consistent, sidesteps the (1+1j) mask bug, and matches what
reconstruct_modified.py does for the res_rob baseline.

Run with the res_rob venv, by ABSOLUTE PATH (see CLAUDE.md):
    RR=.../src/ram/resolution_robust_3d_mri
    $RR/.venv/bin/python .../reconstruct_csgm_skmtea.py \
        --dataset qr_4x --num_volumes 1 --all_slices --slice_batch 32 --no_wandb
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
# Path setup. Walk up to the mri3d root (the dir holding src/datasets/skmtea.py)
# instead of a fixed parents[N] depth, which silently resolves to the wrong
# directory when this file is run from a copy / symlink / git worktree.
# ---------------------------------------------------------------------------
def _find_kno_root(start: Path) -> Path:
    for p in start.parents:
        if (p / "src" / "datasets" / "skmtea.py").exists():
            return p
    return start.parents[3]


_HERE     = Path(os.path.abspath(__file__))
_KNO_ROOT = _find_kno_root(_HERE)
KNO_SRC   = _KNO_ROOT / "src"
KNO_VENV  = _KNO_ROOT / ".venv/lib/python3.10/site-packages"
RES_ROB   = _KNO_ROOT / "src/ram/resolution_robust_3d_mri"

sys.path.insert(0, str(KNO_VENV))
sys.path.append(str(KNO_SRC))
sys.path.insert(0, str(RES_ROB))

# meddlr (imported transitively) uses the deprecated np.complex alias.
if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore[attr-defined]

import meddlr.metrics.functional as _meddlr_metrics

# datasets.skmtea does `from fastmri.subsample import MaskFunc`; the installed
# fastmri exposes it at fastmri.data.subsample -- alias so the import resolves.
import fastmri.data.subsample as _fmri_sub
sys.modules.setdefault("fastmri.subsample", _fmri_sub)

from src.diffmodels.diffmodels_resolver import create_dense_model
from src.diffmodels.ema import ExponentialMovingAverage
from src.diffmodels.sde import DDPM
from src.problem_trafos.trafo_resolver import get_prior_trafo


# ===========================================================================
# Default paths
# ===========================================================================
_RUIBO = Path("/scratch/10846/armeet/datasets/ruibo")

# Naming: {resolution}_{acceleration};  half = 256x256x160, qr = 128x128x80.
# half_4x uses armeet's 0.5first2 set -- the same LMDB the res_rob half-4x
# diffusion baseline was run on. (ruibo/skmtea_val_256x256x160_4x_random_lmdb is
# EMPTY: 363 KB, zero entries. The _equispaced variant is a different sampling
# pattern with density 0.309, i.e. R=3.24, not a 4x Gaussian set.)
SKMTEA_LMDBS = {
    "qr_4x":    _RUIBO / "skmtea_val_128x128x80_4x_lmdb",
    "qr_8x":    _RUIBO / "skmtea_val_128x128x80_8x_lmdb",
    "qr_16x":   _RUIBO / "skmtea_val_128x128x80_16x_lmdb",
    "half_4x":  Path("/scratch/10846/armeet/datasets/skmtea_val_0.5first2_4x_lmdb"),
    "half_8x":  _RUIBO / "skmtea_val_256x256x160_8x_lmdb",
    "half_16x": _RUIBO / "skmtea_val_256x256x160_16x_lmdb",
}

# SKM-TEA DDPM prior (res_rob dense UNet + EMA). ema_model_110.pt is the checkpoint
# used for every published res_rob SKM-TEA reconstruction result, so CSGM uses it
# too for an apples-to-apples comparison.
EMA_CKPT  = RES_ROB / "recon_workdir/ema_model_110.pt"
TRAIN_CFG = RES_ROB / "recon_workdir/.hydra/config.yaml"

NORM_CONSTANT = 3e7   # LMDB storage divisor for SKM-TEA kspace/target


# ===========================================================================
# Forward operator (from csgm-mri-langevin/utils.py, centered-FFT fix applied)
# ===========================================================================
def _ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x


def _fft(x):
    # Centered fft2c: ifftshift -> fft2 -> fftshift, mirroring _ifft so the pair is
    # a true forward/adjoint on ALL sizes. The upstream csgm order
    # (fftshift -> fft2 -> ifftshift) is only equivalent on even axes.
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
                # SKM-TEA / BraTS: 2D mask (B, H, W) -> broadcast over coils
                ksp_coils = ksp_coils * mask[:, None, :, :]
            else:
                raise NotImplementedError("mask orientation not supported")
        return ksp_coils


# ===========================================================================
# Metrics -- identical formulas to reconstruct_csgm_brats.py / res_rob / KIR-NO.
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
    For B=1 this returns exactly `torch.norm(t)`, so --slice_batch 1 is
    byte-identical to the unbatched code path.
    """
    return t.flatten(1).norm(dim=1).view(-1, *([1] * (t.dim() - 1)))


def kno_psnr(rec_hw2, gt_hw2):
    """rec, gt: (H, W, 2). Returns scalar 2D PSNR (dB)."""
    rec_mag = _to_mag(rec_hw2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_hw2).unsqueeze(0).unsqueeze(0)
    return _kno_psnr_fn(rec_mag, gt_mag).squeeze()


def kno_ssim(rec_hw2, gt_hw2):
    rec_mag = _to_mag(rec_hw2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_hw2).unsqueeze(0).unsqueeze(0)
    return _meddlr_metrics.ssim(rec_mag, gt_mag).mean()


def kno_nmse(rec_hw2, gt_hw2):
    """fastmri/evaluate.py convention: ||gt-pred||^2 / ||gt||^2 on magnitudes."""
    rec_mag = _to_mag(rec_hw2).float()
    gt_mag  = _to_mag(gt_hw2).float()
    return (rec_mag - gt_mag).pow(2).sum() / gt_mag.pow(2).sum()


def volume_metrics(rec_3d_mag, gt_3d_mag):
    """3D volume PSNR/SSIM/NMSE, comparable to the KIR-NO / res_rob tables.

    Both inputs are (X, Y, Z) magnitude tensors. meddlr picks its SSIM kernel from
    the tensor rank, so the (1,1,X,Y,Z) shaping here is what makes this a true 3D
    11x11x11 SSIM rather than the 2D one used by the mid-slice path.
    Runs on whatever device the tensors are on -- keep them on GPU; the CPU path
    costs ~75 s per call at half-res.
    """
    rec = rec_3d_mag.unsqueeze(0).unsqueeze(0)
    gt  = gt_3d_mag.unsqueeze(0).unsqueeze(0)
    psnr = _meddlr_metrics.psnr(rec, gt).mean()
    ssim = _meddlr_metrics.ssim(rec, gt).mean()
    nmse = (rec - gt).pow(2).sum() / gt.pow(2).sum()
    return psnr, ssim, nmse


# ===========================================================================
# Score model / prior loaders -- same as reconstruct_csgm_brats.py.
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
    """Load the training prior_trafo (CroppedMagnitudeImagePriorTrafo).

    With the SKM-TEA training settings (center_crop_enabled=false,
    magnitude_enabled=false, normalize_enabled=false, scaling_factor=1.0,
    swap_channels=true, everything else null) this is a pure channel swap
    (B,H,W,2) -> (B,2,H,W), so _predict_eps bypasses it -- see there.
    """
    tcfg = OmegaConf.load(train_cfg_path)
    pt   = tcfg.problem_trafos.prior_trafo
    kwargs = {k: v for k, v in OmegaConf.to_container(pt, resolve=True).items()
              if k not in ("name", "defaults")}
    kwargs["center_crop_enabled"] = False
    kwargs["crop_size"]           = None
    kwargs["scaling_factor"]      = scaling_factor
    return get_prior_trafo(name=pt.name, **kwargs)


# ===========================================================================
# LangevinOptimizer -- identical algorithm to reconstruct_csgm_brats.py.
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
        self.prior_input_scale = prior_input_scale

        # Ablation: --no_prior drops the prior's directional term from the Langevin
        # update ONLY. Default 1.0 -> self.prior_scale*p_grad is byte-identical to p_grad.
        self.prior_scale = 0.0 if config.get("no_prior", False) else 1.0

        t_start = min(int(config["t_start"]), sde.num_steps - 1)
        self.timesteps = list(range(t_start, -1, -1))
        with torch.no_grad():
            abar = sde._compute_alpha_cumprod(
                torch.tensor(self.timesteps, device=device)).squeeze()
            self.sigmas = (1.0 - abar).sqrt().detach()
        # Divide by sigmas[-1] (the SMALLEST sigma, t~0), which plays the role of
        # sigma_L in Song & Ermon annealed Langevin: steps are LARGE at high noise
        # and equal step_lr at the lowest noise level.
        self.sigma_min = float(self.sigmas[-1].item())

    @torch.no_grad()
    def _predict_eps(self, samples_b2hw, t_idx):
        # The SKM-TEA prior_trafo is a pure channel swap (swap_channels=true, no
        # crop/magnitude/normalize, scaling_factor=1.0), so the
        # permute -> prior_trafo -> ... -> trafo_inv -> permute round-trip is an
        # identity. Feed the score at its training distribution scale directly.
        t_vec = torch.full((samples_b2hw.shape[0],), t_idx,
                           device=self.device, dtype=torch.long)
        return self.score(samples_b2hw * self.prior_input_scale, t_vec)

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y

        # ref/gt already live in the training GT domain (volume-norm scaled in
        # skmtea_vol_to_csgm_2d_batch), so samples stay in that same domain -- there
        # is no 99th-percentile MVUE normalize/unnormalize. coil_imgs is the adjoint
        # k-space, reused for the zf init.
        coil_imgs = _ifft(ref)

        logging.info(f"Running {len(self.timesteps)} DDPM timesteps x "
                     f"{self.config['n_steps_each']} inner Langevin steps "
                     f"(total {len(self.timesteps)*self.config['n_steps_each']})")

        # SKM-TEA mask is 2D (B,H,W) over the (Y,Z) phase-encode plane
        # -> orientation='2d' -> mask.ndim==3 broadcast branch.
        forward_operator = lambda x: MulticoilForwardMRI("2d")(
            torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask
        )

        B = ref.shape[0]
        H, W = self.config["image_size"]
        if self.config.get("init_mode", "random") == "zf":
            # Adjoint-SENSE ZF reconstruction -- same as SenseModel.backward().
            zf_complex = (coil_imgs * maps.conj()).sum(dim=1)                          # (B,H,W)
            samples    = torch.view_as_real(zf_complex.contiguous()).permute(0, 3, 1, 2)
        else:
            samples = torch.rand(B, 2, H, W, device=self.device)

        step_lr = float(self.config["step_lr"])
        pbar    = tqdm(self.timesteps, desc="anneal", leave=False)
        pbar_labels = ["t", "step_size", "error", "mean", "max"]

        with torch.no_grad():
            for k, t_idx in enumerate(pbar):
                sigma        = float(self.sigmas[k].item())
                step_size    = step_lr * (sigma / self.sigma_min) ** 2
                n_steps_each = int(self.config["n_steps_each"])

                for _ in range(n_steps_each):
                    noise  = torch.randn_like(samples) * math.sqrt(step_size * 2)

                    # DDPM score:  s(x, sigma_t) = -eps(x, t) / sigma_t
                    eps    = self._predict_eps(samples, t_idx)
                    p_grad = -eps / (sigma + 1e-8)

                    # ---- Data consistency gradient -- verbatim CSGM ----
                    meas      = forward_operator(samples)
                    meas_grad = torch.view_as_real(
                        torch.sum(_ifft(meas - ref) * torch.conj(maps), axis=1)
                    ).permute(0, 3, 1, 2)
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor) \
                                if samples.is_cuda else meas_grad.float()
                    # PER-SAMPLE norms (_bnorm): with --slice_batch > 1 a global
                    # torch.norm would reduce over the batch dim too, making each
                    # slice's DC step depend on the slices stacked beside it.
                    # At B=1, _bnorm(t) == torch.norm(t), so this is byte-identical.
                    meas_grad = meas_grad / _bnorm(meas_grad)
                    meas_grad = meas_grad * _bnorm(p_grad)
                    meas_grad = meas_grad * self.config["mse"]

                    # CSGM Langevin update
                    samples = samples + step_size * (self.prior_scale * p_grad - meas_grad) + noise

                    err = (meas - ref).norm()
                    metrics = [t_idx, step_size, err.item(),
                               (p_grad - meas_grad).abs().mean().item(),
                               (p_grad - meas_grad).abs().max().item()]
                    pbar.set_description("; ".join(
                        f"{lbl}: {m:.6g}" for lbl, m in zip(pbar_labels, metrics)
                    ))

                    if torch.isnan(err):
                        logging.warning(f"NaN at t={t_idx} -- early stop.")
                        return samples

        return samples

    def sample(self, y):
        return self._sample(y)


# ===========================================================================
# LangevinOptimizerStrided -- "multi_step" sampler (n_jumps uniformly-spaced
# DDPM timesteps). Only __init__ is overridden; _sample/_predict_eps inherit.
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
            if abar.dim() == 0:          # guard for n_jumps == 1
                abar = abar.unsqueeze(0)
            self.sigmas = (1.0 - abar).sqrt().detach()
        self.sigma_min = float(self.sigmas[-1].item())


# ===========================================================================
# SKM-TEA LMDB access. Sub-db layout mirrors the BraTS sets:
#   <root>/shapes/  JSON shape records, key = str(vol_idx)
#   <root>/maps/    complex64 (C, X, Y, Z)
#   <root>/target/  complex64 (1, X, Y, Z)
#   <root>/masks/   complex64 (1, X, Y, Z)   -- stored as (1+1j) at sampled bins
#   <root>/kspace/  complex64 (C, X, Y, Z)   -- NOT read (see module docstring)
# ===========================================================================
_ENV_CACHE: dict = {}


def _open_lmdb(path):
    key = str(Path(path))
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = lmdb.open(key, readonly=True, lock=False, readahead=False)
    return _ENV_CACHE[key]


def _lmdb_read(env, vk, shape, dtype=np.complex64):
    with env.begin() as txn:
        buf = txn.get(vk.encode())
    if buf is None:
        raise KeyError(f"key {vk!r} missing")
    return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()


def _volume_keys(lmdb_path):
    """Sorted string keys, matching LMDBVolumeDataset (handles non-contiguous ids)."""
    env = _open_lmdb(Path(lmdb_path) / "shapes")
    with env.begin() as txn:
        return sorted([k.decode() for k, _ in txn.cursor()], key=int)


def load_skmtea_volume(lmdb_path, vol_key):
    """Load maps/target/mask for one SKM-TEA val volume.

    Returns (maps, target, mask) with maps (C,X,Y,Z), target (X,Y,Z), mask (X,Y,Z),
    all complex64, target divided by NORM_CONSTANT to match LMDBVolumeDataset.
    The stored k-space is not read: it is exactly M * fft3c(S_c*target) after the
    (1-1j)/2 mask-bug correction (verified to 1.6e-07), so it is recomputed per
    slice instead -- see the module docstring.
    """
    lmdb_path = Path(lmdb_path)
    env_shapes = _open_lmdb(lmdb_path / "shapes")
    with env_shapes.begin() as txn:
        shp = json.loads(txn.get(vol_key.encode()).decode())
    maps   = _lmdb_read(_open_lmdb(lmdb_path / "maps"),   vol_key, tuple(shp["maps"]))
    target = _lmdb_read(_open_lmdb(lmdb_path / "target"), vol_key, tuple(shp["target"]))
    mask   = _lmdb_read(_open_lmdb(lmdb_path / "masks"),  vol_key, tuple(shp["mask"]))
    return maps, target[0] / NORM_CONSTANT, mask[0]


def _centered_mask(mask_3d):
    """Return a binary (X,Y,Z) mask in the CENTERED convention (DC in the middle).

    fft2c/_fft are centered, so the mask must be too, or data consistency would
    constrain high frequencies and drop the low ones (~ -9 dB at the ZF recon).
    Some LMDB generations store an uncentered (numpy fftn, DC-at-corner) mask.
    Detector ported from reconstruct_modified.py. Measured on the current SKM-TEA
    val sets: all are already CENTERED, so this is a no-op guard there.
    """
    m = (np.abs(mask_3d) > 0).astype(np.float32)
    t = torch.from_numpy(m)
    shifted = torch.fft.fftshift(t, dim=(-3, -2, -1))

    def _center_density(mm):
        vals = []
        for ax in range(mm.ndim):
            n = mm.shape[ax]
            if n < 8:
                continue
            prof = mm.movedim(ax, 0).reshape(n, -1).mean(dim=1)
            if prof.std() > 1e-6:          # skip fully-sampled (flat) axes like readout
                vals.append(prof[n // 2 - n // 8: n // 2 + n // 8].mean())
        return torch.stack(vals).mean() if vals else mm.mean()

    already = bool(_center_density(t) >= _center_density(shifted))
    return (t if already else shifted).numpy(), already


# ===========================================================================
# Per-slice-batch preprocessing -- SKM-TEA volume -> CSGM-shaped tensors.
#   ref:   (B, C, H, W) complex   masked k-space, volume-norm scaled
#   mvue:  None                   (obsolete; kept for call-site stability)
#   maps:  (B, C, H, W) complex   sensitivity maps
#   mask:  (B, H, W)   float      2D (Y,Z) phase-encode mask
#   gt_b:  (B, H, W, 2) float     ground-truth slices, volume-norm scaled
# H, W = (Y, Z) since readout_axis=0 (X).
# ===========================================================================
def skmtea_vol_to_csgm_2d_batch(vol, device, readout_axis, slice_indices):
    maps, target, mask_c, kspace_scale = vol
    idx = [(target.shape[readout_axis] // 2 if s is None else int(s))
           for s in slice_indices]
    sel = np.asarray(idx)
    ax_cxyz = 1 + readout_axis    # coil axis is 0 in (C, X, Y, Z)

    target_b = np.take(target, sel, axis=readout_axis)          # (B,H,W) complex
    maps_b   = np.moveaxis(np.take(maps, sel, axis=ax_cxyz),
                           ax_cxyz, 0)                          # (B,C,H,W)
    # mask is constant along the readout axis -> any index gives the same 2D plane
    mask_2d  = np.take(mask_c, 0, axis=readout_axis)            # (H,W)

    target_t = torch.from_numpy(np.ascontiguousarray(target_b)).to(device)
    maps_t   = torch.from_numpy(np.ascontiguousarray(maps_b)).to(device)
    mask_t   = torch.from_numpy(np.ascontiguousarray(mask_2d)).to(device)
    mask_t   = mask_t.unsqueeze(0).expand(target_t.shape[0], -1, -1).contiguous()

    coil_imgs = target_t.unsqueeze(1) * maps_t                  # (B,C,H,W) complex
    ksp_b     = _fft(coil_imgs)                                 # UNMASKED 2D k-space

    ref    = ksp_b * mask_t.unsqueeze(1) * kspace_scale         # (B,C,H,W)
    gt_b   = torch.view_as_real(target_t.contiguous()) * kspace_scale   # (B,H,W,2)
    return ref, None, maps_t, mask_t, gt_b


# ===========================================================================
# Main
# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qr_4x",
                        choices=list(SKMTEA_LMDBS.keys()),
                        help="SKM-TEA val LMDB. half=256x256x160, qr=128x128x80.")
    parser.add_argument("--lmdb", "--lmdb_path", dest="lmdb", type=str, default=None,
                        help="Override the LMDB path directly (ignores --dataset).")
    parser.add_argument("--ema_ckpt",  type=str, default=str(EMA_CKPT))
    parser.add_argument("--train_cfg", type=str, default=str(TRAIN_CFG))
    parser.add_argument("--num_volumes", type=int, default=1,
                        help="Number of volumes to reconstruct (-1 = all).")
    parser.add_argument("--vol_start", type=int, default=0)
    parser.add_argument("--device",    type=str, default="cuda:0")
    parser.add_argument("--no_wandb",  action="store_true")
    parser.add_argument("--out_json",  type=str, default=None,
                        help="Save per-volume metrics here (rewritten after each volume).")
    parser.add_argument("--readout_axis", type=int, default=0,
                        help="Fully-sampled readout axis in (X,Y,Z). SKM-TEA = 0 (X); "
                             "the 2D slice is the other two axes (Y,Z).")
    # CSGM hyperparameters -- defaults identical to reconstruct_csgm_brats.py.
    parser.add_argument("--t_start",      type=int,   default=399,
                        help="Highest DDPM noise step (replaces NCSNv2 L=232).")
    parser.add_argument("--n_steps_each", type=int,   default=3,
                        help="Inner Langevin steps per noise level.")
    parser.add_argument("--step_lr",      type=float, default=5e-5,
                        help="Base Langevin step size; step_size = step_lr*(sigma/sigma_min)^2.")
    parser.add_argument("--mse",          type=float, default=5.0,
                        help="DC weight lambda on the meas-grad direction.")
    parser.add_argument("--n_jumps",      type=int,   default=None,
                        help="If set, use LangevinOptimizerStrided with this many "
                             "uniformly-spaced timesteps (the 'multi_step' sampler). "
                             "If omitted, use the full dense t_start..0 schedule.")
    parser.add_argument("--prior_scaling_factor", type=float, default=1.0,
                        help="INERT: _predict_eps no longer routes through prior_trafo, so "
                             "this has no effect (must stay 1.0). The score-model amplitude "
                             "is set by --prior_input_scale. Passing !=1.0 fails loud.")
    parser.add_argument("--prior_input_scale", type=float, default=None,
                        help="Override for the score-model input scale. If omitted, resolved "
                             "from train_cfg problem_trafos.dataset_trafo.target_scaling_factor "
                             "(=2.0 for this prior); fails loud if that key is missing.")
    parser.add_argument("--init_mode", type=str, default="zf", choices=["random", "zf"],
                        help="'zf': initialize from the zero-filled SENSE recon (default). "
                             "'random': uniform [0,1) init (original paper default).")
    parser.add_argument("--all_slices", action="store_true",
                        help="Reconstruct ALL readout slices per volume and report 3D "
                             "volume PSNR/SSIM/NMSE (comparable to the KIR-NO/res_rob tables). "
                             "Without it, only the mid-slice 2D metrics are reported.")
    parser.add_argument("--slice_batch", type=int, default=1,
                        help="Reconstruct this many slices per Langevin run (GPU batching). "
                             "1 = serial, byte-identical to the unbatched path.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="If set, save per-volume GT/recon/FBP magnitude volumes (.npy).")
    parser.add_argument("--no_prior", action="store_true",
                        help="Ablation: drop the diffusion-prior term (p_grad) from the "
                             "Langevin update (data consistency + noise only).")
    args = parser.parse_args()
    device = args.device

    if args.prior_scaling_factor != 1.0:
        raise ValueError(
            "--prior_scaling_factor is inert (prior amplitude is handled by "
            f"--prior_input_scale); leave it at 1.0. Got {args.prior_scaling_factor}."
        )

    lmdb_path = Path(args.lmdb) if args.lmdb else SKMTEA_LMDBS[args.dataset]
    if not (lmdb_path / "shapes").is_dir():
        raise FileNotFoundError(
            f"SKM-TEA LMDB not found: {lmdb_path} (missing 'shapes' sub-db). "
            f"Known sets: {[str(v) for v in SKMTEA_LMDBS.values()]}"
        )

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "skmtea_csgm"),
        name=os.environ.get("WANDB_NAME", f"csgm_skmtea_{args.dataset}"),
        mode="disabled" if args.no_wandb else "online",
        config=vars(args),
    )

    # ---- Score model + SDE + prior_trafo ------------------------------------
    score       = load_score_model(args.ema_ckpt, args.train_cfg, device)
    sde         = DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)
    prior_trafo = load_prior_trafo(args.train_cfg,
                                   scaling_factor=args.prior_scaling_factor)

    # Feed the score model at its TRAINING distribution scale. Read the real cfg
    # path and FAIL LOUD on a miss so it can never silently fall back to 1.0
    # (which makes the prior inert -- see CLAUDE.md gotchas #14/#15).
    if args.prior_input_scale is not None:
        prior_input_scale = float(args.prior_input_scale)
        logging.info(f"prior_input_scale = {prior_input_scale}  (CLI override)")
    else:
        _tcfg = OmegaConf.load(args.train_cfg)
        _path = "problem_trafos.dataset_trafo.target_scaling_factor"
        _val  = OmegaConf.select(_tcfg, _path, default=None)
        if _val is None:
            raise ValueError(
                f"Could not resolve '{_path}' in {args.train_cfg}. "
                f"Pass --prior_input_scale explicitly; do NOT let it default to 1.0."
            )
        prior_input_scale = float(_val)
        logging.info(f"prior_input_scale ({_path}) = {prior_input_scale}")

    # ---- Volume loop --------------------------------------------------------
    keys = _volume_keys(lmdb_path)
    n_total = len(keys)
    remaining = n_total - args.vol_start
    num_volumes = remaining if args.num_volumes in (None, -1) \
        else min(args.num_volumes, remaining)
    logging.info(f"SKM-TEA LMDB: {n_total} volumes at {lmdb_path}  "
                 f"({num_volumes} to process from index {args.vol_start})")

    fbp_psnrs, rec_psnrs = [], []
    fbp_ssims, rec_ssims = [], []
    fbp_nmses, rec_nmses = [], []
    per_vol = []

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        logging.info(f"Saving GT/recon/FBP volumes to {args.save_dir}")

    for i in tqdm(range(num_volumes), desc="volumes"):
        vol_key = keys[args.vol_start + i]
        maps, target, mask_raw = load_skmtea_volume(lmdb_path, vol_key)
        mask_c, was_centered = _centered_mask(mask_raw)

        # Training-faithful, mask-independent volume scaling (see module docstring):
        #   kspace_scale = sqrt(prod(vol_shape)) / ||target_3D||_F
        # Element count is the magnitude-image one (no x2): training used
        # sqrt(prod(target_vol_shape)) with target_vol_shape the 3-tuple, so the
        # x2.0 applied in _predict_eps reproduces the exact training input scale.
        target_t3 = torch.from_numpy(target)
        vol_norm  = float(torch.view_as_real(target_t3).norm())
        kspace_scale = math.sqrt(float(target.size)) / max(vol_norm, 1e-8)
        vol = (maps, target, mask_c, kspace_scale)

        if args.all_slices:
            slices_iter = list(range(target.shape[args.readout_axis]))
        else:
            slices_iter = [None]           # None -> mid-slice

        if i == 0:
            logging.info(
                f"[vol {vol_key}] target={tuple(target.shape)}  coils={maps.shape[0]}  "
                f"mask density={float(mask_c.mean()):.4f}  "
                f"mask stored {'CENTERED (no fftshift)' if was_centered else 'UNCENTERED (fftshift applied)'}  "
                f"||target||={vol_norm:.4f}  kspace_scale={kspace_scale:.6f}  "
                f"readout_axis={args.readout_axis} -> {len(slices_iter)} slice(s) of "
                f"{tuple(np.delete(np.array(target.shape), args.readout_axis))}"
            )

        # slice key -> (H,W) magnitude tensor. Keyed (not appended) because slices are
        # reconstructed in GPU batches; materialized in slices_iter order afterwards.
        rec_by_idx, fbp_by_idx = {}, {}
        chunks = [slices_iter[j:j + args.slice_batch]
                  for j in range(0, len(slices_iter), args.slice_batch)]

        for chunk in tqdm(chunks, desc="slice-batches", leave=False,
                          disable=not args.all_slices):
            ref, mvue, maps_b, mask_b, gt_b = skmtea_vol_to_csgm_2d_batch(
                vol, device, args.readout_axis, chunk
            )
            H, W = ref.shape[-2], ref.shape[-1]

            # Drop fully-background readout slices (target ~ 0): their zero DC
            # residual makes `meas_grad /= ||meas_grad||` 0/0 = NaN, which would
            # poison the assembled volume's 3D metrics. Zero-fill them instead.
            # (all_slices only: in mid-slice mode there is nothing to zero-fill and
            # skipping would leave the metric lists empty.)
            keep = [b for b in range(len(chunk))
                    if not args.all_slices or float(gt_b[b].abs().max()) >= 1e-8]
            for b in range(len(chunk)):
                if b not in keep:
                    fbp_by_idx[chunk[b]] = torch.zeros(H, W, device=device)
                    rec_by_idx[chunk[b]] = torch.zeros(H, W, device=device)
            if not keep:
                continue
            if len(keep) < len(chunk):
                s = torch.as_tensor(keep, device=ref.device)
                ref, maps_b, mask_b, gt_b = ref[s], maps_b[s], mask_b[s], gt_b[s]
            ckeys = [chunk[b] for b in keep]

            # ---- FBP (zero-filled SENSE) ------------------------------------
            with torch.no_grad():
                coils_img = _ifft(ref)                                    # (B,C,H,W)
                fbp_hw    = (torch.conj(maps_b) * coils_img).sum(dim=1)   # (B,H,W)
                fbp_sf    = torch.view_as_real(fbp_hw.contiguous())       # (B,H,W,2)

            # ---- CSGM Langevin sampling -------------------------------------
            config = {
                "device":       device,
                "image_size":   (H, W),
                "t_start":      args.t_start,
                "n_steps_each": args.n_steps_each,
                "step_lr":      args.step_lr,
                "mse":          args.mse,
                "init_mode":    args.init_mode,
                "no_prior":     args.no_prior,
            }
            if args.n_jumps is not None:
                config["n_jumps"] = args.n_jumps
                Cls = LangevinOptimizerStrided
            else:
                Cls = LangevinOptimizer
            optim = Cls(config, device, score, sde, prior_trafo,
                        1.0, prior_input_scale=prior_input_scale).to(device)

            samples = optim.sample((ref, mvue, maps_b, mask_b))        # (B,2,H,W)
            rec_sf  = samples.permute(0, 2, 3, 1).contiguous()         # (B,H,W,2)

            if args.all_slices:
                # Unscale back to the original target domain for 3D assembly.
                for b, ck in enumerate(ckeys):
                    fbp_by_idx[ck] = _to_mag(fbp_sf[b]) / kspace_scale
                    rec_by_idx[ck] = _to_mag(rec_sf[b]) / kspace_scale
            else:
                # 2D mid-slice metrics (fast; NOT comparable to the 3D tables).
                fbp_p = kno_psnr(fbp_sf[0], gt_b[0]); fbp_s = kno_ssim(fbp_sf[0], gt_b[0])
                fbp_n = kno_nmse(fbp_sf[0], gt_b[0])
                rec_p = kno_psnr(rec_sf[0], gt_b[0]); rec_s = kno_ssim(rec_sf[0], gt_b[0])
                rec_n = kno_nmse(rec_sf[0], gt_b[0])
                fbp_psnrs.append(fbp_p.item()); fbp_ssims.append(fbp_s.item())
                fbp_nmses.append(fbp_n.item())
                rec_psnrs.append(rec_p.item()); rec_ssims.append(rec_s.item())
                rec_nmses.append(rec_n.item())
                logging.info(f"[vol {vol_key}] FBP  2D PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}  NMSE={fbp_n:.6f}")
                logging.info(f"[vol {vol_key}] CSGM 2D PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}  NMSE={rec_n:.6f}")

        if args.all_slices:
            # Assemble the 3D volume and compute 3D metrics. Per-slice (H,W) mags are
            # the two non-readout axes, so stack along readout_axis to recover (X,Y,Z).
            rec_3d = torch.stack([rec_by_idx[s] for s in slices_iter], dim=args.readout_axis)
            fbp_3d = torch.stack([fbp_by_idx[s] for s in slices_iter], dim=args.readout_axis)
            gt_3d  = target_t3.abs().to(device)

            fbp_p, fbp_s, fbp_n = volume_metrics(fbp_3d, gt_3d)
            rec_p, rec_s, rec_n = volume_metrics(rec_3d, gt_3d)
            fbp_psnrs.append(fbp_p.item()); fbp_ssims.append(fbp_s.item())
            fbp_nmses.append(fbp_n.item())
            rec_psnrs.append(rec_p.item()); rec_ssims.append(rec_s.item())
            rec_nmses.append(rec_n.item())

            logging.info(f"[vol {vol_key}] FBP  3D PSNR={fbp_p:.2f} dB  SSIM={fbp_s:.4f}  NMSE={fbp_n:.6f}")
            logging.info(f"[vol {vol_key}] CSGM 3D PSNR={rec_p:.2f} dB  SSIM={rec_s:.4f}  NMSE={rec_n:.6f}")

            if args.save_dir is not None:
                np.save(os.path.join(args.save_dir, f"vol{vol_key}_gt.npy"), gt_3d.cpu().numpy())
                np.save(os.path.join(args.save_dir, f"vol{vol_key}_recon.npy"), rec_3d.cpu().numpy())
                np.save(os.path.join(args.save_dir, f"vol{vol_key}_fbp.npy"), fbp_3d.cpu().numpy())

        wandb.log({"fbp_psnr": fbp_psnrs[-1], "fbp_ssim": fbp_ssims[-1],
                   "fbp_nmse": fbp_nmses[-1],
                   "rec_psnr": rec_psnrs[-1], "rec_ssim": rec_ssims[-1],
                   "rec_nmse": rec_nmses[-1],
                   "rec_psnr_mean": float(np.mean(rec_psnrs)),
                   "rec_ssim_mean": float(np.mean(rec_ssims)),
                   "rec_nmse_mean": float(np.mean(rec_nmses)),
                   "global_step": i})

        per_vol.append({
            "vol_key": vol_key, "vol_idx": args.vol_start + i,
            "all_slices": bool(args.all_slices),
            "fbp_psnr": fbp_psnrs[-1], "fbp_ssim": fbp_ssims[-1], "fbp_nmse": fbp_nmses[-1],
            "rec_psnr": rec_psnrs[-1], "rec_ssim": rec_ssims[-1], "rec_nmse": rec_nmses[-1],
        })
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump({"args": vars(args), "per_volume": per_vol,
                           "mean": {"fbp_psnr": float(np.mean(fbp_psnrs)),
                                    "fbp_ssim": float(np.mean(fbp_ssims)),
                                    "fbp_nmse": float(np.mean(fbp_nmses)),
                                    "rec_psnr": float(np.mean(rec_psnrs)),
                                    "rec_ssim": float(np.mean(rec_ssims)),
                                    "rec_nmse": float(np.mean(rec_nmses))}}, f, indent=2)

    logging.info(
        f"\n{'='*62}\n"
        f"  n = {len(rec_psnrs)} volumes  ({'3D' if args.all_slices else '2D mid-slice'} metrics)\n"
        f"  FBP  PSNR: {np.mean(fbp_psnrs):.2f} +/- {np.std(fbp_psnrs):.2f} dB\n"
        f"  FBP  SSIM: {np.mean(fbp_ssims):.4f} +/- {np.std(fbp_ssims):.4f}\n"
        f"  FBP  NMSE: {np.mean(fbp_nmses):.6f} +/- {np.std(fbp_nmses):.6f}\n"
        f"  CSGM PSNR: {np.mean(rec_psnrs):.2f} +/- {np.std(rec_psnrs):.2f} dB\n"
        f"  CSGM SSIM: {np.mean(rec_ssims):.4f} +/- {np.std(rec_ssims):.4f}\n"
        f"  CSGM NMSE: {np.mean(rec_nmses):.6f} +/- {np.std(rec_nmses):.6f}\n"
        f"{'='*62}"
    )

    if wandb.run is not None:
        wandb.run.summary["fbp_psnr_mean"] = float(np.mean(fbp_psnrs))
        wandb.run.summary["fbp_ssim_mean"] = float(np.mean(fbp_ssims))
        wandb.run.summary["fbp_nmse_mean"] = float(np.mean(fbp_nmses))
        wandb.run.summary["rec_psnr_mean"] = float(np.mean(rec_psnrs))
        wandb.run.summary["rec_ssim_mean"] = float(np.mean(rec_ssims))
        wandb.run.summary["rec_nmse_mean"] = float(np.mean(rec_nmses))

    wandb.finish()


if __name__ == "__main__":
    main()
