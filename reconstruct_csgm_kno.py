#!/usr/bin/env python3
"""
reconstruct_csgm_kno.py

Faithful CSGM (Jalal et al. 2021, "Robust Compressed Sensing MRI with Deep
Generative Priors") reconstruction adapted to SKM-TEA val data.

Built by starting from csgm-mri-langevin/main.py and changing ONLY:
  1. Diffusion prior: SKM-TEA-trained DDPM (res_rob create_dense_model + EMA)
     replaces brain-trained NCSNv2. The NCSNv2 sigma sweep becomes a DDPM
     timestep sweep with sigma_t = sqrt(1 - abar_t) and p_grad = -eps / sigma_t.
  2. Dataset + mask: LMDBVolumeDataset (3D Gaussian mask, m*(1+j)*k storage with
     the (1-1j)/2 correction) replaces MVU_Estimator_Brain. CSGM is a 2D method
     so we extract the middle Z-slice; the slice's (Y,) mask matches CSGM's
     'vertical' MulticoilForwardMRI orientation.
  3. Metrics: KNO-style PSNR (20*log10(max|gt| / RMSE)) + meddlr SSIM, computed
     on the magnitude of the final reconstruction.

The annealed Langevin loop, normalize/unnormalize via the 99th-percentile MVUE,
the per-step gradient direction normalization
    meas_grad /= ||meas_grad||;  meas_grad *= ||p_grad||;  meas_grad *= mse
and the update
    x_new = x + step_size * (p_grad - meas_grad) + sqrt(2*step_size)*noise
are byte-for-byte the upstream CSGM algorithm.

Run from baselines/resolution_robust_3d_mri/:
    .venv/bin/python reconstruct_csgm_kno.py --num_volumes 1 --no_wandb
"""

import sys
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
KNO_SRC  = Path(__file__).resolve().parents[2]
KNO_VENV = Path(__file__).resolve().parents[3] / ".venv/lib/python3.10/site-packages"
sys.path.insert(0, str(KNO_VENV))
sys.path.append(str(KNO_SRC))

# meddlr (imported transitively) uses the deprecated np.complex alias.
if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore[attr-defined]

import meddlr.metrics.functional as _meddlr_metrics

# datasets.skmtea imports `from fastmri.subsample import MaskFunc` (KNO-local).
# The installed fastmri exposes it at fastmri.data.subsample — alias.
import fastmri.data.subsample as _fmri_sub
sys.modules.setdefault("fastmri.subsample", _fmri_sub)

# ---- res_rob diffusion prior ------------------------------------------------
from src.diffmodels.diffmodels_resolver import create_dense_model
from src.diffmodels.ema import ExponentialMovingAverage
from src.diffmodels.sde import DDPM
from src.problem_trafos.trafo_resolver import get_prior_trafo

# ---- KNO LMDB dataset -------------------------------------------------------
from datasets.skmtea import LMDBVolumeDataset


# ===========================================================================
# Constants — same checkpoint and LMDB as reconstruct_kno_val.py
# ===========================================================================
VAL_LMDB   = Path("/scratch/10846/armeet/datasets/skmtea_val_0.5first2_4x_lmdb")
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


def normalize(gen_img, estimated_mvue):
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling


def unnormalize(gen_img, estimated_mvue):
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling


# ===========================================================================
# KNO metrics — replaces CSGM's RSS / MVUE PSNR/SSIM at evaluation time.
# Identical to reconstruct_kno_val.py.
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


def kno_psnr(rec_xy2, gt_xy2):
    """rec, gt: (X, Y, 2). Returns scalar KNO PSNR (dB)."""
    rec_mag = _to_mag(rec_xy2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_xy2).unsqueeze(0).unsqueeze(0)
    return _kno_psnr_fn(rec_mag, gt_mag).squeeze()


def kno_ssim(rec_xy2, gt_xy2):
    rec_mag = _to_mag(rec_xy2).unsqueeze(0).unsqueeze(0)
    gt_mag  = _to_mag(gt_xy2).unsqueeze(0).unsqueeze(0)
    return _meddlr_metrics.ssim(rec_mag, gt_mag).mean()


# ===========================================================================
# Score model loader — same as reconstruct_kno_val.py.
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
    """SKM-TEA training prior_trafo: scale by 2.0, then move (X,Y,2) → (2,X,Y)."""
    prior_cfg = OmegaConf.load(HYDRA_ROOT / "problem_trafos/prior_trafo/crop_mag.yaml")
    OmegaConf.update(prior_cfg, "scaling_factor", 2.0)
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
        self.sigma_L = float(self.sigmas[0].item())              # CSGM σ_L

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
        logging.info(f"Running {len(self.timesteps)} DDPM timesteps × "
                     f"{self.config['n_steps_each']} inner Langevin steps "
                     f"(total {len(self.timesteps)*self.config['n_steps_each']})")

        forward_operator = lambda x: MulticoilForwardMRI(self.config["orientation"])(
            torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask
        )

        B = ref.shape[0]
        H, W = self.config["image_size"]
        samples = torch.rand(B, 2, H, W, device=self.device)         # CSGM init

        step_lr = float(self.config["step_lr"])
        pbar    = tqdm(self.timesteps, desc="anneal", leave=False)
        pbar_labels = ["t", "step_size", "error", "mean", "max"]

        with torch.no_grad():
            for k, t_idx in enumerate(pbar):
                sigma     = float(self.sigmas[k].item())
                step_size = step_lr * (sigma / self.sigma_L) ** 2
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
                    meas_grad /= torch.norm(meas_grad)
                    meas_grad *= torch.norm(p_grad)
                    meas_grad *= self.config["mse"]

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
# Per-volume preprocessing — KNO LMDB → CSGM-shaped (ref, mvue, maps, mask).
#   ref:  (B=1, C, H=X, W=Y) complex   masked k-space (with (1-1j)/2 correction)
#   mvue: (B=1, H, W) complex          for diagnostic only
#   maps: (B=1, C, H, W) complex       sensitivity maps
#   mask: (B=1, W) float               1D phase-encode mask
#   gt:   (X, Y, 2) real-as-complex    ground-truth target (for KNO metrics)
# ===========================================================================
def lmdb_volume_to_csgm_2d(batch, device):
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
    z_mid = Z // 2

    ksp_2d  = ksp_z_hybrid[:, :, :, z_mid].to(device)        # (C, X, Y) complex
    maps_2d = maps[:, :, :, z_mid].to(device)                # (C, X, Y) complex
    gt_2d   = torch.view_as_real(target.squeeze(0)[:, :, z_mid].to(device))  # (X,Y,2)

    # Phase-encode mask is constant along X. Take a single X-row, real part.
    # Shape: (Y,) — broadcasts as 'vertical' in MulticoilForwardMRI: mask[:, None, None, :]
    mask_y = mask.real[0, 0, :, z_mid].to(device).float()     # (Y,)

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
# Main
# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_volumes",  type=int,   default=1)
    parser.add_argument("--vol_start",    type=int,   default=0)
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--no_wandb",     action="store_true")
    # CSGM hyperparameters (defaults from upstream brain config):
    parser.add_argument("--t_start",      type=int,   default=399,
                        help="Highest DDPM noise step (replaces NCSNv2 L=232).")
    parser.add_argument("--n_steps_each", type=int,   default=3,
                        help="Inner Langevin steps per noise level (CSGM: 3).")
    parser.add_argument("--step_lr",      type=float, default=5e-5,
                        help="Base Langevin step size (CSGM brain default).")
    parser.add_argument("--mse",          type=float, default=5.0,
                        help="DC weight λ on the meas-grad direction (CSGM: 5).")
    parser.add_argument("--orientation",  type=str,   default="vertical",
                        choices=["vertical", "horizontal"])
    args = parser.parse_args()
    device = args.device

    wandb.init(project="skmtea_csgm_kno", name="csgm_kno_midslice",
               mode="disabled" if args.no_wandb else "online",
               config=vars(args))

    # ---- Score model + SDE + prior_trafo ----------------------------------
    score       = load_score_model(EMA_CKPT, TRAIN_CFG, device)
    sde         = DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)
    prior_trafo = load_prior_trafo()

    # ---- Dataset ----------------------------------------------------------
    dataset = LMDBVolumeDataset(root_dir=VAL_LMDB, norm_constant=3e7)
    logging.info(f"Val LMDB: {len(dataset)} volumes at {VAL_LMDB}")
    num_volumes = min(args.num_volumes, len(dataset) - args.vol_start)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    fbp_psnrs, rec_psnrs = [], []
    fbp_ssims, rec_ssims = [], []

    for i, batch in enumerate(
        tqdm(islice(loader, args.vol_start, args.vol_start + num_volumes),
             total=num_volumes, desc="volumes")
    ):
        ref, mvue, maps, mask_b, gt_2d = lmdb_volume_to_csgm_2d(batch, device)

        H, W = ref.shape[-2], ref.shape[-1]

        # ------------------------------------------------------------------
        # Bring ref into a unit-ish coordinate system so CSGM's [0,1]-init
        # samples are at the same scale as the underlying image and the
        # score model sees inputs in its training distribution.
        # Same formula as reconstruct_csgm_val.py / reconstruct_kno_val.py.
        # ------------------------------------------------------------------
        ref_real_norm = torch.view_as_real(ref.squeeze(0)).norm().item()
        scaling_factor = math.sqrt(float(H * W * 2)) / ref_real_norm

        ref_sf  = ref  * scaling_factor
        mvue_sf = mvue * scaling_factor

        if i == 0:
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
        }
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
