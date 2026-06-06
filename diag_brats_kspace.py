#!/usr/bin/env python3
"""
diag_brats_kspace.py — trace kspace amplitude at each BraTS CSGM pipeline step.

Run from resolution_robust_3d_mri/ with .venv active:
    python ../csgm-mri-langevin/diag_brats_kspace.py [--lmdb PATH] [--vol 0]

Prints amplitude at 5 checkpoints:
  1. Raw stored kspace (from LMDB)
  2. Mask values (are they 1+1j due to bug?)
  3. After 1D IFFT along X, mid-X slice   ← suspected near-zero
  4. Kspace recomputed via _fft(target_2d) ← should be ~0.3
  5. ref = mask * _fft(target_2d)          ← expected CSGM input
"""

import sys
import json
import argparse
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.fft as torch_fft

LMDB_QR = Path("/scratch/10471/peterwg/brats2021_lmdb/brats_val_60x60x39_4x_singlecoil_lmdb")


# --------------------------------------------------------------------------
# Minimal _fft/_ifft (same as reconstruct_csgm_brats.py)
# --------------------------------------------------------------------------
def _fft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x


def _ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x


# --------------------------------------------------------------------------
# LMDB helpers (same as reconstruct_csgm_brats.py)
# --------------------------------------------------------------------------
_ENV_CACHE = {}

def _open(path):
    key = str(Path(path).resolve())
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = lmdb.open(key, readonly=True, lock=False, readahead=False)
    return _ENV_CACHE[key]


def _read(env, vk, shape):
    with env.begin() as txn:
        buf = txn.get(vk.encode())
    if buf is None:
        raise KeyError(f"key {vk!r} missing")
    return np.frombuffer(buf, dtype=np.complex64).reshape(shape).copy()


def load_vol(lmdb_path, vol_idx):
    vk = str(vol_idx)
    env_shapes = _open(lmdb_path / "shapes")
    with env_shapes.begin() as txn:
        shp = json.loads(txn.get(vk.encode()).decode())
    kspace = _read(_open(lmdb_path / "kspace"), vk, tuple(shp["kspace"]))
    maps   = _read(_open(lmdb_path / "maps"),   vk, tuple(shp["maps"]))
    target = _read(_open(lmdb_path / "target"), vk, tuple(shp["target"]))
    mask   = _read(_open(lmdb_path / "masks"),  vk, tuple(shp["mask"]))
    return kspace, maps, target[0], mask   # target[0] → (X,Y,Z)


def sep(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def stats(label, arr):
    a = np.abs(arr)
    print(f"  {label}:")
    print(f"    dtype={arr.dtype}  shape={arr.shape}")
    print(f"    abs: max={a.max():.6e}  mean={a.mean():.6e}  nonzero={np.count_nonzero(a)}/{a.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(LMDB_QR))
    ap.add_argument("--vol",  type=int, default=0)
    ap.add_argument("--readout_axis", type=int, default=0)
    args = ap.parse_args()

    lmdb_path = Path(args.lmdb)
    vol_idx   = args.vol
    readout_axis = args.readout_axis

    print(f"LMDB  : {lmdb_path}")
    print(f"Volume: {vol_idx}  readout_axis={readout_axis}")

    kspace, maps, target, mask = load_vol(lmdb_path, vol_idx)
    ax = 1 + readout_axis          # axis in (C,X,Y,Z) layout

    # ------------------------------------------------------------------
    sep("1. Raw stored kspace (masked + stored by convert_brats_halfres_lmdb.py)")
    stats("kspace (C,X,Y,Z)", kspace)

    # ------------------------------------------------------------------
    sep("2. Mask values — checking for 1+1j bug")
    stats("mask (1,X,Y,Z)", mask)
    mask_real = mask.real
    mask_imag = mask.imag
    sampled = mask_real > 0.5
    print(f"    sampled fraction (real>0.5): {sampled.mean():.4f}")
    if sampled.any():
        print(f"    at sampled positions: real={mask_real[sampled].mean():.4f}  "
              f"imag={mask_imag[sampled].mean():.4f}  "
              f"  ← expected 1.0/0.0 if no bug, 1.0/1.0 if bug")

    # ------------------------------------------------------------------
    sep("3. After 1D IFFT along readout axis (old approach in reconstruct_csgm_brats.py)")
    ks_hybrid = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(kspace, axes=ax), axis=ax),
        axes=ax
    ).astype(np.complex64)
    mid = kspace.shape[ax] // 2
    ksp_2d = np.take(ks_hybrid, mid, axis=ax)   # (C, H, W)
    stats(f"ks_hybrid mid-slice (C,H,W), mid={mid}", ksp_2d)

    # Check a few other slices too
    for s_idx in [0, kspace.shape[ax]//4, kspace.shape[ax]-1]:
        sl = np.take(ks_hybrid, s_idx, axis=ax)
        print(f"    slice {s_idx}: max_abs={np.abs(sl).max():.6e}")

    # ------------------------------------------------------------------
    sep("4. Target slice (from LMDB target sub-db)")
    target_2d = np.take(target, mid, axis=readout_axis).real.astype(np.float32)  # (H,W)
    print(f"  target_2d: shape={target_2d.shape}  max={target_2d.max():.4f}  "
          f"mean={target_2d.mean():.4f}")

    # ------------------------------------------------------------------
    sep("5. Kspace recomputed via _fft(target_2d) — proposed fix")
    target_t = torch.from_numpy(target_2d)
    target_c = torch.complex(target_t, torch.zeros_like(target_t))  # (H,W) complex
    ksp_recomputed = _fft(target_c.unsqueeze(0))                    # (1,H,W) complex
    stats("ksp_recomputed = _fft(target_2d)", ksp_recomputed.numpy())

    # ------------------------------------------------------------------
    sep("6. ref = mask_2d * ksp_recomputed — CSGM observation (proposed fix)")
    # mask is uniform along X → take any X slice (use X=0)
    mask_2d = np.take(mask[0].real, 0, axis=readout_axis).astype(np.float32)  # (H,W)
    print(f"  mask_2d: shape={mask_2d.shape}  nonzero_frac={mask_2d.mean():.4f}")

    mask_t = torch.from_numpy(mask_2d)
    ref_2d = ksp_recomputed * mask_t.unsqueeze(0)   # (1,H,W)
    stats("ref = mask * _fft(target)", ref_2d.numpy())

    ref_real_norm = torch.view_as_real(ref_2d).norm().item()
    H, W = target_2d.shape
    sf = np.sqrt(H * W * 2) / (ref_real_norm + 1e-30)
    print(f"\n  sf (scaling_factor) = {sf:.4f}   ← want ~5–20, NOT 1e9")

    # ------------------------------------------------------------------
    sep("7. FBP quality check (zero-filled reconstruction from ref)")
    ifft_ref = _ifft(ref_2d)                  # (1,H,W) complex
    fbp_mag  = ifft_ref.abs().squeeze(0)       # (H,W) real
    tgt_mag  = torch.from_numpy(target_2d)     # (H,W) real

    mse = (fbp_mag - tgt_mag).pow(2).mean().item()
    max_val = tgt_mag.abs().max().item()
    psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-30))
    print(f"  ZF FBP PSNR = {psnr:.2f} dB   ← want ~22–26 dB for 4x BraTS QR")
    print(f"  fbp_mag: max={fbp_mag.max().item():.4f}  "
          f"target_2d: max={target_2d.max():.4f}")

    print("\n" + "="*60)
    print("Summary:")
    print(f"  old 1D-IFFT ksp_2d max_abs = {np.abs(ksp_2d).max():.6e}")
    print(f"  new _fft(target) ref  max_abs = {ref_2d.abs().max().item():.6e}")
    print(f"  sf (new) = {sf:.4f}")
    print(f"  FBP PSNR (new) = {psnr:.2f} dB")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
