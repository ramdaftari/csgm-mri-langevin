#!/bin/bash
#SBATCH -p gh
#SBATCH -A TG-NAIRR240304
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/10471/peterwg/csgm_steps_run/logs/%x_%j.out
#SBATCH -e /scratch/10471/peterwg/csgm_steps_run/logs/%x_%j.err
#
# CSGM BraTS multi-coil (8-coil) 4x — 1-step vs 10-step annealed Langevin,
# full val set, 3D volume metrics.
#
# Driven by two exported vars:
#   RES = qr | hr | fr
#   ARM = 1step | 10step
# Submit via sbatch/submit_all.sh (sets -J and -t per combination).
#
# Shared hyperparameters (fixed): step_lr=1e-5, mse=5.0, n_steps_each=1
#   1step : t_start=0,  n_jumps=1   -> 1 level  x 1 = 1  update
#   10step: t_start=18, n_jumps=10  -> 10 levels x 1 = 10 updates

set -euo pipefail

module load gcc cuda/12.6 openmpi
ulimit -n 4096
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/10471/peterwg/csgm_steps_run
RR=$ROOT/src/ram/resolution_robust_3d_mri
SCRIPT=$ROOT/src/ram/csgm-mri-langevin/reconstruct_csgm_brats.py

case "$RES" in
  qr) LMDB_RES=60x60x39    ;;
  hr) LMDB_RES=120x120x78  ;;
  fr) LMDB_RES=240x240x155 ;;
  *)  echo "bad RES=$RES (want qr|hr|fr)" >&2; exit 1 ;;
esac
LMDB=/scratch/10471/peterwg/brats2021_lmdb/brats_val_${LMDB_RES}_4x_8coil_lmdb

case "$ARM" in
  1step)  T_START=0;  N_JUMPS=1  ;;
  10step) T_START=18; N_JUMPS=10 ;;
  *) echo "bad ARM=$ARM (want 1step|10step)" >&2; exit 1 ;;
esac

# wandb
source /work/10471/peterwg/vista/.wandb_creds.sh
export WANDB_ENTITY=ramdaftari-indian-institute-of-technology-kanpur
export WANDB_PROJECT=brats_csgm_steps
export WANDB_NAME=csgm_brats_8coil_4x_${RES}_${ARM}
export WANDB_DIR=$ROOT/wandb
mkdir -p "$WANDB_DIR"

echo "host=$(hostname)  RES=$RES  ARM=$ARM  t_start=$T_START  n_jumps=$N_JUMPS"
echo "lmdb=$LMDB"
nvidia-smi -L

# --num_volumes 1000 is capped internally by min(n, total): 251 (qr/hr), 100 (fr).
# --all_slices reconstructs every readout slice and reports 3D volume PSNR/SSIM/NMSE.
$RR/.venv/bin/python "$SCRIPT" \
    --lmdb "$LMDB" \
    --num_volumes 1000 \
    --all_slices \
    --t_start   $T_START \
    --n_jumps   $N_JUMPS \
    --n_steps_each 1 \
    --step_lr   1e-5 \
    --mse       5.0 \
    --init_mode zf
