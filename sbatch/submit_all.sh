#!/bin/bash
# Submit all six CSGM BraTS runs (3 resolutions x 2 arms) in parallel.
# MUST be run from a Vista LOGIN node — sbatch is blocked on compute nodes.
#
#   bash /scratch/10471/peterwg/csgm_steps_run/sbatch/submit_all.sh
#
# Walltimes are ~1.8x the measured single-volume timings extrapolated to the
# full val set (251 vols for qr/hr, 100 for fr) on one GH200.
set -eu
ROOT=/scratch/10471/peterwg/csgm_steps_run
S=$ROOT/sbatch/csgm_brats_steps.sbatch

submit () {  # res arm walltime
  jid=$(sbatch --parsable -J "csgm_${1}_${2}" -t "$3" \
        --export=ALL,RES="$1",ARM="$2" "$S")
  printf '%-12s %-7s  t=%-9s  jobid=%s\n' "$1" "$2" "$3" "$jid"
}

#      res  arm     walltime   (measured full-run estimate)
submit qr   1step   02:00:00   # ~46 min
submit qr   10step  02:00:00   # ~46 min
submit hr   1step   03:00:00   # ~1.1 h
submit hr   10step  05:00:00   # ~2.5 h
submit fr   1step   07:00:00   # ~3.8 h
submit fr   10step  09:00:00   # ~5.0 h

echo
echo "watch:   squeue -u peterwg"
echo "logs:    $ROOT/logs/"
echo "wandb:   https://wandb.ai/ramdaftari-indian-institute-of-technology-kanpur/brats_csgm_steps"
