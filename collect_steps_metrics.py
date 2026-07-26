#!/usr/bin/env python3
"""
collect_steps_metrics.py — assemble the CSGM BraTS 1-step / 10-step results table.

Reads the SLURM logs written by sbatch/csgm_brats_steps.sbatch and prints the
final 3D volume PSNR / SSIM / NMSE for each (resolution, arm), plus the shared
zero-filled (FBP) baseline. Runs still in flight are reported from their running
mean so partial progress is visible.

    python collect_steps_metrics.py                       # default log dir
    python collect_steps_metrics.py --logs /path/to/logs
"""
import re
import argparse
from pathlib import Path

DEFAULT_LOGS = "/scratch/10471/peterwg/csgm_steps_run/logs"

RES_LABEL = {"qr": "QR  60x60x39", "hr": "HR  120x120x78", "fr": "FR  240x240x155"}
ARMS      = ["1step", "10step"]

# Final summary block emitted at the end of a completed run.
FINAL_RE = re.compile(
    r"FBP\s+PSNR:\s*([\d.]+).*?FBP\s+SSIM:\s*([\d.]+).*?"
    r"CSGM\s+PSNR:\s*([\d.]+).*?CSGM\s+SSIM:\s*([\d.]+).*?CSGM\s+NMSE:\s*([\d.]+)",
    re.S,
)
# Per-volume lines, used to recover a running mean from an unfinished run.
VOL_RE = re.compile(
    r"CSGM 3D PSNR=([\d.]+) dB\s+SSIM=([\d.]+)\s+NMSE=([\d.]+)"
)
FBPVOL_RE = re.compile(r"FBP\s+3D PSNR=([\d.]+) dB\s+SSIM=([\d.]+)")


def parse(path):
    txt = path.read_text(errors="ignore")
    m = FINAL_RE.search(txt)
    if m:
        f_p, f_s, r_p, r_s, r_n = (float(x) for x in m.groups())
        n = len(VOL_RE.findall(txt))
        return dict(done=True, n=n, fbp_psnr=f_p, fbp_ssim=f_s,
                    psnr=r_p, ssim=r_s, nmse=r_n)
    vols = VOL_RE.findall(txt)
    if not vols:
        return None
    fb = FBPVOL_RE.findall(txt)
    mean = lambda xs: sum(xs) / len(xs)
    return dict(
        done=False, n=len(vols),
        fbp_psnr=mean([float(a) for a, _ in fb]) if fb else float("nan"),
        fbp_ssim=mean([float(b) for _, b in fb]) if fb else float("nan"),
        psnr=mean([float(a) for a, _, _ in vols]),
        ssim=mean([float(b) for _, b, _ in vols]),
        nmse=mean([float(c) for _, _, c in vols]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    args = ap.parse_args()

    logdir = Path(args.logs)
    rows = {}
    for res in RES_LABEL:
        for arm in ARMS:
            # newest log for this combination
            cands = sorted(logdir.glob(f"csgm_{res}_{arm}_*.out"),
                           key=lambda p: p.stat().st_mtime)
            if cands:
                r = parse(cands[-1])
                if r:
                    rows[(res, arm)] = r

    if not rows:
        print(f"no parseable logs in {logdir}")
        return

    print("\nCSGM BraTS 8-coil 4x — 3D volume metrics (full val set)")
    print("step_lr=1e-5  mse=5.0  n_steps_each=1 | 1step: t_start=0,n_jumps=1 | "
          "10step: t_start=18,n_jumps=10\n")
    hdr = f"{'Resolution':<16}{'Arm':<8}{'N':>5}  {'PSNR':>7} {'SSIM':>7} {'NMSE':>9}   {'ZF PSNR':>8} {'ZF SSIM':>8}"
    print(hdr); print("-" * len(hdr))
    for res, label in RES_LABEL.items():
        for arm in ARMS:
            r = rows.get((res, arm))
            if not r:
                print(f"{label:<16}{arm:<8}{'-':>5}  {'(no log yet)':>7}")
                continue
            flag = "" if r["done"] else "  <- in progress"
            print(f"{label:<16}{arm:<8}{r['n']:>5}  {r['psnr']:>7.2f} {r['ssim']:>7.4f} "
                  f"{r['nmse']:>9.6f}   {r['fbp_psnr']:>8.2f} {r['fbp_ssim']:>8.4f}{flag}")
    print()


if __name__ == "__main__":
    main()
