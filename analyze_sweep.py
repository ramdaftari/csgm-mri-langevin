#!/usr/bin/env python3
"""
analyze_sweep.py — find CSGM hyperparameters meeting the goal:

        1-step PSNR  >  ZF PSNR        and        10-step PSNR  >  1-step PSNR

Reads the JSONL written by sweep_steps.py. The 1-step arm is pinned at T=0 so it does
not depend on t_start; the 10-step arm sweeps t_start. A "solution" therefore pairs a
1-step row with a 10-step row that share the SAME (n_steps_each, step_lr, mse) — i.e.
one hyperparameter setting under which both inequalities hold. Global per-arm bests are
also reported, in case you prefer to tune each arm independently.

Usage:
    python analyze_sweep.py sweep.jsonl [--top 20]
"""
import json
import argparse
from pathlib import Path


def hp3(c):
    return (c["n_steps_each"], c["step_lr"], c["mse"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    if not rows:
        print("no rows yet")
        return

    zf = rows[0]["zf_mean_psnr"]
    one = {hp3(r["config"]): r for r in rows if r["config"]["arm"] == "1step"}
    ten = [r for r in rows if r["config"]["arm"] == "10step"]

    print(f"rows={len(rows)}  1step={len(one)}  10step={len(ten)}   ZF PSNR = {zf:.2f} dB\n")

    # ---- global per-arm bests ------------------------------------------------
    def show(r, label):
        c = r["config"]
        print(f"  {label:9} PSNR={r['mean_psnr']:6.2f} dB  SSIM={r['mean_ssim']:.4f}  "
              f"NMSE={r['mean_nmse']:.6f}  | t_start={c['t_start']:2d} "
              f"n_steps={c['n_steps_each']:2d} lr={c['step_lr']:g} mse={c['mse']:g} "
              f"| {r['n_levels']} lvl x {c['n_steps_each']} = {r['total_updates']} upd")

    print("BEST PER ARM (by mean PSNR):")
    if one:
        show(max(one.values(), key=lambda r: r["mean_psnr"]), "1-step")
    if ten:
        show(max(ten, key=lambda r: r["mean_psnr"]), "10-step")

    # ---- matched-HP solutions ------------------------------------------------
    sols = []
    for r10 in ten:
        r1 = one.get(hp3(r10["config"]))
        if r1 is None:
            continue
        if r1["mean_psnr"] > zf and r10["mean_psnr"] > r1["mean_psnr"]:
            sols.append((r10["mean_psnr"] - r1["mean_psnr"],
                         r1["mean_psnr"] - zf, r1, r10))

    print(f"\n{'='*100}")
    print(f"MATCHED-HP SOLUTIONS (1step > ZF AND 10step > 1step): {len(sols)} found")
    print(f"{'='*100}")
    if not sols:
        # Diagnose which half of the goal is failing.
        n_beat = sum(1 for r in one.values() if r["mean_psnr"] > zf)
        print(f"  none yet. 1-step configs beating ZF: {n_beat}/{len(one)}")
        if one:
            b = max(one.values(), key=lambda r: r["mean_psnr"])
            print(f"  best 1-step = {b['mean_psnr']:.2f} dB vs ZF {zf:.2f} dB "
                  f"({'OK' if b['mean_psnr'] > zf else 'BELOW ZF'})")
        ok10 = [r for r in ten if one.get(hp3(r["config"]))
                and r["mean_psnr"] > one[hp3(r["config"])]["mean_psnr"]]
        print(f"  10-step configs beating their matched 1-step: {len(ok10)}/{len(ten)}")
        return

    def table(sols_sorted, title):
        print(f"\n{title}")
        print(f"{'gain10>1':>9} {'gain1>ZF':>9} | {'1step':>7} {'10step':>7} | "
              f"{'t':>3} {'nstep':>5} {'lr':>8} {'mse':>6} | {'upd10':>6}")
        print("-" * 100)
        for d10, d1, r1, r10 in sols_sorted[:args.top]:
            c = r10["config"]
            print(f"{d10:+9.2f} {d1:+9.2f} | {r1['mean_psnr']:7.2f} {r10['mean_psnr']:7.2f} | "
                  f"{c['t_start']:3d} {c['n_steps_each']:5d} {c['step_lr']:8g} {c['mse']:6g} | "
                  f"{r10['total_updates']:6d}")

    # Two rankings: a large margin is easy to get at WEAK hyperparameters (a bad 1-step is
    # easy to beat), so absolute 10-step quality is usually the one you actually want.
    table(sorted(sols, key=lambda x: -x[3]["mean_psnr"]),
          "RANKED BY ABSOLUTE 10-step PSNR  <-- usually the one you want:")
    table(sorted(sols, key=lambda x: -x[0]),
          "RANKED BY 10step-over-1step MARGIN:")

    best = max(sols, key=lambda x: x[3]["mean_psnr"])
    d10, d1, r1, r10 = best
    print(f"\nRECOMMENDED (best absolute 10-step PSNR among solutions):")
    show(r1,  "1-step")
    show(r10, "10-step")
    print(f"\n  ZF {zf:.2f}  ->  1-step {r1['mean_psnr']:.2f} (+{d1:.2f})  "
          f"->  10-step {r10['mean_psnr']:.2f} (+{d10:.2f})")


if __name__ == "__main__":
    main()
