"""Evaluate checkpoints across a training run to pick the best one.

Hover precision can peak before the end of a run: the
aerial_12in_skydreamer_smooth run plateaued at 0.12-0.15 m from ~8M
steps on, with ~0.02 m of scatter between checkpoints, and its last
checkpoint was one of the worse ones (0.19 m vs 0.12 m at 9.4M). Export
the checkpoint this picks with `export_model.py --step <step>`.

Exports the checkpoint nearest each requested step to a scratch dir and
runs eval_bundle.run() on it (mean actions, same seeds every time).

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/ckpt_sweep.py \\
        --logdir ~/logdir/dreamer/<run> --every 1e6 --stage 1c
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np

from aerial.deploy.export_model import export, list_ckpts
from aerial.deploy.inference import PolicyInference
from aerial.train.eval_bundle import run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", required=True, type=Path)
    ap.add_argument("--every", type=float, default=1e6,
                    help="evaluate the checkpoint nearest every N steps")
    ap.add_argument("--steps", type=float, nargs="*", default=None,
                    help="explicit steps instead of --every")
    ap.add_argument("--stage", default="1c")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--cuda-device", default="1",
                    help="GPU index for the policy; '' runs on CPU")
    args = ap.parse_args()

    logdir = args.logdir.expanduser()
    ckpts = list_ckpts(logdir)
    last = max(ckpts.values())
    targets = args.steps or list(np.arange(args.every, last, args.every)) + [last]
    picks = {}
    for t in targets:
        name = min(ckpts, key=lambda n: abs(ckpts[n] - t))
        picks[name] = ckpts[name]

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, step in sorted(picks.items(), key=lambda kv: kv[1]):
            bundle = export(logdir, f"s{step}", tmp, ckpt=name, verbose=False)
            print(f"== step {step:,d} ({name})", flush=True)
            policy = PolicyInference(bundle, cuda_device=args.cuda_device or None)
            med = run(policy, args.stage, args.episodes)
            results.append((step, med["dist"], med["rate"], name))

    print("\nstep        median dist  rate-cmd change  checkpoint")
    best = min(results, key=lambda r: r[1])
    for step, dist, rate, name in results:
        mark = "  <- best" if name == best[3] else ""
        print(f"{step:>10,d}  {dist:8.3f} m  {rate:8.2f} rad/s   {name}{mark}")


if __name__ == "__main__":
    main()
