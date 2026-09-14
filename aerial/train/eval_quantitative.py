"""Quantitative eval of a trained Phase 0 checkpoint.

Runs N episodes headlessly against the trained policy, captures per-step
trajectory data, and reports aggregate hover-quality statistics plus
per-episode plots.

Metrics:
  * Success rate (episodes surviving to EPISODE_STEPS)
  * Mean / median episode length
  * Mean per-step reward
  * Steady-state position error (last 2s of episode)
  * Steady-state speed (last 2s)
  * Steady-state gyro magnitude (last 2s)
  * Mean upright (R[2,2]) over episode
  * Time-to-target for random-init recovery (first step reaching <0.5m)

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/eval_quantitative.py
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/eval_quantitative.py \\
        --logdir ~/logdir/dreamer/aerial-pybullet-<timestamp> --n-episodes 50
"""
import argparse
import glob
import os
import sys
from collections import defaultdict
from functools import partial as bind

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import elements
import embodied
import ruamel.yaml as yaml

# Force env registration before make_env is called.
import aerial.envs.hover_pybullet_env  # noqa: F401
from aerial.envs.hover_pybullet_env import EPISODE_STEPS
from aerial.spec.interface import STEP_RATE_HZ, TARGET_POSITION
from dreamerv3.main import make_agent, make_env


def _resolve_checkpoint(logdir):
    ckpt_dir = os.path.join(logdir, "ckpt")
    latest_file = os.path.join(ckpt_dir, "latest")
    if os.path.isfile(latest_file):
        with open(latest_file) as f:
            return os.path.join(ckpt_dir, f.read().strip())
    subs = sorted(glob.glob(os.path.join(ckpt_dir, "*/")))
    return subs[-1].rstrip("/") if subs else None


def _load_config(config_name="aerial_pybullet_eval"):
    import dreamerv3
    cfg_path = os.path.join(os.path.dirname(dreamerv3.__file__), "configs.yaml")
    with open(cfg_path) as f:
        configs = yaml.YAML(typ="safe").load(f)
    config = elements.Config(configs["defaults"])
    config = config.update(configs[config_name])
    # Override to force headless + a scratch logdir.
    config = config.update({"env.aerial.gui": False})
    config = config.update({"logdir": "/tmp/aerial-eval-quant"})
    return config


def _steady(arr, tail_frac=0.2):
    """Mean of the last tail_frac of a per-step array. If arr is shorter
    than 2 samples, returns np.nan.
    """
    if len(arr) < 2:
        return np.nan
    n = max(1, int(len(arr) * tail_frac))
    return float(np.mean(arr[-n:]))


def _time_to_target(dists, threshold=0.5):
    """First step index at which distance dropped below threshold. None if
    never achieved.
    """
    idx = np.argmax(dists < threshold)
    if dists[idx] < threshold:
        return int(idx)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=None,
                    help="Training logdir. Defaults to newest under "
                         "~/logdir/dreamer/aerial-pybullet-*.")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--out-dir", default="/tmp/aerial-eval-quant")
    ap.add_argument("--config", default="aerial_pybullet_eval",
                    help="Config block from dreamerv3/configs.yaml. Use "
                         "aerial_12in_pybullet_eval for Phase 1.")
    ap.add_argument("--pattern", default="aerial-pybullet-*",
                    help="Glob for auto-discovering training logdirs.")
    args = ap.parse_args()

    if args.logdir is None:
        candidates = sorted(glob.glob(
            os.path.expanduser(f"~/logdir/dreamer/{args.pattern}")))
        candidates = [c for c in candidates if "-eval-" not in c]
        if not candidates:
            print("No training logdir found.", file=sys.stderr)
            sys.exit(1)
        args.logdir = candidates[-1]

    ckpt = _resolve_checkpoint(args.logdir)
    if not ckpt or not os.path.isdir(ckpt):
        print(f"No checkpoint found under {args.logdir}/ckpt", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Logdir     : {args.logdir}")
    print(f"Checkpoint : {ckpt}")
    print(f"Episodes   : {args.n_episodes}")
    print(f"Out dir    : {args.out_dir}")
    print()

    config = _load_config(args.config)

    agent = make_agent(config)

    episodes = []
    current = defaultdict(list)

    def on_step(tran, worker):
        v = np.asarray(tran["vector"])
        current["pos_err"].append(v[0:3].copy())
        current["vel"].append(v[3:6].copy())
        current["R_flat"].append(v[6:15].copy())
        current["gyro"].append(v[15:18].copy())
        current["reward"].append(float(tran["reward"]))
        current["is_terminal"].append(bool(tran["is_terminal"]))
        if tran["is_last"]:
            ep = {k: np.array(v) for k, v in current.items()}
            episodes.append(ep)
            current.clear()

    fns = [bind(make_env, config, 0)]
    driver = embodied.Driver(fns, parallel=False)
    driver.on_step(on_step)

    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(ckpt, keys=["agent"])

    print("Running eval episodes...")
    policy = lambda *a: agent.policy(*a, mode="eval")
    driver.reset(agent.init_policy)
    while len(episodes) < args.n_episodes:
        driver(policy, steps=10)
    episodes = episodes[:args.n_episodes]
    print(f"Collected {len(episodes)} episodes.\n")

    _report(episodes, args.out_dir)


def _report(episodes, out_dir):
    lengths = np.array([len(ep["reward"]) for ep in episodes])
    successes = lengths >= EPISODE_STEPS
    rewards_per_step = np.array([np.mean(ep["reward"]) for ep in episodes])
    total_rewards = np.array([np.sum(ep["reward"]) for ep in episodes])

    dists = [np.linalg.norm(ep["pos_err"], axis=1) for ep in episodes]
    speeds = [np.linalg.norm(ep["vel"], axis=1) for ep in episodes]
    gyros = [np.linalg.norm(ep["gyro"], axis=1) for ep in episodes]
    uprights = [ep["R_flat"].reshape(-1, 3, 3)[:, 2, 2] for ep in episodes]

    ss_dist = np.array([_steady(d) for d in dists])
    ss_speed = np.array([_steady(s) for s in speeds])
    ss_gyro = np.array([_steady(g) for g in gyros])
    mean_upright = np.array([float(np.mean(u)) for u in uprights])

    ttt = [_time_to_target(d, threshold=0.5) for d in dists]
    ttt_reached = [t for t in ttt if t is not None]

    print("=" * 62)
    print(" PHASE 0 QUANTITATIVE EVAL")
    print("=" * 62)
    print(f"Episodes:                 {len(episodes)}")
    print(f"Success rate:             "
          f"{100.0 * successes.mean():.1f}%  "
          f"({int(successes.sum())}/{len(episodes)} reached "
          f"full {EPISODE_STEPS} steps)")
    print()
    print(f"Episode length      steps         mean {lengths.mean():7.1f} "
          f"median {np.median(lengths):7.1f}  min {lengths.min()}  "
          f"max {lengths.max()}")
    print(f"Reward/step                       mean "
          f"{rewards_per_step.mean():7.3f}  std {rewards_per_step.std():.3f}")
    print(f"Total reward/ep                   mean "
          f"{total_rewards.mean():7.1f}  std {total_rewards.std():.1f}")
    print()
    print("Steady-state (last 20% of each episode):")
    print(f"  Position error       (m)         mean "
          f"{np.nanmean(ss_dist):.3f}  median {np.nanmedian(ss_dist):.3f}  "
          f"p90 {np.nanpercentile(ss_dist, 90):.3f}")
    print(f"  Speed                (m/s)       mean "
          f"{np.nanmean(ss_speed):.3f}  median {np.nanmedian(ss_speed):.3f}  "
          f"p90 {np.nanpercentile(ss_speed, 90):.3f}")
    print(f"  Gyro magnitude       (rad/s)     mean "
          f"{np.nanmean(ss_gyro):.3f}  median {np.nanmedian(ss_gyro):.3f}  "
          f"p90 {np.nanpercentile(ss_gyro, 90):.3f}")
    print(f"  Upright (R[2,2] avg over ep)     mean "
          f"{mean_upright.mean():.3f}  min {mean_upright.min():.3f}")
    print()
    print("Recovery from random init (first step within 0.5m of target):")
    if ttt_reached:
        arr = np.array(ttt_reached)
        print(f"  Reached in {len(ttt_reached)}/{len(episodes)} episodes "
              f"({100 * len(ttt_reached) / len(episodes):.0f}%)")
        print(f"  Steps  mean {arr.mean():.1f}  median {np.median(arr):.1f}  "
              f"p90 {np.percentile(arr, 90):.1f}")
        print(f"  Wall time  mean {arr.mean() / STEP_RATE_HZ:.2f}s")
    else:
        print("  Never reached — policy is not tracking target.")
    print("=" * 62)

    _plot(episodes, dists, speeds, gyros, uprights, out_dir)


def _plot(episodes, dists, speeds, gyros, uprights, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    t = lambda n: np.arange(n) / STEP_RATE_HZ

    for i, (d, s, g, u) in enumerate(zip(dists, speeds, gyros, uprights)):
        alpha = 0.35 if len(episodes) > 5 else 0.7
        axes[0, 0].plot(t(len(d)), d, alpha=alpha, linewidth=0.8)
        axes[0, 1].plot(t(len(s)), s, alpha=alpha, linewidth=0.8)
        axes[1, 0].plot(t(len(g)), g, alpha=alpha, linewidth=0.8)
        axes[1, 1].plot(t(len(u)), u, alpha=alpha, linewidth=0.8)

    axes[0, 0].set_ylabel("distance to target (m)")
    axes[0, 0].axhline(0.5, color="k", linestyle="--", alpha=0.4,
                       label="0.5 m")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].set_ylabel("speed (m/s)")
    axes[1, 0].set_ylabel("gyro magnitude (rad/s)")
    axes[1, 0].axhline(3.0, color="r", linestyle="--", alpha=0.4,
                       label="unstable threshold")
    axes[1, 0].legend(loc="upper right")
    axes[1, 1].set_ylabel("upright (R[2,2])")
    axes[1, 1].axhline(0.707, color="r", linestyle="--", alpha=0.4,
                       label="unstable threshold")
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].legend(loc="lower right")

    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 0 policy — {len(episodes)} eval episodes "
                 f"(target = {TARGET_POSITION.tolist()})")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "trajectories.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Wrote trajectory plot: {out_path}")


if __name__ == "__main__":
    main()
