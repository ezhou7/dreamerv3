"""Smoke test for a bundled model.

Loads the model with PolicyInference, feeds 500 random observation
vectors through it, and prints per-step latency + action-range stats.
Run this on the Orange Pi first (after `pip install -e .[cpu]`) to
verify the model loads and hits the 50 Hz budget.

Usage:
    python aerial/deploy/smoke_test.py \\
        --model aerial/deploy/models/skydreamer-17m-1c
"""
import argparse
from pathlib import Path

import numpy as np

from aerial.deploy.inference import PolicyInference, benchmark_latency
from aerial.spec.interface import OBS_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path,
                    help='deployment bundle directory (contains agent.pkl)')
    ap.add_argument('--steps', type=int, default=500)
    args = ap.parse_args()

    print(f'loading model: {args.model}')
    print('(first call triggers JAX trace + compile; can take ~30 s)')
    policy = PolicyInference(args.model)
    print('model loaded.')

    # Deterministic obs -> deterministic action check.
    policy.reset()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    a0 = policy.step(obs, is_first=True)
    a1 = policy.step(obs, is_first=False)  # same obs, state advanced
    print(f'first action (zero obs):    {a0}')
    print(f'second action (zero obs):   {a1}')
    assert a0.shape == (4,), a0.shape
    # DreamerV3's Gaussian policy head can output values slightly outside
    # [-1, 1]; the ArduPilot bridge clips downstream. Assert only that
    # actions are finite and reasonably sized (not exploded).
    assert np.isfinite(a0).all(), f'non-finite action: {a0}'
    assert np.abs(a0).max() < 5.0, f'action magnitude too large: {a0}'

    # Latency benchmark.
    print(f'\nbenchmarking {args.steps} steps of random obs...')
    stats = benchmark_latency(args.model, n_steps=args.steps)
    target_period_ms = 1000.0 / 50  # 50 Hz control rate
    print(f'per-step latency (ms):')
    print(f'  mean {stats["mean_ms"]:.2f}  '
          f'p50 {stats["p50_ms"]:.2f}  '
          f'p90 {stats["p90_ms"]:.2f}  '
          f'p99 {stats["p99_ms"]:.2f}  '
          f'max {stats["max_ms"]:.2f}')
    ok = stats['p99_ms'] < target_period_ms
    print(f'  target {target_period_ms:.2f} ms (50 Hz)  ->  '
          f'{"OK" if ok else "TOO SLOW"}')

    print('\nsmoke test complete.')


if __name__ == '__main__':
    main()
