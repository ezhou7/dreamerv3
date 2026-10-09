"""Phase 0 benchmark: per-step policy latency for each inference backend.

Run on the Orange Pi (and the dev box for comparison) to decide whether
the NPU port is worth it -- see aerial/deploy/NPU_PLAN.md, Phase 0.

    python -m aerial.deploy.npu.bench --model aerial/deploy/models/<bundle>
    python -m aerial.deploy.npu.bench --model <bundle> --backends numpy \\
        --json bench-$(hostname).json

Backends:
  numpy  aerial.deploy.npu.reference (CPU, numpy only)
  jax    the current deploy path (agent.policy on JAX CPU), built without
         the PyBullet env import

No trained model handy? Make a dummy bundle (random weights, same shapes):
    python -m aerial.deploy.npu.jax_agent --out aerial/deploy/models/dummy
"""
import os

# Batch-1 GEMVs are too small to split across threads: multithreaded BLAS
# made p99 ~15x worse on the dev box. Must be set before numpy loads;
# override from the shell to compare.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from aerial.spec.interface import OBS_DIM, STEP_RATE_HZ


def make_backend(name, model_dir, sampling):
    if name == 'numpy':
        from aerial.deploy.npu.reference import NumpyPolicyInference
        return NumpyPolicyInference(model_dir, sampling=sampling, seed=0)
    if name == 'jax':
        from aerial.deploy.npu import jax_agent
        return jax_agent.JaxPolicy(jax_agent.load_bundle_agent(model_dir))
    raise ValueError(f'unknown backend {name!r}')


def time_policy(policy, steps, warmup, seed=0):
    rng = np.random.default_rng(seed)
    policy.reset()
    times = []
    for i in range(warmup + steps):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        t0 = time.perf_counter()
        action = policy.step(obs, is_first=(i == 0))
        dt = time.perf_counter() - t0
        assert np.isfinite(action).all(), action
        if i >= warmup:
            times.append(dt * 1e3)
    t = np.array(times)
    return {
        'n': int(t.size),
        'mean_ms': float(t.mean()),
        'p50_ms': float(np.percentile(t, 50)),
        'p90_ms': float(np.percentile(t, 90)),
        'p99_ms': float(np.percentile(t, 99)),
        'max_ms': float(t.max()),
    }


def system_info():
    info = {
        'host': platform.node(),
        'machine': platform.machine(),
        'system': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'cpu_count': os.cpu_count(),
    }
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        if var in os.environ:
            info[var] = os.environ[var]
    model = Path('/proc/device-tree/model')
    if model.exists():
        info['board'] = model.read_text().strip('\x00\n')
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path,
                    help='bundle dir (agent.pkl + config.yaml)')
    ap.add_argument('--backends', default='numpy,jax')
    ap.add_argument('--sampling', default='stochastic')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--json', type=Path, default=None,
                    help='also write results here')
    args = ap.parse_args()

    if not (args.model / 'policy_weights.npz').exists():
        from aerial.deploy.npu.extract_weights import extract_bundle
        out, _, _ = extract_bundle(args.model)
        print(f'extracted {out}')

    budget = 1000.0 / STEP_RATE_HZ
    results = {'system': system_info(), 'model': str(args.model),
               'budget_ms': budget, 'backends': {}}
    print(json.dumps(results['system'], indent=2))
    for name in args.backends.split(','):
        t0 = time.perf_counter()
        policy = make_backend(name, args.model, args.sampling)
        load_s = time.perf_counter() - t0
        stats = time_policy(policy, args.steps, args.warmup)
        stats['load_s'] = load_s
        results['backends'][name] = stats
        ok = 'OK' if stats['p99_ms'] < budget else 'TOO SLOW'
        print(f'{name:>6}: load {load_s:5.1f}s  mean {stats["mean_ms"]:6.2f}  '
              f'p50 {stats["p50_ms"]:6.2f}  p99 {stats["p99_ms"]:6.2f}  '
              f'max {stats["max_ms"]:6.2f} ms  '
              f'(budget {budget:.0f} ms: {ok})')
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
