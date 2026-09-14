"""Bundle a trained DreamerV3 checkpoint into a self-contained model dir
suitable for `deploy.sh` to push to the Orange Pi.

Usage:
    python aerial/deploy/export_model.py \\
        --logdir ~/logdir/dreamer/aerial-<run>-<ts> \\
        --name skydreamer-17m-1c-2026-09-05

Output directory (default: aerial/deploy/models/<name>/):
    agent.pkl     - the network parameters (from the run's ckpt/latest)
    step.pkl      - training-step counter at the checkpoint
    done          - checkpoint sentinel file (empty)
    config.yaml   - the run's saved config, force-CPU-patched
    meta.json     - export metadata: source path, timestamp, hash
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import ruamel.yaml as yaml


def resolve_ckpt(logdir):
    latest = logdir / 'ckpt' / 'latest'
    if not latest.exists():
        raise SystemExit(f'no ckpt/latest in {logdir}')
    return logdir / 'ckpt' / latest.read_text().strip()


def strip_ref_pol(src_agent_pkl, dst_agent_pkl):
    """Remove ref_pol/* keys from the params dict. Older SkyDreamer
    checkpoints (before commit 9eacbb0) still have these; the current
    agent.py doesn't declare ref_pol so loading errors on shape mismatch.
    Returns True if any keys were stripped.
    """
    with src_agent_pkl.open('rb') as f:
        data = pickle.load(f)
    before = len(data.get('params', {}))
    if 'params' in data:
        data['params'] = {k: v for k, v in data['params'].items()
                          if 'ref_pol' not in k}
    after = len(data.get('params', {}))
    with dst_agent_pkl.open('wb') as f:
        pickle.dump(data, f)
    return before != after, before - after


def force_cpu_config(cfg):
    """Patch a run's saved config so downstream loaders don't try to grab
    a GPU that isn't there, and drop keys that no longer exist in the
    current code paths (config-schema drift from removed experiments
    like BC regularization and continuous curriculum). Idempotent."""
    # Env kwargs the current Hover12inEnv accepts.
    AERIAL_ENV_ALLOWED = {'gui', 'dr_stage'}
    # Kwargs the current imag_loss accepts.
    IMAG_LOSS_ALLOWED = {'slowtar', 'lam', 'actent', 'slowreg', 'smooth_coef'}
    if isinstance(cfg, dict):
        cfg.setdefault('jax', {})
        cfg['jax']['cuda_visible_devices'] = ''
        cfg['jax']['compute_dtype'] = 'float32'
        env = cfg.get('env')
        if isinstance(env, dict):
            aerial = env.get('aerial')
            if isinstance(aerial, dict):
                aerial['gui'] = False
                for k in list(aerial.keys()):
                    if k not in AERIAL_ENV_ALLOWED:
                        aerial.pop(k)
        agent = cfg.get('agent')
        if isinstance(agent, dict):
            imag_loss = agent.get('imag_loss')
            if isinstance(imag_loss, dict):
                for k in list(imag_loss.keys()):
                    if k not in IMAG_LOSS_ALLOWED:
                        imag_loss.pop(k)
    return cfg


def sha256_of(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', required=True, type=Path,
                    help='training run directory')
    ap.add_argument('--name', required=True,
                    help='short deploy name, e.g. skydreamer-17m-1c-2026-09-05')
    ap.add_argument('--out-root', type=Path,
                    default=Path(__file__).parent / 'models')
    args = ap.parse_args()

    logdir = args.logdir.expanduser()
    if not logdir.exists():
        raise SystemExit(f'logdir does not exist: {logdir}')
    src_ckpt = resolve_ckpt(logdir)
    print(f'src checkpoint: {src_ckpt}')

    dst = args.out_root / args.name
    dst.mkdir(parents=True, exist_ok=True)

    # agent.pkl, with ref_pol stripped for compatibility.
    stripped, n_stripped = strip_ref_pol(src_ckpt / 'agent.pkl', dst / 'agent.pkl')
    if stripped:
        print(f'stripped {n_stripped} ref_pol keys from agent.pkl')

    # Sidecar files.
    for name in ('step.pkl', 'done', 'replay.pkl'):
        src = src_ckpt / name
        if src.exists():
            shutil.copy(src, dst / name)

    # Config: rewrite for CPU-only headless.
    src_cfg = logdir / 'config.yaml'
    if not src_cfg.exists():
        raise SystemExit(f'no config.yaml in {logdir}')
    with src_cfg.open() as f:
        cfg = yaml.YAML(typ='safe').load(f)
    cfg = force_cpu_config(cfg)
    with (dst / 'config.yaml').open('w') as f:
        yaml.YAML(typ='safe').dump(cfg, f)

    # Metadata.
    meta = {
        'name': args.name,
        'src_logdir': str(logdir),
        'src_ckpt': str(src_ckpt),
        'exported_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'agent_pkl_sha256': sha256_of(dst / 'agent.pkl'),
        'stripped_ref_pol_keys': n_stripped if stripped else 0,
    }
    with (dst / 'meta.json').open('w') as f:
        json.dump(meta, f, indent=2)

    size_mb = (dst / 'agent.pkl').stat().st_size / (1024 * 1024)
    print(f'exported to {dst}')
    print(f'agent.pkl: {size_mb:.1f} MB')
    print(f'sha256:    {meta["agent_pkl_sha256"][:16]}...')


if __name__ == '__main__':
    main()
