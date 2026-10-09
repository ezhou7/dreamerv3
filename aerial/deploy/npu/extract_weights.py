"""Extract the deploy-time policy weights from a DreamerV3 checkpoint.

At inference time `Agent.policy()` only runs the encoder, one RSSM
posterior step (`_core` + `obs*` layers) and the policy head. This pulls
exactly those parameters out of `agent.pkl` -- no decoder, prior, reward,
value or optimizer state -- together with the hyperparameters the
forward pass needs, into a single `policy_weights.npz`.

Needs only numpy + ruamel.yaml (no JAX), so it can run on the Pi too.

Usage:
    python -m aerial.deploy.npu.extract_weights \\
        --model aerial/deploy/models/<bundle>
    # writes aerial/deploy/models/<bundle>/policy_weights.npz
"""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml

FORMAT_VERSION = 1

# Parameters used by the deploy-time forward pass. Everything else in the
# checkpoint (dec/, rew/, con/, val/, slowval/, dyn/prior*, opt state) is
# training-only.
POLICY_PARAM_RE = re.compile(
    r'^(enc/mlp\d+(norm)?/'
    r'|dyn/(dynin\d+|dynhid\d+|dyngru|obs\d+|obslogit)(norm)?/'
    r'|pol/)')

META_KEY = '__meta__'


def hparams_from_config(cfg):
    """Pull the forward-pass hyperparameters out of a run's config.yaml
    (nested dict). Raises on anything the reference doesn't implement."""
    agent = cfg['agent']
    if agent['enc']['typ'] != 'simple' or agent['dyn']['typ'] != 'rssm':
        raise ValueError(
            f"unsupported enc/dyn: {agent['enc']['typ']}/{agent['dyn']['typ']}")
    if agent['policy_dist_cont'] != 'bounded_normal':
        raise ValueError(
            f"unsupported policy_dist_cont: {agent['policy_dist_cont']}")
    rssm, enc, pol = agent['dyn']['rssm'], agent['enc']['simple'], agent['policy']
    if rssm.get('absolute', False):
        raise ValueError('rssm.absolute=True is not supported')
    for name, block in (('rssm', rssm), ('enc', enc), ('policy', pol)):
        if block['norm'] != 'rms':
            raise ValueError(f'{name}.norm={block["norm"]!r}; only rms supported')
    return {
        'format_version': FORMAT_VERSION,
        'deter': int(rssm['deter']),
        'hidden': int(rssm['hidden']),
        'stoch': int(rssm['stoch']),
        'classes': int(rssm['classes']),
        'blocks': int(rssm['blocks']),
        'obslayers': int(rssm['obslayers']),
        'dynlayers': int(rssm['dynlayers']),
        'unimix': float(rssm['unimix']),
        'rssm_act': rssm['act'],
        'enc_layers': int(enc['layers']),
        'enc_act': enc['act'],
        'enc_symlog': bool(enc['symlog']),
        'pol_layers': int(pol['layers']),
        'pol_act': pol['act'],
        'minstd': float(pol['minstd']),
        'maxstd': float(pol['maxstd']),
        'norm_eps': 1e-4,  # embodied.jax.nets.Norm default
        'compute_dtype': cfg.get('jax', {}).get('compute_dtype', 'unknown'),
    }


def extract_from_params(params, cfg):
    """params: flat {name: array} from agent.pkl. cfg: nested run config.
    Returns (weights dict of float32 arrays, hparams dict)."""
    hp = hparams_from_config(cfg)
    if any(k.startswith('enc/cnn') for k in params):
        raise ValueError('image encoders are not supported (vector obs only)')
    weights = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in params.items() if POLICY_PARAM_RE.match(k)}
    _check_shapes(weights, hp)
    return weights, hp


def _check_shapes(w, hp):
    D, H, S, C, G = (hp[k] for k in ('deter', 'hidden', 'stoch', 'classes', 'blocks'))
    expect = {
        'dyn/dynin0/kernel': (D, H),
        'dyn/dynin1/kernel': (S * C, H),
        'dyn/dyngru/kernel': (G, D // G, 3 * D // G),
        'dyn/obslogit/kernel': (H, S * C),
        'pol/mlp/linear0/kernel': (D + S * C, w['pol/mlp/linear0/kernel'].shape[1]),
    }
    for key, shape in expect.items():
        if key not in w:
            raise KeyError(f'missing parameter {key}')
        if tuple(w[key].shape) != tuple(shape):
            raise ValueError(f'{key}: shape {w[key].shape} != expected {shape}')
    if hp['dynlayers'] > 0:
        shape = w['dyn/dynhid0/kernel'].shape
        if shape != (G, D // G + 3 * H, D // G):
            raise ValueError(f'dyn/dynhid0/kernel: unexpected shape {shape}')


def save_npz(path, weights, hp, extra_meta=None):
    meta = dict(hp, **(extra_meta or {}))
    arrays = dict(weights)
    arrays[META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(path, **arrays)


def load_npz(path):
    """Returns (weights dict, meta dict)."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data[META_KEY]))
        weights = {k: data[k] for k in data.files if k != META_KEY}
    if meta.get('format_version') != FORMAT_VERSION:
        raise ValueError(
            f'{path}: format_version {meta.get("format_version")} '
            f'!= {FORMAT_VERSION}')
    return weights, meta


def extract_bundle(model_dir, out=None):
    """Read <model_dir>/agent.pkl + config.yaml, write policy_weights.npz."""
    model_dir = Path(model_dir)
    agent_pkl = model_dir / 'agent.pkl'
    with agent_pkl.open('rb') as f:
        data = pickle.load(f)
    with (model_dir / 'config.yaml').open() as f:
        cfg = yaml.YAML(typ='safe').load(f)
    weights, hp = extract_from_params(data['params'], cfg)
    sha = hashlib.sha256(agent_pkl.read_bytes()).hexdigest()
    out = Path(out) if out else model_dir / 'policy_weights.npz'
    save_npz(out, weights, hp, {'src_agent_pkl_sha256': sha})
    return out, weights, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path,
                    help='bundle dir containing agent.pkl + config.yaml')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out, weights, hp = extract_bundle(args.model, args.out)
    n = sum(v.size for v in weights.values())
    print(f'wrote {out}')
    print(f'  {len(weights)} tensors, {n / 1e6:.2f}M params, '
          f'{out.stat().st_size / 2**20:.1f} MB')
    print(f'  deter={hp["deter"]} hidden={hp["hidden"]} '
          f'stoch={hp["stoch"]}x{hp["classes"]} blocks={hp["blocks"]} '
          f'compute_dtype={hp["compute_dtype"]}')
    if hp['compute_dtype'] != 'float32':
        print('  WARNING: trained in reduced precision; the float32 reference '
              'will differ slightly from the training-time forward pass.')


if __name__ == '__main__':
    main()
