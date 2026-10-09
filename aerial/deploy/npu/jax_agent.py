"""JAX-side helpers for validating and benchmarking the NPU port.

Builds the DreamerV3 agent straight from a config + the canonical
aerial spaces, bypassing `make_env` (which imports PyBullet). Used by the
parity test and the Phase 0 benchmark; not needed for NumPy/RKNN
inference.

Also creates dummy bundles (random-init weights, perturbed so biases and
norm scales are non-trivial) so the pipeline can be exercised without a
trained checkpoint:

        python -m aerial.deploy.npu.jax_agent --preset aerial_12in_skydreamer_1b \\
                --out aerial/deploy/models/dummy-size12m
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import elements  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from aerial.spec.interface import ACTION_DIM, OBS_DIM  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def aerial_spaces():
    obs_space = {
            'vector': elements.Space(np.float32, (OBS_DIM,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
    }
    act_space = {'action': elements.Space(np.float32, (ACTION_DIM,), -1, 1)}
    return obs_space, act_space


def preset_config(preset):
    """defaults + named block(s) from dreamerv3/configs.yaml, forced to CPU."""
    configs = yaml.YAML(typ='safe').load(
            (REPO / 'dreamerv3' / 'configs.yaml').read_text())
    config = elements.Config(configs['defaults'])
    for name in preset.split(','):
        config = config.update(configs[name])
    return _cpu(config)


def _cpu(config):
    return config.update({
            'jax.platform': 'cpu',
            'jax.cuda_visible_devices': '',
            'jax.compute_dtype': 'float32',
    })


def build_agent(config):
    from dreamerv3.agent import Agent
    obs_space, act_space = aerial_spaces()
    return Agent(obs_space, act_space, elements.Config(
            **config.agent,
            logdir=str(config.logdir),
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=0,
            replicas=1,
    ))


def load_bundle_agent(model_dir):
    """Agent with weights from <model_dir>/agent.pkl + config.yaml."""
    model_dir = Path(model_dir)
    cfg = yaml.YAML(typ='safe').load((model_dir / 'config.yaml').read_text())
    agent = build_agent(_cpu(elements.Config(cfg)))
    with (model_dir / 'agent.pkl').open('rb') as f:
        agent.load(pickle.load(f))
    return agent


def perturb_params(params, seed=0, scale=0.1):
    """Random-init params have zero biases and unit norm scales, which would
    let a reference that forgets them pass. Perturb every tensor."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in params.items():
        v = np.asarray(v)
        if not np.issubdtype(v.dtype, np.floating):
            out[k] = v  # optimizer step counters etc.
            continue
        noise = rng.standard_normal(v.shape).astype(np.float32)
        if k.endswith('/bias'):
            out[k] = v + scale * noise
        elif k.endswith('/scale'):
            out[k] = v * (1 + scale * noise)
        else:
            out[k] = v + scale * (v.std() + 1e-3) * noise
    return out


def make_dummy_bundle(preset, out, seed=0):
    config = preset_config(preset).update(logdir=str(out))
    agent = build_agent(config)
    data = agent.save()
    data['params'] = perturb_params(data['params'], seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'agent.pkl').open('wb') as f:
        pickle.dump(data, f)
    config.save(out / 'config.yaml')
    return out


class JaxPolicy:
    """Minimal stateful wrapper over agent.policy() (same as
    aerial.deploy.inference.PolicyInference, minus the env import)."""

    def __init__(self, agent):
        self.agent = agent
        self._carry = None

    def reset(self):
        self._carry = self.agent.init_policy(batch_size=1)

    def step(self, obs_vector, is_first=False):
        if self._carry is None:
            self.reset()
            is_first = True
        obs = {
                'vector': np.asarray(obs_vector, np.float32).reshape(1, OBS_DIM),
                'is_first': np.array([bool(is_first)]),
                'is_last': np.array([False]),
                'is_terminal': np.array([False]),
                'reward': np.array([0.0], np.float32),
        }
        self._carry, act, _ = self.agent.policy(self._carry, obs, mode='eval')
        return np.asarray(act['action'], np.float32).reshape(ACTION_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='aerial_12in_skydreamer_1b')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out = make_dummy_bundle(args.preset, args.out, args.seed)
    print(f'wrote dummy bundle to {out}')


if __name__ == '__main__':
    main()
