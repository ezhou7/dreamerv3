"""Inference-only wrapper around a trained DreamerV3 policy.

The policy is stateful: an internal carry (encoder + RSSM latents) is
updated on every step. Use one instance per drone / episode:

    policy = PolicyInference('/path/to/model/dir')
    policy.reset()
    action = policy.step(obs_vector)  # obs_vector: (22,) float32
    # ... every 20 ms at 50 Hz ...
    action = policy.step(obs_vector)

The action returned is a (4,) float32 array in [-1, 1] with the canonical
aerial-spec layout:
    action[0:3] -> body-rate roll/pitch/yaw commands, tanh-scaled to
                    ±6 rad/s by aerial.spec.interface.scale_action().
    action[3]   -> collective thrust, linear-scaled to [0, 1] normalized
                    throttle by scale_action().

This module is intended to run on ARM Linux (Orange Pi companion
computer) as well as on the dev machine. It forces JAX onto CPU so the
Orange Pi has no CUDA dependency; on the dev machine you can still call
it from a GPU-preallocating process because we set the flag before jax
is imported.
"""
import os

# Force CPU-only JAX. Must be set before jax is imported anywhere.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=1')

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import elements  # noqa: E402
import embodied  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

# Register the aerial env family so make_agent can size its spaces.
import aerial.envs.hover_12in_env  # noqa: F401,E402
from aerial.spec.interface import OBS_DIM, ACTION_DIM  # noqa: E402
from dreamerv3.main import make_agent, make_env  # noqa: E402


class PolicyInference:
    """Loads a DreamerV3 checkpoint and answers one action per obs."""

    def __init__(self, model_dir, config_name=None):
        """model_dir contains: agent.pkl, config.yaml (bundled by
        export_model.py). config_name defaults to reading config.yaml.
        """
        self.model_dir = Path(model_dir)
        self._load_config(config_name)
        self._make_agent()
        self._load_checkpoint()
        self._carry = None

    def _load_config(self, config_name):
        cfg_path = self.model_dir / 'config.yaml'
        with cfg_path.open() as f:
            cfg = yaml.YAML(typ='safe').load(f)
        # Bundled config is already flat (the run's saved config, not
        # the multi-block configs.yaml). Wrap in an elements.Config.
        self.config = elements.Config(cfg)
        # Force CPU + headless env.
        self.config = self.config.update({
            'jax.cuda_visible_devices': '',
            'jax.compute_dtype': 'float32',
            'env.aerial.gui': False,
        })

    def _make_agent(self):
        # make_agent needs the env's obs/act spaces; build a throwaway
        # env instance to read them (env is discarded, only spaces used).
        env = make_env(self.config, 0)
        self._obs_space = env.obs_space
        self._act_space = env.act_space
        env.close()
        self.agent = make_agent(self.config)

    def _load_checkpoint(self):
        ckpt_path = self.model_dir / 'agent.pkl'
        assert ckpt_path.exists(), f'no agent.pkl at {ckpt_path}'
        cp = elements.Checkpoint()
        cp.agent = self.agent
        # elements.Checkpoint.load expects a directory containing
        # agent.pkl; our bundle IS that directory.
        cp.load(str(self.model_dir), keys=['agent'])

    def reset(self):
        """Call at the start of every episode."""
        self._carry = self.agent.init_policy(batch_size=1)

    def step(self, obs_vector, is_first=False):
        """One inference step.

        Args:
          obs_vector: (22,) float32 obs from aerial.spec.interface.get_obs_vector.
          is_first: True on the very first step of an episode. If False and
            reset() was never called, we auto-reset on the first call.

        Returns:
          (4,) float32 action in [-1, 1].
        """
        if self._carry is None:
            self.reset()
            is_first = True
        obs = _wrap_obs(obs_vector, is_first)
        self._carry, act, _ = self.agent.policy(self._carry, obs, mode='eval')
        # act is a dict {'action': array(shape=(1, 4))}; unwrap.
        action = np.asarray(act['action']).reshape(ACTION_DIM)
        return action.astype(np.float32)


def _wrap_obs(obs_vector, is_first):
    """Pack a 22-dim obs into the batched dict shape the agent expects."""
    obs_vector = np.asarray(obs_vector, dtype=np.float32).reshape(OBS_DIM)
    return {
        'vector': obs_vector[None],  # (1, 22)
        'is_first': np.array([bool(is_first)]),
        'is_last': np.array([False]),
        'is_terminal': np.array([False]),
        'reward': np.array([0.0], dtype=np.float32),
    }


def benchmark_latency(model_dir, n_steps=500):
    """Utility: load the model, run n_steps with random obs, return
    per-step wall-time stats (mean, p50, p90, p99, max) in milliseconds.
    """
    policy = PolicyInference(model_dir)
    policy.reset()
    rng = np.random.default_rng(0)
    times_ms = []
    # Warm up the JIT with a few real steps first (first ~2 calls trace).
    warmup = 10
    for i in range(warmup + n_steps):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        t0 = time.perf_counter()
        _ = policy.step(obs, is_first=(i == 0))
        t1 = time.perf_counter()
        if i >= warmup:
            times_ms.append((t1 - t0) * 1000.0)
    arr = np.array(times_ms)
    return {
        'n': int(arr.size),
        'mean_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p90_ms': float(np.percentile(arr, 90)),
        'p99_ms': float(np.percentile(arr, 99)),
        'max_ms': float(arr.max()),
        'min_ms': float(arr.min()),
    }
