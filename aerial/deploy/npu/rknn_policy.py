"""Phase 4: run the policy on the RK3588 NPU via rknn-toolkit-lite2.

`RKNNPolicyInference` is a drop-in for `NumpyPolicyInference` /
`PolicyInference`: same `reset()` / `step(obs, is_first)` API and the same
`sampling` modes. The two NPU graphs (`core.rknn`, `actor.rknn`, built by
`convert_rknn.py`) do the matmuls; CPU glue (symlog, reset masking,
action normalization, unimix + sampling) reuses `ReferencePolicy`.

The runtime should match the toolkit version in `rknn_manifest.json`.
rknnlite always loads /usr/lib/librknnrt.so; if that one is older, set
the env var LIBRKNNRT_PATH to a matching copy (e.g. from
github.com/airockchip/rknn-toolkit2, rknpu2/runtime/Linux/librknn_api/aarch64).

    python -m aerial.deploy.npu.bench --model <bundle> --backends rknn,numpy
"""
import json
import os
from pathlib import Path

import numpy as np

from aerial.deploy.npu.reference import SAMPLING_MODES, ReferencePolicy, symlog

f32 = np.float32


def _use_runtime_lib(lib_path):
    """rknnlite hardcodes /usr/lib/librknnrt.so; point it elsewhere."""
    from rknnlite.api import rknn_runtime
    rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = lambda self: str(lib_path)


def _load(path, core_mask):
    from rknnlite.api import RKNNLite
    if os.environ.get('LIBRKNNRT_PATH'):
        _use_runtime_lib(os.environ['LIBRKNNRT_PATH'])
    rknn = RKNNLite()
    if rknn.load_rknn(str(path)) != 0:
        raise RuntimeError(f'load_rknn failed: {path}')
    if rknn.init_runtime(core_mask=core_mask) != 0:
        raise RuntimeError(f'init_runtime failed: {path} (librknnrt too old? '
                           'see LIBRKNNRT_PATH in rknn_policy.py)')
    return rknn


class RKNNPolicyInference:

    def __init__(self, model_dir, sampling='mean_action', seed=None, core_mask=None):
        from rknnlite.api import RKNNLite
        if sampling not in SAMPLING_MODES:
            raise ValueError(f'sampling must be one of {SAMPLING_MODES}')
        model_dir = Path(model_dir)
        manifest = model_dir / 'rknn_manifest.json'
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        # CPU-side glue only; the weights it loads are not used for matmuls.
        self.ref = ReferencePolicy.load(model_dir / 'policy_weights.npz')
        core_mask = RKNNLite.NPU_CORE_0 if core_mask is None else core_mask
        self.core = _load(model_dir / 'core.rknn', core_mask)
        self.actor = _load(model_dir / 'actor.rknn', core_mask)
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self.action_dim = self.ref.w['pol/head/action/mean/bias'].shape[0]
        self._carry = None

    def close(self):
        self.core.release()
        self.actor.release()

    def reset(self):
        hp = self.ref.hp
        self._carry = (
                np.zeros((1, hp['deter']), f32),
                np.zeros((1, hp['stoch'], hp['classes']), f32),
                np.zeros((1, self.action_dim), f32))

    def run_core(self, deter, stoch, prevact, obs):
        """Raw NPU core graph; inputs already masked. Returns deter', logits (1,S,C)."""
        deter, logits = self.core.inference(inputs=[
                symlog(obs).astype(f32), deter.astype(f32),
                stoch.reshape(1, -1).astype(f32),
                (prevact / np.maximum(1, np.abs(prevact))).astype(f32)])
        hp = self.ref.hp
        return (np.asarray(deter, f32).reshape(1, hp['deter']),
                np.asarray(logits, f32).reshape(1, hp['stoch'], hp['classes']))

    def run_actor(self, deter, stoch):
        mean, std = self.actor.inference(inputs=[
                deter.astype(f32), stoch.reshape(1, -1).astype(f32)])
        return (np.asarray(mean, f32).reshape(1, -1),
                np.asarray(std, f32).reshape(1, -1))

    def step(self, obs_vector, is_first=False):
        if self._carry is None:
            self.reset()
            is_first = True
        deter, stoch, prevact = self._carry
        if is_first:
            deter, stoch, prevact = (np.zeros_like(x) for x in self._carry)
        obs = np.asarray(obs_vector, f32).reshape(1, -1)
        deter, logits = self.run_core(deter, stoch, prevact, obs)
        if self.sampling == 'deterministic':
            stoch = self.ref.mode_latent(logits)
        else:
            stoch = self.ref.sample_latent(logits, self.rng)
        mean, std = self.run_actor(deter, stoch)
        if self.sampling == 'stochastic':
            action = mean + std * self.rng.standard_normal(mean.shape).astype(f32)
        else:
            action = mean
        action = action.astype(f32)
        self._carry = (deter, stoch, action)
        return action.reshape(-1)


def check(model_dir, steps=3000, seed=0):
    """NPU vs ReferencePolicy (Phase 5 step-level + drift checks).

    step:  same inputs into both backends each step (teacher-forced on the
           reference carry) -> per-step fp16 error.
    drift: independent deterministic carries, closed loop on random obs
           with a reset every 500 steps -> compounding error in deter and
           how often the argmax latent / action diverge.
    """
    ref = ReferencePolicy.load(Path(model_dir) / 'policy_weights.npz')
    npu = RKNNPolicyInference(model_dir, sampling='deterministic')
    hp, rng = ref.hp, np.random.default_rng(seed)
    zeros = lambda: (np.zeros((1, hp['deter']), f32),  # noqa: E731
                     np.zeros((1, hp['stoch'], hp['classes']), f32),
                     np.zeros((1, npu.action_dim), f32))
    r_carry, n_carry = zeros(), zeros()
    step_err = {'deter': 0.0, 'logits': 0.0, 'mean': 0.0, 'std': 0.0}
    drift = {'deter': [], 'action': [], 'latent_flips': 0}
    for i in range(steps):
        obs = rng.standard_normal((1, 22)).astype(f32)
        if i % 500 == 0:
            r_carry, n_carry = zeros(), zeros()
        # Reference closed loop.
        rd, rl = ref.core(*r_carry, obs, np.array([False]))
        rs = ref.mode_latent(rl)
        rm, rsd = ref.actor(rd, rs)
        # Step-level: NPU on the reference carry.
        nd, nl = npu.run_core(*r_carry, obs)
        nm, nsd = npu.run_actor(rd, rs)
        for k, a, b in (('deter', rd, nd), ('logits', rl, nl), ('mean', rm, nm), ('std', rsd, nsd)):
            step_err[k] = max(step_err[k], float(np.abs(a - b).max()))
        # Drift: NPU closed loop on its own carry.
        dd, dl = npu.run_core(*n_carry, obs)
        ds = ref.mode_latent(dl)
        dm, _ = npu.run_actor(dd, ds)
        drift['deter'].append(float(np.abs(dd - rd).max()))
        drift['action'].append(float(np.abs(dm - rm).max()))
        drift['latent_flips'] += int((ds.argmax(-1) != rs.argmax(-1)).sum())
        r_carry, n_carry = (rd, rs, rm), (dd, ds, dm)
    npu.close()
    d, a = np.array(drift['deter']), np.array(drift['action'])
    print('step-level max abs err: ' + ', '.join(f'{k} {v:.2e}' for k, v in step_err.items()))
    print(f'drift over {steps} steps: deter max {d.max():.3f} p99 {np.percentile(d, 99):.3f}; '
          f'action max {a.max():.3f} mean {a.mean():.4f}; '
          f'latent flips {drift["latent_flips"]}/{steps * hp["stoch"]}')
    return step_err, drift


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NPU vs numpy reference check')
    parser.add_argument('--model', required=True)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    check(args.model, args.steps)
