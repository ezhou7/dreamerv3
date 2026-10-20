"""NumPy reference implementation of the deploy-time DreamerV3 policy.

Mirrors exactly what `dreamerv3.agent.Agent.policy()` computes for a
vector-observation agent:

        obs --symlog--> Encoder MLP --> tokens
        (deter, stoch, prevact, tokens) --RSSM._core + obs layers--> deter', logits
        logits --unimix categorical sample--> stoch'
        (deter', stoch') --policy MLP + bounded_normal head--> mean, std --> action

The forward pass is split into the same two deterministic graphs that get
exported to ONNX / RKNN later (`core` and `actor`). All randomness lives
between them, on the CPU, so the NPU graphs stay pure.

This file is the numerical source of truth for the NPU port: every later
backend is validated against it, and it is itself validated against JAX
in aerial/tests/test_npu_reference.py. It also serves as the CPU
fallback backend on the Orange Pi (needs only numpy).

Source references (keep in sync if upstream changes):
    dreamerv3/rssm.py       RSSM._observe, RSSM._core, Encoder.__call__
    embodied/jax/nets.py    Linear, BlockLinear, Norm(rms), MLP, symlog
    embodied/jax/heads.py   Head.bounded_normal
    embodied/jax/outs.py    Categorical(unimix), OneHot, Normal
"""
from pathlib import Path

import numpy as np

from aerial.deploy.npu.extract_weights import load_npz

f32 = np.float32

SAMPLING_MODES = ('stochastic', 'mean_action', 'deterministic')


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def sigmoid(x):
    # tanh form: no exp() overflow for large |x|, no masked indexing.
    return 0.5 * (1 + np.tanh(0.5 * x))


def silu(x):
    return x * sigmoid(x)


def gelu(x):
    # jax.nn.gelu default is the tanh approximation.
    return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi).astype(x.dtype) * (x + 0.044715 * x ** 3)))


ACTS = {'silu': silu, 'gelu': gelu, 'none': lambda x: x}


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


class ReferencePolicy:
    """Stateless forward pass. All arrays carry a leading batch dim B."""

    def __init__(self, weights, meta):
        self.w = {k: np.asarray(v, f32) for k, v in weights.items()}
        self.hp = meta
        self.deter = meta['deter']
        self.stoch = meta['stoch']
        self.classes = meta['classes']
        self.blocks = meta['blocks']
        self.eps = f32(meta['norm_eps'])
        self.unimix = meta['unimix']
        self.minstd, self.maxstd = meta['minstd'], meta['maxstd']
        for key in ('rssm_act', 'enc_act', 'pol_act'):
            if meta[key] not in ACTS:
                raise NotImplementedError(f'{key}={meta[key]!r}')

    @classmethod
    def load(cls, path):
        return cls(*load_npz(path))

    # --- primitives ------------------------------------------------------

    def _linear(self, name, x):
        return x @ self.w[f'{name}/kernel'] + self.w[f'{name}/bias']

    def _block_linear(self, name, x):
        kernel = self.w[f'{name}/kernel']  # (G, in/G, out/G)
        g, _, o = kernel.shape
        B = x.shape[0]
        # == einsum('bki,kio->bko'), as a BLAS-backed batched matmul.
        x = np.matmul(x.reshape(B, g, -1).transpose(1, 0, 2), kernel)
        return x.transpose(1, 0, 2).reshape(B, g * o) + self.w[f'{name}/bias']

    def _norm(self, name, x):
        mean2 = np.square(x).mean(-1, keepdims=True)
        return x * (1 / np.sqrt(mean2 + self.eps)) * self.w[f'{name}/scale']

    def _dense_block(self, lin, norm, act, x):
        return ACTS[act](self._norm(norm, self._linear(lin, x)))

    # --- graphs ----------------------------------------------------------

    def encode(self, obs_vector):
        x = symlog(obs_vector) if self.hp['enc_symlog'] else obs_vector
        for i in range(self.hp['enc_layers']):
            x = self._dense_block(
                    f'enc/mlp{i}', f'enc/mlp{i}norm', self.hp['enc_act'], x)
        return x

    def core(self, deter, stoch, prevact, obs_vector, is_first):
        """One posterior RSSM step.

        deter (B, D), stoch (B, S, C) one-hot, prevact (B, A), obs (B, O),
        is_first (B,) bool. Returns deter' (B, D), logits (B, S, C) (raw
        posterior logits, before unimix).
        """
        deter, stoch, prevact, obs_vector = (
                np.asarray(x, f32) for x in (deter, stoch, prevact, obs_vector))
        keep = ~np.asarray(is_first, bool)
        # RSSM._observe: zero the carry and previous action on episode start.
        deter = deter * keep[:, None]
        stoch = stoch * keep[:, None, None]
        prevact = prevact * keep[:, None]
        tokens = self.encode(obs_vector)
        deter = self._core(deter, stoch, prevact)
        act = self.hp['rssm_act']
        x = np.concatenate([deter, tokens], -1)
        for i in range(self.hp['obslayers']):
            x = self._dense_block(f'dyn/obs{i}', f'dyn/obs{i}norm', act, x)
        logits = self._linear('dyn/obslogit', x)
        return deter, logits.reshape(-1, self.stoch, self.classes)

    def _core(self, deter, stoch, action):
        B, g, act = deter.shape[0], self.blocks, self.hp['rssm_act']
        stoch = stoch.reshape(B, -1)
        action = action / np.maximum(1, np.abs(action))
        x0 = self._dense_block('dyn/dynin0', 'dyn/dynin0norm', act, deter)
        x1 = self._dense_block('dyn/dynin1', 'dyn/dynin1norm', act, stoch)
        x2 = self._dense_block('dyn/dynin2', 'dyn/dynin2norm', act, action)
        x = np.concatenate([x0, x1, x2], -1)
        x = np.repeat(x[:, None, :], g, 1)                          # (B, g, 3H)
        x = np.concatenate([deter.reshape(B, g, -1), x], -1).reshape(B, -1)
        for i in range(self.hp['dynlayers']):
            x = self._block_linear(f'dyn/dynhid{i}', x)
            x = ACTS[act](self._norm(f'dyn/dynhid{i}norm', x))
        x = self._block_linear('dyn/dyngru', x)                     # (B, 3D)
        # Gates are split per block, not globally (flat2group -> split -> flat).
        reset, cand, update = (
                part.reshape(B, -1)
                for part in np.split(x.reshape(B, g, -1), 3, -1))
        reset = sigmoid(reset)
        cand = np.tanh(reset * cand)
        update = sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def actor(self, deter, stoch):
        """Policy head. Returns (mean (B, A) already tanh'd, std (B, A))."""
        x = np.concatenate([deter, stoch.reshape(stoch.shape[0], -1)], -1)
        x = np.asarray(x, f32)
        for i in range(self.hp['pol_layers']):
            x = self._dense_block(
                    f'pol/mlp/linear{i}', f'pol/mlp/norm{i}', self.hp['pol_act'], x)
        mean = np.tanh(self._linear('pol/head/action/mean', x))
        std = self._linear('pol/head/action/stddev', x)
        std = (self.maxstd - self.minstd) * sigmoid(std + 2.0) + self.minstd
        return mean, std

    # --- sampling (CPU side, between the two graphs) -----------------------

    def latent_probs(self, logits):
        """outs.Categorical with unimix: mix softmax with a uniform."""
        probs = softmax(logits)
        if self.unimix:
            probs = (1 - self.unimix) * probs + self.unimix / probs.shape[-1]
        return probs

    def sample_latent(self, logits, rng):
        """Gumbel-max categorical sample -> one-hot (B, S, C)."""
        logp = np.log(self.latent_probs(logits))
        gumbel = -np.log(-np.log(rng.uniform(1e-20, 1.0, logp.shape)))
        return self._onehot(np.argmax(logp + gumbel, -1))

    def mode_latent(self, logits):
        return self._onehot(np.argmax(logits, -1))

    def _onehot(self, index):
        return np.eye(self.classes, dtype=f32)[index]


class NumpyPolicyInference:
    """Drop-in for aerial.deploy.inference.PolicyInference without JAX.

    sampling:
        'stochastic'    -- sample latent + Gaussian action. Matches JAX
                           `agent.policy(mode='train')`.
        'mean_action'   -- sample latent, act with the Gaussian mean.
                           Matches JAX `agent.policy(mode='eval')`.
        'deterministic' -- argmax latent + mean action. Fully repeatable.
    """

    def __init__(self, model_dir, sampling='mean_action', seed=None):
        if sampling not in SAMPLING_MODES:
            raise ValueError(f'sampling must be one of {SAMPLING_MODES}')
        path = Path(model_dir)
        if path.is_dir():
            path = path / 'policy_weights.npz'
        self.ref = ReferencePolicy.load(path)
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self.action_dim = self.ref.w['pol/head/action/mean/bias'].shape[0]
        self._carry = None

    def reset(self):
        hp = self.ref.hp
        self._carry = (
                np.zeros((1, hp['deter']), f32),
                np.zeros((1, hp['stoch'], hp['classes']), f32),
                np.zeros((1, self.action_dim), f32))

    def step(self, obs_vector, is_first=False):
        if self._carry is None:
            self.reset()
            is_first = True
        deter, stoch, prevact = self._carry
        obs = np.asarray(obs_vector, f32).reshape(1, -1)
        deter, logits = self.ref.core(
                deter, stoch, prevact, obs, np.array([bool(is_first)]))
        if self.sampling == 'deterministic':
            stoch = self.ref.mode_latent(logits)
        else:
            stoch = self.ref.sample_latent(logits, self.rng)
        mean, std = self.ref.actor(deter, stoch)
        if self.sampling == 'stochastic':
            action = mean + std * self.rng.standard_normal(mean.shape).astype(f32)
        else:
            action = mean
        action = action.astype(f32)
        self._carry = (deter, stoch, action)
        return action.reshape(-1)
