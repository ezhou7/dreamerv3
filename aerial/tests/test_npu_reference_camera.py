"""Parity of the NumPy reference policy with JAX for a camera policy.

Builds a size12m agent with the camera tracking policy's inputs (a
128x128 RGB image, its depth and a 23-dim vector) on CPU with perturbed
random weights, and checks that aerial.deploy.npu.reference reproduces
each step: the encoder's tokens (image CNN + vector MLP), the posterior
RSSM step and the policy head. Also checks that weight files written
before the image encoder (format 1) still load.

Run:  venv/bin/python -m pytest aerial/tests/test_npu_reference_camera.py -q
"""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ninjax as nj  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aerial.deploy.npu import extract_weights, jax_agent  # noqa: E402
from aerial.deploy.npu.reference import (  # noqa: E402
    NumpyPolicyInference, ReferencePolicy)
from aerial.spec.interface import ACTION_DIM  # noqa: E402

PRESET = 'aerial_12in_skydreamer_1b'
RES, VEC = jax_agent.CAMERA_RES, jax_agent.CAMERA_VECTOR_DIM

# XLA vs NumPy float32 reduction order differs; the CNN adds a few
# thousand-term sums per token.
ATOL = 2e-4


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    return jax_agent.make_dummy_bundle(
        PRESET, tmp_path_factory.mktemp('bundle'), seed=0, camera=True)


@pytest.fixture(scope='module')
def agent(bundle):
    return jax_agent.load_bundle_agent(bundle, camera=True)


@pytest.fixture(scope='module')
def ref(bundle):
    out, _, _ = extract_weights.extract_bundle(bundle)
    return ReferencePolicy.load(out)


@pytest.fixture(scope='module')
def probe(agent):
    model = agent.model

    def step(deter, stoch, prevact, obs, is_first):
        kw = dict(training=False, single=True)
        obs = dict(obs, is_first=is_first)
        _, _, tokens = model.enc({}, obs, is_first, **kw)
        _, _, feat = model.dyn.observe(
            {'deter': deter, 'stoch': stoch}, tokens,
            {'action': prevact}, is_first, **kw)
        return dict(tokens=tokens, deter=feat['deter'], logit=feat['logit'])

    def actor(deter, stoch):
        normal = model.pol(model.feat2tensor(
            {'deter': deter, 'stoch': stoch}), bdims=1)['action'].output
        return normal.mean, normal.stddev

    pure_step = jax.jit(lambda p, *a: nj.pure(step)(
        p, *a, seed=jnp.zeros(2, jnp.uint32), create=False, modify=False)[1])
    pure_actor = jax.jit(lambda p, *a: nj.pure(actor)(
        p, *a, create=False, modify=False)[1])

    def call(fn, *args):
        with jax.transfer_guard('allow'):
            return jax.device_get(fn(agent.params, *args))
    return (lambda *a: call(pure_step, *a), lambda *a: call(pure_actor, *a))


def random_obs(rng, batch=1):
    """Smooth random images (blocks, like a scene) and a random vector."""
    def img(c):
        small = rng.integers(0, 256, (batch, 16, 16, c), dtype=np.uint8)
        return small.repeat(RES // 16, 1).repeat(RES // 16, 2)
    return {'vector': (2.0 * rng.standard_normal((batch, VEC))).astype(np.float32),
            'image': img(3), 'depth': img(1)}


def test_extract_keeps_the_image_encoder(ref):
    hp = ref.hp
    assert hp['format_version'] == 2
    assert hp['cnn_layers'] == len(hp['enc_mults'])
    assert hp['cnn_in_channels'] == 4 and hp['vector_dim'] == VEC
    assert any(k.startswith('enc/cnn0/') for k in ref.w)
    assert not any(k.startswith(('dec/', 'rew/', 'con/', 'val/', 'slowval/'))
                   for k in ref.w)


def test_steps_match_jax(ref, probe):
    step, actor = probe
    rng = np.random.default_rng(0)
    B, C = 2, ref.classes
    deter = np.zeros((B, ref.deter), np.float32)
    stoch = np.zeros((B, ref.stoch, C), np.float32)
    prevact = np.zeros((B, ACTION_DIM), np.float32)
    for t in range(6):
        obs = random_obs(rng, B)
        first = np.array([t == 0, t in (0, 3)])
        out = step(deter, stoch, prevact, obs, first)
        np.testing.assert_allclose(ref.encode(obs), out['tokens'], atol=ATOL)
        d, logits = ref.core(deter, stoch, prevact, obs, first)
        np.testing.assert_allclose(d, out['deter'], atol=ATOL)
        np.testing.assert_allclose(logits, out['logit'], atol=ATOL)
        stoch = ref.mode_latent(logits)            # same latent on both sides
        mean, std = ref.actor(d, stoch)
        mean_j, std_j = actor(d, stoch)
        np.testing.assert_allclose(mean, mean_j, atol=ATOL)
        np.testing.assert_allclose(std, std_j, atol=ATOL)
        deter = d
        prevact = np.clip(mean + 0.3 * rng.standard_normal(mean.shape),
                          -1, 1).astype(np.float32)


def test_inference_wrapper_takes_dict_observations(bundle):
    pol = NumpyPolicyInference(bundle / 'policy_weights.npz',
                               sampling='deterministic')
    rng = np.random.default_rng(1)
    for t in range(3):
        obs = {k: v[0] for k, v in random_obs(rng).items()}
        a = pol.step(obs, is_first=(t == 0))
        assert a.shape == (ACTION_DIM,) and np.isfinite(a).all()


def test_format_1_files_still_load(ref, tmp_path):
    """Vector-only weight files written before the image encoder."""
    vec_only = {k: v for k, v in ref.w.items() if not k.startswith('enc/cnn')}
    meta = {k: v for k, v in ref.hp.items()
            if not k.startswith(('enc_depth', 'enc_mults', 'enc_kernel',
                                 'enc_strided', 'enc_outer', 'enc_skip',
                                 'cnn_', 'vector_dim'))}
    meta['format_version'] = 1
    path = tmp_path / 'v1.npz'
    extract_weights.save_npz(path, vec_only, meta)
    w, hp = extract_weights.load_npz(path)
    assert hp['cnn_layers'] == 0 and set(w) == set(vec_only)
