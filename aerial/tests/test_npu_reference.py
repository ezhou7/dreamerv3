"""Parity tests: NumPy reference policy vs the JAX DreamerV3 agent.

Builds a size12m agent (the deployed aerial preset) on CPU with
perturbed random weights, then checks that aerial.deploy.npu.reference
reproduces the JAX forward pass:

  * per-step: deter, posterior logits, unimix probs, action mean/std
  * closed loop: 1000 steps with an independent NumPy carry (drift)
  * end to end: through the real agent.policy() + a bundle on disk

Run:  venv/bin/python -m pytest aerial/tests/test_npu_reference.py -q
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
from aerial.spec.interface import ACTION_DIM, OBS_DIM  # noqa: E402

PRESET = 'aerial_12in_skydreamer_1b'

# XLA vs NumPy float32 reduction order differs; errors are ~1e-6.
ATOL = 1e-4


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    return jax_agent.make_dummy_bundle(
        PRESET, tmp_path_factory.mktemp('bundle'), seed=0)


@pytest.fixture(scope='module')
def agent(bundle):
    return jax_agent.load_bundle_agent(bundle)


@pytest.fixture(scope='module')
def ref(bundle):
    out, _, _ = extract_weights.extract_bundle(bundle)
    return ReferencePolicy.load(out)


@pytest.fixture(scope='module')
def probe(agent):
    """Jitted JAX function returning the intermediates of one policy step,
    computed with the same module calls Agent.policy() makes."""
    model = agent.model

    def step(deter, stoch, prevact, vector, is_first):
        kw = dict(training=False, single=True)
        obs = {'vector': vector, 'is_first': is_first}
        _, _, tokens = model.enc({}, obs, is_first, **kw)
        _, _, feat = model.dyn.observe(
            {'deter': deter, 'stoch': stoch}, tokens,
            {'action': prevact}, is_first, **kw)
        mixed = model.dyn._dist(feat['logit']).output.dist.logits
        normal = model.pol(model.feat2tensor(feat), bdims=1)['action'].output
        return dict(deter=feat['deter'], logit=feat['logit'],
                    probs=jnp.exp(mixed), mean=normal.mean, std=normal.stddev)

    def actor(deter, stoch):
        normal = model.pol(
            model.feat2tensor({'deter': deter, 'stoch': stoch}),
            bdims=1)['action'].output
        return normal.mean, normal.stddev

    params = agent.params
    pure_step = jax.jit(lambda p, *a: nj.pure(step)(
        p, *a, seed=jnp.zeros(2, jnp.uint32), create=False, modify=False)[1])
    pure_actor = jax.jit(lambda p, *a: nj.pure(actor)(
        p, *a, create=False, modify=False)[1])
    # The embodied agent enables a strict transfer guard; the probe feeds
    # host numpy arrays on purpose.
    def call(fn, *args):
        with jax.transfer_guard('allow'):
            return jax.device_get(fn(params, *args))

    return (lambda *a: call(pure_step, *a),
            lambda *a: call(pure_actor, *a))


def random_obs(rng, batch=1):
    return (2.0 * rng.standard_normal((batch, OBS_DIM))).astype(np.float32)


def test_extract_keeps_only_policy_path(bundle, ref):
    keys = set(ref.w)
    assert any(k.startswith('enc/') for k in keys)
    assert any(k.startswith('dyn/dyngru') for k in keys)
    assert any(k.startswith('pol/head/action') for k in keys)
    assert not any(k.startswith(('dec/', 'rew/', 'con/', 'val/', 'slowval/'))
                   for k in keys)
    assert not any(k.startswith('dyn/prior') for k in keys)
    hp = ref.hp
    assert (hp['deter'], hp['hidden'], hp['stoch'], hp['classes']) == (
        2048, 256, 32, 16)


def test_single_step_matches_jax(ref, probe):
    step, _ = probe
    rng = np.random.default_rng(1)
    B = 4
    deter = np.tanh(rng.standard_normal((B, ref.deter))).astype(np.float32)
    idx = rng.integers(0, ref.classes, (B, ref.stoch))
    stoch = np.eye(ref.classes, dtype=np.float32)[idx]
    prevact = (1.5 * rng.standard_normal((B, ACTION_DIM))).astype(np.float32)
    obs = random_obs(rng, B)
    is_first = np.array([True, False, False, False])

    out = step(deter, stoch, prevact, obs, is_first)
    d, logits = ref.core(deter, stoch, prevact, obs, is_first)
    np.testing.assert_allclose(d, out['deter'], atol=ATOL)
    np.testing.assert_allclose(logits, out['logit'], atol=ATOL)
    np.testing.assert_allclose(ref.latent_probs(logits), out['probs'],
                               atol=ATOL)
    # Actor on the JAX-sampled latent isn't observable here, so check it
    # on our own inputs against the JAX actor separately (next test).
    assert np.isfinite(d).all() and np.isfinite(logits).all()


def test_actor_matches_jax(ref, probe):
    _, actor = probe
    rng = np.random.default_rng(2)
    B = 4
    deter = np.tanh(rng.standard_normal((B, ref.deter))).astype(np.float32)
    idx = rng.integers(0, ref.classes, (B, ref.stoch))
    stoch = np.eye(ref.classes, dtype=np.float32)[idx]
    mean_j, std_j = actor(deter, stoch)
    mean, std = ref.actor(deter, stoch)
    np.testing.assert_allclose(mean, mean_j, atol=ATOL)
    np.testing.assert_allclose(std, std_j, atol=ATOL)
    assert (std >= ref.minstd - 1e-6).all() and (std <= ref.maxstd + 1e-6).all()


def test_closed_loop_drift(ref, probe):
    """1000 steps where NumPy keeps its OWN deter state. Latent samples and
    action noise are shared so both sides see the same trajectory; any
    numerical drift in the recurrent state accumulates here."""
    step, actor = probe
    rng = np.random.default_rng(3)
    D, S, C = ref.deter, ref.stoch, ref.classes
    jd = nd = np.zeros((1, D), np.float32)
    stoch = np.zeros((1, S, C), np.float32)
    act = np.zeros((1, ACTION_DIM), np.float32)
    worst = 0.0
    flips = 0
    for t in range(1000):
        obs = random_obs(rng)
        first = np.array([t in (0, 500)])  # includes a mid-run reset
        out = step(jd, stoch, act, obs, first)
        nd, nlogits = ref.core(nd, stoch, act, obs, first)
        jd = out['deter']
        worst = max(worst, float(np.abs(nd - jd).max()))
        # Shared latent sample: drawn once from the JAX probs.
        logp = np.log(out['probs'])
        g = -np.log(-np.log(rng.uniform(1e-20, 1.0, logp.shape)))
        idx = np.argmax(logp + g, -1)
        flips += int((np.argmax(np.log(ref.latent_probs(nlogits)) + g, -1)
                      != idx).sum())
        stoch = np.eye(C, dtype=np.float32)[idx]
        mean_j, std_j = actor(jd, stoch)
        mean_n, std_n = ref.actor(nd, stoch)
        np.testing.assert_allclose(mean_n, mean_j, atol=1e-3)
        np.testing.assert_allclose(std_n, std_j, atol=1e-3)
        eps = rng.standard_normal(mean_j.shape).astype(np.float32)
        act = (mean_j + std_j * eps).astype(np.float32)
    print(f'\nclosed loop: max |deter| error {worst:.2e}, latent flips {flips}')
    assert worst < 1e-3, f'deter drifted to {worst:.2e}'
    # A latent flip happens only when two Gumbel-perturbed logits are within
    # ~1e-5 of each other; allow a handful over 32k draws.
    assert flips <= 5, flips


def test_against_agent_policy(agent, bundle):
    """End to end through the real agent.policy(): replay its carry into
    the NumPy core and check the produced deter state."""
    ref = ReferencePolicy.load(bundle / 'policy_weights.npz')
    rng = np.random.default_rng(4)
    carry = agent.init_policy(batch_size=1)
    for t in range(50):
        obs = {
            'vector': random_obs(rng),
            'is_first': np.array([t == 0]),
            'is_last': np.array([False]),
            'is_terminal': np.array([False]),
            'reward': np.zeros(1, np.float32),
        }
        _, dyn_c, _, prevact = carry
        new_carry, _, _ = agent.policy(carry, obs, mode='eval')
        d, _ = ref.core(np.asarray(dyn_c['deter']),
                        np.asarray(dyn_c['stoch']),
                        np.asarray(prevact['action']),
                        obs['vector'], obs['is_first'])
        np.testing.assert_allclose(d, np.asarray(new_carry[1]['deter']),
                                   atol=ATOL)
        stoch = np.asarray(new_carry[1]['stoch'])
        assert np.allclose(stoch.sum(-1), 1.0) and set(np.unique(stoch)) <= {0, 1}
        carry = new_carry


def test_eval_mode_acts_with_policy_mean(agent, bundle):
    """mode='eval' returns the Gaussian action mean for the sampled latent
    (same as NumpyPolicyInference(sampling='mean_action')); mode='train'
    samples around it."""
    ref = ReferencePolicy.load(bundle / 'policy_weights.npz')
    obs = {
        'vector': np.zeros((1, OBS_DIM), np.float32),
        'is_first': np.array([True]),
        'is_last': np.array([False]),
        'is_terminal': np.array([False]),
        'reward': np.zeros(1, np.float32),
    }
    for mode, expect_mean in [('eval', True), ('train', False)]:
        carry, act, _ = agent.policy(
            agent.init_policy(batch_size=1), obs, mode=mode)
        mean, _ = ref.actor(np.asarray(carry[1]['deter']),
                            np.asarray(carry[1]['stoch']))
        is_mean = np.allclose(np.asarray(act['action']), mean, atol=ATOL)
        assert is_mean == expect_mean, mode


@pytest.mark.parametrize('sampling', ['stochastic', 'mean_action',
                                      'deterministic'])
def test_numpy_inference_api(bundle, sampling):
    pol = NumpyPolicyInference(bundle, sampling=sampling, seed=0)
    pol.reset()
    rng = np.random.default_rng(5)
    acts = [pol.step(random_obs(rng)[0], is_first=(t == 0)) for t in range(20)]
    assert all(a.shape == (ACTION_DIM,) and a.dtype == np.float32 for a in acts)
    assert all(np.isfinite(a).all() for a in acts)
    if sampling == 'deterministic':
        pol2 = NumpyPolicyInference(bundle, sampling=sampling, seed=123)
        pol2.reset()
        rng = np.random.default_rng(5)
        acts2 = [pol2.step(random_obs(rng)[0], is_first=(t == 0))
                 for t in range(20)]
        np.testing.assert_array_equal(np.stack(acts), np.stack(acts2))
