"""Parity tests: ONNX core/actor graphs vs the NumPy reference policy.

Builds small random weights with the same layout as `policy_weights.npz`
(no JAX, no bundle), exports them with aerial.deploy.npu.export_onnx and
checks onnxruntime against ReferencePolicy:

  * core: deter', posterior logits (incl. a nonzero stoch and prevact)
  * actor: action mean, std
  * closed loop: 50 steps with the ONNX deter fed back

The RKNN step (convert_rknn.py, rknn_policy.py) needs the converter and
the NPU; check it on the Pi with `python -m aerial.deploy.npu.rknn_policy`.

Run:  python -m pytest aerial/tests/test_npu_onnx.py -q
"""
import numpy as np
import pytest

ort = pytest.importorskip('onnxruntime')
pytest.importorskip('onnx')

from aerial.deploy.npu import export_onnx  # noqa: E402
from aerial.deploy.npu.reference import ReferencePolicy, symlog  # noqa: E402

D, G, H, S, C, A, O = 32, 4, 8, 4, 4, 4, 22
ATOL = 1e-5

META = {
    'deter': D, 'hidden': H, 'stoch': S, 'classes': C, 'blocks': G,
    'dynlayers': 1, 'obslayers': 1, 'enc_layers': 3, 'pol_layers': 3,
    'enc_symlog': True, 'enc_act': 'silu', 'rssm_act': 'silu',
    'pol_act': 'silu', 'norm_eps': 1e-4, 'unimix': 0.01,
    'minstd': 0.1, 'maxstd': 1.0,
}


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    w = {}

    def lin(name, i, o):
        w[f'{name}/kernel'] = rng.normal(0, 1 / np.sqrt(i), (i, o))
        w[f'{name}/bias'] = rng.normal(0, 0.1, o)

    def norm(name, n):
        w[f'{name}/scale'] = 1 + rng.normal(0, 0.1, n)

    for i in range(3):
        lin(f'enc/mlp{i}', O if i == 0 else H, H)
        norm(f'enc/mlp{i}norm', H)
    for i, n in enumerate((D, S * C, A)):
        lin(f'dyn/dynin{i}', n, H)
        norm(f'dyn/dynin{i}norm', H)
    blk_in = D // G + 3 * H
    w['dyn/dynhid0/kernel'] = rng.normal(0, 1 / np.sqrt(blk_in), (G, blk_in, D // G))
    w['dyn/dynhid0/bias'] = rng.normal(0, 0.1, D)
    norm('dyn/dynhid0norm', D)
    w['dyn/dyngru/kernel'] = rng.normal(0, 1 / np.sqrt(D // G), (G, D // G, 3 * D // G))
    w['dyn/dyngru/bias'] = rng.normal(0, 0.1, 3 * D)
    lin('dyn/obs0', D + H, H)
    norm('dyn/obs0norm', H)
    lin('dyn/obslogit', H, S * C)
    for i in range(3):
        lin(f'pol/mlp/linear{i}', D + S * C if i == 0 else H, H)
        norm(f'pol/mlp/norm{i}', H)
    lin('pol/head/action/mean', H, A)
    lin('pol/head/action/stddev', H, A)
    return {k: v.astype(np.float32) for k, v in w.items()}


@pytest.fixture(scope='module')
def graphs():
    w = _weights()
    core = ort.InferenceSession(
        export_onnx.build_core(w, META).SerializeToString())
    actor = ort.InferenceSession(
        export_onnx.build_actor(w, META).SerializeToString())
    return ReferencePolicy(w, META), core, actor


def _run_core(core, deter, stoch, prevact, obs):
    return core.run(None, {
        'obs_symlog': symlog(obs).astype(np.float32),
        'deter': deter.astype(np.float32),
        'stoch': stoch.reshape(1, -1).astype(np.float32),
        'prevact': (prevact / np.maximum(1, np.abs(prevact))).astype(np.float32)})


def test_core_matches_reference(graphs):
    ref, core, _ = graphs
    rng = np.random.default_rng(1)
    deter = rng.normal(0, 0.5, (1, D)).astype(np.float32)
    stoch = ref.mode_latent(rng.normal(size=(1, S, C)))
    prevact = rng.uniform(-1.5, 1.5, (1, A)).astype(np.float32)  # incl. |a| > 1
    obs = rng.normal(0, 3, (1, O)).astype(np.float32)
    r_deter, r_logits = ref.core(deter, stoch, prevact, obs, np.array([False]))
    o_deter, o_logits = _run_core(core, deter, stoch, prevact, obs)
    np.testing.assert_allclose(o_deter, r_deter, atol=ATOL)
    np.testing.assert_allclose(o_logits, r_logits.reshape(1, -1), atol=ATOL)


def test_actor_matches_reference(graphs):
    ref, _, actor = graphs
    rng = np.random.default_rng(2)
    deter = rng.normal(0, 0.5, (1, D)).astype(np.float32)
    stoch = ref.mode_latent(rng.normal(size=(1, S, C)))
    r_mean, r_std = ref.actor(deter, stoch)
    o_mean, o_std = actor.run(None, {
        'deter': deter, 'stoch': stoch.reshape(1, -1)})
    np.testing.assert_allclose(o_mean, r_mean, atol=ATOL)
    np.testing.assert_allclose(o_std, r_std, atol=ATOL)
    assert (o_std >= META['minstd']).all() and (o_std <= META['maxstd']).all()


def test_closed_loop_drift(graphs):
    ref, core, actor = graphs
    rng = np.random.default_rng(3)
    r = (np.zeros((1, D), np.float32), np.zeros((1, S, C), np.float32),
         np.zeros((1, A), np.float32))
    o_deter = r[0]
    for _ in range(50):
        obs = rng.normal(size=(1, O)).astype(np.float32)
        r_deter, r_logits = ref.core(*r, obs, np.array([False]))
        o_deter, _ = _run_core(core, o_deter, r[1], r[2], obs)
        stoch = ref.mode_latent(r_logits)
        mean, _ = ref.actor(r_deter, stoch)
        r = (r_deter, stoch, mean)
    np.testing.assert_allclose(o_deter, r[0], atol=1e-4)
