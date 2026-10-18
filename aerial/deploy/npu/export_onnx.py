"""Phase 2: build the NPU-shaped ONNX graphs from `policy_weights.npz`.

Two static-shape, batch-1 graphs that mirror `ReferencePolicy` exactly
(see aerial/deploy/NPU_PLAN.md, Phase 2):

    core:  (obs_symlog 1x22, deter 1x2048, stoch 1x512, prevact 1x4)
               -> (deter_out 1x2048, logits 1x512)
    actor: (deter 1x2048, stoch 1x512) -> (mean 1x4, std 1x4)

CPU glue stays in numpy (`rknn_policy.py`): symlog, reset masking,
`act / max(1, |act|)`, unimix + sampling. `prevact` is fed already
normalized.

Graphs are written with `onnx.helper` straight from the npz -- no torch
or JAX needed. BlockLinear is a batched MatMul (G,1,in/G) @ (G,in/G,out/G),
which is a plain reshape at batch 1 (no transpose, no block-diagonal).

Usage:
    python -m aerial.deploy.npu.export_onnx --model aerial/deploy/models/<bundle>
    # writes <bundle>/core.onnx and <bundle>/actor.onnx, checks them
    # against ReferencePolicy with onnxruntime if it is installed.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from aerial.deploy.npu.extract_weights import load_npz

OPSET = 13
f32 = np.float32


class _Graph:
    """Tiny ONNX builder: every op returns the name of its output."""

    def __init__(self, weights):
        self.w = weights
        self.nodes, self.inits, self._n = [], [], 0

    def _name(self, prefix):
        self._n += 1
        return f'{prefix}_{self._n}'

    def const(self, value, name=None):
        name = name or self._name('c')
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, op_type, *inputs, **attrs):
        out = self._name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def ops(self, op_type, inputs, n_out, **attrs):
        outs = [self._name(op_type.lower()) for _ in range(n_out)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, **attrs))
        return outs

    def reshape(self, x, shape):
        return self.op('Reshape', x, self.const(np.array(shape, np.int64)))

    def weight(self, key, shape=None):
        value = self.w[key].astype(f32)
        if shape is not None:
            value = value.reshape(shape)
        return self.const(value, key.replace('/', '.'))

    # --- layers (see ReferencePolicy for the numerical source of truth) ---

    def linear(self, name, x):
        y = self.op('MatMul', x, self.weight(f'{name}/kernel'))
        return self.op('Add', y, self.weight(f'{name}/bias'))

    def block_linear(self, name, x, groups):
        """x (1, G*i) -> (1, G*o); == einsum('bki,kio->bko') at batch 1."""
        kernel = self.w[f'{name}/kernel']
        g, _, o = kernel.shape
        assert g == groups, (name, g, groups)
        y = self.op('MatMul', self.reshape(x, [g, 1, -1]), self.weight(f'{name}/kernel'))
        y = self.op('Add', y, self.weight(f'{name}/bias', (g, 1, o)))
        return self.reshape(y, [1, g * o])

    def norm(self, name, x, eps):
        mean2 = self.op('ReduceMean', self.op('Mul', x, x), axes=[-1], keepdims=1)
        # Div, not Mul(Reciprocal): RKNN runs Reciprocal on the CPU.
        rms = self.op('Sqrt', self.op('Add', mean2, self.const(f32(eps))))
        return self.op('Mul', self.op('Div', x, rms), self.weight(f'{name}/scale'))

    def act(self, kind, x):
        if kind == 'silu':
            return self.op('Mul', x, self.op('Sigmoid', x))
        if kind == 'none':
            return x
        raise NotImplementedError(kind)  # gelu: not used by aerial recipes

    def dense_block(self, lin, norm, act, x, eps):
        return self.act(act, self.norm(norm, self.linear(lin, x), eps))

    def build(self, name, inputs, outputs):
        graph = helper.make_graph(
                self.nodes, name,
                [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
                [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
                self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model

    def output(self, x, name):
        self.nodes.append(helper.make_node('Identity', [x], [name]))


def build_core(weights, hp):
    g = _Graph(weights)
    D, S, C, G = hp['deter'], hp['stoch'], hp['classes'], hp['blocks']
    A = weights['pol/head/action/mean/bias'].shape[0]
    O = weights['enc/mlp0/kernel'].shape[0]
    eps, act = hp['norm_eps'], hp['rssm_act']

    # Encoder (obs arrives already symlog'd).
    tokens = 'obs_symlog'
    for i in range(hp['enc_layers']):
        tokens = g.dense_block(f'enc/mlp{i}', f'enc/mlp{i}norm', hp['enc_act'], tokens, eps)

    # RSSM._core
    x0 = g.dense_block('dyn/dynin0', 'dyn/dynin0norm', act, 'deter', eps)
    x1 = g.dense_block('dyn/dynin1', 'dyn/dynin1norm', act, 'stoch', eps)
    x2 = g.dense_block('dyn/dynin2', 'dyn/dynin2norm', act, 'prevact', eps)
    x = g.op('Concat', x0, x1, x2, axis=-1)                         # (1, 3H)
    # Concat of G copies, not Tile: RKNN runs Tile (as Expand) on the CPU.
    x = g.reshape(x, [1, 1, -1])
    x = g.op('Concat', *[x] * G, axis=0)                            # (G, 1, 3H)
    x = g.op('Concat', g.reshape('deter', [G, 1, -1]), x, axis=-1)  # (G, 1, D/G+3H)
    x = g.reshape(x, [1, -1])
    for i in range(hp['dynlayers']):
        x = g.block_linear(f'dyn/dynhid{i}', x, G)
        x = g.act(act, g.norm(f'dyn/dynhid{i}norm', x, eps))
    x = g.block_linear('dyn/dyngru', x, G)                          # (1, 3D)
    # Gates are split per block, not globally.
    parts = g.ops('Split', [g.reshape(x, [G, 1, -1])], 3, axis=-1)
    reset, cand, update = (g.reshape(p, [1, D]) for p in parts)
    reset = g.op('Sigmoid', reset)
    cand = g.op('Tanh', g.op('Mul', reset, cand))
    update = g.op('Sigmoid', g.op('Sub', update, g.const(f32(1))))
    # update * cand + (1 - update) * deter == deter + update * (cand - deter)
    deter = g.op('Add', 'deter', g.op('Mul', update, g.op('Sub', cand, 'deter')))
    g.output(deter, 'deter_out')

    # Posterior logits.
    x = g.op('Concat', deter, tokens, axis=-1)
    for i in range(hp['obslayers']):
        x = g.dense_block(f'dyn/obs{i}', f'dyn/obs{i}norm', act, x, eps)
    g.output(g.linear('dyn/obslogit', x), 'logits')

    return g.build('core',
                   [('obs_symlog', [1, O]), ('deter', [1, D]),
                    ('stoch', [1, S * C]), ('prevact', [1, A])],
                   [('deter_out', [1, D]), ('logits', [1, S * C])])


def build_actor(weights, hp):
    g = _Graph(weights)
    D, S, C = hp['deter'], hp['stoch'], hp['classes']
    A = weights['pol/head/action/mean/bias'].shape[0]
    eps = hp['norm_eps']
    x = g.op('Concat', 'deter', 'stoch', axis=-1)
    for i in range(hp['pol_layers']):
        x = g.dense_block(f'pol/mlp/linear{i}', f'pol/mlp/norm{i}', hp['pol_act'], x, eps)
    g.output(g.op('Tanh', g.linear('pol/head/action/mean', x)), 'mean')
    std = g.op('Sigmoid', g.op('Add', g.linear('pol/head/action/stddev', x), g.const(f32(2))))
    std = g.op('Add', g.op('Mul', std, g.const(f32(hp['maxstd'] - hp['minstd']))),
               g.const(f32(hp['minstd'])))
    g.output(std, 'std')
    return g.build('actor', [('deter', [1, D]), ('stoch', [1, S * C])],
                   [('mean', [1, A]), ('std', [1, A])])


def export(model_dir):
    model_dir = Path(model_dir)
    weights, hp = load_npz(model_dir / 'policy_weights.npz')
    paths = {}
    for name, build in (('core', build_core), ('actor', build_actor)):
        paths[name] = model_dir / f'{name}.onnx'
        onnx.save(build(weights, hp), str(paths[name]))
        print(f'wrote {paths[name]}')
    return paths


def check(model_dir, steps=200, seed=0):
    """Closed-loop parity of the ONNX graphs vs ReferencePolicy (fp32)."""
    import onnxruntime as ort
    from aerial.deploy.npu.reference import ReferencePolicy, symlog
    model_dir = Path(model_dir)
    ref = ReferencePolicy.load(model_dir / 'policy_weights.npz')
    core = ort.InferenceSession(str(model_dir / 'core.onnx'))
    actor = ort.InferenceSession(str(model_dir / 'actor.onnx'))
    rng = np.random.default_rng(seed)
    hp = ref.hp
    deter = np.zeros((1, hp['deter']), f32)
    stoch = np.zeros((1, hp['stoch'], hp['classes']), f32)
    act = np.zeros((1, 4), f32)
    err = {'deter': 0.0, 'logits': 0.0, 'mean': 0.0, 'std': 0.0}
    for _ in range(steps):
        obs = rng.standard_normal((1, 22)).astype(f32)
        r_deter, r_logits = ref.core(deter, stoch, act, obs, np.array([False]))
        o_deter, o_logits = core.run(None, {
                'obs_symlog': symlog(obs).astype(f32), 'deter': deter,
                'stoch': stoch.reshape(1, -1),
                'prevact': (act / np.maximum(1, np.abs(act))).astype(f32)})
        stoch = ref.mode_latent(r_logits)
        r_mean, r_std = ref.actor(r_deter, stoch)
        o_mean, o_std = actor.run(None, {'deter': r_deter, 'stoch': stoch.reshape(1, -1)})
        for k, a, b in (('deter', r_deter, o_deter), ('logits', r_logits.reshape(1, -1), o_logits),
                        ('mean', r_mean, o_mean), ('std', r_std, o_std)):
            err[k] = max(err[k], float(np.abs(a - b).max()))
        deter, act = r_deter, r_mean
    print('onnxruntime vs reference, max abs err over', steps, 'steps:',
          ', '.join(f'{k} {v:.2e}' for k, v in err.items()))
    return err


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', required=True, help='bundle dir with policy_weights.npz')
    parser.add_argument('--no-check', action='store_true')
    args = parser.parse_args()
    export(args.model)
    if not args.no_check:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print('onnxruntime not installed; skipping parity check')
            return
        err = check(args.model)
        if max(err.values()) > 1e-4:
            raise SystemExit('ONNX parity check FAILED')


if __name__ == '__main__':
    main()
