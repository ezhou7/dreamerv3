"""Phase 3: convert core.onnx / actor.onnx to RK3588 .rknn models (fp16).

Needs `rknn-toolkit2` (Linux x86_64 / aarch64 only). The toolkit version
is recorded in `<bundle>/rknn_manifest.json`; the Pi's librknnrt.so must
be the same version or newer (see NPU_PLAN.md, Phase 3).

Usage:
    python -m aerial.deploy.npu.convert_rknn --model aerial/deploy/models/<bundle>
"""
import argparse
import hashlib
import json
from pathlib import Path

GRAPHS = ('core', 'actor')


def convert(model_dir, target='rk3588', verbose=False):
    from rknn.api import RKNN
    from importlib.metadata import version
    model_dir = Path(model_dir)
    manifest = {'target_platform': target, 'dtype': 'float16',
                'rknn_toolkit2': version('rknn-toolkit2'), 'files': {}}
    for name in GRAPHS:
        onnx_path, out = model_dir / f'{name}.onnx', model_dir / f'{name}.rknn'
        rknn = RKNN(verbose=verbose)
        rknn.config(target_platform=target)
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise SystemExit(f'load_onnx failed: {onnx_path}')
        if rknn.build(do_quantization=False) != 0:
            raise SystemExit(f'build failed: {onnx_path}')
        if rknn.export_rknn(str(out)) != 0:
            raise SystemExit(f'export_rknn failed: {out}')
        rknn.release()
        manifest['files'][out.name] = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f'wrote {out}')
    (model_dir / 'rknn_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', required=True, help='bundle dir with core.onnx/actor.onnx')
    parser.add_argument('--target', default='rk3588')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    convert(args.model, args.target, args.verbose)


if __name__ == '__main__':
    main()
