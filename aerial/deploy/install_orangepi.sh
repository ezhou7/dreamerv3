#!/usr/bin/env bash
# One-time setup on the Orange Pi companion computer.
# Assumes: Debian-based ARM64 Linux, Python 3.10+.
# Run this from the deployment root (e.g. ~/aerial-drone) AFTER
# deploy.sh has copied the code. Code runs from the repo root; nothing
# here pip-installs the repo itself.
#
# Usage:
#   ./aerial/deploy/install_orangepi.sh          # numpy backend only
#   ./aerial/deploy/install_orangepi.sh --jax    # + JAX baseline backend
#   ./aerial/deploy/install_orangepi.sh --npu    # + RKNN converter/runtime
#
# Default installs only what aerial.deploy.npu (numpy reference policy +
# bench) needs. --jax adds the JAX stack used by
# `aerial.deploy.npu.bench --backends jax`, which builds the agent
# without the PyBullet env. The sim stack (pybullet, gym_pybullet_drones)
# is intentionally NOT installed: the old inference.py / smoke_test.py
# path still imports it, the npu path does not.
#
# --npu adds rknn-toolkit2 (ONNX -> .rknn converter; pulls in torch) and
# rknn-toolkit-lite2 (on-device runtime), and downloads the matching
# librknnrt.so into .venv/rknnrt/ -- /usr/lib is left alone. Run the NPU
# backend with LIBRKNNRT_PATH=$PWD/.venv/rknnrt/librknnrt.so.

set -euo pipefail

WITH_JAX=0
WITH_NPU=0
# Converter, lite runtime and librknnrt.so must all be the same version.
RKNN_VERSION=2.3.2
for arg in "$@"; do
  case "$arg" in
    --jax) WITH_JAX=1 ;;
    --npu) WITH_NPU=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "== aerial-drone Orange Pi install =="
echo "python: $(python3 --version)"
echo "arch:   $(uname -m)"

if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  echo ""
  echo "-- apt packages (python3-venv missing) --"
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-dev build-essential
fi

echo ""
echo "-- virtualenv --"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
./.venv/bin/pip install --upgrade pip wheel setuptools

echo ""
echo "-- python packages (numpy backend) --"
# numpy<2 matches training (pyproject.toml); agent.pkl holds numpy arrays.
./.venv/bin/pip install "numpy<2" "ruamel.yaml"

if [[ "$WITH_JAX" == 1 ]]; then
  echo ""
  echo "-- python packages (JAX backend) --"
  # Do NOT pip install `embodied`: the repo ships its own embodied/, and
  # the PyPI sdist is broken (setup.py reads a missing requirements.txt).
  # jax pinned to the training version (pyproject.toml). Repeat the numpy
  # pin: without it pip upgrades numpy to 2.x to satisfy newer deps.
  ./.venv/bin/pip install \
    "numpy<2" \
    "jax[cpu]==0.4.38" \
    "chex>=0.1.90" \
    "optax>=0.2.5" \
    "elements>=3.19.1" \
    "ninjax>=3.5.1" \
    "portal>=3.5.0" \
    einops \
    cloudpickle \
    colored_traceback
fi

if [[ "$WITH_NPU" == 1 ]]; then
  echo ""
  echo "-- python packages (RKNN NPU backend) --"
  ./.venv/bin/pip install "numpy<2" \
    "rknn-toolkit2==$RKNN_VERSION" "rknn-toolkit-lite2==$RKNN_VERSION"
  mkdir -p .venv/rknnrt
  curl -L --fail -o .venv/rknnrt/librknnrt.so \
    "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v$RKNN_VERSION/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
fi

echo ""
echo "-- verify --"
./.venv/bin/python -c "import numpy, ruamel.yaml; print('numpy', numpy.__version__)"
./.venv/bin/python -c "import aerial.deploy.npu.reference; print('numpy backend OK')"
if [[ "$WITH_JAX" == 1 ]]; then
  ./.venv/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
  ./.venv/bin/python -c "import aerial.deploy.npu.jax_agent, dreamerv3.agent; print('jax backend OK')"
fi
if [[ "$WITH_NPU" == 1 ]]; then
  ./.venv/bin/python -c "import rknn.api, rknnlite.api; print('rknn toolkit + lite OK')"
  strings .venv/rknnrt/librknnrt.so | grep 'librknnrt version' || true
fi

echo ""
echo "install complete. Next step:"
echo "  ./.venv/bin/python -m aerial.deploy.npu.bench --model models/<name> --backends numpy"
