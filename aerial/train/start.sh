#!/usr/bin/env bash
# Launch DreamerV3 training on the PyBullet hover environment.
#
# This script is the Phase 0 entry point. It uses the .venv-pybullet
# virtual environment (which has gym-pybullet-drones installed)
# instead of the main .venv. The training itself runs on a CUDA GPU
# selected by jax.cuda_visible_devices in the `aerial_pybullet` config.
#
# Pass extra dreamerv3 flags after this script's args, e.g.:
#   ./aerial/train/start.sh --run.train_ratio 64
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV="${PWD}/.venv-pybullet"
if [ ! -d "${VENV}" ]; then
  echo "Missing ${VENV}. Run: uv venv --python 3.12 .venv-pybullet" >&2
  exit 1
fi

# .venv-pybullet has no project install, so we point Python at the repo
# root so `import dreamerv3`, `import embodied`, `import aerial` all
# resolve.
PYTHONPATH="${PWD}:${PYTHONPATH:-}" \
  "${VENV}/bin/python" dreamerv3/main.py \
    --logdir ~/logdir/dreamer/aerial-pybullet-{timestamp} \
    --configs aerial_pybullet \
    "$@"
