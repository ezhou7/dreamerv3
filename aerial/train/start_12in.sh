#!/usr/bin/env bash
# Phase 1 training: 12" airframe hover with domain randomization.
#
# Config: dreamerv3/configs.yaml -> aerial_12in_pybullet
# Env:    aerial/envs/hover_12in_env.py (Aerial-hover-12in-v0)
# Logs:   ~/logdir/dreamer/aerial-12in-<timestamp>/
#
# Extra flags pass through:
#   ./aerial/train/start_12in.sh --run.train_ratio 64
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV="${PWD}/.venv-pybullet"
if [ ! -d "${VENV}" ]; then
  echo "Missing ${VENV}. Run: uv venv --python 3.12 .venv-pybullet" >&2
  exit 1
fi

PYTHONPATH="${PWD}:${PYTHONPATH:-}" \
  "${VENV}/bin/python" dreamerv3/main.py \
    --logdir ~/logdir/dreamer/aerial-12in-{timestamp} \
    --configs aerial_12in_pybullet \
    "$@"
