#!/usr/bin/env bash
# Load a trained DreamerV3 checkpoint and run the policy in PyBullet
# with the GUI on, so you can watch the drone fly.
#
# Pick checkpoint:
#   ./aerial/train/eval_visual.sh                    # latest training run
#   ./aerial/train/eval_visual.sh /path/to/logdir    # specific logdir
#
# Press Ctrl-C in this terminal to stop. Closing the PyBullet window
# will also exit the process.
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV="${PWD}/.venv-pybullet"
if [ ! -d "${VENV}" ]; then
  echo "Missing ${VENV}." >&2
  exit 1
fi

# Find a checkpoint to load.
if [ $# -ge 1 ]; then
  TRAIN_LOGDIR="$1"
else
  TRAIN_LOGDIR=$(ls -dt ~/logdir/dreamer/aerial-pybullet-* 2>/dev/null \
    | grep -v -- '-eval-' | head -1)
fi

if [ -z "${TRAIN_LOGDIR:-}" ] || [ ! -d "${TRAIN_LOGDIR}" ]; then
  echo "No training logdir found. Run training first or pass one as arg." >&2
  exit 1
fi

CKPT_DIR="${TRAIN_LOGDIR}/ckpt"
LATEST_FILE="${CKPT_DIR}/latest"
if [ -f "${LATEST_FILE}" ]; then
  # `latest` is a text file with the name of the most recent checkpoint
  # subdir, per elements.Checkpoint's storage format.
  CKPT="${CKPT_DIR}/$(cat "${LATEST_FILE}")"
else
  # Fall back to newest subdir directly (e.g. if `latest` wasn't written
  # yet because save_every hasn't elapsed).
  CKPT=$(ls -dt "${CKPT_DIR}"/*/ 2>/dev/null | head -1)
fi
if [ -z "${CKPT}" ] || [ ! -d "${CKPT}" ]; then
  echo "No checkpoint directory in ${CKPT_DIR}." >&2
  exit 1
fi

EVAL_LOGDIR="${TRAIN_LOGDIR}-eval-$(date +%Y%m%dT%H%M%S)"
echo "Training logdir : ${TRAIN_LOGDIR}"
echo "Loading ckpt    : ${CKPT}"
echo "Eval logdir     : ${EVAL_LOGDIR}"
echo

PYTHONPATH="${PWD}:${PYTHONPATH:-}" \
  "${VENV}/bin/python" dreamerv3/main.py \
    --logdir "${EVAL_LOGDIR}" \
    --configs aerial_pybullet_eval \
    --run.from_checkpoint "${CKPT}" \
    "${@:2}"
