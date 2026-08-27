#!/usr/bin/env bash
# Phase 1 curriculum training entry point.
#
# Usage:
#   ./aerial/train/start_12in_curriculum.sh 1a               # kick off Phase 1a
#   ./aerial/train/start_12in_curriculum.sh 1b --run.from_checkpoint ...
#   ./aerial/train/start_12in_curriculum.sh 1c --run.from_checkpoint ...
#
# Stage picks the config block: aerial_12in_curriculum_<stage>.
# 1b and 1c should always be warm-started from the previous stage's
# checkpoint via --run.from_checkpoint.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
  echo "Usage: $0 <stage> [extra dreamerv3 flags...]" >&2
  echo "  stage: 1a | 1b | 1c" >&2
  exit 1
fi

STAGE="$1"; shift
case "${STAGE}" in
  1a|1b|1c) ;;
  *) echo "Unknown stage '${STAGE}'. Expected 1a, 1b, or 1c." >&2; exit 1 ;;
esac

VENV="${PWD}/.venv-pybullet"
if [ ! -d "${VENV}" ]; then
  echo "Missing ${VENV}." >&2
  exit 1
fi

PYTHONPATH="${PWD}:${PYTHONPATH:-}" \
  "${VENV}/bin/python" dreamerv3/main.py \
    --logdir ~/logdir/dreamer/aerial-12in-curriculum-${STAGE}-{timestamp} \
    --configs aerial_12in_curriculum_${STAGE} \
    "$@"
