#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv run python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs liftoff \
  --run.train_ratio 32 \
  "$@"
