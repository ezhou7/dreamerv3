#!/usr/bin/env bash
# Push code + model bundles to an Orange Pi companion computer.
#
# Usage:
#   ./aerial/deploy/deploy.sh <ssh_target> [model_name]
# Examples:
#   ./aerial/deploy/deploy.sh pi@orangepi.local
#   ./aerial/deploy/deploy.sh pi@orangepi.local skydreamer-17m-1c
#
# Behavior:
#   - Rsyncs the minimum code needed to run inference (aerial/, dreamerv3/,
#     embodied/, ninjax/, pyproject.toml)
#   - Rsyncs the requested model bundle (or all bundles if none named)
#   - Preserves timestamps so unchanged files aren't re-copied
#   - Uses SSH multiplexing so the pipe is reused across rsync calls

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ssh_target> [model_name]" >&2
  exit 1
fi

SSH_TARGET="$1"
MODEL_NAME="${2:-}"
REMOTE_ROOT="${REMOTE_ROOT:-~/aerial-drone}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "local repo:    $REPO_ROOT"
echo "remote target: $SSH_TARGET:$REMOTE_ROOT"

# Ensure remote dir exists.
ssh "$SSH_TARGET" "mkdir -p $REMOTE_ROOT/{aerial,dreamerv3,embodied,ninjax,models}"

# Code sync. Exclude caches + local logdirs + venvs.
RSYNC_FLAGS=(-avz --delete
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='.venv'
  --exclude='.venv-*'
  --exclude='aerial/deploy/models'  # models handled below
  --exclude='PLAN.md' --exclude='RESULTS.md'  # gitignored private notes
  --exclude='*.log'
  --exclude='logdir'
)

echo ""
echo "-- syncing code --"
for DIR in aerial dreamerv3 embodied ninjax; do
  if [[ -d "$REPO_ROOT/$DIR" ]]; then
    rsync "${RSYNC_FLAGS[@]}" \
      "$REPO_ROOT/$DIR/" "$SSH_TARGET:$REMOTE_ROOT/$DIR/"
  fi
done

# Top-level project files needed for pip install -e.
for FILE in pyproject.toml setup.py setup.cfg README.md; do
  if [[ -f "$REPO_ROOT/$FILE" ]]; then
    rsync -avz "$REPO_ROOT/$FILE" "$SSH_TARGET:$REMOTE_ROOT/"
  fi
done

echo ""
echo "-- syncing models --"
MODELS_LOCAL="$REPO_ROOT/aerial/deploy/models"
if [[ -n "$MODEL_NAME" ]]; then
  if [[ ! -d "$MODELS_LOCAL/$MODEL_NAME" ]]; then
    echo "ERROR: model bundle not found: $MODELS_LOCAL/$MODEL_NAME" >&2
    echo "export it first with:" >&2
    echo "  python aerial/deploy/export_model.py --logdir <path> --name $MODEL_NAME" >&2
    exit 2
  fi
  rsync -avz --delete \
    "$MODELS_LOCAL/$MODEL_NAME/" \
    "$SSH_TARGET:$REMOTE_ROOT/models/$MODEL_NAME/"
else
  # Sync all bundles present.
  if [[ -d "$MODELS_LOCAL" ]] && [[ -n "$(ls -A "$MODELS_LOCAL" 2>/dev/null)" ]]; then
    rsync -avz --delete "$MODELS_LOCAL/" "$SSH_TARGET:$REMOTE_ROOT/models/"
  else
    echo "no local model bundles found in $MODELS_LOCAL (skipping)"
  fi
fi

echo ""
echo "deploy complete."
echo ""
echo "next on the Orange Pi:"
echo "  cd $REMOTE_ROOT"
echo "  # first-time only:  ./aerial/deploy/install_orangepi.sh"
echo "  ./.venv/bin/python -m aerial.deploy.smoke_test --model models/$MODEL_NAME"
