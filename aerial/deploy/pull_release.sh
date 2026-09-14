#!/usr/bin/env bash
# Pull a model bundle from GitHub Releases onto the Orange Pi.
# Public releases: no auth needed. Private releases: run 'gh auth login'
# first (works on ARM64 via the official gh package).
#
# Usage:
#   ./aerial/deploy/pull_release.sh <bundle_name> [--tag <tag>]
# Example:
#   ./aerial/deploy/pull_release.sh skydreamer-17m-1c-2026-09-05
#
# Downloads the release's <bundle>.tar.gz, verifies sha256 against
# the accompanying .sha256 file, and extracts into aerial/deploy/models/.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundle_name> [--tag <tag>] [--repo owner/name]" >&2
  exit 1
fi

BUNDLE="$1"; shift
TAG="model-${BUNDLE}"
REPO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)  TAG="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="$REPO_ROOT/aerial/deploy/models"
mkdir -p "$MODELS_DIR"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "== pulling model bundle =="
echo "tag: $TAG"

# Prefer 'gh' if available and authenticated -- it handles private repos
# and rate limits better. Fall back to curl for public releases.
USE_GH=0
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  USE_GH=1
fi

if [[ $USE_GH -eq 1 ]]; then
  ARGS=(release download "$TAG"
    --pattern "$BUNDLE.tar.gz"
    --pattern "$BUNDLE.tar.gz.sha256"
    --dir "$STAGE_DIR")
  [[ -n "$REPO" ]] && ARGS+=(--repo "$REPO")
  gh "${ARGS[@]}"
else
  # Guess repo from local git remote if not given.
  if [[ -z "$REPO" ]]; then
    if git remote get-url origin >/dev/null 2>&1; then
      URL=$(git remote get-url origin)
      # Handles both https and ssh remotes.
      REPO=$(echo "$URL" | sed -E 's|.*github.com[:/]([^/]+/[^/.]+)(\.git)?|\1|')
    fi
  fi
  if [[ -z "$REPO" ]]; then
    echo "ERROR: pass --repo <owner>/<name> or authenticate 'gh' first." >&2
    exit 3
  fi
  BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
  echo "downloading from $BASE_URL"
  curl -L --fail -o "$STAGE_DIR/$BUNDLE.tar.gz"       "$BASE_URL/$BUNDLE.tar.gz"
  curl -L --fail -o "$STAGE_DIR/$BUNDLE.tar.gz.sha256" "$BASE_URL/$BUNDLE.tar.gz.sha256"
fi

echo ""
echo "== verifying sha256 =="
EXPECTED=$(cat "$STAGE_DIR/$BUNDLE.tar.gz.sha256" | awk '{print $1}')
ACTUAL=$(sha256sum "$STAGE_DIR/$BUNDLE.tar.gz" | awk '{print $1}')
if [[ "$EXPECTED" != "$ACTUAL" ]]; then
  echo "ERROR: sha256 mismatch." >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 4
fi
echo "sha256 OK: ${ACTUAL:0:16}..."

echo ""
echo "== extracting =="
DST="$MODELS_DIR/$BUNDLE"
if [[ -d "$DST" ]]; then
  echo "removing existing bundle at $DST"
  rm -rf "$DST"
fi
tar -xzf "$STAGE_DIR/$BUNDLE.tar.gz" -C "$MODELS_DIR"

echo ""
echo "installed bundle at: $DST"
ls -la "$DST"
echo ""
echo "next step:"
echo "  ./.venv/bin/python -m aerial.deploy.smoke_test --model models/$BUNDLE"
