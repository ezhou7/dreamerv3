#!/usr/bin/env bash
# Publish a bundled model to GitHub Releases so the remote Orange Pi
# (not on this LAN) can pull it via HTTPS.
#
# Prerequisites (one-time on the dev machine):
#   gh auth login    # authenticates the GitHub CLI
#
# Usage:
#   ./aerial/deploy/publish_release.sh <bundle_name> [--tag <tag>]
# Example:
#   ./aerial/deploy/publish_release.sh skydreamer-17m-1c-2026-09-05
#
# Creates a GitHub release named "model-<bundle_name>" (unless --tag is
# given), packages the bundle into a gzipped tar, uploads it, and
# uploads a .sha256 sidecar so the Orange Pi can verify the download.
# Prints the release URL at the end.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundle_name> [--tag <tag>]" >&2
  exit 1
fi

BUNDLE="$1"; shift
TAG="model-${BUNDLE}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUNDLE_DIR="$REPO_ROOT/aerial/deploy/models/$BUNDLE"
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "ERROR: bundle dir not found: $BUNDLE_DIR" >&2
  echo "export it first with:" >&2
  echo "  python aerial/deploy/export_model.py --logdir <path> --name $BUNDLE" >&2
  exit 2
fi
if [[ ! -f "$BUNDLE_DIR/agent.pkl" ]]; then
  echo "ERROR: no agent.pkl inside $BUNDLE_DIR" >&2
  exit 2
fi

# gh must be authenticated.
if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh not authenticated. run 'gh auth login' first." >&2
  exit 3
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
TARBALL="$STAGE_DIR/$BUNDLE.tar.gz"
SHA256_FILE="$STAGE_DIR/$BUNDLE.tar.gz.sha256"

echo "== packaging =="
echo "bundle:  $BUNDLE_DIR"
echo "tarball: $TARBALL"

# -C so entries in the tar are relative to the bundle dir.
tar -czf "$TARBALL" -C "$REPO_ROOT/aerial/deploy/models" "$BUNDLE"
TARBALL_BYTES=$(stat -c %s "$TARBALL")
TARBALL_MB=$(( (TARBALL_BYTES + 1024*1024 - 1) / (1024*1024) ))
echo "size:    ${TARBALL_MB} MB"

sha256sum "$TARBALL" | awk '{print $1}' > "$SHA256_FILE"
SHA=$(cat "$SHA256_FILE")
echo "sha256:  $SHA"

echo ""
echo "== publishing release =="
echo "tag: $TAG"

# Build release notes from meta.json if present.
NOTES_FILE="$STAGE_DIR/notes.md"
{
  echo "# Model bundle: \`$BUNDLE\`"
  echo ""
  echo "Deploy-ready DreamerV3 hover policy bundle. Pull with:"
  echo ""
  echo '```bash'
  echo "./aerial/deploy/pull_release.sh $BUNDLE"
  echo '```'
  echo ""
  echo "Or manually:"
  echo '```bash'
  REPO_SLUG=$(gh repo view --json nameWithOwner -q .nameWithOwner)
  BASE_URL="https://github.com/${REPO_SLUG}/releases/download/${TAG}"
  echo "curl -L -o ${BUNDLE}.tar.gz ${BASE_URL}/${BUNDLE}.tar.gz"
  echo "sha256sum -c <(echo \"$SHA  ${BUNDLE}.tar.gz\")"
  echo "tar -xzf ${BUNDLE}.tar.gz -C aerial/deploy/models/"
  echo '```'
  echo ""
  echo "**File:** \`${BUNDLE}.tar.gz\` (${TARBALL_MB} MB)"
  echo "**sha256:** \`$SHA\`"
  echo ""
  if [[ -f "$BUNDLE_DIR/meta.json" ]]; then
    echo "## Export metadata"
    echo ""
    echo '```json'
    cat "$BUNDLE_DIR/meta.json"
    echo '```'
  fi
} > "$NOTES_FILE"

# Create the release + upload the tarball and sha256. If the release
# already exists, use gh release upload instead.
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "release $TAG already exists; uploading new assets (--clobber)"
  gh release upload "$TAG" "$TARBALL" "$SHA256_FILE" --clobber
else
  gh release create "$TAG" \
    --title "Model: $BUNDLE" \
    --notes-file "$NOTES_FILE" \
    "$TARBALL" "$SHA256_FILE"
fi

echo ""
echo "release URL:"
gh release view "$TAG" --json url -q .url
