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

# Create the release + upload the tarball and sha256 via the REST API.
# We use `gh api` rather than `gh release create` because the latter
# requires the "workflow" scope while the former works with the default
# "repo" scope from `gh auth login`.
REPO_SLUG=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# Check whether the release already exists; if so, reuse its id and
# clobber the assets.
if RELEASE_JSON=$(gh api "repos/$REPO_SLUG/releases/tags/$TAG" 2>/dev/null); then
  RELEASE_ID=$(printf '%s' "$RELEASE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
  echo "release $TAG already exists (id=$RELEASE_ID); replacing assets"
  # Delete existing assets with the same names so re-upload doesn't 422.
  for NAME in "$BUNDLE.tar.gz" "$BUNDLE.tar.gz.sha256"; do
    ASSET_ID=$(printf '%s' "$RELEASE_JSON" | python3 -c "
import sys,json
d = json.load(sys.stdin)
for a in d.get('assets', []):
    if a['name'] == '$NAME':
        print(a['id']); break
")
    if [[ -n "$ASSET_ID" ]]; then
      gh api "repos/$REPO_SLUG/releases/assets/$ASSET_ID" -X DELETE
    fi
  done
else
  BODY=$(cat "$NOTES_FILE")
  RELEASE_JSON=$(gh api "repos/$REPO_SLUG/releases" -X POST \
    -f "tag_name=$TAG" \
    -f "name=Model: $BUNDLE" \
    -f "body=$BODY" \
    -F "draft=false" \
    -F "prerelease=false")
  RELEASE_ID=$(printf '%s' "$RELEASE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
  echo "created release id=$RELEASE_ID"
fi

echo ""
echo "-- uploading $BUNDLE.tar.gz --"
gh api "https://uploads.github.com/repos/$REPO_SLUG/releases/$RELEASE_ID/assets?name=$BUNDLE.tar.gz" \
  -X POST -H 'Content-Type: application/gzip' --input "$TARBALL" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(' ', d.get('name'), d.get('size'), 'bytes')"

echo "-- uploading $BUNDLE.tar.gz.sha256 --"
gh api "https://uploads.github.com/repos/$REPO_SLUG/releases/$RELEASE_ID/assets?name=$BUNDLE.tar.gz.sha256" \
  -X POST -H 'Content-Type: text/plain' --input "$SHA256_FILE" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(' ', d.get('name'), d.get('size'), 'bytes')"

echo ""
echo "release URL:"
printf '  https://github.com/%s/releases/tag/%s\n' "$REPO_SLUG" "$TAG"
