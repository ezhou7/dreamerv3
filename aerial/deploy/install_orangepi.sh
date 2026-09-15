#!/usr/bin/env bash
# One-time setup on the Orange Pi companion computer.
# Assumes: Debian-based ARM64 Linux, Python 3.11 or 3.12 available.
# Run this from the deployment root (default ~/aerial-drone) AFTER
# deploy.sh has copied the code.

set -euo pipefail

echo "== aerial-drone Orange Pi install =="
echo "python: $(python3 --version)"
echo "arch:   $(uname -m)"

# System packages needed for JAX CPU backend + numpy build.
if command -v apt-get >/dev/null 2>&1; then
  echo ""
  echo "-- apt packages --"
  sudo apt-get update
  sudo apt-get install -y \
    python3-venv python3-dev build-essential \
    libopenblas-dev
fi

echo ""
echo "-- virtualenv --"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
./.venv/bin/pip install --upgrade pip wheel setuptools

echo ""
echo "-- python packages --"
# Minimal inference-only set. No pybullet on the Orange Pi -- the
# aerial env is imported for its space registration only; we make that
# import optional in inference.py in a follow-up.
./.venv/bin/pip install \
  "jax[cpu]" \
  "numpy" \
  "ruamel.yaml" \
  "chex" \
  "optax" \
  "flax" \
  "elements" \
  "embodied" \
  "gymnasium" \
  "pybullet"   # still needed for env registration; strip later

echo ""
echo "-- verify --"
./.venv/bin/python -c "import jax; print('jax devices:', jax.devices())"
./.venv/bin/python -c "import numpy, ruamel.yaml, gymnasium; print('imports OK')"

echo ""
echo "install complete. Next step:"
echo "  ./.venv/bin/python -m aerial.deploy.smoke_test --model models/<name>"
