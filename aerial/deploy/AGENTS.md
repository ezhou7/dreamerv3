# AGENTS.md — `aerial/deploy/` reference

Guidance for AI agents working in this directory. Human-facing usage
docs live in [`README.md`](README.md); this file explains the design,
invariants, and pitfalls so an agent can extend or debug the pipeline
without stepping on the same rakes twice.

---

## Purpose

Ship a trained DreamerV3 hover policy from the training machine to a
remote Orange Pi companion computer, run inference-only there at 50 Hz,
and iterate on that loop as new models are trained. **Nothing here
trains, gradient-updates, or connects to a flight controller.** MAVLink
and safety-envelope work belongs in a future `aerial/hardware/` module
(see `PLAN.md` Phase 4).

Two delivery paths coexist:

- **Same-LAN:** rsync over SSH (`deploy.sh`) — fast, ~2 seconds over
  gigabit.
- **Cross-network:** GitHub Releases (`publish_release.sh` on the dev
  machine, `pull_release.sh` on the Orange Pi) — needed when the two
  machines are on different networks or the Pi is intermittently
  online. Bundle sha256 is verified before extraction.

---

## File map

| File | Runs on | Purpose | Depends on |
|---|---|---|---|
| `inference.py` | Both | `PolicyInference` class: stateful `.reset()` + `.step(obs)` wrapper. Forces `JAX_PLATFORMS=cpu` before importing JAX so the Orange Pi has no CUDA dependency. | `dreamerv3.main.make_agent`, `dreamerv3.main.make_env`, `aerial.envs.hover_12in_env`, `aerial.spec.interface` |
| `export_model.py` | Dev | Bundle a training-run's `ckpt/latest` into `models/<name>/`. Strips `ref_pol/*` params (pre-cleanup legacy), sanitizes `/home/<user>/` paths in `meta.json` and `config.yaml`, filters out obsolete config keys from removed experiments. | none beyond the checkpoint files |
| `smoke_test.py` | Both | Loads a bundle, feeds random obs, measures per-step latency vs the 50 Hz budget. Passes if `p99 < 20 ms`. No drone hardware required. | `inference.py`, `numpy` |
| `deploy.sh` | Dev → Pi | rsync code (aerial/, dreamerv3/, embodied/, ninjax/) + one named model bundle to the Orange Pi over SSH. LAN path. | `rsync`, ssh access |
| `publish_release.sh` | Dev | Tar-gz a bundle, sha256 it, create/refresh a GitHub Release with both assets. Uses `gh api` (works with default `repo` scope) — NOT `gh release create` (needs `workflow` scope). | `gh` authenticated, `tar`, `sha256sum` |
| `pull_release.sh` | Pi | Download release assets, verify sha256, extract into `aerial/deploy/models/`. Uses `gh` when authenticated, `curl` otherwise. | `gh` OR `curl`, `sha256sum` |
| `install_orangepi.sh` | Pi | One-shot Python env setup on Orange Pi (apt deps, venv, pip install jax[cpu] + friends). | Debian-based ARM64 Linux, Python 3.11+ |
| `README.md` | — | Human-facing usage instructions. | — |
| `AGENTS.md` | — | This file. | — |

`models/` is gitignored — see [Invariants](#invariants).

---

## End-to-end workflow

### One-time Orange Pi bootstrap

```bash
# On the Orange Pi
git clone https://github.com/ezhou7/dreamerv3 ~/aerial-drone
cd ~/aerial-drone
./aerial/deploy/install_orangepi.sh
# (optional, only if pulling from a private repo)
gh auth login
```

### After every training run — publish loop

On the dev machine:

```bash
# 1. Bundle the checkpoint.
python aerial/deploy/export_model.py \
  --logdir ~/logdir/dreamer/aerial-12in-<recipe>-<ts> \
  --name <recipe>-<steps>-<stage>-<yyyy-mm-dd>

# 2. Publish to GitHub Releases (cross-network path).
./aerial/deploy/publish_release.sh <recipe>-<steps>-<stage>-<yyyy-mm-dd>

#    Or, if the Pi is on the same LAN:
./aerial/deploy/deploy.sh pi@orangepi.local <recipe>-<steps>-<stage>-<yyyy-mm-dd>
```

On the Orange Pi:

```bash
cd ~/aerial-drone
git pull                                      # picks up code updates
./aerial/deploy/pull_release.sh <bundle>      # for the Releases path
./.venv/bin/python -m aerial.deploy.smoke_test \
    --model aerial/deploy/models/<bundle>
```

### Expected smoke_test.py output

```
model loaded.
first action (zero obs):    [x.xx, x.xx, x.xx, x.xx]
second action (zero obs):   [y.yy, y.yy, y.yy, y.yy]     # different — RSSM state advanced
benchmarking 500 steps of random obs...
per-step latency (ms):
  mean 15.x  p50 12.x  p90 20.x  p99 25.x  max 30.x
  target 20.00 ms (50 Hz)  ->  OK  or TOO SLOW
```

**Dev machine reference:** mean ~3 ms, p99 ~9 ms on Ryzen. Orange Pi is
5-10× slower; expect p99 25-80 ms. If TOO SLOW, options are:
1. Drop `STEP_RATE_HZ` in `aerial/spec/interface.py` to 25 Hz.
2. Reimplement the forward pass in pure NumPy (planned; ~500 lines).

---

## I/O contract (canonical, don't drift)

Defined in `aerial/spec/interface.py`. **Anything that touches the
policy must respect this schema exactly** — the trained weights depend
on these dimensions and layouts.

### Observation (22 floats)

| Index | Field | Frame / units |
|---|---|---|
| 0-2 | `pos - target` | world, meters |
| 3-5 | linear velocity | world, m/s |
| 6-14 | rotation matrix (body→world) | flattened row-major |
| 15-17 | gyro | body, rad/s |
| 18-21 | previous action | policy output, [-1, 1] |

### Action (4 floats, [-1, 1])

| Index | Field | Physical scaling (applied downstream) |
|---|---|---|
| 0 | body roll rate | `× ±6 rad/s` |
| 1 | body pitch rate | `× ±6 rad/s` |
| 2 | body yaw rate | `× ±6 rad/s` |
| 3 | collective thrust | linear map to [0, 1] normalized throttle |

Values may exceed [-1, 1] because the policy head is Gaussian — clip
downstream at the ArduPilot bridge or in a safety layer, not here.

Control rate: **50 Hz** (`STEP_RATE_HZ`). Matches ArduPilot's MAVLink
offboard command ceiling; deployment sends `SET_ATTITUDE_TARGET` in
body-rate mode. **Policy output is NOT motor PWMs** — ArduPilot's inner
rate loop handles mixing.

---

## Invariants

Break any of these and something silently corrupts.

1. **`models/` is gitignored.** Bundles are 100+ MB binaries; they
   belong in Releases, not git. `.gitignore` has `aerial/deploy/models/`.

2. **All paths in exported bundles are `~/`-prefixed, never absolute.**
   `export_model.py:sanitize_path()` handles this. Public release
   tarballs must not leak the trainer's Linux username.

3. **JAX is forced onto CPU in `inference.py` before `import jax`.**
   Setting `JAX_PLATFORMS=cpu` mid-import doesn't work — it must be an
   env var at Python startup.

4. **The bundle contains exactly:** `agent.pkl`, `config.yaml`,
   `meta.json`, `step.pkl`, `done`, `replay.pkl`. Never commit any of
   these to git.

5. **Release tag format is `model-<bundle-name>`.** Both scripts assume
   this. Bundle names follow `<recipe>-<steps>-<stage>-<yyyy-mm-dd>`
   e.g. `skydreamer-17m-1c-2026-09-05`.

6. **Sha256 verification is mandatory on pull.** `pull_release.sh`
   downloads a `.tar.gz.sha256` sidecar alongside the tarball, verifies,
   and refuses to extract on mismatch.

7. **`gh api` uses the default `repo` scope; `gh release create` does
   not.** Always use `gh api repos/OWNER/REPO/releases ...` for release
   creation. Otherwise `gh auth refresh -s workflow` is required and
   the script fails on default installs.

8. **The trained agent expects the exact obs/action schema above.**
   Changing dims, order, or units silently breaks a loaded policy.

---

## Common pitfalls (empirically hit)

### 1. Loading a checkpoint from before commit `9eacbb0` fails with `assert_trees_all_equal_shapes`

Pre-cleanup checkpoints have `ref_pol/*` parameter keys that no longer
exist in the current `agent.py`. `export_model.py:strip_ref_pol()`
handles this — it strips those keys during bundling. Never load such a
checkpoint directly through the eval script without going through
`export_model.py` first.

### 2. Config schema drift breaks agent construction at load time

A training run's saved `config.yaml` may contain keys that no longer
exist in the current code:
- `env.aerial.dr_stage_end` and `env.aerial.dr_ramp_steps` — from the
  removed continuous-curriculum experiment
- `agent.imag_loss.bc_coef` — from the removed BC-anchor experiment

`export_model.py:force_cpu_config()` filters these out. When the current
code removes another config key, update the `_ALLOWED` sets in that
function.

### 3. `gh` push over HTTPS fails with "could not read Username"

`gh auth login` may pick SSH protocol; git remote may be HTTPS.
`gh auth setup-git` configures git's credential helper to reuse the
`gh` token for HTTPS. Run it once per machine.

### 4. `set -e` trips on `if RELEASE_JSON=$(gh api ...); then`

When the release doesn't exist, `gh api ... /releases/tags/$TAG`
returns 404 → non-zero exit. In some bash versions `set -e` fires from
the command substitution before `if` gets a chance to branch. Use
`gh api ... || true` and check for empty output instead. Fixed in
`publish_release.sh` — pattern to follow elsewhere.

### 5. Deterministic-mode eval vs stochastic-mode training

`eval_quantitative.py` uses `mode='eval'` (policy distribution mean);
training rollouts use `mode='train'` (sample). If training rewards
climb but eval regresses, suspect this divergence — especially at low
`actent`. Not a `deploy/` concern directly, but the smoke_test also
uses `mode='eval'` (via inference.py), so it will show the same
signature.

### 6. PyBullet is imported at env-registration time

`aerial/envs/hover_12in_env.py` triggers a pybullet import even when
we only want inference. `install_orangepi.sh` currently pip-installs
pybullet on the Pi as a workaround. Eventual cleanup: gate the pybullet
import behind an env-registration flag so real hardware can run without
the sim library.

---

## Where NOT to touch

### Don't touch these from `aerial/deploy/`:

- **`aerial/spec/interface.py`** — the obs/action schema. Changing dims
  or order breaks every trained policy. If you must extend it, versioning
  the bundle format is the right move, not patching in place.
- **`dreamerv3/agent.py`** — the model class. Adding params here (like
  the removed `ref_pol`) creates checkpoint-compat headaches downstream.
- **`aerial/configs/domain_randomization.py`** — training-time only.
  Deployment doesn't care about DR ranges.

### Don't create these:

- MAVLink client / ArduPilot bridge — belongs in `aerial/hardware/`
  (Phase 4+, not yet started).
- Safety clamps (rate/thrust ceilings, watchdogs) — same, hardware
  concern.
- Training tooling — this is inference-only.

---

## What's still missing (as of 2026-09-14)

Deploy scaffolding is complete but doesn't yet reach a spinning motor.
Missing pieces in likely order:

1. **MAVLink bridge** (`aerial/hardware/ardupilot_bridge.py`) — subscribe
   to `LOCAL_POSITION_NED`, `ATTITUDE_QUATERNION`, `HIGHRES_IMU`; pack
   the 22-dim obs; send `SET_ATTITUDE_TARGET` at 50 Hz.
2. **Safety layer** — clamp body rates to a conservative envelope
   (3 rad/s), thrust ≤ 0.5, watchdog on step-rate slippage. Never fly
   raw policy output.
3. **Real hardware measurements** — replace placeholders in
   `aerial/configs/airframe_12in.py` (mass, arm length, thrust curve,
   motor τ), retrain with those, then export the improved bundle.
4. **SITL validation** (Phase 4) — real ArduPilot firmware in
   simulation via MAVLink, catches encoding/rate/EKF issues before
   real flight.

Only after all four does real-drone flight enter the picture. Every
model bundle in Releases today is for **pipeline integration testing**
— running inference on the Pi, measuring latency, verifying the
end-to-end deploy loop — not for flying.

---

## Quick command reference

```bash
# --- Dev machine ---

# Export a bundle
python aerial/deploy/export_model.py \
  --logdir ~/logdir/dreamer/<run-name> \
  --name <bundle-name>

# Publish to GitHub Releases (cross-network)
./aerial/deploy/publish_release.sh <bundle-name>

# rsync to a same-LAN Pi
./aerial/deploy/deploy.sh <ssh-target> <bundle-name>

# --- Orange Pi ---

# One-time setup
./aerial/deploy/install_orangepi.sh

# Pull a bundle from GitHub Releases
./aerial/deploy/pull_release.sh <bundle-name>

# Smoke test
./.venv/bin/python -m aerial.deploy.smoke_test \
  --model aerial/deploy/models/<bundle-name>

# --- Both ---

# Programmatic inference (in Python)
from aerial.deploy.inference import PolicyInference
policy = PolicyInference('aerial/deploy/models/<bundle-name>')
policy.reset()
for step in range(500):
    obs = get_current_22_dim_obs()  # see aerial/spec/interface.py
    action = policy.step(obs, is_first=(step == 0))
    send_to_flight_controller(action)  # not implemented yet
```
