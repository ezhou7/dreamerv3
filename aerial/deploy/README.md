# aerial/deploy — companion-computer deployment

Push a trained DreamerV3 hover policy to an Orange Pi companion
computer and run it (inference only). Two workflows:

1. **Export + push a new model** (continuous deployment loop)
2. **Smoke-test a pushed model** on the Orange Pi

Nothing here trains, saves gradients, or connects to the flight
controller yet — that lives in `aerial/hardware/` (Phase 4+).

## Files

| File | Purpose |
|---|---|
| `inference.py` | `PolicyInference` class — stateful `.reset()` + `.step(obs)` wrapper over the trained checkpoint. Forces JAX onto CPU. |
| `export_model.py` | Bundle a training-run's checkpoint into `models/<name>/`. Strips `ref_pol/*` params for pre-cleanup checkpoints. |
| `smoke_test.py` | Load a bundle, run 500 random-obs steps, print latency + action stats. No drone hardware needed. |
| `deploy.sh` | rsync code + a named model bundle to the Orange Pi over SSH. **Same-LAN path.** |
| `publish_release.sh` | Tar + upload a bundle as a GitHub Release asset. **Cross-network path (dev side).** |
| `pull_release.sh` | Download + sha256-verify a bundle from a GitHub Release. **Cross-network path (Orange Pi side).** |
| `install_orangepi.sh` | One-shot Python env setup on the Orange Pi. |

## Model I/O reference

Trained policy expects **22-dim float32 observation** and returns
**4-dim float32 action in [-1, 1]** per step, at 50 Hz.

Obs layout (`aerial/spec/interface.py:58`):

| Index | Field | Frame / units |
|---|---|---|
| 0-2 | `pos - target` | world, meters |
| 3-5 | linear velocity | world, m/s |
| 6-14 | rotation matrix (body→world) | flattened row-major |
| 15-17 | gyro | body, rad/s |
| 18-21 | previous action | policy output, [-1, 1] |

Action layout (`aerial/spec/interface.py:28`):

| Index | Field | Physical scaling |
|---|---|---|
| 0 | body roll rate  | tanh × ±6 rad/s |
| 1 | body pitch rate | tanh × ±6 rad/s |
| 2 | body yaw rate   | tanh × ±6 rad/s |
| 3 | collective thrust | linear → [0, 1] normalized throttle |

These match ArduPilot `SET_ATTITUDE_TARGET` body-rate mode. **The policy
does NOT output motor PWMs.** ArduPilot's inner rate loop handles motor
mixing and ESC signaling downstream.

## Continuous deployment — two paths

### Path A: same-LAN (rsync over SSH)

Fastest when the Orange Pi is on the same network. About 2 seconds over gigabit.

On the dev machine:
```bash
# 1. Export the run's final checkpoint into a bundle.
python aerial/deploy/export_model.py \
  --logdir ~/logdir/dreamer/aerial-12in-skydreamer-1b-<ts> \
  --name skydreamer-1b-17m-<yyyymmdd>

# 2. Push code + this bundle to the Orange Pi.
./aerial/deploy/deploy.sh pi@orangepi.local skydreamer-1b-17m-<yyyymmdd>
```

### Path B: cross-network (GitHub Releases)

Use when the Orange Pi is remote — different network, or offline until you connect it up. About 105 MB tarball per bundle; expect a few seconds to a minute to transfer depending on the Pi's internet.

**One-time dev-side setup:**
```bash
gh auth login   # authenticates the GitHub CLI
```

**Publish a bundle from the dev machine:**
```bash
python aerial/deploy/export_model.py --logdir <path> --name <bundle>
./aerial/deploy/publish_release.sh <bundle>
```

This creates a release tagged `model-<bundle>` with:
- `<bundle>.tar.gz` — the packaged bundle (~105 MB)
- `<bundle>.tar.gz.sha256` — hash sidecar for integrity check

The release URL is printed at the end.

**Pull on the Orange Pi:**
```bash
cd ~/aerial-drone
git pull                            # picks up latest code
./aerial/deploy/pull_release.sh <bundle>
```

`pull_release.sh` uses `gh` if authenticated (works for private repos on ARM64) and falls back to `curl` for public releases. It verifies the sha256 before extracting.

### Orange Pi one-time setup (either path)

```bash
git clone https://github.com/<owner>/dreamerv3 ~/aerial-drone
cd ~/aerial-drone
./aerial/deploy/install_orangepi.sh
```

### Test the pulled model

```bash
./.venv/bin/python -m aerial.deploy.smoke_test \
  --model aerial/deploy/models/<bundle>
```

Expected output:
```
model loaded.
first action (zero obs):    [ 0.02 -0.01  0.00  0.71]
second action (zero obs):   [ 0.02 -0.01  0.00  0.71]
benchmarking 500 steps of random obs...
per-step latency (ms):
  mean 3.42  p50 3.35  p90 3.71  p99 4.20  max 5.11
  target 20.00 ms (50 Hz)  ->  OK
```

Numbers on the Blackwell dev machine are ~3-5 ms per step. Orange Pi
CPU is slower — expect ~15-30 ms depending on model. If p99 exceeds
20 ms, either drop to 25 Hz control (`STEP_RATE_HZ` in
`aerial/spec/interface.py`) or reimplement the forward pass in pure
NumPy (planned follow-up).

## What's NOT yet done

- **No MAVLink bridge.** The Orange Pi can produce actions but nothing
  wires them to ArduPilot yet. See `PLAN.md` Phase 4 SITL and Phase 5
  hardware for the wiring plan.
- **PyBullet on the Orange Pi.** `install_orangepi.sh` pip-installs
  pybullet because `aerial/envs/hover_12in_env.py` triggers the import
  at env-registration time. Trim this later — Orange Pi should not
  need any sim library.
- **No safety limits.** For tethered flight, wrap `policy.step()` in a
  clamp (max body rate 3 rad/s, thrust ceiling 0.5, watchdog on
  step-rate slippage). Don't fly with raw policy output.

## Safety before real flight

The current best model is 42% success at Stage 1c. That's a 58% crash
rate on random init conditions in sim — DO NOT fly untethered until:

1. A policy passes 90%+ sim eval AND passes SITL validation (Phase 4).
2. Airframe params in `aerial/configs/airframe_12in.py` are replaced
   with real-hardware measurements (mass, arm length, thrust curve,
   motor time constant).
3. Retraining includes the measured airframe.

The current bundle is for **inference-latency and integration testing
on the Orange Pi** — running the code and confirming the pipeline works,
not for flying.
