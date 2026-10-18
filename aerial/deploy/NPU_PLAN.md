# Orange Pi 5 (RK3588) NPU adapter — plan

Goal: run the deployed DreamerV3 hover policy on the RK3588 NPU (via
`rknn-toolkit-lite2`) with the same `reset()` / `step(obs, is_first)`
interface as `aerial/deploy/inference.py:PolicyInference`, without JAX on
the companion computer.

Status (2026-10-19): **Phases 0–4 done; the policy runs on the NPU**
(trained bundle `skydreamer-f450-4.4m-1c-2026-10-17`). Decision unchanged:
the numpy CPU backend stays the default. At batch 1 the NPU is
dispatch-bound and *slower* than numpy on an A76 core (see Phase 4
results). Phases 5–7 not started.

---

## Background: what actually runs at deploy time

`Agent.policy()` (`dreamerv3/agent.py`) runs encoder → one RSSM posterior
step → policy head. The decoder is skipped (`Decoder.initial()` returns
`{}`), and reward/continue/value heads and the RSSM prior are unused.

For the `size12m` preset used by every aerial recipe:

| Stage | Ops | Params |
|---|---|---|
| CPU prep | `symlog(obs22)`, reset masking, `act / max(1, \|act\|)` | – |
| Encoder | 3 × [Linear 256 → RMSNorm → SiLU] | 0.14M |
| `_core` inputs | Linear(deter 2048→256), Linear(stoch 512→256), Linear(act 4→256), each Norm+SiLU | 0.66M |
| `_core` blocks | BlockLinear 8×(1024→256) → Norm → SiLU; BlockLinear 8×(256→768) → GRU gates (per block split) | 3.67M |
| Posterior | Linear(2304→256) → Norm → SiLU → Linear(256→512) → 32×16 logits | 0.72M |
| Latent sample | unimix(1%) softmax → categorical → one-hot | – |
| Policy head | 3 × [Linear 256, Norm, SiLU] on 2560-d feat → `tanh(mean)`, `0.9·σ(s+2)+0.1` → Gaussian sample | 0.79M |

≈ 6.0M MACs/step, 22.9 MB fp32 weights, batch 1. Recurrent state:
deter (2048), stoch (32×16 one-hot), previous action (4).

### Findings that shape the design

1. **`mode='eval'` is ignored** — `Agent.policy` never reads `mode`, so
   the deployed policy *samples* both the latent and the action
   (`aerial/tests/test_npu_reference.py::test_eval_mode_is_stochastic`).
   `aerial/deploy/AGENTS.md` pitfall 5 assumes otherwise. The adapter
   exposes `sampling = stochastic | mean_action | deterministic`; pick by
   closed-loop sim eval (Phase 5), not by assumption.
2. **The old JAX deploy install was broken** — it pip-installed the
   PyPI `embodied` sdist (fails to build; the repo ships its own) and
   missed `ninjax`, `einops`, `portal`, `cloudpickle`,
   `colored_traceback`. Fixed on 2026-10-11: `install_orangepi.sh` now
   installs the numpy backend by default and the JAX stack with `--jax`
   (verified on the Pi). `inference.py` / `smoke_test.py` still import
   the PyBullet env (`gym_pybullet_drones`), which is deliberately not
   installed on the Pi; the NumPy/RKNN path needs none of it.
3. **`rknn-toolkit2` (the converter) is Linux-only** (x86_64 / aarch64).
   Conversion runs on the training box or on the Pi, not macOS.
4. **Multithreaded BLAS wrecks tail latency for batch-1 GEMVs** — on the
   dev box OpenBLAS with 10 threads gave p99 16 ms vs 0.99 ms with 1
   thread. Pin `OPENBLAS_NUM_THREADS=1` for any CPU backend.

---

## Phase 0 — measure, then decide  ✅

Tool: `python -m aerial.deploy.npu.bench --model <bundle>`

Dev box (Apple M-series, arm64, numpy 1.26, dummy size12m bundle,
2000 steps, BLAS 1 thread):

| Backend | mean | p50 | p99 | max |
|---|---|---|---|---|
| numpy reference | 0.83 ms | 0.83 ms | 0.99 ms | 1.16 ms |
| JAX CPU (current path) | 1.33 ms | 1.29 ms | 1.64 ms | 8.67 ms |

Orange Pi 5 (2026-10-11; RK3588, cpu0-3 Cortex-A55 1.8 GHz, cpu4-7
Cortex-A76 2.4 GHz, `ondemand` governor; Python 3.10.12, numpy 1.26.4,
jax 0.4.38; dummy size12m bundle, 2000 steps, BLAS 1 thread). Load =
one shell busy-loop per listed core. Raw JSON:
`npu/bench_results/2026-10-11-orangepi5/`.

| # | Setup | Backend | mean | p50 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | idle, unpinned | numpy | 2.40 | 2.40 | 2.53 | 2.61 |
| 1 | idle, unpinned | JAX | 6.08 | 6.09 | 6.40 | 8.22 |
| 2 | idle, pinned A76 (cpu4) | numpy | 2.42 | 2.42 | 2.49 | 2.63 |
| 3 | idle, pinned A55 (cpu0) | numpy | 11.39 | 11.35 | 11.88 | 13.26 |
| 4 | load on all 8 cores, unpinned | numpy | 5.54 | 2.78 | **19.50** | **29.49** |
| 5 | load on 7 cores, pinned to isolated cpu4 | numpy | 3.02 | 3.03 | 3.19 | 3.66 |
| 5 | load on 7 cores, pinned to isolated cpu4 | JAX | 6.65 | 6.70 | 7.24 | 9.11 |

(ms; budget 20 ms at 50 Hz.) JAX load time 29–54 s vs 0.1 s for numpy.
SoC temperature reached 84 °C after the load runs — near RK3588's
throttling point, so the deployed board needs a heatsink + fan.

**Decision.**
- The numpy backend meets the budget with ~8× headroom (p99 2.5 ms) and
  is 2.5× faster than the current JAX path. **Ship it as the default
  deploy backend; the NPU is not needed for latency at size12m.**
- **Core placement is the real risk**, not compute: an A55 core is 4.7×
  slower, and unpinned under full contention the max step (29.5 ms)
  broke the budget. The control loop must run pinned to an isolated
  A76 core (`taskset -c 4`, later `isolcpus=4` + `SCHED_FIFO`), with
  MAVLink/telemetry/logging on other cores.
- Phases 2–5 continue as an optimization: free the A76 cores for the
  MAVLink bridge + safety layer, headroom for size25m/50m, and NPU CNN
  encoders if camera observations are added. Re-measure on a trained
  bundle (shapes are identical, so numbers should carry over).

Reproduce on the Pi:

```bash
cd ~/Documents/dev/aerial-drone
./aerial/deploy/install_orangepi.sh            # numpy backend; --jax for JAX
taskset -c 4 ./.venv/bin/python -m aerial.deploy.npu.bench \
    --model models/<bundle> --backends numpy --steps 2000 --json bench.json
# No trained bundle? On the dev box:
#   python -m aerial.deploy.npu.jax_agent --out aerial/deploy/models/dummy-size12m
#   REMOTE_ROOT='~/Documents/dev/aerial-drone' ./aerial/deploy/deploy.sh orangepi5 dummy-size12m
```

## Phase 1 — framework-free reference  ✅

| File | Purpose |
|---|---|
| `npu/extract_weights.py` | `agent.pkl` + `config.yaml` → `policy_weights.npz` (41 tensors, 5.99M params, 22.9 MB vs 114.7 MB `agent.pkl`). Only `enc/`, `dyn/{dynin,dynhid,dyngru,obs,obslogit}`, `pol/`. Hyperparams stored as JSON `__meta__`; refuses configs it can't reproduce (non-rms norm, image encoder, `absolute`, non-`bounded_normal`). |
| `npu/reference.py` | `ReferencePolicy` — `core()` (deter′, posterior logits) and `actor()` (mean, std) as two pure functions, sampling between them on CPU. `NumpyPolicyInference` — drop-in for `PolicyInference`, numpy only. |
| `npu/jax_agent.py` | Builds the JAX agent without `make_env`/PyBullet; `JaxPolicy`; dummy-bundle generator (random weights, perturbed so biases/norm scales are non-trivial). |
| `npu/bench.py` | Phase 0 latency benchmark. |
| `tests/test_npu_reference.py` | Parity suite vs JAX (below). |

Parity results (size12m, perturbed weights, float32):

- single step (B=4, incl. one reset row): deter, logits, unimix probs,
  action mean/std all within 1e-4
- closed loop, 1000 steps with an independent NumPy deter carry and a
  mid-run reset: **max |deter| error 2.4e-7, 0 latent flips**
- end to end through the real `agent.policy()` + on-disk bundle: match
- mutation check: breaking the per-block gate split or the `update − 1`
  gate bias fails 3 tests each

```bash
PYTHONPATH=$PWD venv/bin/python -m pytest aerial/tests/test_npu_reference.py -q
```

Not yet validated against a **trained** checkpoint (none on the dev
box). First thing to do with a real bundle:
`python -m aerial.deploy.npu.extract_weights --model <bundle>`, then run
the parity suite pointed at it (or `bench` with both backends).

---

## Running on the NPU (Phases 2–4)  ✅

```bash
./aerial/deploy/install_orangepi.sh --npu   # toolkit + lite 2.3.2, .venv/rknnrt/librknnrt.so
export LIBRKNNRT_PATH=$PWD/.venv/rknnrt/librknnrt.so PYTHONPATH=$PWD
M=aerial/deploy/models/<bundle>
./aerial/deploy/pull_release.sh <bundle>
.venv/bin/python -m aerial.deploy.npu.extract_weights --model $M  # policy_weights.npz
.venv/bin/python -m aerial.deploy.npu.export_onnx --model $M      # core/actor.onnx + ORT parity
.venv/bin/python -m aerial.deploy.npu.convert_rknn --model $M     # core/actor.rknn (fp16)
.venv/bin/python -m aerial.deploy.npu.rknn_policy --model $M      # NPU vs reference check
taskset -c 4 .venv/bin/python -m aerial.deploy.npu.bench --model $M --backends rknn,numpy
.venv/bin/python -m aerial.deploy.npu.run_policy --model $M --mode sim --backend rknn
```

| File | Purpose |
|---|---|
| `npu/export_onnx.py` | Phase 2. `policy_weights.npz` → `core.onnx` (obs_symlog, deter, stoch, normalized prevact → deter′, logits) and `actor.onnx` (deter′, stoch′ → mean, std), built with `onnx.helper` (no torch). BlockLinear = batched MatMul (G,1,i)@(G,i,o). Checks against `ReferencePolicy` with onnxruntime (max err 1.6e-5). |
| `npu/convert_rknn.py` | Phase 3. ONNX → `.rknn`, `target_platform=rk3588`, fp16, no quantization; writes `rknn_manifest.json` (toolkit version + sha256s). |
| `npu/rknn_policy.py` | Phase 4. `RKNNPolicyInference` (`RKNNLite`, `NPU_CORE_0`), same API and `sampling` modes (default `mean_action`) as `NumpyPolicyInference`; `python -m` runs the step-level + drift check. `bench.py --backends rknn`. |
| `npu/run_policy.py` | Prints a bundle's actions in the terminal. `--mode probe`: hand-built states. `--mode sim`: closed loop in a NumPy rigid-body model using `airframe_12in`, `ArduPilotRateController` (400 Hz) and the Stage 1c init/latency ranges. `--backend numpy\|rknn`. |
| `tests/test_npu_onnx.py` | ONNX graphs vs `ReferencePolicy` on small random weights (no JAX). |
| `tests/test_run_policy_sim.py` | Sim sign checks: hover trim, +pitch = nose-down (→ +x), +roll = right bank (→ −y). |

**Runtime version.** The board image ships `/usr/lib/librknnrt.so`
2.0.0b0 (kernel driver 0.8.8); the oldest aarch64 `rknn-toolkit2`
wheel is 2.3.0, so models are built with 2.3.2 and loaded with a 2.3.2
`librknnrt.so` from airockchip. rknnlite hardcodes the `/usr/lib` path
(and ignores `LD_PRELOAD`); `rknn_policy.py` overrides it when
`LIBRKNNRT_PATH` is set. The 2.0.0b0 runtime also runs the models, with a
version-mismatch warning.

**Graph shaping.** `Reciprocal` and `Tile` are placed on the CPU by
RKNN (as `Div` / `Expand`), splitting the graph; the exporter uses
`Div(x, Sqrt(·))` and a `Concat` of G copies instead, so every compute
op runs on the NPU. Per-op profile: `RKNN_LOG_LEVEL=4`.

**Accuracy** (trained bundle, fp16, deterministic sampling, 3000
steps of random obs): step-level max error deter 4.7e-3, logits 5.4e-2,
action mean 2.0e-3, std 2.3e-3. Closed-loop deterministic trajectories
diverge (deter Δ ≈ 1, ~2% latent flips), but the fp32 reference diverges
identically from itself under 1e-5 injected deter noise, so this is the
trajectory's sensitivity to tiny perturbations, not an fp16 defect.
The real gate is still closed-loop sim eval (Phase 5).

**Closed loop** (`run_policy.py --mode sim`, `mean_action`, 30 random
Stage 1c starts, 8 s, no wind/noise/DR): **30/30 on numpy and 30/30 on
rknn**, no crashes, median final error 0.13 / 0.12 m. Not a substitute
for the PyBullet eval (PR #7: 29/30), but it shows the NPU policy flies
the same as the CPU one. The policy depends on the FC's rate-setpoint
slew (19.2 rad/s² roll/pitch): without it, a first-order rate model
over-rotated and crashed.

**Latency** (Orange Pi 5, 2026-10-19, 2000 steps, stochastic sampling,
ms; raw JSON: `npu/bench_results/2026-10-19-orangepi5-rknn/`):

| Setup (Python thread pinned to) | Backend | mean | p50 | p99 | max |
|---|---|---|---|---|---|
| idle, A76 (cpu4) | rknn | 6.04 | 5.83 | 9.25 | 23.21 |
| idle, A76 (cpu4) | numpy | 2.62 | 2.49 | 5.41 | 6.93 |
| idle, A55 (cpu0) | rknn | 6.68 | 6.53 | 8.61 | 46.00 |
| idle, A55 (cpu0) | numpy | 12.47 | 11.72 | 17.02 | 23.70 |
| load on 7 cores, cpu4 | rknn | 5.56 | 4.72 | 19.60 | 30.91 |
| load on 7 cores, cpu4 | numpy | 3.81 | 3.54 | 6.28 | 9.45 |

Percentiles, idle, pinned to cpu4, 3 runs × 5000 steps (ranges over
runs; `4-a76-percentiles-run*.json`):

| Backend | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|
| numpy | 2.45–2.46 | 2.51–2.53 | 2.61 | 5.30–5.43 | 6.1–6.4 |
| rknn | 5.82–6.07 | 6.60–7.04 | 7.51–8.12 | 9.09–10.38 | 24.2–27.6 |

`core` ≈ 4.4 ms, `actor` ≈ 0.95 ms per call. Each of the ~87 core ops
costs a near-fixed ~50–70 µs of NPU dispatch, whatever its size, so the
graph is overhead-bound, not compute-bound. The NPU path's tail latency
(max 23–46 ms) breaks the 20 ms budget; numpy's does not. The NPU only
wins when the policy has to run on an A55 core.

Next levers, if the NPU is still wanted: fewer ops (one fused graph
with noise as inputs, see below; fold reshapes), RKNN multi-core / async
mode, `performance` NPU devfreq governor, and larger presets or CNN
encoders where per-op work outweighs dispatch.

Original Phase 2–4 design notes:

- Two static-shape graphs, batch 1. CPU glue stays in numpy: symlog,
  reset masking, action normalization, unimix + sampling. (Optional later:
  single graph with noise passed in as inputs — Gumbel-max latent,
  `mean + std·ε` — saves one NPU call but needs ArgMax/Equal support.)
- BlockLinear alternative: grouped 1×1 `Conv` (`group=8`). Never a
  block-diagonal dense matrix (8× the weights).
- `rknn.accuracy_analysis` per layer vs ONNX if fp16 error grows
  (RMSNorm over 2048, GRU gates).

## Phase 5 — validation gate

- Step level: RKNN vs reference within fp16 tolerance.
- **Drift**: 3000+ steps (60 s) comparing deter trajectories — compounding
  error in the recurrent state is the main numerical risk.
- **Closed loop**: `eval_quantitative.py` with the RKNN backend, ~200
  episodes per DR stage; success rate within the CI of the JAX baseline.
  Evaluate all three sampling modes here.
- Timing: p99 < 5 ms under CPU load; control loop on an isolated core,
  `SCHED_FIFO`, GC disabled in the loop.

## Phase 6 — packaging

`export_model.py` also emits `policy_weights.npz`, `core.onnx`,
`actor.onnx`, `core.rknn`, `actor.rknn`, `manifest.json` (obs/action
schema version, model size, toolkit version, sampling mode, sha256s);
drop `replay.pkl`. Bundle ~15–30 MB instead of ~105 MB. Slim
`install_orangepi.sh` to the numpy/RKNN path.

## Phase 7 — optional

INT8 / hybrid quantization for feed-forward layers only (keep GRU fp16),
only after Phase 5 passes; bigger presets; NPU CNN encoder for camera
observations (where the NPU earns its keep).

## Risks

| Risk | Mitigation |
|---|---|
| fp16 drift in the recurrent state | Phase 5 drift test; keep GRU path fp16/fp32 |
| Op support (grouped conv / batched matmul) | Two designed fallbacks |
| Toolkit ↔ runtime version mismatch | Pinned in manifest, checked at load |
| NPU call overhead ≥ compute saved | Phase 0 gate; numpy backend ships regardless |
| Sampling semantics differ from training | Explicit `sampling` flag, chosen by sim eval |
