# Orange Pi 5 (RK3588) NPU adapter — plan

Goal: run the deployed DreamerV3 hover policy on the RK3588 NPU (via
`rknn-toolkit-lite2`) with the same `reset()` / `step(obs, is_first)`
interface as `aerial/deploy/inference.py:PolicyInference`, without JAX on
the companion computer.

Status (2026-10-09): **Phase 1 done, Phase 0 done on the dev box —
waiting on Orange Pi measurements.** Phases 2–7 not started.

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
2. **The current JAX deploy install is incomplete** — `inference.py`
   imports the PyBullet env (needs `gym_pybullet_drones`,
   `pkg_resources`) and `install_orangepi.sh` doesn't install `ninjax`,
   `einops`, `portal`, `cloudpickle`, `colored_traceback` or `gym_pybullet_drones`. The NumPy/RKNN path
   needs none of them.
3. **`rknn-toolkit2` (the converter) is Linux-only** (x86_64 / aarch64).
   Conversion runs on the training box or on the Pi, not macOS.
4. **Multithreaded BLAS wrecks tail latency for batch-1 GEMVs** — on the
   dev box OpenBLAS with 10 threads gave p99 16 ms vs 0.99 ms with 1
   thread. Pin `OPENBLAS_NUM_THREADS=1` for any CPU backend.

---

## Phase 0 — measure, then decide  ⏳ (dev box done, Pi pending)

Tool: `python -m aerial.deploy.npu.bench --model <bundle>`

Dev box (Apple M-series, arm64, numpy 1.26, dummy size12m bundle,
2000 steps, BLAS 1 thread):

| Backend | mean | p50 | p99 | max |
|---|---|---|---|---|
| numpy reference | 0.83 ms | 0.83 ms | 0.99 ms | 1.16 ms |
| JAX CPU (current path) | 1.33 ms | 1.29 ms | 1.64 ms | 8.67 ms |

**To do on the Orange Pi 5:**

```bash
cd ~/aerial-drone && git pull
# numpy-only (works with the existing venv):
./.venv/bin/python -m aerial.deploy.npu.bench \
    --model aerial/deploy/models/<bundle> --backends numpy \
    --steps 2000 --json bench-orangepi.json
# JAX baseline too (needs: pip install ninjax einops portal cloudpickle colored_traceback):
./.venv/bin/python -m aerial.deploy.npu.bench \
    --model aerial/deploy/models/<bundle> --steps 2000 \
    --json bench-orangepi.json
# No trained bundle on the Pi? Make a dummy one on the dev box:
python -m aerial.deploy.npu.jax_agent --out aerial/deploy/models/dummy-size12m
```

Also run once with `taskset -c 4` (an A76 big core) and once with a
`stress-ng --cpu 4` load in the background to see tail latency under
contention.

**Decision gate.** Expect the A76 cores at a few × the dev-box time.
- numpy p99 ≪ 20 ms on the Pi (likely): the NPU is **not needed for
  latency**. Its value is offloading the CPU for MAVLink + safety, room
  for size25m/50m, and future vision encoders. Proceed with Phases 2–5
  as an optimization, ship the numpy backend first.
- numpy p99 near/over budget: NPU becomes the critical path.

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

## Phase 2 — ONNX graphs shaped for the NPU

- Two static-shape graphs, batch 1, opset 13–17:
  - **core**: (obs_symlog 22, deter 2048, stoch 512, prevact_norm 4) →
    (deter′ 2048, logits 512)
  - **actor**: (deter′ 2048, stoch′ 512) → (mean 4, std 4)
- CPU glue stays in numpy: symlog, reset masking, action normalization,
  unimix + sampling. (Optional later: single graph with noise passed in
  as inputs — Gumbel-max latent, `mean + std·ε` — saves one NPU call but
  needs ArgMax/Equal support.)
- BlockLinear → grouped 1×1 `Conv` (`group=8`) — the NPU's best-
  optimized path. Fallback: batched `MatMul` (8,1,1024)@(8,1024,256).
  Never a block-diagonal dense matrix (8× the weights).
- Build via a small PyTorch module loading `policy_weights.npz`, then
  `torch.onnx.export`. Validate ONNX Runtime output against
  `ReferencePolicy` (same harness as Phase 1, ≤1e-5).

## Phase 3 — RKNN conversion (Linux host)

- `rknn.config(target_platform='rk3588')`, `do_quantization=False`
  (**fp16 first**).
- `rknn.accuracy_analysis` per layer vs ONNX; watch RMSNorm reductions
  (mean of squares over 2048) and GRU gates in fp16.
- Pin the toolkit version; record it in the bundle manifest. The Pi's
  `librknnrt.so` must match.

## Phase 4 — runtime adapter on the Pi

- `npu/rknn_policy.py`: `RKNNPolicy` (`RKNNLite`, `core_mask=NPU_CORE_0`),
  carry in preallocated numpy buffers, same sampling modes and API.
- `make_policy(model_dir, backend='numpy'|'onnx'|'rknn'|'jax', sampling=…)`;
  `smoke_test.py --backend`.
- `install_orangepi_npu.sh`: numpy, ruamel.yaml, `rknn-toolkit-lite2`
  (cp312 aarch64 wheel); check `/sys/kernel/debug/rknpu/version`.

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
