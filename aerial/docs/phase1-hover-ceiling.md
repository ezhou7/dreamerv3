# Phase 1 hover — what the training ceiling actually is

Empirical findings from training a DreamerV3 policy for hover recovery
on a 12" ArduPilot-based multirotor, over ~six weeks of experiments.

**Short version.** The best 12M policy scores 36-42% on Stage 1c (the
widest domain-randomization setting), and that's not a training
failure — it's close to the Bayes limit for that DR distribution.
Bigger models (400M) hit the same ceiling. Training on narrower DR
does worse. The realistic deployment target is Stage 1b conditions.

---

## The metric

"Success rate" = fraction of episodes where the drone stays airborne
for the full 500 steps (10 seconds at 50 Hz) starting from a
randomized initial condition. The env terminates on ground contact,
exit from a bounding box, or catastrophic instability. See
`aerial/train/eval_quantitative.py` for the exact definition.

## Domain randomization stages

| Stage | Init pos | Init tilt | Mass | Motor | Rate Kp | Wind σ | Latency |
|---|---|---|---|---|---|---|---|
| 1a | ±0.5 m | ±10° | ±10% | ±8% | ±10% | 0.15 N | 0-15 ms |
| 1a5 | ±0.75 m | ±15° | ±12% | ±10% | ±12% | 0.18 N | 0-18 ms |
| 1b | ±1.0 m | ±20° | ±15% | ±12% | ±15% | 0.22 N | 0-22 ms |
| 1c | ±2.0 m | ±30° | ±20% | ±15% | ±20% | 0.30 N | 0-30 ms |

Defined in `aerial/configs/domain_randomization.py`. Each parameter is
sampled independently per episode.

## The winning recipe

The `aerial_12in_skydreamer` config in `dreamerv3/configs.yaml`:
- `size12m` DreamerV3 preset (~12M params)
- Full Stage 1c DR from step 0 (no curriculum)
- No warm-starting from any prior checkpoint (fresh init)
- Motor smoothness regularization (`smooth_coef=0.002`)
- Default DreamerV3 hyperparameters otherwise (actent 3e-4, batch_length 64)
- 17M training steps
- ~20-hour training run on an RTX PRO 6000 Blackwell

Recipe adapted from SkyDreamer (arxiv 2510.14783). Model bundle
published as a GitHub release:
`skydreamer-17m-1c-2026-09-05` at
https://github.com/ezhou7/dreamerv3/releases/tag/model-skydreamer-17m-1c-2026-09-05

## What the policy actually does

Evaluated the same 17M-step checkpoint against each stage in sequence:

| Stage | Success rate | Recovery within 0.5m of target | Median position error |
|---|---|---|---|
| 1a | 94% | 100% | 0.38 m |
| 1a5 | 84% | 88% | 0.45 m |
| 1b | 72% | 78% | 0.54 m |
| 1c | 42% | 50% | 1.09 m |

Graceful degradation. The policy is competent everywhere and gets
progressively cornered by the wider distribution as init pose and
physics variance widen.

## Why 1c success plateaus at ~42%

Stage 1c samples every parameter independently. A single episode can
draw all of:
- Init 2 m off-target (upper limit)
- Init tilt −30° face-down (upper limit)
- Body 20% heavier (upper limit)
- Motors 15% weaker (upper limit)
- Wind gust
- Control latency 30 ms + motor τ up to 52 ms

For that draw, the physics:
- Nominal thrust-to-weight ratio (TWR) 3.06 becomes effective TWR
  ~2.17 (0.85 motor × 1.20 mass)
- At −30° tilt, only cos(30°) = 87% of thrust points up
- Actions can lag reality by ~80 ms (motor τ + control latency)
- Episode budget is 10 seconds

Some of those combinations hit the ground before any policy could
recover. The 1c distribution is a **stress test** — it deliberately
includes physically-unrecoverable inits. `~40%` unrecoverable per
episode roughly matches the observed failure fraction.

## Evidence this is a task-difficulty ceiling, not a training failure

**1. Same policy degrades cleanly across stages.** If it were policy
holes, we'd expect uneven degradation (some stages surprisingly bad,
some fine). Instead the 94%/84%/72%/42% cascade tracks the widening
distribution smoothly.

**2. Training on narrower DR does worse, not better.** A dedicated
17M-step run on Stage 1b DR scored 60% on Stage 1b — the 1c-trained
model already scored 72% zero-shot on that distribution. Wider
training makes the policy *more* transferable, not less.

**3. 30× more parameters hit the same p50 asymptote.** A 400M-parameter
run of the same recipe converges to the same training-time p50 (~-8)
and p90 (~350) as the 12M by step 1-2M. Capacity accelerates learning
the good mode but doesn't fix the failure mode. See
`RESULTS.md` for the head-to-head trajectory data (private, project
notes).

## What to do about it

**For real-hardware deployment.** Match the flight envelope to Stage
1b conditions — don't spawn 2 m off-target, don't tilt beyond 20°,
don't carry payloads outside ±15% of nominal mass, don't rely on the
policy in gusts above ~4% of hover thrust. Expected sim success rate
in that envelope: ~72%. Real-world transfer is a separate question;
see the sim2real notes in `PLAN.md`.

**"One Net to Rule Them All" (arxiv 2504.21586) confirms this direction.**
Their model-free study found ~10-15% DR gave the best real-world
transfer; wider DR reduced flight speed and success. Our 1b is
~15% DR, our 1c is ~20% DR — 1b sits closer to their sweet spot.

**Ways to push the 1c number that we haven't tried and probably won't.**
- Increase airframe TWR (hardware change, not training). A TWR-4.0
  airframe with the same DR ranges would push 1c higher because the
  effective-TWR floor rises above 2.5.
- Longer episodes. 20-second episodes give more time to recover from
  bad starts. Changes reward math, requires retraining.
- Non-uniform DR sampling. Correlate mass and motor so "heavy body +
  weak motors" doesn't co-occur. Sacrifices sim2real coverage.

**Ways NOT to push the 1c number.** These we tried and they made
things worse or the same, in descending order of "how strongly the
data rejects it":
- SkyDreamer's published three-phase hyperparameter schedule (batch
  length 64→256, actent 3e-4→1e-5, lr 4e-5→2e-6): scored 8% on Stage
  1c. Deterministic-eval collapse from the low actent — the paper
  probably samples stochastically in eval too.
- Warm-starts through DR distribution shifts (any variant of
  curriculum 1a → 1a5 → 1b → 1c with checkpoint chaining): capped at
  ~24% because the actor forgets its skill on wider-DR training.
- BC-regularization anchor: 0% because the reference actor sees
  drifted world-model features and gives meaningless gradients.
- Continuous DR ramp: 16%, worse than the zero-shot baseline.
- Training targeted at Stage 1b: 60% on Stage 1b, worse than the
  1c-trained model's 72% zero-shot on the same distribution.

## Reproducing the winning recipe

```bash
PYTHONPATH=$PWD .venv-pybullet/bin/python dreamerv3/main.py \
    --logdir ~/logdir/dreamer/aerial-12in-skydreamer-<yourtag> \
    --configs aerial_12in_skydreamer
```

Wait ~20 hours. Then:

```bash
PYTHONPATH=$PWD .venv-pybullet/bin/python aerial/train/eval_quantitative.py \
    --logdir ~/logdir/dreamer/aerial-12in-skydreamer-<yourtag> \
    --n-episodes 50 --config aerial_12in_skydreamer_eval
```

Should score in the 35-45% band on Stage 1c and 65-75% on Stage 1b.

## Reproducing the ceiling test

If someone wants to independently verify the "bigger model doesn't
help" finding on a machine with more RAM headroom:

```bash
# Would need to add back to configs.yaml (reverted after our run):
#   aerial_12in_skydreamer_400m: <<: *size400m, replay.size 5e5,
#     dr_stage 1c, smooth_coef 0.002, 17M steps
# Then:
systemd-run --user --scope --property=MemoryMax=80G --property=MemoryHigh=70G \
  bash -c '...python dreamerv3/main.py --configs aerial_12in_skydreamer_400m ...'
```

Expected finding at 1-2M steps: same training-time p50 and p90 as the
12M. Full 17M run would take ~4-5 days on the Blackwell.

## References

- **SkyDreamer** (arxiv 2510.14783) — the base recipe. Achieves 100%
  real-world success on 3 small race tracks + 83% on big track,
  using the exact same size12m + 17M steps + smoothness reg we do.
  Does NOT publish sim success rates in %, so their numbers don't
  directly bound ours.
- **Dream to Fly** (arxiv 2501.14377) — larger DreamerV3, curriculum
  on reward parameter (not DR). 20M steps.
- **One Net to Rule Them All** (arxiv 2504.21586) — model-free but
  the DR study is directly relevant. 10% DR = best real transfer.
- **The World Model Remembers, the Actor Forgets** (arxiv 2607.19749) —
  documents the failure mode we chased for weeks before switching to
  the SkyDreamer recipe.
