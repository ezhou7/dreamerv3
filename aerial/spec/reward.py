"""Sim-agnostic reward function for hover task.

Ported from steam/liftoff/hover_env_uinput.py. Same shape, same
weights, but operates on the canonical state struct (no telemetry
indices) so the same function runs against PyBullet, Aerial Gym, or
ArduPilot SITL with no changes.

Reward design:
  * Dense terms (position, velocity, gyro, upright) sum to ~1.0/step at
    perfect hover.
  * Smoothness penalty discourages thrash.
  * Survival bonus + stability-streak bonus reward staying alive.
  * Unstable steps return a fixed penalty instead of the dense sum, so
    the model can't farm dense reward while flailing.
  * Crash penalty applied externally on terminal step.
"""
from dataclasses import dataclass

import numpy as np


# --- Thresholds ---------------------------------------------------------
UPRIGHT_THRESHOLD = 0.707          # cos(45 deg)
GYRO_THRESHOLD = 3.0               # rad/s, body-frame magnitude
UNSTABLE_STEP_PENALTY = -1.0
CRASH_PENALTY = -10.0

# --- Stability streak bonus ---------------------------------------------
# Side objective: reward staying upright continuously. Saturates so it
# can't dominate position-tracking reward. TAU = 50 steps = 1.0s at 50 Hz
# means 63% of plateau at 1s, 95% at 3s.
STABILITY_STREAK_MAX = 0.15
STABILITY_STREAK_TAU = 50.0

# --- Dense-term weights -------------------------------------------------
W_POSITION = 0.30
W_VELOCITY = 0.15
W_GYRO = 0.15
W_UPRIGHT = 0.30
W_SMOOTHNESS = 0.02
SURVIVAL_BONUS = 0.10

# --- Shaping tuned from Phase 0 quantitative eval -----------------------
# Phase 0 policy hovered at ~0.58m offset with ~0.5m/s drift because the
# original exp(-dist) and exp(-0.5*speed) gave too much partial credit
# for "close enough". Sharpening the decay forces the policy to actually
# nail the target instead of settling into a stable orbit.
POSITION_DECAY = 2.0        # was 1.0 in Phase 0
VELOCITY_DECAY = 1.0        # was 0.5 in Phase 0

# Tight-target bonus: flat additive reward when the drone is within
# TIGHT_TARGET_RADIUS meters of the target. Small enough not to dominate
# but large enough to break ties in favor of "actually on target".
TIGHT_TARGET_BONUS = 0.10
TIGHT_TARGET_RADIUS = 0.2   # meters


@dataclass
class RewardState:
    """Per-episode mutable state passed back in to step()."""
    prev_action: np.ndarray
    stable_streak: int = 0

    @classmethod
    def initial(cls, action_dim=4):
        return cls(prev_action=np.zeros(action_dim, dtype=np.float32),
                   stable_streak=0)


def is_unstable(up_axis, gyro):
    """True iff the drone is tilted past UPRIGHT_THRESHOLD or spinning
    above GYRO_THRESHOLD."""
    gyro_mag = float(np.linalg.norm(gyro))
    return bool(up_axis < UPRIGHT_THRESHOLD) or bool(gyro_mag > GYRO_THRESHOLD)


def step_reward(pos_err, vel, gyro, up_axis, action, state):
    """Compute one step's reward and update streak counter in `state`.

    Args:
      pos_err: (3,) position error from target, world frame, meters.
      vel:     (3,) world linear velocity, m/s.
      gyro:    (3,) body angular velocity, rad/s.
      up_axis: scalar in [-1, 1], world-z component of body-z (R[2,2]).
      action:  (4,) current policy action in [-1, 1].
      state:   RewardState (mutated in place).

    Returns:
      reward (float), terms (dict for logging).
    """
    pos_err = np.asarray(pos_err, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)

    dist = float(np.linalg.norm(pos_err))
    speed = float(np.linalg.norm(vel))
    gyro_mag = float(np.linalg.norm(gyro))

    unstable = is_unstable(up_axis, gyro)
    action_delta = float(np.linalg.norm(action - state.prev_action))
    state.prev_action = action.copy()

    if unstable:
        state.stable_streak = 0
        return UNSTABLE_STEP_PENALTY, {
            "unstable": 1.0,
            "dist": dist,
            "speed": speed,
            "gyro_mag": gyro_mag,
            "up_axis": float(up_axis),
        }

    state.stable_streak += 1
    r_position = float(np.exp(-POSITION_DECAY * dist))
    r_velocity = float(np.exp(-VELOCITY_DECAY * speed))
    r_gyro = float(np.exp(-0.5 * gyro_mag))
    r_upright = float(up_axis)
    r_smoothness = -W_SMOOTHNESS * action_delta
    r_streak = STABILITY_STREAK_MAX * (
        1.0 - float(np.exp(-state.stable_streak / STABILITY_STREAK_TAU))
    )
    r_tight = TIGHT_TARGET_BONUS if dist < TIGHT_TARGET_RADIUS else 0.0

    reward = (
        W_POSITION * r_position
        + W_VELOCITY * r_velocity
        + W_GYRO * r_gyro
        + W_UPRIGHT * r_upright
        + r_smoothness
        + SURVIVAL_BONUS
        + r_streak
        + r_tight
    )

    return float(reward), {
        "unstable": 0.0,
        "dist": dist,
        "speed": speed,
        "gyro_mag": gyro_mag,
        "up_axis": float(up_axis),
        "r_position": r_position,
        "r_velocity": r_velocity,
        "r_gyro": r_gyro,
        "r_upright": r_upright,
        "r_smoothness": r_smoothness,
        "r_streak": r_streak,
        "r_tight": r_tight,
        "streak": float(state.stable_streak),
    }
