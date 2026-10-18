"""Sim-agnostic reward function for hover task.

Ported from steam/liftoff/hover_env_uinput.py. Same shape, same
weights, but operates on the canonical state struct (no telemetry
indices) so the same function runs against PyBullet, Aerial Gym, or
ArduPilot SITL with no changes.

Reward design:
  * Dense terms (position, velocity, gyro, upright, hold) sum to ~1.25/step
    at perfect hover.
  * Hold term pays only for being near the target AND nearly still, so a
    loose orbit around the target earns much less than holding it.
  * Ground term penalizes flying below GROUND_SAFE_ALTITUDE_M. Ground
    contact itself is a crash (enforced by the env).
  * Smoothness penalty discourages thrash.
  * Survival bonus + stability-streak bonus reward staying alive. Any
    stable step still earns > 0, so crashing is never preferable.
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
STABILITY_STREAK_MAX = 0.10       # was 0.15 before 2026-10-16
STABILITY_STREAK_TAU = 50.0

# --- Dense-term weights -------------------------------------------------
# Before 2026-10-16, terms earned just for staying upright and alive
# (upright 0.30, gyro 0.15, survival 0.10, streak 0.15) were ~56% of the
# max reward, so orbiting 0.5 m off target or skimming 20 cm above the
# floor still earned ~70-74% of max.
W_POSITION = 0.30
W_VELOCITY = 0.10           # was 0.15
W_GYRO = 0.10               # was 0.15
W_UPRIGHT = 0.15            # was 0.30
W_SMOOTHNESS = 0.02
SURVIVAL_BONUS = 0.15       # was 0.10

# --- Shaping tuned from Phase 0 quantitative eval -----------------------
POSITION_DECAY = 2.0        # was 1.0 in Phase 0
VELOCITY_DECAY = 1.0        # was 0.5 in Phase 0

# Hold term: exp(-8*dist) * exp(-4*speed). Replaces the flat +0.10 bonus
# inside 0.2 m, which gave no gradient toward the center and none for
# holding still. Worth 0.35 at d=0, v=0; ~0.10 at d=0.1 m, v=0.1 m/s;
# ~0 beyond 0.4 m, so it doesn't distort far-field recovery.
W_HOLD = 0.35
HOLD_POS_DECAY = 8.0        # 1/m
HOLD_VEL_DECAY = 4.0        # s/m

# Ground term: linear penalty below GROUND_SAFE_ALTITUDE_M, reaching
# -0.175 at the env's 0.15 m ground-crash altitude.
W_GROUND = 0.25
GROUND_SAFE_ALTITUDE_M = 0.5


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


def step_reward(pos_err, vel, gyro, up_axis, altitude, action, state):
    """Compute one step's reward and update streak counter in `state`.

    Args:
      pos_err: (3,) position error from target, world frame, meters.
      vel:     (3,) world linear velocity, m/s.
      gyro:    (3,) body angular velocity, rad/s.
      up_axis: scalar in [-1, 1], world-z component of body-z (R[2,2]).
      altitude: height of the body above the ground, meters.
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
    r_hold = float(np.exp(-HOLD_POS_DECAY * dist - HOLD_VEL_DECAY * speed))
    r_ground = -W_GROUND * max(
        0.0, (GROUND_SAFE_ALTITUDE_M - float(altitude)) / GROUND_SAFE_ALTITUDE_M)

    reward = (
        W_POSITION * r_position
        + W_VELOCITY * r_velocity
        + W_GYRO * r_gyro
        + W_UPRIGHT * r_upright
        + W_HOLD * r_hold
        + r_ground
        + r_smoothness
        + SURVIVAL_BONUS
        + r_streak
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
        "r_hold": r_hold,
        "r_ground": r_ground,
        "streak": float(state.stable_streak),
    }
