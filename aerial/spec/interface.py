"""Sim-agnostic observation and action spec.

This module defines the canonical interface every drone sim adapter
(PyBullet, Aerial Gym, ArduPilot SITL, real hardware) must conform to.
Changes here ripple to every adapter; nothing else should redefine these
constants.

Conventions:
  * World frame: z-up (matches PyBullet, Aerial Gym, ROS REP-103 ENU).
  * Rotation order: rotation matrix R is body->world.
  * Body angular velocity (gyro): rad/s in body frame.
  * Action layout matches ArduPilot SET_ATTITUDE_TARGET body-rate mode +
    PX4 offboard body-rate mode, so a policy trained against this spec
    can deploy to either FC without retraining the head.
"""
import numpy as np

# Control rate. Matches ArduPilot offboard-MAVLink ceiling so the trained
# policy can deploy without changing step rate. PyBullet's PYB_FREQ must
# be set to a multiple of this (we use 250 -> 5 phys steps per ctrl step).
STEP_RATE_HZ = 50

# Action limits. Body-rate ceiling is a safety + agility knob; 6 rad/s
# (~343 deg/s) is reasonable for general-purpose flight. Raise for racing
# / acro phases (Phase 3+).
MAX_BODY_RATE_RPY = 6.0  # rad/s

# Action layout (4 dims, range [-1, 1]):
#   0: body rate roll  (tanh-scaled to +-MAX_BODY_RATE_RPY)
#   1: body rate pitch (tanh-scaled to +-MAX_BODY_RATE_RPY)
#   2: body rate yaw   (tanh-scaled to +-MAX_BODY_RATE_RPY)
#   3: collective thrust normalized (linear-scaled to [0, 1])
ACTION_DIM = 4
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Observation layout (22 dims). Indices below match get_obs_vector().
# All fields are policy-relative (no absolute world coords) so the policy
# generalizes across launch points and target locations.
OBS_DIM = 22

# Default hover target relative to spawn. The env may override per episode.
TARGET_POSITION = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # z = 1m


def scale_action(raw_action):
    """Map policy output [-1, 1]^4 to physical units.

    Returns (body_rates_rps[3], thrust_normalized[0..1]).
    """
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    body_rates = raw[:3] * MAX_BODY_RATE_RPY
    thrust_norm = (raw[3] + 1.0) * 0.5
    thrust_norm = float(np.clip(thrust_norm, 0.0, 1.0))
    return body_rates, thrust_norm


def get_obs_vector(pos, vel, R, gyro, last_action, target=TARGET_POSITION):
    """Pack the canonical 22-dim observation.

    Args:
      pos: (3,) world position, meters.
      vel: (3,) world linear velocity, m/s.
      R:   (3,3) rotation matrix body->world.
      gyro: (3,) body angular velocity, rad/s.
      last_action: (4,) previous policy action in [-1, 1].
      target: (3,) world position of hover target.

    Layout:
       [0:3]   position relative to target (world frame)
       [3:6]   linear velocity (world frame)
       [6:15]  rotation matrix flattened row-major (body->world)
       [15:18] gyro (body frame)
       [18:22] last action
    """
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    vel = np.asarray(vel, dtype=np.float32).reshape(3)
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    gyro = np.asarray(gyro, dtype=np.float32).reshape(3)
    last_action = np.asarray(last_action, dtype=np.float32).reshape(4)
    target = np.asarray(target, dtype=np.float32).reshape(3)

    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0:3] = pos - target
    obs[3:6] = vel
    obs[6:15] = R.reshape(-1)
    obs[15:18] = gyro
    obs[18:22] = last_action
    return obs


def upright_from_R(R):
    """Return the world-z component of the body-z axis.

    1.0 = perfectly upright, 0.0 = on its side, -1.0 = inverted.
    R is body->world, so R[2, 2] is body-z's projection onto world-z.
    """
    return float(R[2, 2])
