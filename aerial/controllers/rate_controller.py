"""Body-rate + collective thrust controller.

Sits between the RL policy and a sim that takes per-motor commands.
The policy outputs desired body angular rates and a normalized collective
thrust; this module turns that into four motor commands.

Matches the ArduPilot SET_ATTITUDE_TARGET / PX4 offboard body-rate
interface, so the policy is FC-portable. In Phase 1, these PD gains
should be randomized per-episode as part of sim2real domain
randomization.

Mixer uses the canonical CF2X X-frame layout from gym-pybullet-drones'
DSLPIDControl (which matches PyBullet's cf2x URDF prop positions):

  Body frame: x=forward, y=left, z=up.
  Motor 0: (+x, -y)  front-right (CW)
  Motor 1: (-x, -y)  back-right  (CCW)
  Motor 2: (-x, +y)  back-left   (CW)
  Motor 3: (+x, +y)  front-left  (CCW)

  MIXER_MATRIX columns = [roll, pitch, yaw]:
      m0: [-0.5, -0.5, -1.0]
      m1: [-0.5, +0.5, +1.0]
      m2: [+0.5, +0.5, -1.0]
      m3: [+0.5, -0.5, +1.0]

  Note the sign conventions: +roll torque lifts the +y (left) side;
  +pitch torque lifts the back (nose down); +yaw torque spins CCW
  about world-z. The policy spec uses standard FPV (+roll = right
  bank, +pitch = nose up, +yaw = CCW) and the rate axis order matches
  PyBullet's gyro output, so the spec layer just hands rate setpoints
  through to this controller without sign flips — the mixer absorbs
  the convention difference.
"""
import numpy as np


# Default rate-loop PD gains. Tuned for the normalized motor-command
# output range. With max body-rate command ~6 rad/s and Kp~0.05, peak
# tau is ~0.3, which after the 0.5 mixer coefficient gives ~0.15
# differential thrust per axis — enough authority without saturating.
DEFAULT_KP = np.array([0.008, 0.008, 0.02], dtype=np.float32)  # roll, pitch, yaw
DEFAULT_KD = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# DSLPIDControl's CF2X mixer matrix. Columns are [roll, pitch, yaw].
_MIXER = np.array([
    [-0.5, -0.5, -1.0],
    [-0.5, +0.5, +1.0],
    [+0.5, +0.5, -1.0],
    [+0.5, -0.5, +1.0],
], dtype=np.float32)


class BodyRateController:
    """Cascaded PD rate controller producing normalized motor commands.

    Inputs per step:
        desired_body_rates (3,) — rad/s, body frame, from policy
        measured_gyro (3,)      — rad/s, body frame, from sim
        thrust_normalized scalar in [0, 1] — collective from policy

    Output:
        motor_cmd (4,) in [0, 1] — normalized motor commands.
    """

    def __init__(self, kp=None, kd=None):
        self.kp = np.asarray(DEFAULT_KP if kp is None else kp, dtype=np.float32)
        self.kd = np.asarray(DEFAULT_KD if kd is None else kd, dtype=np.float32)
        self.prev_rate_error = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.prev_rate_error.fill(0.0)

    def step(self, desired_body_rates, measured_gyro, thrust_normalized, dt):
        desired = np.asarray(desired_body_rates, dtype=np.float32).reshape(3)
        measured = np.asarray(measured_gyro, dtype=np.float32).reshape(3)
        rate_error = desired - measured
        d_error = (rate_error - self.prev_rate_error) / max(dt, 1e-6)
        self.prev_rate_error = rate_error

        tau = self.kp * rate_error + self.kd * d_error  # (3,)
        t = float(np.clip(thrust_normalized, 0.0, 1.0))
        motor = t + _MIXER @ tau  # (4,)
        return np.clip(motor.astype(np.float32), 0.0, 1.0)
