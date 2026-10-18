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
MIXER = np.array([
    [-0.5, -0.5, -1.0],
    [-0.5, +0.5, +1.0],
    [+0.5, +0.5, -1.0],
    [+0.5, -0.5, +1.0],
], dtype=np.float32)


class ArduPilotRateController:
    """Body-rate controller modeled on ArduPilot Copter, simplified.

    Runs at the flight-controller loop rate (every physics substep), not
    the 50 Hz policy rate. Per substep:
      1. Slew the rate setpoint with an angular-acceleration limit
         (ATC_ACCEL_R/P/Y_MAX).
      2. Low-pass the gyro (INS_GYRO_FILTER).
      3. PID on rate error, I-term clamped (ATC_RAT_*_IMAX).
      4. Mix: roll/pitch first, yaw gets the remaining headroom, then
         collective shifts to keep motors in range (AP_MotorsMatrix).
      5. Armed idle floor (MOT_SPIN_MIN).

    Gains are derived from the airframe so they rescale when measured
    specs replace the placeholders: the rate loop closes with a time
    constant of RP_TC_PER_MOTOR_TAU (roll/pitch) or YAW_TC_PER_MOTOR_TAU
    (yaw) times the motor time constant. `gain_scale` models tuning
    mismatch between the FC and the actual vehicle.
    """

    RP_TC_PER_MOTOR_TAU = 1.25
    YAW_TC_PER_MOTOR_TAU = 2.5
    I_TIME_S = 0.5
    D_TIME_S = 0.01
    I_MAX = 0.2
    ACCEL_MAX_RAD_S2 = np.array([19.2, 19.2, 4.7], dtype=np.float32)
    GYRO_FILTER_HZ = 20.0
    # ArduPilot maps a zero thrust request to MOT_SPIN_MIN (~0.15
    # throttle), which is ~6% of max thrust on a typical prop curve.
    MIN_MOTOR = 0.06

    def __init__(self, airframe, loop_hz, gain_scale=1.0):
        self.dt = 1.0 / float(loop_hz)
        lever = airframe.arm_length_m / np.sqrt(2.0)
        thrust = airframe.max_thrust_per_motor_N
        inertia = np.asarray(airframe.inertia_diag_kg_m2, dtype=np.float32)
        # Angular acceleration (rad/s^2) per unit of mixer input per axis.
        plant_gain = np.array([
            2.0 * thrust * lever,
            2.0 * thrust * lever,
            4.0 * airframe.torque_coeff_per_thrust * thrust,
        ], dtype=np.float32) / inertia
        tau_m = airframe.motor_time_constant_s
        closed_loop_tc = np.array([
            self.RP_TC_PER_MOTOR_TAU * tau_m,
            self.RP_TC_PER_MOTOR_TAU * tau_m,
            self.YAW_TC_PER_MOTOR_TAU * tau_m,
        ], dtype=np.float32)
        self.kp = (gain_scale / (plant_gain * closed_loop_tc)).astype(np.float32)
        self.ki = self.kp / self.I_TIME_S
        self.kd = self.kp * self.D_TIME_S
        self._gyro_alpha = self.dt / (
            self.dt + 1.0 / (2.0 * np.pi * self.GYRO_FILTER_HZ))
        self.reset()

    def reset(self):
        self.rate_target = np.zeros(3, dtype=np.float32)
        self.gyro_filt = np.zeros(3, dtype=np.float32)
        self.integ = np.zeros(3, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)

    def step(self, desired_body_rates, measured_gyro, thrust_normalized):
        desired = np.asarray(desired_body_rates, dtype=np.float32).reshape(3)
        max_delta = self.ACCEL_MAX_RAD_S2 * self.dt
        self.rate_target += np.clip(
            desired - self.rate_target, -max_delta, max_delta)
        self.gyro_filt += self._gyro_alpha * (
            np.asarray(measured_gyro, dtype=np.float32) - self.gyro_filt)

        error = self.rate_target - self.gyro_filt
        self.integ = np.clip(
            self.integ + self.ki * error * self.dt, -self.I_MAX, self.I_MAX)
        d_error = (error - self.prev_error) / self.dt
        self.prev_error = error
        tau = self.kp * error + self.integ + self.kd * d_error
        return self._mix(float(np.clip(thrust_normalized, 0.0, 1.0)), tau)

    def _mix(self, thrust, tau):
        rp = MIXER[:, :2] @ tau[:2]
        rp_span = float(rp.max() - rp.min())
        if rp_span > 1.0:
            rp = rp / rp_span
            rp_span = 1.0
        yaw = MIXER[:, 2] * tau[2]
        yaw_span = float(yaw.max() - yaw.min())
        if rp_span + yaw_span > 1.0 and yaw_span > 0.0:
            yaw = yaw * ((1.0 - rp_span) / yaw_span)
        out = rp + yaw
        thrust = float(np.clip(thrust, -out.min(), 1.0 - out.max()))
        motor = np.clip(thrust + out, 0.0, 1.0)
        return np.maximum(motor, self.MIN_MOTOR).astype(np.float32)


class BodyRateController:
    """Cascaded PD rate controller producing normalized motor commands.

    Phase 0 (Crazyflie) env only. The 12" env uses ArduPilotRateController.

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
        motor = t + MIXER @ tau  # (4,)
        return np.clip(motor.astype(np.float32), 0.0, 1.0)
