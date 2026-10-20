"""Gymnasium env for the target airframe with full domain randomization.

Phase 1 production env. Reuses aerial/spec/ and PyBullet BaseAviary but
swaps the Crazyflie URDF for aerial/assets/cf12in.urdf and applies
per-episode + per-step domain randomization. The "12in" names are
historical: since 2026-10-17 the airframe is the project's F450 (10"
props), see aerial/configs/airframe_12in.py.

Registered as Gymnasium env `Aerial-hover-12in-v0`.

Simulation loop (since 2026-10-16). The policy acts at STEP_RATE_HZ;
everything below it runs every physics substep at PYB_FREQ, the default
ArduPilot Copter loop rate:
  * ArduPilot-style rate controller (aerial/controllers/rate_controller.py)
    reads a biased, noisy body-frame gyro.
  * First-order motor lag, per-motor thrust scale, armed idle floor.
  * Rotor drag + ground effect (gym-pybullet-drones PYB_GND_DRAG_DW).
  * Mean wind + gusts applied at the center of mass; CG-offset torque.
  * Action latency and observation latency at substep resolution.
The policy observes a delayed state with estimator errors: attitude
bias/noise, gyro bias/noise, drifting position/velocity error. Reward
and termination use the true state.

URDF loading gotcha:
  gym-pybullet-drones hardcodes URDF path via pkg_resources against its
  own package's assets/ directory. We work around this by (a) using a
  duck-typed DroneModel-like object with .value = 'cf12in', and
  (b) copying our URDF into the gym_pybullet_drones assets dir at module
  import time. The copy is idempotent — safe to re-run.
"""
import os
import shutil
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, Dict as DictSpace

import pkg_resources
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import Physics

from aerial.spec.interface import (
    OBS_DIM, ACTION_DIM, STEP_RATE_HZ, TARGET_POSITION,
    scale_action, get_obs_vector, upright_from_R,
)
from aerial.spec.reward import (
    RewardState, step_reward, CRASH_PENALTY, is_unstable,
)
from aerial.controllers.rate_controller import (
    ArduPilotRateController, MIXER,
)
from aerial.configs.airframe_12in import DEFAULT_12IN
from aerial.configs.domain_randomization import (
    GUST_AXIS_SCALE,
    get_stage as get_dr_stage,
    sample_episode_params,
    sample_init_pose,
)

# --- URDF install at import time ---------------------------------------
_MY_URDF_NAME = "cf12in.urdf"
_MY_URDF_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "assets", _MY_URDF_NAME,
)


def _install_urdf():
    dst = pkg_resources.resource_filename(
        "gym_pybullet_drones", f"assets/{_MY_URDF_NAME}")
    if not os.path.isfile(dst) or (
        os.path.getmtime(_MY_URDF_SRC) > os.path.getmtime(dst)
    ):
        shutil.copyfile(_MY_URDF_SRC, dst)


_install_urdf()


class _DroneModel12in:
    """Duck-typed DroneModel: only needs .value for BaseAviary URDF lookup."""
    value = "cf12in"


CF12IN = _DroneModel12in()

# --- Timing ------------------------------------------------------------
PYB_FREQ = 400
SUBSTEPS = PYB_FREQ // STEP_RATE_HZ
PHYS_DT = 1.0 / PYB_FREQ
CTRL_DT = 1.0 / STEP_RATE_HZ
assert PYB_FREQ % STEP_RATE_HZ == 0
# Delayed-state buffer, in physics substeps (covers up to ~77 ms).
HISTORY_LEN = 32
# center_of_mass_link in cf12in.urdf. gym-pybullet-drones applies body
# torques here because LINK_FRAME torques on the base (-1) are unreliable.
COM_LINK = 4

# --- Workspace / termination ------------------------------------------
# 12" quad flies faster + we spawn it up to 2m from target, so bounds
# need to be wider than Phase 0's cramped Crazyflie workspace.
BOUND_XY = 10.0
BOUND_Z_MAX = 8.0
# Before 2026-10-16 the floor bound sat below the resting height, so
# sitting on the ground never ended an episode and the floor worked as a
# safety net. Now any ground contact, or dropping to within 10 cm of
# touching down (the F450 stands at z = 0.155 m on its legs), is an
# immediate crash.
GROUND_CRASH_ALTITUDE_M = DEFAULT_12IN.rest_height_m + 0.10

EPISODE_STEPS = 500          # 10s at 50 Hz
UNSTABLE_WINDOW = 10         # consecutive unstable steps before crash
OOB_WINDOW = 5               # consecutive OOB steps before crash


def rotor_rpm(thrust_frac, max_rpm):
    """Rotor speed for a per-motor thrust fraction (thrust ~ rpm^2)."""
    return np.sqrt(np.maximum(np.asarray(thrust_frac), 0.0)) * max_rpm


# Yaw sign of each motor's reaction torque on the frame, in the order of
# BaseAviary._physics (motors 0 and 2 spin CW, 1 and 3 CCW).
ROTOR_YAW_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])


class _HoverAviary12in(BaseAviary):
    """BaseAviary subclass loading the 12" URDF instead of cf2x.

    One sim.step() is one physics substep (ctrl_freq == pyb_freq); the
    env runs the flight controller between substeps. The action is the
    per-motor thrust as a fraction of nominal max thrust; it may exceed
    1 for stronger-than-nominal motors.
    """

    ACTION_BUFFER_SIZE = 1

    def __init__(self, initial_xyzs, initial_rpys, gui=False):
        super().__init__(
            drone_model=CF12IN,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=Physics.PYB_GND_DRAG_DW,
            pyb_freq=PYB_FREQ,
            ctrl_freq=PYB_FREQ,
            gui=gui,
            record=False,
            user_debug_gui=False,
        )

    def _actionSpace(self):
        return Box(low=0.0, high=np.inf, shape=(1, 4), dtype=np.float32)

    def _observationSpace(self):
        return Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def _computeObs(self):
        return None

    def _preprocessAction(self, action):
        return rotor_rpm(action, self.MAX_RPM).reshape(1, 4)

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}


class Hover12inEnv(gym.Env):
    """Gymnasium env — 12" quad hover with domain randomization.

    Action:  policy body-rate + collective thrust in [-1, 1]^4
    Obs:     Dict{'vector': float32[22]}  (position-relative, sim-agnostic)
    Reward:  aerial/spec/reward.py
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gui=False, dr_stage="1c"):
        super().__init__()
        self.render_mode = render_mode
        self._gui = bool(gui) or (render_mode == "human")
        # Curriculum stage picks the domain-randomization range table.
        # "1a" = easy, "1b" = medium, "1c" = full target ranges.
        self._dr_ranges = get_dr_stage(dr_stage)
        self._dr_stage = dr_stage

        self.action_space = Box(
            low=-1.0, high=1.0,
            shape=(ACTION_DIM,), dtype=np.float32,
        )
        self.observation_space = DictSpace({
            "vector": Box(low=-np.inf, high=np.inf,
                          shape=(OBS_DIM,), dtype=np.float32),
        })

        self._sim = None
        self._nominal_drag = None
        self._fc = None
        self._reward_state = RewardState.initial(action_dim=ACTION_DIM)
        self._elapsed = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._unstable_window = []
        self._oob_window = []
        self._rng = np.random.default_rng()
        self._episode_params = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._elapsed = 0
        self._unstable_window.clear()
        self._oob_window.clear()
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._reward_state = RewardState.initial(action_dim=ACTION_DIM)

        # 1. Episode-wide randomization.
        ep = self._episode_params = sample_episode_params(
            self._rng, self._dr_ranges)
        r = self._dr_ranges

        # 2. Flight controller tuned for the nominal airframe; gain_scale
        #    models the tuning mismatch with the actual vehicle.
        self._fc = ArduPilotRateController(
            DEFAULT_12IN, PYB_FREQ, gain_scale=ep["rate_kp_scale"])
        motor_tau = DEFAULT_12IN.motor_time_constant_s * ep["motor_tau_scale"]
        self._motor_alpha = 1.0 - np.exp(-PHYS_DT / motor_tau)

        # 3. Latency: actions are queued with a release time; observations
        #    are read from a delayed-state buffer.
        self._t = 0.0
        self._pending = deque()
        self._applied = np.zeros(ACTION_DIM, dtype=np.float32)
        self._obs_delay_steps = int(round(ep["obs_latency_s"] * PYB_FREQ))

        # 4. Slow random processes start from their stationary spread.
        self._gust_sigma = r.gust_force_sigma_N * GUST_AXIS_SCALE
        self._gust = self._rng.normal(0.0, self._gust_sigma)
        self._pos_drift = self._rng.normal(0.0, r.pos_drift_sigma_m, 3)
        self._vel_drift = self._rng.normal(0.0, r.vel_drift_sigma_m_s, 3)

        # 5. Sample init pose (wide: ±2m xy, ±1m z, ±30° rpy).
        init_pos, init_rpy_rad = sample_init_pose(
            self._rng, TARGET_POSITION, self._dr_ranges)

        # 6. Create or repose the sim.
        if self._sim is not None and not self._gui:
            self._sim.close()
            self._sim = None
        if self._sim is None:
            self._sim = _HoverAviary12in(
                initial_xyzs=init_pos.reshape(1, 3),
                initial_rpys=init_rpy_rad.reshape(1, 3),
                gui=self._gui,
            )
            self._nominal_drag = self._sim.DRAG_COEFF.copy()
        else:
            self._sim.INIT_XYZS = init_pos.reshape(1, 3)
            self._sim.INIT_RPYS = init_rpy_rad.reshape(1, 3)
        self._sim.reset()

        # 7. Body + aero randomization.
        self._apply_body_randomization()

        # 8. Motors start at the thrust that holds this vehicle's weight
        #    (the drone is handed over in flight).
        mass = DEFAULT_12IN.mass_kg * ep["mass_scale"]
        hover = mass * 9.81 / (
            4.0 * DEFAULT_12IN.max_thrust_per_motor_N
            * float(np.mean(ep["motor_scales"])))
        self._motor = np.full(
            4, np.clip(hover, ArduPilotRateController.MIN_MOTOR, 1.0),
            dtype=np.float32)
        self._rotor_w = rotor_rpm(self._motor * ep["motor_scales"],
                                  self._sim.MAX_RPM) * (2.0 * np.pi / 60.0)
        # ArduPilot keeps rate I-terms across mode changes, so at handover
        # the FC has already trimmed out CG offset and motor imbalance.
        self._fc.integ = self._hover_trim(hover)

        self._history = deque(maxlen=HISTORY_LEN)
        self._history.append(self._snapshot())
        obs_vec, _ = self._observe(self._last_action)
        return {"vector": obs_vec}, {}

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        ep = self._episode_params
        r = self._dr_ranges
        self._pending.append((self._t + ep["control_latency_s"], raw))

        min_z = np.inf
        for _ in range(SUBSTEPS):
            while self._pending and self._pending[0][0] <= self._t + 1e-9:
                self._applied = self._pending.popleft()[1]
            body_rates, thrust_norm = scale_action(self._applied)
            gyro_meas = (self._body_gyro() + ep["gyro_bias"]
                         + self._rng.normal(0.0, r.gyro_noise_sigma_rad_s, 3))
            cmd = self._fc.step(body_rates, gyro_meas, thrust_norm)
            self._motor += self._motor_alpha * (cmd - self._motor)
            thrust_frac = self._motor * ep["motor_scales"]
            self._apply_disturbances(thrust_frac, self._rotor_yaw_torque(thrust_frac))
            self._sim.step(thrust_frac.reshape(1, 4))
            self._t += PHYS_DT
            self._history.append(self._snapshot())
            min_z = min(min_z, float(self._sim.pos[0][2]))
        self._advance_slow_processes()

        self._elapsed += 1
        obs_vec, obs_gyro = self._observe(raw)
        terminated, truncated = self._check_done(obs_gyro, min_z)

        R = _quat_to_R(self._sim.quat[0])
        reward, info = step_reward(
            pos_err=self._sim.pos[0] - TARGET_POSITION,
            vel=self._sim.vel[0],
            gyro=R.T @ self._sim.ang_v[0],
            up_axis=upright_from_R(R),
            altitude=self._sim.pos[0][2],
            action=raw,
            state=self._reward_state,
        )
        if terminated:
            reward = CRASH_PENALTY

        self._last_action = raw
        return ({"vector": obs_vec},
                float(reward),
                bool(terminated),
                bool(truncated),
                info)

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # --- Randomization helpers -----------------------------------------
    def _apply_body_randomization(self):
        """Scale mass, inertia and drag by the per-episode multipliers."""
        ep = self._episode_params
        base_mass = DEFAULT_12IN.mass_kg * ep["mass_scale"]
        inertia = DEFAULT_12IN.inertia_diag_kg_m2 * ep["inertia_scale"]
        p.changeDynamics(
            int(self._sim.DRONE_IDS[0]), -1,
            mass=float(base_mass),
            localInertiaDiagonal=inertia.tolist(),
            physicsClientId=self._sim.CLIENT,
        )
        self._sim.DRAG_COEFF = self._nominal_drag * ep["drag_scale"]

    def _rotor_yaw_torque(self, thrust_frac):
        """Reaction torque of the rotors changing speed, J * dw/dt.

        gym-pybullet-drones only applies the steady drag torque
        (km * rpm^2); a motor that speeds its prop up also pushes the
        frame the other way. Same sign as the drag torque of that motor.
        """
        w = rotor_rpm(thrust_frac, self._sim.MAX_RPM) * (2.0 * np.pi / 60.0)
        dw = (w - self._rotor_w) / PHYS_DT
        self._rotor_w = w
        return float(DEFAULT_12IN.rotor_inertia_kg_m2 * ROTOR_YAW_SIGN @ dw)

    def _apply_disturbances(self, thrust_frac, rotor_yaw_torque=0.0):
        """Wind + gust at the center of mass, the CG-offset torque and
        the rotor reaction torque.

        PyBullet clears external forces after every stepSimulation, so
        this runs before each substep. In WORLD_FRAME, posObj is a world
        position: it must be the drone's position, or the force also
        applies a torque about the world origin.
        """
        ep = self._episode_params
        drone_id = int(self._sim.DRONE_IDS[0])
        force = ep["wind_force_world"] + self._gust
        p.applyExternalForce(
            drone_id, -1,
            forceObj=force.tolist(),
            posObj=self._sim.pos[0].tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self._sim.CLIENT,
        )
        # Thrust acts at the geometric center; with the CG displaced by
        # (rx, ry) that adds torque -(r x F) about the CG, body frame.
        total_thrust = (float(thrust_frac.sum())
                        * DEFAULT_12IN.max_thrust_per_motor_N)
        rx, ry = ep["cg_offset_m"]
        p.applyExternalTorque(
            drone_id, COM_LINK,
            torqueObj=[-ry * total_thrust, rx * total_thrust, rotor_yaw_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self._sim.CLIENT,
        )

    def _hover_trim(self, collective):
        """Mixer input that zeroes body torque at hover for this vehicle.

        Solves A·diag(s)·(c + M·tau)·T_max + tau_cg = 0 for tau, where A
        maps per-motor thrust to body torque and s are the motor scales.
        """
        ep = self._episode_params
        a = DEFAULT_12IN.arm_length_m / np.sqrt(2.0)
        x = np.array([a, -a, -a, a])
        y = np.array([-a, -a, a, a])
        spin = np.array([-1.0, 1.0, -1.0, 1.0])
        A = np.stack([y, -x, DEFAULT_12IN.torque_coeff_per_thrust * spin])
        s = np.asarray(ep["motor_scales"], dtype=np.float64)
        t_max = DEFAULT_12IN.max_thrust_per_motor_N
        rx, ry = ep["cg_offset_m"]
        weight = DEFAULT_12IN.mass_kg * ep["mass_scale"] * 9.81
        tau_cg = np.array([-ry * weight, rx * weight, 0.0])
        lhs = t_max * A @ np.diag(s) @ MIXER
        rhs = -(t_max * collective * A @ s + tau_cg)
        tau = np.linalg.solve(lhs, rhs)
        return np.clip(tau, -ArduPilotRateController.I_MAX,
                       ArduPilotRateController.I_MAX).astype(np.float32)

    def _advance_slow_processes(self):
        """Gusts and estimator drift: Ornstein-Uhlenbeck, per control step."""
        r = self._dr_ranges
        self._gust = self._ou(self._gust, self._gust_sigma, r.gust_tau_s)
        self._pos_drift = self._ou(
            self._pos_drift, r.pos_drift_sigma_m, r.pos_drift_tau_s)
        self._vel_drift = self._ou(
            self._vel_drift, r.vel_drift_sigma_m_s, r.vel_drift_tau_s)

    def _ou(self, x, sigma, tau):
        a = np.exp(-CTRL_DT / tau)
        return a * x + sigma * np.sqrt(1.0 - a * a) * self._rng.normal(size=3)

    # --- State access ---------------------------------------------------
    def _body_gyro(self):
        # PyBullet reports angular velocity in the world frame; a real
        # gyro measures it in the body frame.
        return _quat_to_R(self._sim.quat[0]).T @ self._sim.ang_v[0]

    def _snapshot(self):
        s = self._sim
        return (s.pos[0].copy(), s.vel[0].copy(),
                s.quat[0].copy(), s.ang_v[0].copy())

    # --- Obs / termination ---------------------------------------------
    def _observe(self, last_action):
        """Delayed state + estimator errors, as the policy would see it."""
        ep = self._episode_params
        r = self._dr_ranges
        delay = min(self._obs_delay_steps, len(self._history) - 1)
        pos, vel, quat, ang_v = self._history[-1 - delay]
        R_true = _quat_to_R(quat)

        att_err = ep["att_bias_rpy"] + self._rng.normal(
            0.0, r.att_noise_sigma_rad, 3)
        R_est = _rpy_to_R(att_err) @ R_true
        pos_est = pos + self._pos_drift + self._rng.normal(
            0.0, r.pos_noise_sigma_m, 3)
        vel_est = vel + self._vel_drift + self._rng.normal(
            0.0, r.vel_noise_sigma_m_s, 3)
        gyro_est = (R_true.T @ ang_v + ep["gyro_bias"]
                    + self._rng.normal(0.0, r.gyro_noise_sigma_rad_s, 3))

        obs_vec = get_obs_vector(
            pos=pos_est, vel=vel_est, R=R_est, gyro=gyro_est,
            last_action=last_action, target=TARGET_POSITION,
        )
        return obs_vec.astype(np.float32), gyro_est

    def _check_done(self, obs_gyro, min_z):
        # Stability uses the observed gyro, matching what the policy sees.
        up = upright_from_R(_quat_to_R(self._sim.quat[0]))
        unstable = is_unstable(up, obs_gyro)
        self._unstable_window.append(unstable)
        if len(self._unstable_window) > UNSTABLE_WINDOW:
            self._unstable_window.pop(0)
        unstable_too_long = (
            len(self._unstable_window) == UNSTABLE_WINDOW
            and all(self._unstable_window)
        )

        # Bounds check uses TRUE position (safety, not noisy).
        pos_true = self._sim.pos[0]
        oob = (
            abs(pos_true[0]) > BOUND_XY
            or abs(pos_true[1]) > BOUND_XY
            or pos_true[2] > BOUND_Z_MAX
        )
        self._oob_window.append(oob)
        if len(self._oob_window) > OOB_WINDOW:
            self._oob_window.pop(0)
        oob_too_long = (
            len(self._oob_window) == OOB_WINDOW
            and all(self._oob_window)
        )

        ground_hit = (min_z < GROUND_CRASH_ALTITUDE_M
                      or self._ground_contact())

        terminated = bool(unstable_too_long or oob_too_long or ground_hit)
        truncated = bool(self._elapsed >= EPISODE_STEPS)
        return terminated, truncated

    def _ground_contact(self):
        return len(p.getContactPoints(
            bodyA=int(self._sim.DRONE_IDS[0]),
            bodyB=int(self._sim.PLANE_ID),
            physicsClientId=self._sim.CLIENT,
        )) > 0


def _quat_to_R(quat_xyzw):
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)


def _rpy_to_R(rpy):
    """Rotation matrix from roll/pitch/yaw (ZYX convention)."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)


register(
    id="Aerial-hover-12in-v0",
    entry_point=Hover12inEnv,
    max_episode_steps=EPISODE_STEPS,
)
