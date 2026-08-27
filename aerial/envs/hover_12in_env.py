"""Gymnasium env for the 12" airframe with full domain randomization.

Phase 1 production env. Reuses everything from Phase 0
(aerial/spec/, aerial/controllers/, PyBullet BaseAviary) but swaps the
Crazyflie URDF for the 12" quad URDF (aerial/assets/cf12in.urdf) and
applies per-episode + per-step domain randomization.

Registered as Gymnasium env `Aerial-hover-12in-v0`.

Differences from hover_pybullet_env.py:
  * 12" airframe (0.8 kg, 0.15m arm) instead of Crazyflie (30g, 40mm)
  * Full domain randomization from configs/domain_randomization.py
  * Wider init randomization (±2m xy, ±1m z, ±30° rpy)
  * Larger workspace bounds
  * Sharpened reward already lives in aerial/spec/reward.py (shared)

URDF loading gotcha:
  gym-pybullet-drones hardcodes URDF path via pkg_resources against its
  own package's assets/ directory. We work around this by (a) using a
  duck-typed DroneModel-like object with .value = 'cf12in', and
  (b) copying our URDF into the gym_pybullet_drones assets dir at module
  import time. The copy is idempotent — safe to re-run.
"""
import os
import shutil

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, Dict as DictSpace

import pkg_resources
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import Physics

from aerial.spec.interface import (
    OBS_DIM, ACTION_DIM, STEP_RATE_HZ, TARGET_POSITION,
    scale_action, get_obs_vector, upright_from_R,
)
from aerial.spec.reward import (
    RewardState, step_reward, CRASH_PENALTY, is_unstable,
)
from aerial.controllers.rate_controller import BodyRateController, DEFAULT_KP
from aerial.configs.airframe_12in import DEFAULT_12IN
from aerial.configs.domain_randomization import (
    get_stage as get_dr_stage,
    sample_episode_params,
    sample_init_pose,
    add_sensor_noise,
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
PYB_FREQ = 250
PYB_STEPS_PER_CTRL = PYB_FREQ // STEP_RATE_HZ
assert PYB_FREQ % STEP_RATE_HZ == 0

# --- Workspace / termination ------------------------------------------
# 12" quad flies faster + we spawn it up to 2m from target, so bounds
# need to be wider than Phase 0's cramped Crazyflie workspace.
BOUND_XY = 10.0
BOUND_Z_MIN = 0.05
BOUND_Z_MAX = 8.0

EPISODE_STEPS = 500          # 10s at 50 Hz
UNSTABLE_WINDOW = 10         # consecutive unstable steps before crash
OOB_WINDOW = 5               # consecutive OOB steps before crash


class _HoverAviary12in(BaseAviary):
    """BaseAviary subclass loading the 12" URDF instead of cf2x."""

    ACTION_BUFFER_SIZE = 1

    def __init__(self, initial_xyzs, initial_rpys, gui=False):
        super().__init__(
            drone_model=CF12IN,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=Physics.PYB,
            pyb_freq=PYB_FREQ,
            ctrl_freq=STEP_RATE_HZ,
            gui=gui,
            record=False,
            user_debug_gui=False,
        )

    def _actionSpace(self):
        return Box(low=0.0, high=1.0, shape=(1, 4), dtype=np.float32)

    def _observationSpace(self):
        return Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def _computeObs(self):
        return self._getDroneStateVector(0)

    def _preprocessAction(self, action):
        cmd = np.clip(np.asarray(action), 0.0, 1.0).reshape(1, 4)
        rpm = np.sqrt(cmd) * self.MAX_RPM
        return rpm

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
    Reward:  aerial/spec/reward.py (sharpened for Phase 1)
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
        self._controller = BodyRateController()
        self._reward_state = RewardState.initial(action_dim=ACTION_DIM)
        self._elapsed = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._unstable_window = []
        self._oob_window = []
        self._rng = np.random.default_rng()
        # Per-episode randomization state.
        self._episode_params = None
        # Per-step control latency buffer (holds the raw action received
        # `latency_steps` calls ago; released to physics this step).
        self._action_buffer = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._elapsed = 0
        self._unstable_window.clear()
        self._oob_window.clear()
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._reward_state = RewardState.initial(action_dim=ACTION_DIM)
        self._controller.reset()

        # 1. Draw episode-wide randomization: mass/motor/rate-gain/etc.
        self._episode_params = sample_episode_params(self._rng, self._dr_ranges)

        # 2. Apply rate-controller gain jitter.
        self._controller.kp = (
            DEFAULT_KP.copy() * self._episode_params["rate_kp_scale"]
        )

        # 3. Compute how many control steps of latency this episode.
        latency_steps = int(round(
            self._episode_params["control_latency_s"] * STEP_RATE_HZ
        ))
        self._action_buffer = [
            np.zeros(ACTION_DIM, dtype=np.float32)
            for _ in range(latency_steps)
        ]

        # 4. Sample init pose (wide: ±2m xy, ±1m z, ±30° rpy).
        init_pos, init_rpy_rad = sample_init_pose(
            self._rng, TARGET_POSITION, self._dr_ranges)

        # 5. Create or repose the sim.
        if self._sim is None:
            self._sim = _HoverAviary12in(
                initial_xyzs=init_pos.reshape(1, 3),
                initial_rpys=init_rpy_rad.reshape(1, 3),
                gui=self._gui,
            )
        else:
            if not self._gui:
                self._sim.close()
                self._sim = _HoverAviary12in(
                    initial_xyzs=init_pos.reshape(1, 3),
                    initial_rpys=init_rpy_rad.reshape(1, 3),
                    gui=self._gui,
                )
            else:
                self._sim.INIT_XYZS = init_pos.reshape(1, 3)
                self._sim.INIT_RPYS = init_rpy_rad.reshape(1, 3)
        self._sim.reset()

        # 6. Apply per-episode mass/inertia scaling to the drone body.
        self._apply_body_randomization()

        obs_vec, _, _ = self._build_obs_and_done(self._last_action)
        return {"vector": obs_vec}, {}

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)

        # Latency: policy action goes into buffer, physics reads the
        # `latency`-old action off the buffer.
        if self._action_buffer:
            self._action_buffer.append(raw.copy())
            applied = self._action_buffer.pop(0)
        else:
            applied = raw

        body_rates, thrust_norm = scale_action(applied)

        # Motor thrust scaling: reduce effective thrust command by the
        # episode's thrust_scale so the policy has to compensate for
        # weak/strong motors.
        thrust_norm = float(np.clip(
            thrust_norm * self._episode_params["motor_thrust_scale"],
            0.0, 1.0,
        ))

        # Rate loop reads the (noisy) gyro from the sim.
        state = self._sim._getDroneStateVector(0)
        gyro_true = state[13:16]
        # Noisy gyro for rate loop (mimics real IMU that the FC would see).
        gyro_noisy = gyro_true + self._rng.normal(
            0.0, self._dr_ranges.gyro_noise_sigma_rad_s, size=3
        ).astype(gyro_true.dtype)

        dt = 1.0 / STEP_RATE_HZ
        motor_cmd = self._controller.step(body_rates, gyro_noisy,
                                          thrust_norm, dt)
        self._sim.step(motor_cmd.reshape(1, 4))

        # Apply constant per-episode wind gust as an external force on
        # base_link (in world frame).
        self._apply_wind()

        self._elapsed += 1
        obs_vec, terminated, truncated = self._build_obs_and_done(raw)
        reward, info = step_reward(
            pos_err=obs_vec[0:3],
            vel=obs_vec[3:6],
            gyro=obs_vec[15:18],
            up_axis=upright_from_R(obs_vec[6:15].reshape(3, 3)),
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
        """Scale drone mass + inertia by the per-episode multipliers.

        PyBullet exposes mass/inertia via changeDynamics on the base link.
        """
        import pybullet as p
        drone_id = int(self._sim.DRONE_IDS[0])
        client = self._sim.CLIENT

        base_mass = DEFAULT_12IN.mass_kg * self._episode_params["mass_scale"]
        inertia = (DEFAULT_12IN.inertia_diag_kg_m2
                   * self._episode_params["inertia_scale"])
        p.changeDynamics(
            drone_id, -1,
            mass=float(base_mass),
            localInertiaDiagonal=inertia.tolist(),
            physicsClientId=client,
        )
        # Note: motor_tau randomization is only meaningful if the sim
        # uses a first-order motor model. gym-pybullet-drones' default
        # PYB physics applies instantaneous thrust, so motor_tau is
        # currently a no-op. Kept in the DR spec for when we swap to
        # Physics.DYN or a custom motor model.

    def _apply_wind(self):
        """Apply constant wind gust force to the drone body (world frame)."""
        wind = self._episode_params.get("wind_force_world")
        if wind is None or float(np.linalg.norm(wind)) < 1e-6:
            return
        import pybullet as p
        drone_id = int(self._sim.DRONE_IDS[0])
        p.applyExternalForce(
            objectUniqueId=drone_id,
            linkIndex=-1,
            forceObj=wind.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self._sim.CLIENT,
        )

    # --- Obs / termination ---------------------------------------------
    def _build_obs_and_done(self, last_action):
        state = self._sim._getDroneStateVector(0)
        pos_true = state[0:3]
        quat_xyzw = state[3:7]
        vel_true = state[10:13]
        gyro_true = state[13:16]
        R = _quat_to_R(quat_xyzw)

        # Sensor noise on the observations the POLICY sees (separate from
        # the noisy gyro fed to the rate loop above — different sensor).
        noisy = add_sensor_noise(
            self._rng,
            gyro=gyro_true,
            accel=None,
            pos=pos_true,
            vel=vel_true,
            ranges=self._dr_ranges,
        )

        obs_vec = get_obs_vector(
            pos=noisy["pos"], vel=noisy["vel"], R=R, gyro=noisy["gyro"],
            last_action=last_action, target=TARGET_POSITION,
        )

        # Stability check uses the noisy gyro (matches what the policy
        # would see, so the reward and observation are consistent).
        up = upright_from_R(R)
        unstable = is_unstable(up, noisy["gyro"])
        self._unstable_window.append(unstable)
        if len(self._unstable_window) > UNSTABLE_WINDOW:
            self._unstable_window.pop(0)
        unstable_too_long = (
            len(self._unstable_window) == UNSTABLE_WINDOW
            and all(self._unstable_window)
        )

        # Bounds check uses TRUE position (safety, not noisy).
        oob = (
            abs(pos_true[0]) > BOUND_XY
            or abs(pos_true[1]) > BOUND_XY
            or pos_true[2] < BOUND_Z_MIN
            or pos_true[2] > BOUND_Z_MAX
        )
        self._oob_window.append(oob)
        if len(self._oob_window) > OOB_WINDOW:
            self._oob_window.pop(0)
        oob_too_long = (
            len(self._oob_window) == OOB_WINDOW
            and all(self._oob_window)
        )

        terminated = bool(unstable_too_long or oob_too_long)
        truncated = bool(self._elapsed >= EPISODE_STEPS)
        return obs_vec, terminated, truncated


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


register(
    id="Aerial-hover-12in-v0",
    entry_point=Hover12inEnv,
    max_episode_steps=EPISODE_STEPS,
)
