"""Gymnasium env adapter for gym-pybullet-drones HoverAviary.

Phase 0 deliverable: a known-good DreamerV3 training pipeline for
quadrotor hover, used to validate the spec/reward/controller stack
before committing infra effort to Aerial Gym.

Registered as Gymnasium env `Aerial-hover-pybullet-v0` so DreamerV3 can
load it via `embodied.envs.from_gym:FromGym`, mirroring the pattern
used by steam/liftoff/hover_env_uinput.py.

Pipeline per step:
    policy action ([-1, 1]^4) ->
      scale_action -> (body_rates rad/s, thrust [0,1]) ->
      BodyRateController -> motor commands [0, 1]^4 ->
      PyBullet physics ->
      canonical obs ([22]) ->
      reward.step_reward ->
      Gymnasium (obs, reward, terminated, truncated, info)

The PyBullet drone is a Crazyflie 2.X (cf2x) because that's the
calibrated model. The 12" airframe target is NOT modeled here — Phase 0
validates the pipeline, not the specific airframe.
"""
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, Dict as DictSpace

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from aerial.spec.interface import (
    OBS_DIM, ACTION_DIM, STEP_RATE_HZ, TARGET_POSITION,
    scale_action, get_obs_vector, upright_from_R,
)
from aerial.spec.reward import (
    RewardState, step_reward, CRASH_PENALTY, is_unstable,
)
from aerial.controllers.rate_controller import BodyRateController


# PyBullet physics rate must be an integer multiple of the control rate.
# 250 Hz physics / 50 Hz control = 5 substeps per env step. Close to
# stock 240 Hz, well within PyBullet's stable range for cf2x.
PYB_FREQ = 250
PYB_STEPS_PER_CTRL = PYB_FREQ // STEP_RATE_HZ
assert PYB_FREQ % STEP_RATE_HZ == 0

# Episode termination bounds. Crazyflie has a small workspace; this is
# tight enough to terminate fast on bad policies but loose enough to
# allow ~1m hover with slop.
BOUND_XY = 3.0
BOUND_Z_MIN = 0.05
BOUND_Z_MAX = 3.0

EPISODE_STEPS = 500          # 10s at 50 Hz
UNSTABLE_WINDOW = 10         # consecutive unstable steps before crash
OOB_WINDOW = 5               # consecutive OOB steps before crash

# Per-episode init randomization (minimal — production randomization
# happens in Aerial Gym).
INIT_POS_NOISE = 0.3         # +- meters in xy
INIT_Z_NOISE = 0.2           # +- meters in z (around target z)
INIT_RPY_NOISE_DEG = 15.0


class _HoverAviaryPhysics(BaseAviary):
    """Minimal BaseAviary subclass: handles physics, exposes raw motor
    command input. All RL logic lives in HoverPybulletEnv below.
    """

    ACTION_BUFFER_SIZE = 1  # unused but BaseRLAviary parent reads it

    def __init__(self, initial_xyzs, initial_rpys, gui=False):
        super().__init__(
            drone_model=DroneModel.CF2X,
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
        # action: (1, 4) in [0, 1]. Convert to RPM via sqrt mapping so
        # thrust is linear in motor_cmd:
        #   rpm = sqrt(motor_cmd) * MAX_RPM
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


class HoverPybulletEnv(gym.Env):
    """Gymnasium env. Body-rate + collective-thrust action interface.

    Observation: Dict with key 'vector' (22-dim) so DreamerV3's
    multi-modal encoder treats it like a vector input. Single-key dict
    keeps the door open for adding 'image' in Phase 2 without changing
    the spec.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gui=False):
        super().__init__()
        self.render_mode = render_mode
        # GUI is forced on when render_mode='human', otherwise follows
        # the explicit gui kwarg. Off by default for training throughput.
        self._gui = bool(gui) or (render_mode == "human")
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

        init_pos = TARGET_POSITION + self._rng.uniform(
            low=[-INIT_POS_NOISE, -INIT_POS_NOISE, -INIT_Z_NOISE],
            high=[INIT_POS_NOISE, INIT_POS_NOISE, INIT_Z_NOISE],
        ).astype(np.float32)
        init_pos[2] = max(init_pos[2], BOUND_Z_MIN + 0.05)

        init_rpy_rad = np.deg2rad(self._rng.uniform(
            low=-INIT_RPY_NOISE_DEG, high=INIT_RPY_NOISE_DEG, size=3
        )).astype(np.float32)

        # Recreating the sim on every reset is the cheap path when there
        # is no GUI (headless training). With the GUI on it makes the
        # window flash off and on between episodes — instead, create the
        # sim once and just update INIT_XYZS / INIT_RPYS so its reset()
        # places the drone at the new pose.
        if self._sim is None:
            self._sim = _HoverAviaryPhysics(
                initial_xyzs=init_pos.reshape(1, 3),
                initial_rpys=init_rpy_rad.reshape(1, 3),
                gui=self._gui,
            )
        else:
            if not self._gui:
                self._sim.close()
                self._sim = _HoverAviaryPhysics(
                    initial_xyzs=init_pos.reshape(1, 3),
                    initial_rpys=init_rpy_rad.reshape(1, 3),
                    gui=self._gui,
                )
            else:
                self._sim.INIT_XYZS = init_pos.reshape(1, 3)
                self._sim.INIT_RPYS = init_rpy_rad.reshape(1, 3)
        self._sim.reset()

        obs_vec, _, _ = self._build_obs_and_done(self._last_action)
        return {"vector": obs_vec}, {}

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        body_rates, thrust_norm = scale_action(raw)

        state = self._sim._getDroneStateVector(0)
        gyro = state[13:16]

        dt = 1.0 / STEP_RATE_HZ
        motor_cmd = self._controller.step(body_rates, gyro, thrust_norm, dt)
        self._sim.step(motor_cmd.reshape(1, 4))

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

    def _build_obs_and_done(self, last_action):
        state = self._sim._getDroneStateVector(0)
        pos = state[0:3]
        quat_xyzw = state[3:7]
        vel = state[10:13]
        gyro = state[13:16]
        R = _quat_to_R(quat_xyzw)

        obs_vec = get_obs_vector(
            pos=pos, vel=vel, R=R, gyro=gyro,
            last_action=last_action, target=TARGET_POSITION,
        )

        up = upright_from_R(R)
        unstable = is_unstable(up, gyro)
        self._unstable_window.append(unstable)
        if len(self._unstable_window) > UNSTABLE_WINDOW:
            self._unstable_window.pop(0)
        unstable_too_long = (
            len(self._unstable_window) == UNSTABLE_WINDOW
            and all(self._unstable_window)
        )

        oob = (
            abs(pos[0]) > BOUND_XY
            or abs(pos[1]) > BOUND_XY
            or pos[2] < BOUND_Z_MIN
            or pos[2] > BOUND_Z_MAX
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
    """Convert a (x, y, z, w) quaternion to a body->world rotation
    matrix. PyBullet's quaternion convention is xyzw.
    """
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)
    return R


register(
    id="Aerial-hover-pybullet-v0",
    entry_point=HoverPybulletEnv,
    max_episode_steps=EPISODE_STEPS,
)
