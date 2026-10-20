"""Physics checks for the hover env on a nominal (un-randomized) vehicle.

Covers bugs fixed 2026-10-16: world-frame gyro, wind applied at the world
origin (spurious torque) and on 1 of 5 substeps, a 50 Hz P-only rate loop
with a ~0.8 s time constant, and latency rounded to whole control steps.

Run:  .venv-pybullet/bin/python -m pytest aerial/tests/test_hover_12in_env.py -q
"""
import dataclasses

import numpy as np
import pybullet as p
import pytest

from aerial.envs import hover_12in_env as H
from aerial.configs.airframe_12in import DEFAULT_12IN
from aerial.controllers.rate_controller import ArduPilotRateController
from aerial.spec.interface import MAX_BODY_RATE_RPY

HOVER = DEFAULT_12IN.hover_thrust_normalized


def nominal_env(yaw=0.0, z=3.0, **episode_overrides):
    env = H.Hover12inEnv(dr_stage="1a")
    env._dr_ranges = dataclasses.replace(
        env._dr_ranges, gust_force_sigma_N=0.0, pos_drift_sigma_m=0.0,
        vel_drift_sigma_m_s=0.0, gyro_noise_sigma_rad_s=0.0,
        att_noise_sigma_rad=0.0, pos_noise_sigma_m=0.0,
        vel_noise_sigma_m_s=0.0)
    env.reset(seed=0)
    ep = env._episode_params
    ep.update(mass_scale=1.0, inertia_scale=1.0, motor_tau_scale=1.0,
              rate_kp_scale=1.0, drag_scale=1.0, control_latency_s=0.0,
              obs_latency_s=0.0, motor_scales=np.ones(4, np.float32),
              cg_offset_m=np.zeros(2, np.float32),
              gyro_bias=np.zeros(3, np.float32),
              att_bias_rpy=np.zeros(3, np.float32),
              wind_force_world=np.zeros(3, np.float32))
    ep.update(episode_overrides)
    env._gust[:] = 0
    env._pos_drift[:] = 0
    env._vel_drift[:] = 0
    env._fc = ArduPilotRateController(DEFAULT_12IN, H.PYB_FREQ, 1.0)
    env._motor_alpha = 1 - np.exp(
        -H.PHYS_DT / DEFAULT_12IN.motor_time_constant_s)
    env._obs_delay_steps = int(round(ep["obs_latency_s"] * H.PYB_FREQ))
    env._apply_body_randomization()
    sim = env._sim
    p.resetBasePositionAndOrientation(
        int(sim.DRONE_IDS[0]), [0, 0, z],
        p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=sim.CLIENT)
    p.resetBaseVelocity(int(sim.DRONE_IDS[0]), [0, 0, 0], [0, 0, 0],
                        physicsClientId=sim.CLIENT)
    sim._updateAndStoreKinematicInformation()
    env._motor[:] = HOVER
    env._rotor_w = H.rotor_rpm(env._motor, sim.MAX_RPM) * (2 * np.pi / 60)
    env._fc.integ = env._hover_trim(HOVER)
    env._history.clear()
    env._history.append(env._snapshot())
    return env


def act(roll=0.0, thrust=HOVER, yaw=0.0):
    return np.array([roll / MAX_BODY_RATE_RPY, 0.0, yaw / MAX_BODY_RATE_RPY,
                     2 * thrust - 1], np.float32)


@pytest.fixture
def make_env():
    envs = []

    def make(**kw):
        envs.append(nominal_env(**kw))
        return envs[-1]
    yield make
    for env in envs:
        env.close()


def test_rate_loop_tracks_within_150ms(make_env):
    env = make_env()
    rates = []
    for _ in range(10):
        env.step(act(roll=1.0))
        rates.append(env._body_gyro()[0])
    t63 = next(i for i, r in enumerate(rates) if r >= 0.632)
    assert (t63 + 1) * 20 <= 150, rates
    assert max(rates) < 1.4, rates


def test_gyro_is_body_frame(make_env):
    env = make_env(yaw=np.pi / 2)
    for _ in range(5):
        obs, *_ = env.step(act(roll=0.5))
    gyro = obs["vector"][15:18]
    assert gyro[0] > 0.3 and abs(gyro[1]) < 0.05, gyro


def test_wind_pushes_without_torque(make_env):
    env = make_env(wind_force_world=np.array([1.0, 0.0, 0.0], np.float32))
    for _ in range(25):
        env.step(act())
    accel = env._sim.vel[0][0] / 0.5
    tilt = np.degrees(np.arccos(H._quat_to_R(env._sim.quat[0])[2, 2]))
    # 1 N / mass, less ~6% from rotor drag over the 0.5 s.
    expected = 1.0 / DEFAULT_12IN.mass_kg
    assert 0.85 * expected < accel < 1.0 * expected, accel
    assert tilt < 0.5, tilt


def test_action_latency_resolution(make_env):
    env = make_env(control_latency_s=0.030)
    env.step(act())
    env._pending.append((env._t + 0.030, act(roll=3.0)))
    for k in range(40):
        while env._pending and env._pending[0][0] <= env._t + 1e-9:
            env._applied = env._pending.popleft()[1]
        rates, thrust = H.scale_action(env._applied)
        cmd = env._fc.step(rates, env._body_gyro(), thrust)
        if np.abs(cmd - cmd.mean()).max() > 1e-3:
            break
        env._sim.step(cmd.reshape(1, 4))
        env._t += H.PHYS_DT
    assert k * H.PHYS_DT == pytest.approx(0.030)


def test_hover_trim_cancels_cg_offset_and_imbalance(make_env):
    scales = np.array([1.05, 0.95, 1.0, 0.97], np.float32)
    env = make_env(cg_offset_m=np.array([0.015, 0.0], np.float32),
                   motor_scales=scales)
    hover = HOVER / float(np.mean(scales))
    env._fc.integ = env._hover_trim(hover)
    for _ in range(100):
        env.step(act(thrust=hover))
    tilt = np.degrees(np.arccos(H._quat_to_R(env._sim.quat[0])[2, 2]))
    assert tilt < 2.0, tilt


def test_ground_is_a_crash(make_env):
    env = make_env(z=0.6)
    for _ in range(100):
        _, reward, terminated, _, _ = env.step(act(thrust=0.0))
        if terminated:
            break
    assert terminated and reward == -10.0


def test_rests_on_its_legs(make_env):
    env = make_env(z=0.3)
    sim = env._sim
    for _ in range(400):
        sim.step(np.zeros((1, 4)))
    assert sim.pos[0][2] == pytest.approx(DEFAULT_12IN.rest_height_m, abs=0.01)
    assert env._ground_contact()
    assert H.GROUND_CRASH_ALTITUDE_M > DEFAULT_12IN.rest_height_m + 0.05


def test_rotor_speedup_twists_frame_like_drag_torque(make_env):
    """Speeding up motors 1 and 3 pushes the frame the same way their
    drag torque does in BaseAviary._physics (+z), by J * dw/dt."""
    env = make_env()
    w0 = env._rotor_w.copy()
    frac = np.full(4, HOVER)
    frac[[1, 3]] *= 1.1
    tau = env._rotor_yaw_torque(frac)
    w1 = H.rotor_rpm(frac, env._sim.MAX_RPM) * (2 * np.pi / 60)
    expected = DEFAULT_12IN.rotor_inertia_kg_m2 * (
        (w1 - w0)[1] + (w1 - w0)[3]) / H.PHYS_DT
    assert tau > 0
    assert tau == pytest.approx(expected, rel=1e-4)
    assert env._rotor_yaw_torque(frac) == pytest.approx(0.0, abs=1e-9)


def test_yaw_rate_loop_tracks(make_env):
    env = make_env()
    rates = []
    for _ in range(25):
        env.step(act(yaw=1.0))
        rates.append(env._body_gyro()[2])
    t63 = next(i for i, r in enumerate(rates) if r >= 0.632)
    assert (t63 + 1) * 20 <= 300, rates
    assert max(rates) < 1.3, rates
