"""Run a policy bundle in the terminal and print what it does.

Two modes, both need only numpy (+ rknn-toolkit-lite2 for --backend rknn):

  probe (default): feed hand-built states (hovering, displaced, tilted,
      spinning, ...) for a few steps each and print the actions. The
      previous action is fed back into the obs; the rest is held fixed.
  sim: fly the policy closed-loop in a numpy rigid-body model of the
      Aerial-hover-12in env (F450 airframe, the env's ArduPilot-style rate
      loop at 400 Hz, motor lag, latency, drag; no wind, noise or DR) from
      a few start states, print the trajectories, then score random
      Stage-1c starts. A sanity check, not the PyBullet training sim.

    python -m aerial.deploy.npu.run_policy --model aerial/deploy/models/<bundle>
    python -m aerial.deploy.npu.run_policy --model <bundle> --mode sim --backend rknn

Conventions (aerial/spec/interface.py; body x fwd, y left, z up): action
= [roll rate, pitch rate, yaw rate] x 6 rad/s + throttle (a3 + 1) / 2.
+roll banks right, +pitch is nose-DOWN (rotation about +y), +yaw is CCW.
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from aerial.configs.airframe_12in import DEFAULT_12IN as AIRFRAME  # noqa: E402
from aerial.configs.domain_randomization import (  # noqa: E402
    STAGE_1C, sample_init_pose)
from aerial.controllers.rate_controller import (  # noqa: E402
    MIXER, ArduPilotRateController)
from aerial.spec.interface import (  # noqa: E402
    MAX_BODY_RATE_RPY, STEP_RATE_HZ, TARGET_POSITION, get_obs_vector)

f32 = np.float32
G = 9.81


def rot(axis, angle):
    """Body->world rotation about a body axis ('x', 'y', 'z'), radians."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = {'x': (1, 2), 'y': (2, 0), 'z': (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def make_policy(backend, model_dir, sampling, seed=0):
    if backend == 'numpy':
        from aerial.deploy.npu.reference import NumpyPolicyInference
        return NumpyPolicyInference(model_dir, sampling=sampling, seed=seed)
    if backend == 'rknn':
        from aerial.deploy.npu.rknn_policy import RKNNPolicyInference
        return RKNNPolicyInference(model_dir, sampling=sampling, seed=seed)
    raise ValueError(backend)


def fmt_action(a):
    rates = a[:3] * MAX_BODY_RATE_RPY
    thr = float(np.clip((a[3] + 1) / 2, 0, 1))
    return (f'raw [{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f} {a[3]:+.3f}]  '
            f'rates roll {rates[0]:+5.2f} pitch {rates[1]:+5.2f} yaw {rates[2]:+5.2f} rad/s  '
            f'throttle {thr:.3f}')


# name -> (pos rel. target, vel, R, gyro, what a sensible policy does)
PROBES = {
    'hover':      ([0, 0, 0], [0, 0, 0], np.eye(3), [0, 0, 0],
                   'rates ~0, throttle ~hover (1/TWR)'),
    'low_1m':     ([0, 0, -1], [0, 0, 0], np.eye(3), [0, 0, 0],
                   'throttle above hover'),
    'high_1m':    ([0, 0, 1], [0, 0, 0], np.eye(3), [0, 0, 0],
                   'throttle below hover'),
    'ahead_1m':   ([1, 0, 0], [0, 0, 0], np.eye(3), [0, 0, 0],
                   'pitch rate < 0 (nose up, fly back -x)'),
    'left_1m':    ([0, 1, 0], [0, 0, 0], np.eye(3), [0, 0, 0],
                   'roll rate > 0 (bank right, fly -y)'),
    'falling':    ([0, 0, 0], [0, 0, -1.5], np.eye(3), [0, 0, 0],
                   'throttle well above hover'),
    'rolled_20':  ([0, 0, 0], [0, 0, 0], rot('x', np.radians(20)), [0, 0, 0],
                   'roll rate < 0 (level out)'),
    'pitched_20': ([0, 0, 0], [0, 0, 0], rot('y', np.radians(20)), [0, 0, 0],
                   'pitch rate < 0 (level out)'),
    'yawed_90':   ([0, 0, 0], [0, 0, 0], rot('z', np.radians(90)), [0, 0, 0],
                   'little roll/pitch; yaw may or may not be corrected'),
    'spin_roll':  ([0, 0, 0], [0, 0, 0], np.eye(3), [3, 0, 0],
                   'roll rate < 0 (damp the spin)'),
}


def run_probes(policy, steps, twr):
    print(f'hover throttle for TWR {twr}: {1 / twr:.3f}  (raw a3 {2 / twr - 1:+.3f})\n')
    for name, (pos, vel, R, gyro, expect) in PROBES.items():
        policy.reset()
        prev = np.zeros(4, f32)
        print(f'== {name}: expect {expect}')
        for i in range(steps):
            obs = get_obs_vector(np.add(pos, [0, 0, 1]), vel, R, gyro, prev)
            prev = policy.step(obs, is_first=(i == 0))
            if i in (0, steps // 2, steps - 1):
                print(f'  step {i:2d}  {fmt_action(prev)}')
        print()


# Same physics rate and crash height as aerial/envs/hover_12in_env.py
# (not imported: it pulls in PyBullet).
PHYS_HZ = 400
CRASH_ALTITUDE_M = AIRFRAME.rest_height_m + 0.10
# Motor positions (FR, BR, BL, FL), matching MIXER rows and the env's
# _hover_trim().
_LEVER = AIRFRAME.arm_length_m / np.sqrt(2)
MOTOR_XY = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]]) * _LEVER
DRAG = np.array([AIRFRAME.drag_per_s_xy, AIRFRAME.drag_per_s_xy,
                 AIRFRAME.drag_per_s_z])


def sim_episode(policy, pos, vel, R, steps, ctrl_latency=0.02, obs_latency=0.01):
    """Fly one episode; drone handed over in flight with motors at hover."""
    sub = PHYS_HZ // STEP_RATE_HZ
    dt = 1 / PHYS_HZ
    target = np.asarray(TARGET_POSITION, float)
    pos, vel, R = np.array(pos, float) + target, np.array(vel, float), np.array(R, float)
    omega = np.zeros(3)
    fc = ArduPilotRateController(AIRFRAME, PHYS_HZ)
    hover = AIRFRAME.hover_thrust_normalized
    motor = np.full(4, hover)
    m_alpha = 1 - np.exp(-dt / AIRFRAME.motor_time_constant_s)
    I = np.asarray(AIRFRAME.inertia_diag_kg_m2, float)
    pending, applied = [], np.array([0, 0, 0, 2 * hover - 1])
    history = [(pos.copy(), vel.copy(), R.copy(), omega.copy())]
    obs_delay = int(round(obs_latency * PHYS_HZ))
    prev = np.zeros(4, f32)
    policy.reset()
    log, t = [], 0.0
    for i in range(steps):
        p_o, v_o, R_o, w_o = history[max(0, len(history) - 1 - obs_delay)]
        act = policy.step(get_obs_vector(p_o, v_o, R_o, w_o, prev), is_first=(i == 0))
        pending.append((t + ctrl_latency, act.copy()))
        for _ in range(sub):
            while pending and pending[0][0] <= t + 1e-9:
                applied = pending.pop(0)[1]
            rates = np.asarray(applied[:3], float) * MAX_BODY_RATE_RPY
            cmd = fc.step(rates, omega, (applied[3] + 1) / 2)
            motor += m_alpha * (cmd - motor)
            T = motor * AIRFRAME.max_thrust_per_motor_N
            tau = np.array([MOTOR_XY[:, 1] @ T, -MOTOR_XY[:, 0] @ T,
                            AIRFRAME.torque_coeff_per_thrust * (MIXER[:, 2] @ T)])
            omega += (tau - np.cross(omega, I * omega)) / I * dt
            w = omega * dt
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            R = R @ (np.eye(3) + K + K @ K / 2)
            u, _, vt = np.linalg.svd(R)
            R = u @ vt
            acc = R[:, 2] * T.sum() / AIRFRAME.mass_kg - [0, 0, G] - DRAG * vel
            vel = vel + acc * dt
            pos = pos + vel * dt
            t += dt
            history.append((pos.copy(), vel.copy(), R.copy(), omega.copy()))
        del history[:-64]
        prev = act
        log.append((i, pos - target, vel, R[2, 2], act))
        if pos[2] < CRASH_ALTITUDE_M or R[2, 2] < 0:
            break
    return log


SIM_STARTS = {
    'at_target':   ([0, 0, 0], [0, 0, 0], np.eye(3)),
    'offset_xyz':  ([1, -1, -0.5], [0, 0, 0], np.eye(3)),
    'moving_tilt': ([0, 0, 0], [1, 0, 0], rot('x', np.radians(15))),
    'far_tilted':  ([-2, 1.5, 0.8], [0, 0, 0], rot('y', np.radians(-25)) @ rot('x', np.radians(20))),
}


def run_sim(policy, seconds, episodes, seed=0):
    steps = int(seconds * STEP_RATE_HZ)
    print('F450 sim: airframe_12in + ArduPilotRateController at 400 Hz, motor lag,\n'
          '20 ms action / 10 ms obs latency. No wind, sensor noise or DR, and not\n'
          'PyBullet -- a sanity check, not the training sim.\n')
    for name, (pos, vel, R) in SIM_STARTS.items():
        log = sim_episode(policy, pos, vel, R, steps)
        print(f'== {name}  (start pos err {pos}, vel {vel})')
        print('   t[s]   pos err x/y/z [m]       |err|   speed  upright  throttle')
        for i, err, v, up, a in log:
            if i % 25 == 0 or i == len(log) - 1:
                print(f'  {i / STEP_RATE_HZ:5.2f}  [{err[0]:+5.2f} {err[1]:+5.2f} {err[2]:+5.2f}]'
                      f'  {np.linalg.norm(err):6.2f}  {np.linalg.norm(v):5.2f}'
                      f'  {up:7.3f}  {(a[3] + 1) / 2:7.3f}')
        _, err, _, up, _ = log[-1]
        crashed = len(log) < steps
        print(f'  -> {"CRASHED" if crashed else "survived"} {len(log) / STEP_RATE_HZ:.1f}s, '
              f'final |err| {np.linalg.norm(err):.2f} m\n')
    if episodes:
        # Stage 1c init pose and latency ranges, sampled as in training
        # (altitude clamped to MIN_INIT_ALTITUDE_M).
        rng = np.random.default_rng(seed)
        final, crashes = [], 0
        for _ in range(episodes):
            init_pos, (r, p_, y) = sample_init_pose(rng, TARGET_POSITION, STAGE_1C)
            R = rot('z', y) @ rot('y', p_) @ rot('x', r)
            log = sim_episode(policy, init_pos - TARGET_POSITION, [0, 0, 0], R, steps,
                              ctrl_latency=rng.uniform(*STAGE_1C.control_latency_s),
                              obs_latency=rng.uniform(*STAGE_1C.obs_latency_s))
            crashes += len(log) < steps
            if len(log) == steps:
                final.append(np.linalg.norm(log[-1][1]))
        final = np.array(final)
        print(f'== {episodes} random Stage-1c starts, {seconds:.0f} s each: {crashes} crashed; '
              f'survivors final |err| median {np.median(final):.2f} m, '
              f'{(final < 0.3).sum()}/{episodes} end within 0.3 m')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True, help='bundle dir with policy_weights.npz')
    ap.add_argument('--backend', default='numpy', choices=('numpy', 'rknn'))
    ap.add_argument('--mode', default='probe', choices=('probe', 'sim'))
    # mean_action is what eval/deploy use for the F450 release.
    ap.add_argument('--sampling', default='mean_action',
                    choices=('stochastic', 'mean_action', 'deterministic'))
    ap.add_argument('--steps', type=int, default=25, help='probe: steps per state')
    ap.add_argument('--seconds', type=float, default=8.0, help='sim: episode length')
    ap.add_argument('--episodes', type=int, default=20, help='sim: random Stage-1c starts')
    ap.add_argument('--twr', type=float, default=AIRFRAME.thrust_to_weight)
    args = ap.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    policy = make_policy(args.backend, args.model, args.sampling)
    print(f'backend {args.backend}, sampling {args.sampling}\n')
    if args.mode == 'probe':
        run_probes(policy, args.steps, args.twr)
    else:
        run_sim(policy, args.seconds, args.episodes)


if __name__ == '__main__':
    main()
