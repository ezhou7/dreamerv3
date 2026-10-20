"""Hand-written PID hover on the 12in env's observations (estimated state).

Feasibility check for the domain randomization: if this survives every
stage, crashes by a learned policy are not physics limits. Reports
survival and the median true distance to target over the last 5 s of
each surviving episode. Seeds are 1000 + i, as in eval_bundle.py.

Cascade: position PID -> desired acceleration -> desired body z axis
-> body-rate command; thrust is tilt-compensated. Body rates are capped
at 2.5 rad/s because the env ends an episode after 10 steps above
3 rad/s.

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/scripted_hover_pid.py \\
        --stages 1a 1a5 1b 1c --episodes 30
"""
import argparse

import numpy as np

from aerial.configs.airframe_12in import DEFAULT_12IN as A
from aerial.envs import hover_12in_env as H
from aerial.spec.interface import MAX_BODY_RATE_RPY

KP = np.array([1.5, 1.5, 4.0])
KD = np.array([1.8, 1.8, 3.0])
KI = np.array([0.3, 0.3, 1.5])
K_ATT = 8.0
K_YAW = 2.0
MAX_RATE = 2.5


def run(stage, episodes, seed0=1000):
    env = H.Hover12inEnv(dr_stage=stage)
    survived, dists, thrusts = 0, [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        integ = np.zeros(3)
        d_ep = []
        term = False
        for t in range(H.EPISODE_STEPS):
            o = obs["vector"]
            e, v = o[0:3], o[3:6]
            R = o[6:15].reshape(3, 3)
            integ = np.clip(integ + e * H.CTRL_DT, -1.0, 1.0)
            acc = -KP * e - KD * v - KI * integ
            acc[:2] = np.clip(acc[:2], -4, 4)
            acc[2] += 9.81
            b3 = R[:, 2]
            # Hold the vertical component of thrust when tilted.
            thrust = (A.mass_kg * acc[2] / max(b3[2], 0.5)
                      / (4 * A.max_thrust_per_motor_N))
            z_des = acc / np.linalg.norm(acc)
            err = R.T @ np.cross(b3, z_des)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            rates = np.array([K_ATT * err[0], K_ATT * err[1], -K_YAW * yaw])
            rates *= min(1.0, MAX_RATE / max(np.linalg.norm(rates), 1e-6))
            act = np.concatenate([np.clip(rates / MAX_BODY_RATE_RPY, -1, 1),
                                  [np.clip(2 * thrust - 1, -1, 1)]])
            obs, _, term, trunc, _ = env.step(act.astype(np.float32))
            thrusts.append(thrust)
            if t >= H.EPISODE_STEPS - 250:
                d_ep.append(np.linalg.norm(
                    env._sim.pos[0] - H.TARGET_POSITION))
            if term or trunc:
                break
        if not term:
            survived += 1
            dists.append(np.median(d_ep))
    env.close()
    med = np.median(dists) if dists else float("nan")
    p90 = np.percentile(dists, 90) if dists else float("nan")
    print(f"stage {stage}: survived {survived}/{episodes}  median dist "
          f"{med:.3f} m  p90 {p90:.3f} m  mean thrust cmd "
          f"{np.mean(thrusts):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stages", nargs="+", default=["1a", "1a5", "1b", "1c"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    for stage in args.stages:
        run(stage, args.episodes, args.seed0)


if __name__ == "__main__":
    main()
