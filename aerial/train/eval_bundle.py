"""Hover quality and command chatter of an exported bundle on the 12in env.

Flies the policy with mean actions (what deployment uses) on fixed seeds
and reports, over the last 5 s of each surviving episode, from the true
sim state (not the noisy observation):

  survived   episodes that did not crash in 10 s
  dist       median distance to target (median over episodes)
  wobble     RMS distance from the episode's own average position
  speed      RMS speed
  low        episodes that went below 0.40 m after the first second

and the command chatter:

  da2        mean sum_k (a_t - a_{t-1})^2 in policy units [-1, 1]
             (what agent.imag_loss smooth_coef penalizes)
  rate       RMS step-to-step change of the body-rate command, rad/s
  thr        mean step-to-step change of the thrust command (0-1)
  motor      mean change per control step of the motor outputs (0-1)

Seeds are 1000 + i, shared with scripted_hover_pid.py and
plot_trajectories.py so results line up episode by episode.

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/eval_bundle.py \\
        aerial/deploy/models/<bundle> --stages 1b 1c --episodes 30
"""
import argparse
import dataclasses

import numpy as np

from aerial.deploy.inference import PolicyInference
from aerial.envs import hover_12in_env as H
from aerial.spec.interface import scale_action

TAIL = H.EPISODE_STEPS - 250   # last 5 s at 50 Hz


def run(policy, stage, episodes, seed0=1000, sensor_noise=True):
    env = H.Hover12inEnv(dr_stage=stage)
    if not sensor_noise:
        env._dr_ranges = dataclasses.replace(
            env._dr_ranges, gyro_noise_sigma_rad_s=0.0,
            accel_noise_sigma_m_s2=0.0, pos_noise_sigma_m=0.0,
            vel_noise_sigma_m_s=0.0, att_noise_sigma_rad=0.0)
    target = np.asarray(H.TARGET_POSITION)
    stats = {k: [] for k in
             ("dist", "wobble", "speed", "da2", "rate", "thr", "motor")}
    survived = low = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        policy.reset()
        acts, motors, pos, vel = [], [], [], []
        term = False
        for t in range(H.EPISODE_STEPS):
            a = policy.step(obs["vector"], is_first=(t == 0))
            obs, _, term, trunc, _ = env.step(a)
            acts.append(a)
            motors.append(env._motor.copy())
            pos.append(np.array(env._sim.pos[0]))
            vel.append(np.array(env._sim.vel[0]))
            if term or trunc:
                break
        pos = np.array(pos)
        if pos[50:, 2].min(initial=np.inf) < 0.40:
            low += 1
        if term:
            continue
        survived += 1
        p, v = pos[TAIL:], np.array(vel)[TAIL:]
        a, m = np.array(acts)[TAIL:], np.array(motors)[TAIL:]
        rates = np.array([scale_action(x)[0] for x in a])
        stats["dist"].append(np.median(np.linalg.norm(p - target, axis=1)))
        stats["wobble"].append(
            np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))))
        stats["speed"].append(np.sqrt(np.mean(np.sum(v ** 2, 1))))
        stats["da2"].append(np.mean(np.sum(np.diff(a, axis=0) ** 2, 1)))
        stats["rate"].append(
            np.sqrt(np.mean(np.sum(np.diff(rates, axis=0) ** 2, 1))))
        stats["thr"].append(np.mean(np.abs(np.diff((a[:, 3] + 1) / 2))))
        stats["motor"].append(np.mean(np.abs(np.diff(m, axis=0))))
    env.close()
    med = {k: float(np.median(x)) if x else float("nan")
           for k, x in stats.items()}
    p90 = float(np.percentile(stats["dist"], 90)) if stats["dist"] else float("nan")
    print(f"stage {stage}{'' if sensor_noise else ' (no sensor noise)'}: "
          f"survived {survived}/{episodes}  dist {med['dist']:.3f} m "
          f"(p90 {p90:.3f})  wobble {med['wobble']:.3f} m  "
          f"speed {med['speed']:.3f} m/s  low {low}", flush=True)
    print(f"  chatter: da2 {med['da2']:.3f}  rate {med['rate']:.2f} rad/s  "
          f"thr {med['thr']:.3f}  motor {med['motor']:.4f}", flush=True)
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", help="dir from aerial/deploy/export_model.py")
    ap.add_argument("--stages", nargs="+", default=["1b", "1c"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--no-sensor-noise", action="store_true",
                    help="zero the white sensor noise (bias/drift stay)")
    ap.add_argument("--cuda-device", default="1",
                    help="GPU index for the policy; '' runs on CPU")
    args = ap.parse_args()
    policy = PolicyInference(args.bundle, cuda_device=args.cuda_device or None)
    for stage in args.stages:
        run(policy, stage, args.episodes, args.seed0,
            sensor_noise=not args.no_sensor_noise)


if __name__ == "__main__":
    main()
