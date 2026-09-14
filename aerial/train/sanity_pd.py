"""Hand-coded position-PD sanity check.

Drives the HoverPybulletEnv with a simple position PD controller (not a
neural net) to confirm the env wrapper, rate controller, and reward
function are all sensible. If this hovers around the target and earns
mean per-step reward in the 0.7-1.0 range, the pipeline is good.

If this fails, the bug is in the env/spec/controller — not in
DreamerV3. Always run this before kicking off a real training job.

Run:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/sanity_pd.py
"""
import sys

import numpy as np
import gymnasium as gym

import aerial.envs.hover_pybullet_env  # noqa: F401  (registers env)
from aerial.spec.interface import TARGET_POSITION, MAX_BODY_RATE_RPY


# Position-PD gains. These control how aggressively the PD pushes the
# drone toward the target by setting body-rate setpoints. Yaw is left at
# 0 since hover doesn't need heading control.
KP_POS = np.array([1.5, 1.5, 0.6], dtype=np.float32)  # x, y, z
KD_VEL = np.array([0.9, 0.9, 0.4], dtype=np.float32)

# Crazyflie hover thrust normalized (motor_cmd at hover). Since thrust =
# motor_cmd * t2w * mg in our action mapping, hover_motor_cmd = 1/t2w.
# t2w = 2.25 for cf2x, so hover_thrust_normalized ~= 0.444.
HOVER_THRUST = 1.0 / 2.25


def pd_action(obs_vec):
    """Map obs to policy action [-1, 1]^4 using a position PD law.

    obs_vec[0:3] = position error (drone - target), world frame.
    obs_vec[3:6] = velocity, world frame.
    obs_vec[6:15] = R (body->world) flattened row-major.
    """
    pos_err = obs_vec[0:3]
    vel = obs_vec[3:6]
    R = obs_vec[6:15].reshape(3, 3)

    # Desired world-frame acceleration to drive position error to zero.
    a_desired = -KP_POS * pos_err - KD_VEL * vel
    # Map XY accel to desired roll/pitch in body frame.
    # Tiny-angle approximation: a_y ~= g * roll, a_x ~= -g * pitch.
    # That gives us target roll/pitch; we use the difference between
    # current and target as the body-rate command (rough cascading).
    target_roll = np.clip(a_desired[1] / 9.81, -0.5, 0.5)
    target_pitch = np.clip(-a_desired[0] / 9.81, -0.5, 0.5)
    # Current roll/pitch from R (small-angle): roll ~= R[2,1], pitch ~= -R[2,0]
    current_roll = R[2, 1]
    current_pitch = -R[2, 0]
    body_rate_roll = np.clip(2.0 * (target_roll - current_roll),
                             -MAX_BODY_RATE_RPY, MAX_BODY_RATE_RPY)
    body_rate_pitch = np.clip(2.0 * (target_pitch - current_pitch),
                              -MAX_BODY_RATE_RPY, MAX_BODY_RATE_RPY)

    # Vertical thrust: hover + feed-forward from z-acceleration.
    thrust_norm = HOVER_THRUST + 0.05 * a_desired[2]
    thrust_norm = float(np.clip(thrust_norm, 0.0, 1.0))

    # Convert physical units back to policy [-1, 1] scale.
    action = np.array([
        body_rate_roll / MAX_BODY_RATE_RPY,
        body_rate_pitch / MAX_BODY_RATE_RPY,
        0.0,
        2.0 * thrust_norm - 1.0,
    ], dtype=np.float32)
    return action


def main():
    env = gym.make("Aerial-hover-pybullet-v0")
    n_episodes = 3
    target = np.asarray(TARGET_POSITION)

    print(f"Target hover position: {target}")
    print(f"Hover thrust normalized: {HOVER_THRUST:.3f}")
    print()

    all_means = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        rewards = []
        pos_errs = []
        for t in range(500):
            action = pd_action(obs["vector"])
            obs, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            pos_errs.append(np.linalg.norm(obs["vector"][0:3]))
            if term or trunc:
                break
        mean_r = float(np.mean(rewards))
        mean_err = float(np.mean(pos_errs[-50:]))  # last 1s
        all_means.append(mean_r)
        print(f"ep {ep}: steps={len(rewards)}, "
              f"mean_reward/step={mean_r:.3f}, "
              f"final_pos_err(last 50)={mean_err:.3f}m, "
              f"term={term} trunc={trunc}")

    env.close()
    overall = float(np.mean(all_means))
    print(f"\nOverall mean per-step reward: {overall:.3f}")
    print("PD sanity:", "PASS" if overall > 0.5 else "FAIL")
    return 0 if overall > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
