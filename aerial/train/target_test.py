"""Can the hover policy fly to a new target? Target jumps vs a moving
"carrot" target, using an exported bundle with mean actions.

The policy only sees position error (obs = pos - target), so moving the
target looks to it like being knocked off target. It was trained with
spawns up to ~2.2 m (xy) and 1 m (z) off target.

Hovers 3 s at the home target, then changes the target and flies 14 s.
Reports survival, time to get within 30 cm, max tilt and speed, and the
median distance over the last 3 s.

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/target_test.py \
        aerial/deploy/models/<bundle> --stage 1b --episodes 10
"""
import argparse
import numpy as np
from aerial.deploy.inference import PolicyInference
from aerial.envs import hover_12in_env as H

HOME = np.array(H.TARGET_POSITION, dtype=np.float32)
DT = H.CTRL_DT


def fly(policy, env, seed, goal, carrot_speed=None, settle_s=3.0, fly_s=14.0):
    H.TARGET_POSITION = HOME.copy()
    obs, _ = env.reset(seed=seed)
    policy.reset()
    n_settle, n_fly = int(settle_s / DT), int(fly_s / DT)
    goal = HOME + np.asarray(goal, np.float32)
    target = HOME.copy()
    reach, tilt, speed, tail = None, 0.0, 0.0, []
    for t in range(n_settle + n_fly):
        if t >= n_settle:
            if carrot_speed is None:
                target = goal
            else:
                step = goal - target
                d = np.linalg.norm(step)
                if d > 0:
                    target = target + step * min(1.0, carrot_speed * DT / d)
            H.TARGET_POSITION = target.astype(np.float32)
        a = policy.step(obs["vector"], is_first=(t == 0))
        obs, r, term, trunc, _ = env.step(a)
        if term:
            return dict(ok=False, t=(t - n_settle) * DT)
        if t >= n_settle:
            pos = np.array(env._sim.pos[0])
            dg = np.linalg.norm(pos - goal)
            if reach is None and dg < 0.3:
                reach = (t - n_settle) * DT
            R = H._quat_to_R(env._sim.quat[0])
            tilt = max(tilt, np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            speed = max(speed, np.linalg.norm(env._sim.vel[0]))
            if t >= n_settle + n_fly - int(3.0 / DT):
                tail.append(dg)
    return dict(ok=True, reach=reach, tilt=tilt, speed=speed,
                hold=float(np.median(tail)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", help="dir from aerial/deploy/export_model.py")
    ap.add_argument("--stage", default="1b")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--cuda-device", default="1",
                    help="GPU index for the policy; '' runs on CPU")
    args = ap.parse_args()
    policy = PolicyInference(args.bundle, cuda_device=args.cuda_device or None)
    stage, n = args.stage, args.episodes
    env = H.Hover12inEnv(dr_stage=stage)
    cases = [("jump 1 m", (1, 0, 0), None), ("jump 2 m", (1.4, 1.4, 0), None),
             ("jump 3 m", (0, -3, 0), None), ("jump 5 m", (-5, 0, 0), None),
             ("jump 8 m", (5.7, 5.7, 0), None),
             ("jump up 2 m", (0, 0, 2), None), ("jump up 4 m", (0, 0, 4), None),
             ("jump down 0.5 m", (0, 0, -0.5), None),
             ("carrot 8 m @ 1 m/s", (5.7, 5.7, 0), 1.0),
             ("carrot 8 m @ 2 m/s", (5.7, 5.7, 0), 2.0),
             ("carrot up 4 m @ 1 m/s", (0, 0, 4), 1.0)]
    for name, goal, cs in cases:
        res = [fly(policy, env, 2000 + i, goal, cs) for i in range(n)]
        ok = [r for r in res if r["ok"]]
        reach = [r["reach"] for r in ok if r["reach"] is not None]
        f = lambda k: np.median([r[k] for r in ok]) if ok else float("nan")
        print(f"{name:24s} survived {len(ok)}/{n}  reached(<30 cm) {len(reach)}/{n}"
              f"  median time {np.median(reach) if reach else float('nan'):5.1f} s"
              f"  max tilt {f('tilt'):4.0f} deg  max speed {f('speed'):4.1f} m/s"
              f"  final hold {f('hold'):.2f} m", flush=True)
    env.close()


if __name__ == "__main__":
    main()
