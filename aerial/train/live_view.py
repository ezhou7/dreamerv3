"""Watch an exported bundle fly the 12in env in the PyBullet window, in
real time, with mean actions.

Draws the target (gold cross, 20 cm ring) and the flight path (grey for
the first 5 s, blue for the last 5 s), with time, distance and thrust
command as text. Seeds are 1000 + i, as in eval_bundle.py, so a seed
from an eval or plot can be replayed. Close the window or Ctrl-C to stop.

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/live_view.py \
        aerial/deploy/models/<bundle> --stage 1c --seed 1022 --episodes 1 \
        --slow 10
"""
import argparse
import time
import numpy as np
import pybullet as p
from aerial.deploy.inference import PolicyInference
from aerial.envs import hover_12in_env as H
from aerial.spec.interface import scale_action

TGT = np.asarray(H.TARGET_POSITION, dtype=float)


def draw_target(cid):
    for ax in np.eye(3):
        p.addUserDebugLine(TGT - 0.08 * ax, TGT + 0.08 * ax, [1, 0.8, 0], 3,
                           physicsClientId=cid)
    ang = np.linspace(0, 2 * np.pi, 33)
    ring = TGT + 0.2 * np.stack([np.cos(ang), np.sin(ang), 0 * ang], 1)
    for a, b in zip(ring[:-1], ring[1:]):
        p.addUserDebugLine(a, b, [0.2, 0.2, 0.2], 1, physicsClientId=cid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", help="dir from aerial/deploy/export_model.py")
    ap.add_argument("--stage", default="1c")
    ap.add_argument("--seed", type=int, default=1000, help="first seed")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--slow", type=float, default=1.0,
                    help="slow-motion factor (10 plays 1 s over 10 s)")
    ap.add_argument("--cuda-device", default="1",
                    help="GPU index for the policy; '' runs on CPU")
    args = ap.parse_args()
    stage, seed0, n, slow = args.stage, args.seed, args.episodes, args.slow
    bundle = args.bundle
    policy = PolicyInference(bundle, cuda_device=args.cuda_device or None)
    env = H.Hover12inEnv(dr_stage=stage, gui=True)
    for ep in range(n):
        seed = seed0 + ep
        obs, _ = env.reset(seed=seed)
        cid = env._sim.CLIENT
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
        p.resetDebugVisualizerCamera(3.2, 40, -25, TGT.tolist(), physicsClientId=cid)
        p.removeAllUserDebugItems(physicsClientId=cid)
        draw_target(cid)
        text = p.addUserDebugText("", TGT + [0, 0, 0.9], [0, 0, 0], 1.3,
                                  physicsClientId=cid)
        policy.reset()
        prev = np.array(env._sim.pos[0]); term = False; tail = []
        t0 = time.perf_counter()
        for t in range(H.EPISODE_STEPS):
            act = policy.step(obs["vector"], is_first=(t == 0))
            obs, r, term, trunc, _ = env.step(act)
            pos = np.array(env._sim.pos[0])
            d = float(np.linalg.norm(pos - TGT))
            if t >= H.EPISODE_STEPS - 250:
                tail.append(d)
            if t % 2 == 0:
                color = [0.1, 0.4, 1.0] if t >= H.EPISODE_STEPS - 250 else [0.6, 0.6, 0.6]
                p.addUserDebugLine(prev, pos, color, 2, physicsClientId=cid)
                prev = pos
            if t % 5 == 0:
                _, thrust = scale_action(act)
                text = p.addUserDebugText(
                    f"seed {seed}  t {t * H.CTRL_DT:4.1f}s  dist {d * 100:5.1f} cm"
                    f"  thrust cmd {thrust:.2f}",
                    TGT + [0, 0, 0.9], [0, 0, 0], 1.3,
                    replaceItemUniqueId=text, physicsClientId=cid)
            # real time (x slow): one control step = 20 ms
            lag = t0 + (t + 1) * H.CTRL_DT * slow - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            if term or trunc:
                break
        msg = (f"seed {seed}: CRASH at {(t + 1) * H.CTRL_DT:.2f} s" if term else
               f"seed {seed}: survived, last-5-s median {np.median(tail) * 100:.0f} cm")
        print(msg, flush=True)
        p.addUserDebugText(msg, TGT + [0, 0, 0.9], [0.8, 0, 0] if term else [0, 0.5, 0],
                           1.4, replaceItemUniqueId=text, physicsClientId=cid)
        time.sleep(2.0 if ep < n - 1 else 6.0)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, p.error):
        pass
