"""Flight-path plot of an exported bundle on the 12in env (mean actions).

Four panels: top-down paths (zoom on the last 5 s), altitude, distance
to target (true and as the policy sees it), and the policy's commands
in a typical episode. Seeds are 1000 + i, as in eval_bundle.py.

Usage:
    PYTHONPATH=. .venv-pybullet/bin/python aerial/train/plot_trajectories.py \
        aerial/deploy/models/<bundle> --stage 1c --out traj.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aerial.deploy.inference import PolicyInference
from aerial.envs import hover_12in_env as H
from aerial.spec.interface import scale_action

DT = H.CTRL_DT
TGT = np.asarray(H.TARGET_POSITION)


def fly(policy, stage, episodes, seed0=1000):
    env = H.Hover12inEnv(dr_stage=stage)
    eps = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        policy.reset()
        pos, est, rates, thrust = [], [], [], []
        crashed = False
        for t in range(H.EPISODE_STEPS):
            est.append(TGT + obs["vector"][0:3])
            act = policy.step(obs["vector"], is_first=(t == 0))
            br, th = scale_action(act)
            rates.append(br); thrust.append(th)
            obs, r, term, trunc, _ = env.step(act)
            pos.append(np.array(env._sim.pos[0]))
            if term or trunc:
                crashed = bool(term)
                break
        eps.append(dict(pos=np.array(pos), est=np.array(est),
                        rates=np.array(rates), thrust=np.array(thrust),
                        crashed=crashed, seed=seed0 + ep))
        print(f"seed {seed0 + ep}: {'CRASH at %.2f s' % (len(pos) * DT) if crashed else 'ok'}",
              flush=True)
    env.close()
    return eps


def plot(eps, out, title):
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    tail = H.EPISODE_STEPS - 250
    ok = [e for e in eps if not e["crashed"]]
    med_tail = [np.median(np.linalg.norm(e["pos"][tail:] - TGT, axis=1)) for e in ok]
    typical = ok[int(np.argsort(med_tail)[len(ok) // 2])]

    # (a) top-down view
    a = ax[0, 0]
    for e in eps:
        p = e["pos"]
        if e["crashed"]:
            a.plot(p[:, 0], p[:, 1], color="tab:red", lw=1.2)
            a.plot(p[-1, 0], p[-1, 1], "x", color="tab:red", ms=9, mew=2)
        else:
            a.plot(p[:tail, 0], p[:tail, 1], color="0.75", lw=0.7)
            a.plot(p[tail:, 0], p[tail:, 1], color="tab:blue", lw=0.9, alpha=0.8)
        a.plot(p[0, 0], p[0, 1], "o", color="0.5", ms=3)
    a.plot(*TGT[:2], "*", color="gold", ms=18, mec="k", zorder=5)
    a.set_aspect("equal"); a.grid(alpha=0.3)
    a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    a.set_title("Top-down view: grey = first 5 s (recovery from spawn), "
                "blue = last 5 s,\nred = crash, dot = spawn, star = target",
                fontsize=10)

    # inset: zoom on last 5 s
    ins = a.inset_axes([0.62, 0.62, 0.36, 0.36])
    for e in ok:
        p = e["pos"][tail:]
        ins.plot(p[:, 0], p[:, 1], color="tab:blue", lw=0.6, alpha=0.7)
    ins.plot(*TGT[:2], "*", color="gold", ms=12, mec="k", zorder=5)
    circ = plt.Circle(TGT[:2], 0.2, fill=False, ls="--", color="k", lw=0.8)
    ins.add_patch(circ)
    ins.set_xlim(-0.5, 0.5); ins.set_ylim(-0.5, 0.5); ins.set_aspect("equal")
    ins.set_title("last 5 s, zoom (dashed = 20 cm)", fontsize=8)
    ins.tick_params(labelsize=7)

    # (b) altitude
    a = ax[0, 1]
    for e in eps:
        t = np.arange(len(e["pos"])) * DT
        a.plot(t, e["pos"][:, 2], color="tab:red" if e["crashed"] else "tab:blue",
               lw=1.2 if e["crashed"] else 0.7, alpha=1 if e["crashed"] else 0.6)
    a.axhline(TGT[2], color="gold", lw=2, label="target 1.0 m")
    a.axhline(H.GROUND_CRASH_ALTITUDE_M, color="k", ls="--", lw=1,
              label=f"crash below {H.GROUND_CRASH_ALTITUDE_M:.3f} m")
    a.set_xlabel("time (s)"); a.set_ylabel("altitude (m)"); a.grid(alpha=0.3)
    a.legend(loc="upper right", fontsize=9); a.set_title("Altitude", fontsize=10)

    # (c) distance to target
    a = ax[1, 0]
    L = H.EPISODE_STEPS
    D = np.full((len(ok), L), np.nan)
    for i, e in enumerate(ok):
        D[i] = np.linalg.norm(e["pos"] - TGT, axis=1)
    t = np.arange(L) * DT
    for e in eps:
        d = np.linalg.norm(e["pos"] - TGT, axis=1)
        a.plot(np.arange(len(d)) * DT, d,
               color="tab:red" if e["crashed"] else "0.6",
               lw=1.2 if e["crashed"] else 0.5)
    a.plot(t, np.median(D, 0), color="k", lw=2, label="median (true position)")
    Dest = np.array([np.linalg.norm(e["est"] - TGT, axis=1) for e in ok])
    a.plot(t, np.median(Dest, 0), color="tab:green", lw=1.5, ls="--",
           label="median (position the policy sees)")
    a.axvspan(tail * DT, L * DT, color="tab:blue", alpha=0.07)
    a.set_yscale("log"); a.set_ylim(0.02, 4)
    a.set_xlabel("time (s)"); a.set_ylabel("distance to target (m, log)")
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=9)
    a.set_title(f"Distance to target; last-5-s median over survivors "
                f"{np.median(med_tail):.2f} m", fontsize=10)

    # (d) commands of a typical episode
    a = ax[1, 1]
    tt = np.arange(len(typical["rates"])) * DT
    for k, (lab, c) in enumerate(zip(["roll", "pitch", "yaw"],
                                     ["tab:blue", "tab:orange", "tab:green"])):
        a.plot(tt, typical["rates"][:, k], color=c, lw=0.8, label=f"{lab} rate cmd")
    a.set_xlabel("time (s)"); a.set_ylabel("body-rate command (rad/s)")
    a.grid(alpha=0.3)
    b = a.twinx()
    b.plot(tt, typical["thrust"], color="k", lw=1.2, label="thrust cmd")
    b.set_ylabel("collective thrust command (0-1)"); b.set_ylim(0, 1)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", ncol=2)
    a.set_title(f"Policy commands, typical episode (seed {typical['seed']})",
                fontsize=10)

    n_ok = len(ok)
    fig.suptitle(f"{title}\n{n_ok}/{len(eps)} survived 10 s", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", help="dir from aerial/deploy/export_model.py")
    ap.add_argument("--stage", default="1c")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="trajectories.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--cuda-device", default="1",
                    help="GPU index for the policy; '' runs on CPU")
    args = ap.parse_args()
    title = args.title or f"{args.bundle} on stage {args.stage}"
    policy = PolicyInference(args.bundle, cuda_device=args.cuda_device or None)
    plot(fly(policy, args.stage, args.episodes), args.out, title)


if __name__ == "__main__":
    main()
