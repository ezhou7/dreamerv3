"""Sign and trim checks for the NumPy rigid-body sim in run_policy.py.

Fixed-action stub policies pin the conventions the sim (and later the
MAVLink bridge) must share with the training env, body x forward, y left,
z up:

  * hover thrust with zero rates holds position
  * +pitch rate is nose-DOWN, so the drone accelerates toward +x
  * +roll rate banks right, so the drone accelerates toward -y

Run:  python -m pytest aerial/tests/test_run_policy_sim.py -q
"""
import numpy as np

from aerial.deploy.npu import run_policy


class _FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, np.float32)

    def reset(self):
        pass

    def step(self, obs, is_first=False):
        return self.action.copy()


HOVER_RAW = 2 * run_policy.AIRFRAME.hover_thrust_normalized - 1


def _final_err(action, steps):
    log = run_policy.sim_episode(
        _FixedPolicy(action), [0, 0, 0], [0, 0, 0], np.eye(3), steps)
    assert len(log) == steps, 'crashed'
    return log[-1][1]


def test_hover_thrust_holds_position():
    err = _final_err([0, 0, 0, HOVER_RAW], steps=50)
    assert np.abs(err).max() < 0.05


def test_positive_pitch_is_nose_down():
    err = _final_err([0, 0.05, 0, HOVER_RAW], steps=25)
    assert err[0] > 0.01 and abs(err[1]) < 0.3 * err[0]


def test_positive_roll_banks_right():
    err = _final_err([0.05, 0, 0, HOVER_RAW], steps=25)
    assert err[1] < -0.01 and abs(err[0]) < 0.3 * -err[1]
