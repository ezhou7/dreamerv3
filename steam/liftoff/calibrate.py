"""Liftoff controller calibration helper.

Drives the virtual FPV transmitter through Liftoff's calibration flow:
  1. Edge detection      — sweep all sticks through their full range
  2. Throttle max detect — deflect throttle MIN -> MAX -> 0
  3. Pitch max detect    — deflect pitch 0 -> MAX -> 0
  4. Roll max detect     — deflect roll 0 -> MAX -> 0
  5. Yaw max detect      — deflect yaw 0 -> MAX -> 0

Throttle gets a full MIN->MAX stroke during its stage so Liftoff calibrates
the full bidirectional range; afterward throttle returns to 0 so Liftoff
displays it as centered (rather than "throttle off" at the left edge).

Each stage ramps cleanly back to neutral before the script waits for the
next Enter press, so sticks are at 0 while you click "Next" in Liftoff.

If an axis ends up inverted in Liftoff, fix it via Fine-Tune > Invert.

Press the restart hotkey ('r' by default) at any time — from either the
terminal or the Liftoff window — to abort the current pass, recenter the
sticks, and restart from the edge sweep. Use --restart-key to pick a
different binding if 'r' collides with something in Liftoff.

Usage:
    python -m steam.liftoff.calibrate
    python -m steam.liftoff.calibrate --auto
    python -m steam.liftoff.calibrate --auto --stage-delay 2.5
    python -m steam.liftoff.calibrate --restart-key F8
"""
import argparse
import math
import select
import sys
import threading
import time

import keyboard

from steam.liftoff.transmitter import EvdevTransmitter

MAX_AXIS = 32767
MIN_AXIS = -32768

AXIS_INDEX = {"roll": 0, "pitch": 1, "throttle": 2, "yaw": 3}

_restart_event = threading.Event()


class RestartCalibration(Exception):
    """Raised internally to abort the current calibration pass and restart."""


def _check_restart():
    if _restart_event.is_set():
        raise RestartCalibration()


def _on_restart_key():
    if _restart_event.is_set():
        return
    _restart_event.set()
    print("\n[!] Restart hotkey pressed — aborting current pass...")


def _wait(auto, delay, prompt):
    """Wait for either an Enter keystroke (interactive) or the auto-mode
    delay to elapse. Polls the restart flag throughout so the hotkey
    interrupts even a blocked input prompt.
    """
    if auto:
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            _check_restart()
            time.sleep(0.05)
        return

    print(prompt, end="", flush=True)
    while True:
        _check_restart()
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            sys.stdin.readline()
            return


def _send(tx, sticks):
    tx.set_sticks(roll=sticks[0], pitch=sticks[1],
                  throttle=sticks[2], yaw=sticks[3])
    tx.update()
    _check_restart()


def _ramp_to(tx, from_sticks, to_sticks, duration=0.4, settle=0.3, rate_hz=60):
    """Smoothly ramp from `from_sticks` to `to_sticks`, then optionally
    hold there with a small dither so events continue to fire past kernel
    EV_ABS deduplication.
    """
    steps = max(1, int(duration * rate_hz))
    for i in range(steps + 1):
        alpha = i / steps
        sticks = [int(f + (t - f) * alpha)
                  for f, t in zip(from_sticks, to_sticks)]
        _send(tx, sticks)
        time.sleep(1.0 / rate_hz)
    if settle > 0:
        settle_steps = int(settle * rate_hz)
        for i in range(settle_steps):
            dithered = [v + (1 if (i % 2) else 0) for v in to_sticks]
            _send(tx, dithered)
            time.sleep(1.0 / rate_hz)
        _send(tx, to_sticks)


def stage_edges(tx, duration=6.0, rate_hz=60):
    """Sweep sticks through their full range in a Lissajous-style pattern."""
    print(f"[1/5] Edge sweep ({duration:.1f}s)...")
    steps = int(duration * rate_hz)
    last = [0, 0, 0, 0]
    for i in range(steps):
        t = i / rate_hz
        last = [
            int(MAX_AXIS * math.sin(2 * math.pi * 0.50 * t)),  # roll
            int(MAX_AXIS * math.cos(2 * math.pi * 0.37 * t)),  # pitch
            int(MAX_AXIS * math.sin(2 * math.pi * 0.23 * t)),  # throttle
            int(MAX_AXIS * math.cos(2 * math.pi * 0.61 * t)),  # yaw
        ]
        _send(tx, last)
        time.sleep(1.0 / rate_hz)
    print("  -> returning to neutral")
    rest = [0, 0, 0, 0]
    _ramp_to(tx, last, rest)
    return rest


def stage_assign(tx, step_label, axis, rest_sticks, hold=2.5, ramp=0.4):
    """Run one axis through its calibration motion, returning the new rest
    state.

    Throttle: rest -> MIN -> MAX -> 0 (full-range stroke so Liftoff
    calibrates throttle as bidirectional; final 0 is the new center).
    Others:   rest -> 0   -> MAX -> 0 (center-detent axes).

    The non-target axes are held at whatever `rest_sticks` says they
    should be (so e.g. if a future change parks throttle at MIN between
    stages, it would stay there during pitch/roll/yaw stages).
    """
    idx = AXIS_INDEX[axis]
    print(f"[{step_label}] Deflecting {axis} to MAX ({hold:.1f}s)...")

    start_val = MIN_AXIS if axis == "throttle" else 0
    end_val = 0

    start_sticks = list(rest_sticks)
    start_sticks[idx] = start_val
    peak_sticks = list(rest_sticks)
    peak_sticks[idx] = MAX_AXIS
    new_rest = list(rest_sticks)
    new_rest[idx] = end_val

    if start_sticks != rest_sticks:
        _ramp_to(tx, rest_sticks, start_sticks, duration=ramp, settle=0)
    _ramp_to(tx, start_sticks, peak_sticks, duration=ramp, settle=0)
    hold_end = time.monotonic() + hold
    while time.monotonic() < hold_end:
        _check_restart()
        time.sleep(0.05)
    print("  -> returning to neutral")
    _ramp_to(tx, peak_sticks, new_rest)
    return new_rest


def _run_once(tx, auto, stage_delay):
    print("Virtual transmitter created. Open Liftoff > Options > Controls > "
          "Controller > Calibrate.")
    print()
    _wait(auto, stage_delay,
          "Click 'Start Calibration' in Liftoff, then press Enter here... ")
    rest = stage_edges(tx)
    _wait(auto, stage_delay,
          "Edge sweep done (sticks neutral). Click 'Next' in Liftoff, "
          "then press Enter... ")

    for step_label, axis in [
        ("2/5", "throttle"),
        ("3/5", "pitch"),
        ("4/5", "roll"),
        ("5/5", "yaw"),
    ]:
        rest = stage_assign(tx, step_label, axis, rest)
        _wait(auto, stage_delay,
              f"{axis.capitalize()} done (sticks neutral). Click 'Next' "
              "in Liftoff, then press Enter... ")

    print("\nAll stages done. In Liftoff: click SAVE (and Fine-Tune any "
          "inverted axes / deadband).")


def run(auto=False, stage_delay=2.0, restart_key="r"):
    tx = EvdevTransmitter(with_buttons=True)
    hotkey_registered = False
    try:
        try:
            keyboard.add_hotkey(restart_key, _on_restart_key)
            hotkey_registered = True
            print(f"(Press '{restart_key}' at any time — in this terminal or "
                  f"in Liftoff — to restart calibration from the beginning.)")
        except Exception as e:
            print(f"[warn] Could not register restart hotkey '{restart_key}': "
                  f"{e}. Restart hotkey unavailable; Ctrl-C to exit.",
                  file=sys.stderr)

        while True:
            _restart_event.clear()
            try:
                _run_once(tx, auto, stage_delay)
                return
            except RestartCalibration:
                tx.center_all()
                print("[*] Sticks recentered. Cancel Liftoff's current "
                      "calibration dialog (or click 'Start Calibration' "
                      "again to restart it) before continuing.")
                time.sleep(0.5)
                _restart_event.clear()
                continue
    finally:
        if hotkey_registered:
            try:
                keyboard.remove_hotkey(restart_key)
            except Exception:
                pass
        tx.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-advance with fixed delays instead of "
                             "waiting for Enter between stages.")
    parser.add_argument("--stage-delay", type=float, default=2.0,
                        help="Seconds between stages in --auto mode "
                             "(default: 2.0).")
    parser.add_argument("--restart-key", default="r",
                        help="Global hotkey that aborts the current pass "
                             "and restarts from the edge sweep (default: "
                             "'r'). Use any name the 'keyboard' library "
                             "accepts, e.g. 'F8' or 'ctrl+r'.")
    args = parser.parse_args()
    try:
        run(auto=args.auto, stage_delay=args.stage_delay,
            restart_key=args.restart_key)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
