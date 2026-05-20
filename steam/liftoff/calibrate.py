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

Usage:
    python -m steam.liftoff.calibrate
    python -m steam.liftoff.calibrate --auto
    python -m steam.liftoff.calibrate --auto --stage-delay 2.5
"""
import argparse
import math
import sys
import time

from steam.liftoff.transmitter import EvdevTransmitter

MAX_AXIS = 32767
MIN_AXIS = -32768

AXIS_INDEX = {"roll": 0, "pitch": 1, "throttle": 2, "yaw": 3}


def _wait(auto, delay, prompt):
    if auto:
        time.sleep(delay)
    else:
        input(prompt)


def _send(tx, sticks):
    tx.set_sticks(roll=sticks[0], pitch=sticks[1],
                  throttle=sticks[2], yaw=sticks[3])
    tx.update()


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
    time.sleep(hold)
    print("  -> returning to neutral")
    _ramp_to(tx, peak_sticks, new_rest)
    return new_rest


def run(auto=False, stage_delay=2.0):
    tx = EvdevTransmitter(with_buttons=True)
    print("Virtual transmitter created. Open Liftoff > Options > Controls > "
          "Controller > Calibrate.")
    print()
    _wait(auto, stage_delay,
          "Click 'Start Calibration' in Liftoff, then press Enter here... ")
    try:
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
    finally:
        tx.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-advance with fixed delays instead of "
                             "waiting for Enter between stages.")
    parser.add_argument("--stage-delay", type=float, default=2.0,
                        help="Seconds between stages in --auto mode "
                             "(default: 2.0).")
    args = parser.parse_args()
    try:
        run(auto=args.auto, stage_delay=args.stage_delay)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
