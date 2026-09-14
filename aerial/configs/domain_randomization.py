"""Per-episode / per-step domain randomization ranges.

Applied inside aerial/envs/hover_aerial_env.py when Aerial Gym resets
each parallel env. Purpose: force the learned policy to be robust to
plant variation so it survives sim2real transfer.

Two categories:
  * Per-episode: sampled once at env reset and held constant for the
    episode. Physical parameters that change slowly relative to control
    (mass, motor strength, gains).
  * Per-step: sampled each control step. Sensor and actuator noise.

Ranges are conservative starting points. Widen once the policy learns
hover under narrow randomization; too wide from the start prevents
convergence.

Rationale for the ranges:
  * Mass ±20%: covers payload variation (cameras, batteries, gimbals)
  * Motor thrust ±15%: motor wear, battery voltage sag, temperature
  * Motor tau ±30%: sim2real slop, no good baseline for real motors
  * Control latency 0-30ms: MAVLink offboard ceiling on ArduPilot
    (~50Hz max cmd rate = 20ms nominal; add jitter)
  * IMU gyro noise: σ 0.05 rad/s ≈ typical MEMS IMU spec sheet
  * IMU accel noise: σ 0.1 m/s² ≈ same
  * Rate-loop Kp ±20%: covers ArduPilot vs PX4 default gain differences
    plus per-vehicle tuning variance
  * Init pos ±2m, init rpy ±30°: exercises recovery from significant
    displacement, unlike Phase 0 where init was within success radius
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DomainRandomizationRanges:
    """All ranges are (low, high) tuples for uniform sampling, or scalar
    sigmas for normal-noise sampling.
    """
    # --- Per-episode (relative multipliers on nominal airframe params) --
    mass_scale: Tuple[float, float] = (0.80, 1.20)
    inertia_scale: Tuple[float, float] = (0.80, 1.20)
    motor_thrust_scale: Tuple[float, float] = (0.85, 1.15)
    motor_tau_scale: Tuple[float, float] = (0.70, 1.30)
    rate_kp_scale: Tuple[float, float] = (0.80, 1.20)  # rate controller gain jitter

    # --- Per-episode (absolute) -----------------------------------------
    control_latency_s: Tuple[float, float] = (0.000, 0.030)  # 0-30 ms
    init_pos_range_m: Tuple[float, float] = (-2.0, 2.0)      # per xy axis
    init_z_range_m: Tuple[float, float] = (-1.0, 1.0)        # around target z
    init_rpy_range_deg: Tuple[float, float] = (-30.0, 30.0)  # per axis

    # --- Per-step sensor noise (Gaussian sigma) -------------------------
    gyro_noise_sigma_rad_s: float = 0.05
    accel_noise_sigma_m_s2: float = 0.10
    pos_noise_sigma_m: float = 0.02      # position estimator drift
    vel_noise_sigma_m_s: float = 0.05    # velocity estimator drift

    # --- External disturbance (per-step wind gust) ---------------------
    # Simple constant-per-episode wind, sampled at reset. World-frame
    # force applied every step. Zero-mean so nominal case has no wind.
    wind_force_sigma_N: float = 0.3      # ~4% of hover thrust


# --- Full target ranges (Phase 1c = original DEFAULT) ------------------
STAGE_1C = DomainRandomizationRanges()

# --- Phase 1a: easy start for curriculum learning ----------------------
# Roughly half the range of Phase 1c on every dimension. Phase 1's first
# training attempt showed that presenting the full DR + init range from
# step 0 kept the policy bimodal — it never consolidated because too many
# starting configurations were essentially unrecoverable.
STAGE_1A = DomainRandomizationRanges(
    mass_scale=(0.90, 1.10),
    inertia_scale=(0.90, 1.10),
    motor_thrust_scale=(0.92, 1.08),
    motor_tau_scale=(0.85, 1.15),
    rate_kp_scale=(0.90, 1.10),
    control_latency_s=(0.000, 0.015),
    init_pos_range_m=(-0.5, 0.5),
    init_z_range_m=(-0.3, 0.3),
    init_rpy_range_deg=(-10.0, 10.0),
    gyro_noise_sigma_rad_s=0.025,
    accel_noise_sigma_m_s2=0.05,
    pos_noise_sigma_m=0.01,
    vel_noise_sigma_m_s=0.025,
    wind_force_sigma_N=0.15,
)

# --- Phase 1b: intermediate (added later when 1a converges) ------------
# Placeholder; will be tuned based on 1a results.
STAGE_1B = DomainRandomizationRanges(
    mass_scale=(0.85, 1.15),
    inertia_scale=(0.85, 1.15),
    motor_thrust_scale=(0.88, 1.12),
    motor_tau_scale=(0.78, 1.22),
    rate_kp_scale=(0.85, 1.15),
    control_latency_s=(0.000, 0.022),
    init_pos_range_m=(-1.0, 1.0),
    init_z_range_m=(-0.5, 0.5),
    init_rpy_range_deg=(-20.0, 20.0),
    gyro_noise_sigma_rad_s=0.038,
    accel_noise_sigma_m_s2=0.075,
    pos_noise_sigma_m=0.015,
    vel_noise_sigma_m_s=0.038,
    wind_force_sigma_N=0.22,
)

_STAGES = {
    "1a": STAGE_1A,
    "1b": STAGE_1B,
    "1c": STAGE_1C,
}


def get_stage(name: str) -> DomainRandomizationRanges:
    """Look up a named curriculum stage's DR ranges.

    Accepts "1a", "1b", "1c" (case-insensitive). Falls back to full-range
    STAGE_1C if the name is unrecognized so training never silently runs
    with no randomization.
    """
    key = name.strip().lower()
    if key not in _STAGES:
        raise ValueError(
            f"Unknown DR stage {name!r}. Expected one of {sorted(_STAGES)}."
        )
    return _STAGES[key]


# Backwards compatibility: existing code uses `DEFAULT`. Point it at the
# full-difficulty stage so old callers keep the same behavior.
DEFAULT = STAGE_1C


def sample_episode_params(rng: np.random.Generator,
                          ranges: DomainRandomizationRanges = DEFAULT) -> dict:
    """Draw one episode's frozen randomization values.

    Returns a dict of concrete parameters that the env applies at reset:
      mass_scale, inertia_scale, motor_thrust_scale, motor_tau_scale,
      rate_kp_scale, control_latency_s, wind_force_world (3,).
    """
    return {
        "mass_scale":         float(rng.uniform(*ranges.mass_scale)),
        "inertia_scale":      float(rng.uniform(*ranges.inertia_scale)),
        "motor_thrust_scale": float(rng.uniform(*ranges.motor_thrust_scale)),
        "motor_tau_scale":    float(rng.uniform(*ranges.motor_tau_scale)),
        "rate_kp_scale":      float(rng.uniform(*ranges.rate_kp_scale)),
        "control_latency_s":  float(rng.uniform(*ranges.control_latency_s)),
        "wind_force_world":   rng.normal(
            0.0, ranges.wind_force_sigma_N, size=3).astype(np.float32),
    }


MIN_INIT_ALTITUDE_M = 0.10  # avoid spawning below/on the ground


def sample_init_pose(rng: np.random.Generator,
                     target: np.ndarray,
                     ranges: DomainRandomizationRanges = DEFAULT
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample initial (position, rpy_radians) around `target`.

    Altitude is clamped to at least MIN_INIT_ALTITUDE_M so the drone
    never spawns inside the ground plane.
    """
    pos = np.array([
        target[0] + rng.uniform(*ranges.init_pos_range_m),
        target[1] + rng.uniform(*ranges.init_pos_range_m),
        max(target[2] + rng.uniform(*ranges.init_z_range_m),
            MIN_INIT_ALTITUDE_M),
    ], dtype=np.float32)
    rpy_deg = rng.uniform(*ranges.init_rpy_range_deg, size=3)
    rpy_rad = np.deg2rad(rpy_deg).astype(np.float32)
    return pos, rpy_rad


def add_sensor_noise(rng: np.random.Generator,
                     gyro: np.ndarray,
                     accel: np.ndarray | None,
                     pos: np.ndarray,
                     vel: np.ndarray,
                     ranges: DomainRandomizationRanges = DEFAULT
                     ) -> dict:
    """Return noise-corrupted copies of the sensor readings."""
    return {
        "gyro":  gyro + rng.normal(
            0.0, ranges.gyro_noise_sigma_rad_s, size=3).astype(gyro.dtype),
        "accel": (None if accel is None
                  else accel + rng.normal(
                      0.0, ranges.accel_noise_sigma_m_s2,
                      size=3).astype(accel.dtype)),
        "pos":   pos + rng.normal(
            0.0, ranges.pos_noise_sigma_m, size=3).astype(pos.dtype),
        "vel":   vel + rng.normal(
            0.0, ranges.vel_noise_sigma_m_s, size=3).astype(vel.dtype),
    }
