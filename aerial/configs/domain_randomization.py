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

Added 2026-10-16 (sim-realism pass, airframe-independent). Values are
judgment calls until the real vehicle and positioning system are known:
  * Per-motor thrust imbalance ±5% and CG offset ±1.5 cm: build
    variation; creates a steady torque the FC's I-term must trim.
  * Drag scale 0.5-2x: drag coefficients are scaled-up cf2x guesses.
  * Action latency 5-40 ms + observation latency 0-20 ms: companion
    computer <-> MAVLink <-> FC round trip, at physics-step resolution.
  * Estimator error: gyro bias, attitude bias + noise (yaw bias up to
    ~5 deg, as from compass error), slowly drifting position/velocity
    error (Gauss-Markov), on top of white noise.
  * Wind gusts: Ornstein-Uhlenbeck force on top of the per-episode mean.

Changed 2026-10-17 for the F450 airframe (1.51 kg, was a 0.8 kg
placeholder): wind and gust forces scaled by the weight ratio (x1.89)
so they stay the same fraction of weight. With the F450's rotor drag
(~0.38 N per m/s), the Stage 1c sigmas are about 1.5 m/s of mean wind
and 0.75 m/s of gust. Mass +-20% covers adding a camera (~+8%).
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
    drag_scale: Tuple[float, float] = (0.5, 2.0)
    motor_imbalance: float = 0.05        # each motor 1 ± this, on top of motor_thrust_scale
    cg_offset_m: float = 0.015           # x and y, uniform ± this

    # --- Per-episode (absolute) -----------------------------------------
    control_latency_s: Tuple[float, float] = (0.005, 0.040)  # action delay
    obs_latency_s: Tuple[float, float] = (0.000, 0.020)      # observation delay
    init_pos_range_m: Tuple[float, float] = (-2.0, 2.0)      # per xy axis
    init_z_range_m: Tuple[float, float] = (-1.0, 1.0)        # around target z
    init_rpy_range_deg: Tuple[float, float] = (-30.0, 30.0)  # per axis

    # --- Per-step sensor noise (Gaussian sigma) -------------------------
    gyro_noise_sigma_rad_s: float = 0.05
    accel_noise_sigma_m_s2: float = 0.10
    pos_noise_sigma_m: float = 0.02
    vel_noise_sigma_m_s: float = 0.05
    att_noise_sigma_rad: float = 0.004

    # --- Estimator bias / drift ------------------------------------------
    gyro_bias_sigma_rad_s: float = 0.015     # per episode, per axis
    att_bias_sigma_rad: float = 0.015        # roll/pitch, per episode
    yaw_bias_max_rad: float = 0.09           # uniform ±, per episode
    pos_drift_sigma_m: float = 0.06          # Gauss-Markov steady-state sigma
    pos_drift_tau_s: float = 3.0
    vel_drift_sigma_m_s: float = 0.03
    vel_drift_tau_s: float = 1.0

    # --- External disturbance ---------------------------------------------
    # Mean wind: constant world-frame force per episode, zero-mean so the
    # nominal case has no wind. Gusts: Ornstein-Uhlenbeck force on top.
    wind_force_sigma_N: float = 0.57     # ~4% of weight; ~1.5 m/s of wind
    gust_force_sigma_N: float = 0.28     # ~0.75 m/s
    gust_tau_s: float = 1.5


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
    drag_scale=(0.8, 1.25),
    motor_imbalance=0.02,
    cg_offset_m=0.005,
    control_latency_s=(0.005, 0.020),
    obs_latency_s=(0.000, 0.010),
    init_pos_range_m=(-0.5, 0.5),
    init_z_range_m=(-0.3, 0.3),
    init_rpy_range_deg=(-10.0, 10.0),
    gyro_noise_sigma_rad_s=0.025,
    accel_noise_sigma_m_s2=0.05,
    pos_noise_sigma_m=0.01,
    vel_noise_sigma_m_s=0.025,
    att_noise_sigma_rad=0.002,
    gyro_bias_sigma_rad_s=0.005,
    att_bias_sigma_rad=0.005,
    yaw_bias_max_rad=0.03,
    pos_drift_sigma_m=0.02,
    vel_drift_sigma_m_s=0.01,
    wind_force_sigma_N=0.28,
    gust_force_sigma_N=0.095,
)

# --- Phase 1a5: bridge stage between 1a and 1b -------------------------
# Added after the direct 1a -> 1b jump plateaued at ~60% success in eval:
# the ±1.0m / ±20 deg init range was too big a step from 1a's ±0.5m /
# ±10 deg for the policy to cross by training longer alone. 1a5 sits at
# roughly the midpoint on every dimension, so the policy can climb the
# curriculum instead of getting stuck.
STAGE_1A5 = DomainRandomizationRanges(
    mass_scale=(0.88, 1.12),
    inertia_scale=(0.88, 1.12),
    motor_thrust_scale=(0.90, 1.10),
    motor_tau_scale=(0.82, 1.18),
    rate_kp_scale=(0.88, 1.12),
    drag_scale=(0.7, 1.4),
    motor_imbalance=0.03,
    cg_offset_m=0.0075,
    control_latency_s=(0.005, 0.025),
    obs_latency_s=(0.000, 0.012),
    init_pos_range_m=(-0.75, 0.75),
    init_z_range_m=(-0.4, 0.4),
    init_rpy_range_deg=(-15.0, 15.0),
    gyro_noise_sigma_rad_s=0.031,
    accel_noise_sigma_m_s2=0.063,
    pos_noise_sigma_m=0.012,
    vel_noise_sigma_m_s=0.031,
    att_noise_sigma_rad=0.0025,
    gyro_bias_sigma_rad_s=0.0075,
    att_bias_sigma_rad=0.0075,
    yaw_bias_max_rad=0.045,
    pos_drift_sigma_m=0.03,
    vel_drift_sigma_m_s=0.015,
    wind_force_sigma_N=0.34,
    gust_force_sigma_N=0.15,
)

# --- Phase 1b: intermediate (added later when 1a converges) ------------
# Placeholder; will be tuned based on 1a results.
STAGE_1B = DomainRandomizationRanges(
    mass_scale=(0.85, 1.15),
    inertia_scale=(0.85, 1.15),
    motor_thrust_scale=(0.88, 1.12),
    motor_tau_scale=(0.78, 1.22),
    rate_kp_scale=(0.85, 1.15),
    drag_scale=(0.6, 1.6),
    motor_imbalance=0.04,
    cg_offset_m=0.01,
    control_latency_s=(0.005, 0.030),
    obs_latency_s=(0.000, 0.015),
    init_pos_range_m=(-1.0, 1.0),
    init_z_range_m=(-0.5, 0.5),
    init_rpy_range_deg=(-20.0, 20.0),
    gyro_noise_sigma_rad_s=0.038,
    accel_noise_sigma_m_s2=0.075,
    pos_noise_sigma_m=0.015,
    vel_noise_sigma_m_s=0.038,
    att_noise_sigma_rad=0.003,
    gyro_bias_sigma_rad_s=0.01,
    att_bias_sigma_rad=0.01,
    yaw_bias_max_rad=0.06,
    pos_drift_sigma_m=0.04,
    vel_drift_sigma_m_s=0.02,
    wind_force_sigma_N=0.42,
    gust_force_sigma_N=0.19,
)

_STAGES = {
    "1a": STAGE_1A,
    "1a5": STAGE_1A5,
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

    motor_scales (4,) already includes motor_thrust_scale times each
    motor's individual imbalance.
    """
    thrust_scale = float(rng.uniform(*ranges.motor_thrust_scale))
    imbalance = rng.uniform(
        -ranges.motor_imbalance, ranges.motor_imbalance, size=4)
    return {
        "mass_scale":         float(rng.uniform(*ranges.mass_scale)),
        "inertia_scale":      float(rng.uniform(*ranges.inertia_scale)),
        "motor_thrust_scale": thrust_scale,
        "motor_scales":       (thrust_scale * (1.0 + imbalance)).astype(np.float32),
        "motor_tau_scale":    float(rng.uniform(*ranges.motor_tau_scale)),
        "rate_kp_scale":      float(rng.uniform(*ranges.rate_kp_scale)),
        "drag_scale":         float(rng.uniform(*ranges.drag_scale)),
        "cg_offset_m":        rng.uniform(
            -ranges.cg_offset_m, ranges.cg_offset_m, size=2).astype(np.float32),
        "control_latency_s":  float(rng.uniform(*ranges.control_latency_s)),
        "obs_latency_s":      float(rng.uniform(*ranges.obs_latency_s)),
        "gyro_bias":          rng.normal(
            0.0, ranges.gyro_bias_sigma_rad_s, size=3).astype(np.float32),
        "att_bias_rpy":       np.array([
            rng.normal(0.0, ranges.att_bias_sigma_rad),
            rng.normal(0.0, ranges.att_bias_sigma_rad),
            rng.uniform(-ranges.yaw_bias_max_rad, ranges.yaw_bias_max_rad),
        ], dtype=np.float32),
        "wind_force_world":   (rng.normal(0.0, ranges.wind_force_sigma_N, size=3)
                               * WIND_AXIS_SCALE).astype(np.float32),
    }


# Per-axis scale (x, y, z) on the wind and gust sigmas. Near flat ground
# mean wind is close to horizontal, and low-altitude turbulence models
# (Dryden, MIL-F-8785C) put vertical gusts at about half the horizontal
# ones. Before 2026-10-17 all three axes used the full sigma, so a weak
# vehicle could meet a ~3 m/s downdraft.
WIND_AXIS_SCALE = np.array([1.0, 1.0, 0.2])
GUST_AXIS_SCALE = np.array([1.0, 1.0, 0.5])


# Must stay well above the env's ground-crash altitude (0.255 m for the
# F450, which stands at 0.155 m on its legs).
MIN_INIT_ALTITUDE_M = 0.40


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
