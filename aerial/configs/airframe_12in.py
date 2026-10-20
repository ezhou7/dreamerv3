"""Airframe parameters for the target vehicle.

Since 2026-10-17 this is the project's QWinOut F450 kit (450 mm motor-to-
motor, 10" props), not the generic 12" placeholder the module name
refers to. Mass and inertia come from the parts table F450_PARTS below.
The kit listing gives no weights, so each mass there is a typical value
for that part. To refine:

  * Mass: weigh each part (or the whole vehicle) and edit F450_PARTS
  * Max thrust: after the first hover, MOT_THST_HOVER (ArduPilot) is the
    hover thrust fraction, so max thrust per motor = m*g / (4 * hover)
  * Inertia: bifilar pendulum, or leave it to the parts table
  * Motor time constant: bench log of RPM vs commanded, first-order fit

Keep aerial/assets/cf12in.urdf in step with DEFAULT_12IN (checked by
aerial/tests/test_airframe.py). Controller gains in
aerial/controllers/rate_controller.py derive from these values.

Convention: body frame x=forward, y=left, z=up (matches Phase 0 and
Aerial Gym / ROS REP-103).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AirframeParams:
    """All fields SI units."""
    # --- Mass & inertia -----------------------------------------------
    mass_kg: float
    inertia_diag_kg_m2: np.ndarray  # (3,) [ixx, iyy, izz]

    # --- Geometry -----------------------------------------------------
    arm_length_m: float          # motor mount to CG
    # X-frame motor layout (matches DSLPID cf2x convention):
    #   0: front-right (+x, -y)  CW
    #   1: back-right  (-x, -y)  CCW
    #   2: back-left   (-x, +y)  CW
    #   3: front-left  (+x, +y)  CCW
    prop_radius_m: float
    rest_height_m: float         # body origin height when on its legs

    # --- Motor & prop -------------------------------------------------
    max_thrust_per_motor_N: float
    max_rpm: float               # loaded, at full throttle
    motor_time_constant_s: float
    torque_coeff_per_thrust: float  # m, yaw torque = k * thrust
    rotor_inertia_kg_m2: float   # prop + motor bell, about the shaft

    # --- Aerodynamics -------------------------------------------------
    # Rotor drag at hover, as deceleration per unit airspeed (1/s):
    # F = -mass * coeff * v. Not measured; DR drag_scale covers 0.5-2x.
    drag_per_s_xy: float
    drag_per_s_z: float

    @property
    def hover_thrust_normalized(self) -> float:
        """Collective thrust in [0, 1] required to hover at 1g.

        motor_cmd is normalized so 1.0 = max_thrust_per_motor.
        Total thrust at hover = mass * g = 4 * motor_cmd_hover * max_thrust.
        """
        g = 9.81
        return (self.mass_kg * g) / (4.0 * self.max_thrust_per_motor_N)

    @property
    def thrust_to_weight(self) -> float:
        g = 9.81
        return (4.0 * self.max_thrust_per_motor_N) / (self.mass_kg * g)


# --- F450 parts table ----------------------------------------------------
@dataclass(frozen=True)
class Part:
    name: str
    mass_kg: float
    pos_m: Tuple[float, float, float]   # part center, body frame
    box_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # extents; 0 = point


ARM_LENGTH_M = 0.225                   # 450 mm wheelbase / 2
_D = ARM_LENGTH_M / np.sqrt(2.0)       # motor x/y offset


def _on_arms(name, mass_each, radius, z):
    """One part per arm at `radius` from the center along the diagonals."""
    k = radius / np.sqrt(2.0)
    return [Part(f"{name}{i}", mass_each, (sx * k, sy * k, z))
            for i, (sx, sy) in enumerate([(1, -1), (-1, -1), (-1, 1), (1, 1)])]


def _arm_rods(mass_each, r0, r1, z, segments=4):
    """Arms as uniform rods from r0 to r1, split into point masses."""
    step = (r1 - r0) / segments
    parts = []
    for s in range(segments):
        parts += _on_arms(f"arm_seg{s}_", mass_each / segments,
                          r0 + (s + 0.5) * step, z)
    return parts


# Origin: frame center in the arm plane. Masses are typical listing values
# for each part, not weighed (total frame 282 g for the DJI F450 the HJ450
# copies). Placement (confirmed 2026-10-17): battery and Orange Pi stacked on
# top, camera in the gimbal under the bottom plate. Assumed order, bottom
# to top: Pixhawk on the top plate, battery, Orange Pi. The other orders
# change roll/pitch inertia by about +-5%, well inside DR.
F450_PARTS = (
    [Part("top_plate", 0.040, (0, 0, 0.015), (0.12, 0.12, 0.002)),
     Part("bottom_plate_pdb", 0.060, (0, 0, -0.015), (0.14, 0.14, 0.002))]
    + _arm_rods(0.045, 0.05, ARM_LENGTH_M, 0.0)
    + _on_arms("motor_2212_920kv_", 0.055, ARM_LENGTH_M, 0.025)
    + _on_arms("prop_1045_", 0.011, ARM_LENGTH_M, 0.055)
    + _on_arms("esc_30a_", 0.025, 0.13, -0.015)
    + _on_arms("leg_140mm_", 0.015, 0.12, -0.085)
    + [Part("pixhawk_2.4.8", 0.050, (0, 0, 0.035), (0.08, 0.05, 0.016)),
       Part("receiver_r9ds", 0.010, (-0.05, 0, 0.03)),
       Part("power_module", 0.020, (0, 0, -0.025)),
       Part("gps_mast", 0.025, (-0.05, 0, 0.08)),
       Part("gps_6m", 0.035, (-0.05, 0, 0.15), (0.05, 0.05, 0.015)),
       Part("battery_3s_4400", 0.330, (0, 0, 0.065), (0.136, 0.043, 0.03)),
       Part("orange_pi_5_bec_cooling", 0.100, (0, 0, 0.095),
            (0.10, 0.062, 0.025)),
       Part("gimbal_2axis_brushless", 0.200, (0, 0, -0.10),
            (0.08, 0.08, 0.08)),
       # Model not known yet; GoPro-class (Hero4 88 g, Hero12 154 g).
       Part("camera", 0.120, (0, 0, -0.10), (0.06, 0.04, 0.045)),
       Part("wiring_straps", 0.040, (0, 0, 0))]
)


def mass_properties(parts):
    """Total mass, CG, and inertia diagonal about the CG.

    Off-diagonal terms are dropped; they are near zero for this
    left-right symmetric layout.
    """
    m = np.array([q.mass_kg for q in parts])
    pos = np.array([q.pos_m for q in parts])
    total = m.sum()
    cg = (m[:, None] * pos).sum(0) / total
    inertia = np.zeros((3, 3))
    for mi, ri, q in zip(m, pos - cg, parts):
        inertia += mi * (ri @ ri * np.eye(3) - np.outer(ri, ri))
        a, b, c = q.box_m
        inertia += np.diag([b*b + c*c, a*a + c*c, a*a + b*b]) * mi / 12.0
    return float(total), cg, np.diag(inertia).astype(np.float32)


_MASS, _CG, _INERTIA = mass_properties(F450_PARTS)

# --- F450 propulsion -----------------------------------------------------
# 2212 920KV + 1045 on 3S, full throttle. Listings disagree: 642 g at
# 9.5 A (vendor, likely part throttle), ~700 g (generic A2212 10x4.5),
# ~1000 g at 16 A (blog). A motor + prop model (Ct 0.10-0.11,
# Cp 0.040-0.045, 9.9-11.1 V at the motor) gives 800-1100 g at
# 7400-8800 RPM. 8.0 N (~815 g) is a conservative middle; DR
# motor_thrust_scale +-15% spans 690-940 g, including battery sag.
DEFAULT_12IN = AirframeParams(
    mass_kg=_MASS,
    inertia_diag_kg_m2=_INERTIA,
    arm_length_m=ARM_LENGTH_M,
    prop_radius_m=0.127,                 # 10" prop
    # Legs mount under the bottom plate (z = -0.015) and are 140 mm.
    rest_height_m=0.155,
    max_thrust_per_motor_N=8.0,
    max_rpm=8000.0,
    # Rotor + bell inertia ~4.5e-5 kg m^2 against prop and back-EMF
    # damping at hover gives ~40 ms; spin-down is slower without active
    # braking, so 50 ms. DR motor_tau_scale +-30%.
    motor_time_constant_s=0.05,
    # Cq/Ct * D for a 10x4.5 prop: (Cp / 2pi) * D / Ct ~ 0.016-0.018 m.
    torque_coeff_per_thrust=0.017,
    # 1045 prop (11 g, tapered blade ~0.7 * m L^2 / 12 = 4.1e-5) plus the
    # 2212 bell (~20 g ring at 14 mm, 0.4e-5). Speeding a prop up or down
    # twists the frame the other way by J * dw/dt; for a 0.4 -> 0.6
    # thrust step at this motor lag that is ~0.1 N m per motor, as large
    # as the props' drag torque (0.017 m * 4 N).
    rotor_inertia_kg_m2=4.5e-5,
    # Rotor drag for ~1-2 kg quads is typically 0.2-0.5 1/s horizontally.
    drag_per_s_xy=0.25,
    drag_per_s_z=0.30,
)


def sanity_check(params: AirframeParams = DEFAULT_12IN) -> None:
    """Print derived quantities as a sanity check on the parameters."""
    print(f"Mass:                    {params.mass_kg:.3f} kg")
    print(f"CG (parts table):        {np.round(_CG, 4).tolist()} m")
    print(f"Total max thrust:        "
          f"{4 * params.max_thrust_per_motor_N:.2f} N")
    print(f"Thrust-to-weight:        {params.thrust_to_weight:.2f}")
    print(f"Hover motor_cmd (norm):  {params.hover_thrust_normalized:.3f}")
    print(f"Inertia (Ixx, Iyy, Izz): {params.inertia_diag_kg_m2.tolist()} kg·m²")
    print(f"Arm length:              {params.arm_length_m:.3f} m")


if __name__ == "__main__":
    sanity_check()
