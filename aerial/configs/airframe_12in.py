"""12-inch multirotor airframe parameters.

Placeholder estimates for a generic 12" (motor-to-motor diagonal) quad
in the long-range / cine-FPV / prosumer surveying class. Refine with
measured values from the physical airframe once available:

  * Mass: kitchen scale, no battery + battery separately
  * Arm length: motor mount center to CG, direct measurement
  * Inertia: bifilar pendulum method, or estimate from CAD
  * Motor thrust: from motor/prop combo thrust table (ecalc.ch,
    manufacturer spec sheet, or bench test)
  * Motor time constant: bench log of RPM vs commanded, first-order fit

These values get consumed by aerial/envs/hover_aerial_env.py to
parameterize the Aerial Gym multirotor model. Also consumed by any
future SITL/HITL loops so sim matches airframe.

Convention: body frame x=forward, y=left, z=up (matches Phase 0 and
Aerial Gym / ROS REP-103).
"""
from dataclasses import dataclass

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

    # --- Motor & prop -------------------------------------------------
    max_thrust_per_motor_N: float
    motor_time_constant_s: float
    # kf and km in physical units (thrust/torque per RPM^2). If the
    # simulator uses a rotor speed model instead of a first-order thrust
    # model, these can be derived from t2w and drag_coefficient.
    torque_coeff_per_thrust: float  # dimensionless, tau = k * thrust

    # --- Aerodynamics -------------------------------------------------
    linear_drag_coeff: float     # rough parasitic drag; refine with real CFD data

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


# --- 12" placeholder ---------------------------------------------------
# Generic long-range / cine-class 12" multirotor. Motor class ~2207 2000KV
# on 6S with 5-6" props, thrust ~1.5 kg per motor. Real 12" would use
# longer arm + smaller relative motors. Refine when hardware specs known.
DEFAULT_12IN = AirframeParams(
    mass_kg=0.8,
    inertia_diag_kg_m2=np.array([0.008, 0.008, 0.015], dtype=np.float32),
    arm_length_m=0.15,
    max_thrust_per_motor_N=6.0,          # ~600g per motor at max
    motor_time_constant_s=0.04,          # 40ms first-order response
    torque_coeff_per_thrust=0.016,       # typical propeller reaction torque
    linear_drag_coeff=0.05,              # rough guess; refine
)


def sanity_check(params: AirframeParams = DEFAULT_12IN) -> None:
    """Print derived quantities as a sanity check on the parameters."""
    print(f"Mass:                    {params.mass_kg:.3f} kg")
    print(f"Total max thrust:        "
          f"{4 * params.max_thrust_per_motor_N:.2f} N")
    print(f"Thrust-to-weight:        {params.thrust_to_weight:.2f}")
    print(f"Hover motor_cmd (norm):  {params.hover_thrust_normalized:.3f}")
    print(f"Inertia (Ixx, Iyy, Izz): {params.inertia_diag_kg_m2.tolist()} kg·m²")
    print(f"Arm length:              {params.arm_length_m:.3f} m")


if __name__ == "__main__":
    sanity_check()
