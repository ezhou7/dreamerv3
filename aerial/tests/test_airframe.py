"""The URDF and airframe_12in.py must describe the same vehicle.

gym-pybullet-drones reads mass, thrust and drag from the URDF; the rate
controller, hover trim and DR read airframe_12in.py.

Run:  .venv-pybullet/bin/python -m pytest aerial/tests/test_airframe.py -q
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from aerial.configs.airframe_12in import DEFAULT_12IN as A

URDF = os.path.join(os.path.dirname(__file__), "..", "assets", "cf12in.urdf")
G_PYBULLET = 9.8   # BaseAviary.G


@pytest.fixture(scope="module")
def urdf():
    root = ET.parse(URDF).getroot()
    props = {k: float(v) for k, v in root.find("properties").attrib.items()}
    inertial = root.find("link[@name='base_link']/inertial")
    props["mass"] = float(inertial.find("mass").get("value"))
    inertia = inertial.find("inertia")
    props["inertia"] = np.array(
        [float(inertia.get(k)) for k in ("ixx", "iyy", "izz")])
    return props


def test_mass_and_inertia(urdf):
    assert urdf["mass"] == pytest.approx(A.mass_kg, rel=1e-3)
    np.testing.assert_allclose(urdf["inertia"], A.inertia_diag_kg_m2, rtol=2e-3)


def test_thrust_and_yaw_torque(urdf):
    max_rpm = np.sqrt(urdf["thrust2weight"] * G_PYBULLET * urdf["mass"]
                      / (4 * urdf["kf"]))
    assert max_rpm == pytest.approx(A.max_rpm, rel=1e-3)
    assert urdf["kf"] * max_rpm**2 == pytest.approx(
        A.max_thrust_per_motor_N, rel=2e-3)
    assert urdf["km"] / urdf["kf"] == pytest.approx(
        A.torque_coeff_per_thrust, rel=1e-3)


def test_geometry(urdf):
    assert urdf["arm"] == pytest.approx(A.arm_length_m)
    assert urdf["prop_radius"] == pytest.approx(A.prop_radius_m)


def test_drag_matches_per_second_coefficients(urdf):
    hover_rad_s = (4 * A.max_rpm * np.sqrt(A.hover_thrust_normalized)
                   * 2 * np.pi / 60)
    assert urdf["drag_coeff_xy"] * hover_rad_s / A.mass_kg == pytest.approx(
        A.drag_per_s_xy, rel=5e-3)
    assert urdf["drag_coeff_z"] * hover_rad_s / A.mass_kg == pytest.approx(
        A.drag_per_s_z, rel=5e-3)
