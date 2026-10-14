"""Rigid-rotor kinematics, independent of CUDA and of learned deformation."""

import math
import numpy as np


def spin_profile(time, rpm, ramp_seconds):
    """Return angle, angular velocity and acceleration in rad, rad/s, rad/s².

    Speed follows a quintic smoothstep from rest to rpm. Its first two
    derivatives vanish at both ramp boundaries. Angle is its exact integral,
    not an accumulated Euler step. This prescribes motion; it is not a motor
    torque/airflow simulation. Set ramp_seconds=0 for constant speed.
    """
    t = np.asarray(time, dtype=np.float64)
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("time must be finite and nonnegative")
    if not math.isfinite(rpm) or rpm < 0:
        raise ValueError("rpm must be finite and nonnegative")
    if not math.isfinite(ramp_seconds) or ramp_seconds < 0:
        raise ValueError("ramp_seconds must be finite and nonnegative")
    omega = rpm * 2 * math.pi / 60
    if ramp_seconds == 0:
        return omega * t, np.full_like(t, omega), np.zeros_like(t)
    s = np.clip(t / ramp_seconds, 0, 1)
    speed = omega * (10 * s**3 - 15 * s**4 + 6 * s**5)
    acceleration = omega / ramp_seconds * 30 * s**2 * (1 - s)**2
    ramp_angle = omega * ramp_seconds * (2.5 * s**4 - 3 * s**5 + s**6)
    angle = np.where(t < ramp_seconds, ramp_angle, omega * (t - ramp_seconds / 2))
    return angle, speed, acceleration


def axis_angle(axis, angle):
    """Right-handed rotation matrix and scalar-first unit quaternion."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("axis must be a finite nonzero 3-vector")
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    c, s = math.cos(angle), math.sin(angle)
    matrix = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * skew
    quaternion = np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]
    return matrix, quaternion
