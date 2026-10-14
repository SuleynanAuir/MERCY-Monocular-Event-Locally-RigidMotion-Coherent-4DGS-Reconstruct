import math
import unittest

import numpy as np

from utils.rigid_motion import axis_angle, spin_profile


class RigidMotionTests(unittest.TestCase):
    def test_profile_boundaries(self):
        angle, speed, acceleration = spin_profile(np.array([0., 1., 2., 3.]), 60, 2)
        np.testing.assert_allclose(speed, [0, math.pi, 2 * math.pi, 2 * math.pi])
        np.testing.assert_allclose(angle[[0, 2, 3]], [0, 2 * math.pi, 4 * math.pi])
        np.testing.assert_allclose(acceleration[[0, 2, 3]], 0)

    def test_angle_is_integral_of_speed(self):
        t = np.array([.1, .7, 1.5, 2., 3.])
        h = 1e-5
        derivative = (spin_profile(t + h, 73, 2)[0] - spin_profile(t - h, 73, 2)[0]) / (2 * h)
        np.testing.assert_allclose(derivative, spin_profile(t, 73, 2)[1], atol=1e-8)

    def test_speed_derivative_and_monotonicity(self):
        t = np.linspace(.01, 3, 100)
        h = 1e-5
        derivative = (spin_profile(t + h, 60, 2)[1] - spin_profile(t - h, 60, 2)[1]) / (2 * h)
        np.testing.assert_allclose(derivative, spin_profile(t, 60, 2)[2], atol=1e-8)
        self.assertTrue(np.all(np.diff(spin_profile(t, 60, 2)[1]) >= -1e-12))

    def test_constant_and_stopped_profiles(self):
        np.testing.assert_allclose(spin_profile(1.5, 60, 0), [3 * math.pi, 2 * math.pi, 0])
        np.testing.assert_allclose(spin_profile(np.arange(4), 0, 2), 0)

    def test_rigid_geometry_and_full_turn(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(100, 3))
        center = np.array([2., -1., .3])
        axis = np.array([1., 2., 3.])
        axis /= np.linalg.norm(axis)
        rotation, quaternion = axis_angle(axis, .73)
        moved = (points - center) @ rotation.T + center
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)
        self.assertAlmostEqual(np.linalg.norm(quaternion), 1)
        np.testing.assert_allclose((moved - center) @ axis, (points - center) @ axis, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(moved - moved[0], axis=1), np.linalg.norm(points - points[0], axis=1))
        np.testing.assert_allclose(axis_angle(axis, 2 * math.pi)[0], np.eye(3), atol=1e-14)

    def test_clockwise_sign(self):
        rotation, _ = axis_angle([0, 0, 1], -math.pi / 2)
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, -1, 0], atol=1e-14)

    def test_quaternion_matches_position_rotation(self):
        rotation, (w, x, y, z) = axis_angle([1, -2, 3], -1.2)
        from_quaternion = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        np.testing.assert_allclose(rotation, from_quaternion, atol=1e-14)

    def test_invalid_inputs(self):
        for time, rpm, ramp in [(-1, 60, 2), (0, -1, 2), (0, 60, -1), (np.nan, 60, 2)]:
            with self.assertRaises(ValueError):
                spin_profile(time, rpm, ramp)
        with self.assertRaises(ValueError):
            axis_angle([0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()
