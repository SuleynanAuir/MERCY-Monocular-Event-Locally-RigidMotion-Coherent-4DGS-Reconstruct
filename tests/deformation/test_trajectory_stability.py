import unittest
import numpy as np
from utils.trajectory_stability import trajectory_metrics


class TrajectoryStabilityTests(unittest.TestCase):
    def test_constant_speed_has_no_acceleration_or_highpass(self):
        x = np.zeros((120, 2, 3)); x[:, :, 0] = np.arange(120)[:, None]/60
        result = trajectory_metrics(x, 60, 1)
        self.assertLess(result['acceleration_rms_scene_per_s2'], 1e-10)
        self.assertLess(result['cubic_highpass_rms_scene'], 1e-12)

    def test_small_fast_oscillation_is_detected_on_smooth_motion(self):
        time = np.arange(400)/60
        smooth = np.zeros((400, 1, 3)); smooth[:, 0, 0] = np.sin(2*np.pi*.5*time)
        noisy = smooth.copy(); noisy[:, 0, 0] += .01*np.sin(2*np.pi*15*time)
        a, b = [trajectory_metrics(x, 60, 1) for x in (smooth, noisy)]
        self.assertGreater(b['cubic_highpass_rms_scene'], a['cubic_highpass_rms_scene']*10)
        self.assertGreater(b['velocity_power_above_cutoff_fraction'], a['velocity_power_above_cutoff_fraction']*10)

    def test_translation_and_scene_scale_do_not_change_scores(self):
        x = np.random.default_rng(7).normal(size=(40, 3, 3))
        a, b = trajectory_metrics(x, 60, 2), trajectory_metrics(x*3+10, 60, 6)
        for key in ('jerk_rms_scene_per_s3', 'cubic_highpass_rms_scene', 'velocity_power_above_cutoff_fraction'):
            self.assertAlmostEqual(a[key], b[key])

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_metrics(np.zeros((3, 1, 3)), 60, 1)
        with self.assertRaises(ValueError):
            trajectory_metrics(np.full((9, 1, 3), np.nan), 60, 1)


if __name__ == '__main__':
    unittest.main()
