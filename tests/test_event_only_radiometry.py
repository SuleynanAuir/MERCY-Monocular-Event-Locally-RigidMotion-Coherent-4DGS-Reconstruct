import json
import unittest
from unittest.mock import patch

import numpy as np

from utils.event_only_radiometry import infer_background_from_events


class EventOnlyRadiometryTests(unittest.TestCase):
    def scene(self):
        gray = np.zeros((20, 20), dtype=np.float64)
        alpha = np.zeros_like(gray)
        gray[4:16, 4:16] = np.linspace(.25, .9, 144).reshape(12, 12)
        alpha[4:16, 4:16] = 1
        departed = np.zeros_like(gray)
        mask = alpha == 1
        departed[mask] = np.log(.04 / gray[mask])
        return gray, alpha, departed

    def test_known_background_after_foreground_vacates(self):
        gray, alpha, departed = self.scene()
        result = infer_background_from_events(gray, alpha, iter([np.zeros_like(gray), departed]))
        self.assertAlmostEqual(result["estimate"], .04, places=12)
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["usable_support_pixels"], 144)
        self.assertEqual(diagnostics["status"], "heuristic_candidate")
        for key in ("calibrated", "contrast_thresholds_confirmed", "identity_and_time_confirmed", "motion_confirmed"):
            self.assertFalse(diagnostics[key])
        json.dumps(result, allow_nan=False)

    def test_minimum_is_retained_after_foreground_returns(self):
        gray, alpha, departed = self.scene()
        result = infer_background_from_events(gray, alpha, [departed, np.zeros_like(gray)])
        self.assertAlmostEqual(result["estimate"], .04)

    def test_alpha_aware_compositing_recovers_background_without_pixel_offsets(self):
        gray, alpha, _ = self.scene()
        alpha *= .95
        departed = np.zeros_like(gray)
        selected = alpha > 0
        initial_sensor_intensity = gray[selected] + (1 - alpha[selected]) * .04
        departed[selected] = np.log(.04 / initial_sensor_intensity)
        result = infer_background_from_events(gray, alpha, [departed])
        self.assertAlmostEqual(result["estimate"], .04, places=12)
        self.assertTrue(result["diagnostics"]["background_contrast_coupled"])
        self.assertFalse(result["diagnostics"]["per_pixel_correction_fitted"])
        self.assertEqual(result["diagnostics"]["estimated_parameter"], "one_global_static_background_scalar")

    def test_empty_zero_and_brightening_events_do_not_invent_background(self):
        gray, alpha, _ = self.scene()
        for changes in ([], [np.zeros_like(gray)], [np.ones_like(gray)]):
            with self.subTest(windows=len(changes)):
                result = infer_background_from_events(gray, alpha, changes)
                self.assertIsNone(result["estimate"])
                self.assertEqual(result["diagnostics"]["status"], "insufficient_evidence")

    def test_small_dimming_is_not_taken_as_departure(self):
        gray, alpha, _ = self.scene()
        result = infer_background_from_events(gray, alpha, [-.1 * np.ones_like(gray)])
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["diagnostics"]["dimming_support_pixels"], 0)

    def test_insufficient_motion_support_returns_none(self):
        gray, alpha, departed = self.scene()
        departed[6:, :] = 0
        result = infer_background_from_events(gray, alpha, [departed])
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["diagnostics"]["reason"], "insufficient_dimming_support")

    def test_support_fraction_is_enforced_separately_from_absolute_count(self):
        gray, alpha, departed = self.scene()
        departed[6:, :] = 0
        result = infer_background_from_events(gray, alpha, [departed], minimum_support=8, minimum_support_fraction=.5)
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["diagnostics"]["required_support_pixels"], 72)

    def test_low_alpha_and_dark_pixels_are_not_foreground_anchors(self):
        gray, alpha, departed = self.scene()
        for changed_gray, changed_alpha in ((gray, alpha * .5), (gray * .01, alpha)):
            result = infer_background_from_events(changed_gray, changed_alpha, [departed])
            self.assertIsNone(result["estimate"])
            self.assertEqual(result["diagnostics"]["reason"], "insufficient_bright_opaque_foreground")

    def test_low_quantile_resists_a_few_dark_outliers_and_nonvacated_surface(self):
        gray, alpha, departed = self.scene()
        indices = np.flatnonzero(alpha)
        departed.flat[indices[:8]] = np.log(.0001 / gray.flat[indices[:8]])
        departed.flat[indices[100:]] = np.log(.12 / gray.flat[indices[100:]])
        result = infer_background_from_events(gray, alpha, [departed])
        self.assertAlmostEqual(result["estimate"], .04)
        self.assertGreater(result["diagnostics"]["candidate_quantiles"]["q75"], .04)

    def test_singleton_rasterizer_alpha_is_supported(self):
        gray, alpha, departed = self.scene()
        result = infer_background_from_events(gray, alpha[None], [departed])
        self.assertAlmostEqual(result["estimate"], .04)

    def test_streaming_iterator_can_reuse_one_buffer_without_materialization(self):
        gray, alpha, departed = self.scene()

        class ReusingStream:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Do not materialize the time stack")

            def __iter__(self):
                buffer = np.zeros_like(gray)
                yield buffer
                buffer[:] = departed
                yield buffer
                buffer[:] = 0
                yield buffer

        gray_copy, alpha_copy = gray.copy(), alpha.copy()
        with patch("builtins.open", side_effect=AssertionError("No file access is permitted")):
            result = infer_background_from_events(gray, alpha, ReusingStream())
        self.assertAlmostEqual(result["estimate"], .04)
        self.assertEqual(result["diagnostics"]["cumulative_windows_seen"], 3)
        np.testing.assert_array_equal(gray, gray_copy)
        np.testing.assert_array_equal(alpha, alpha_copy)

    def test_invalid_data_fails_closed_even_after_valid_support(self):
        gray, alpha, departed = self.scene()
        bad_cases = [(gray, alpha, [departed, np.full_like(gray, np.nan)]),
                     (gray, alpha, [np.ones((2, 2))]), (gray, alpha, None),
                     (-gray, alpha, [departed]), (gray, alpha * 2, [departed]),
                     (gray[None], alpha, [departed]), (gray, alpha[:, :-1], [departed]),
                     (gray.astype(complex), alpha, [departed])]
        for first, second, changes in bad_cases:
            with self.subTest(shape=np.shape(first)), self.assertRaises(ValueError):
                infer_background_from_events(first, second, changes)

    def test_numeric_underflow_does_not_become_a_calibrated_zero_background(self):
        gray, alpha, _ = self.scene()
        result = infer_background_from_events(gray, alpha, [-1000 * np.ones_like(gray)])
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["diagnostics"]["numerically_unrepresentable_candidates"], 144)

    def test_invalid_selection_parameters_are_rejected(self):
        gray, alpha, departed = self.scene()
        for parameters in ({"minimum_support": True}, {"minimum_support": 0},
                           {"minimum_alpha": 1.1}, {"minimum_reference_gray": 0},
                           {"minimum_log_decrease": -1}, {"minimum_support_fraction": 2},
                           {"estimate_quantile": 0}, {"estimate_quantile": .8},
                           {"minimum_log_decrease": float("nan")}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                infer_background_from_events(gray, alpha, [departed], **parameters)


if __name__ == "__main__":
    unittest.main()
