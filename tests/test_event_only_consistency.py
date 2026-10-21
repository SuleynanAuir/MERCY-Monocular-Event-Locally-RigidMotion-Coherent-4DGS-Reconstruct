import json
import unittest
from unittest.mock import patch

import numpy as np

from utils.event_only_consistency import event_consistency_diagnostics
from utils.event_only_supervision import EVENT_ONLY_LIMITS, audit_event_prediction


class EventOnlyConsistencyTests(unittest.TestCase):
    def test_static_prediction_with_large_accumulated_events_exceeds_bound(self):
        counts = np.zeros((5, 2, 2, 2), dtype=np.uint32)
        counts[:, 0, 0, 0] = 1
        result = event_consistency_diagnostics(np.zeros((5, 2, 2)), counts)
        native = result["temporal_scales"][0]
        self.assertEqual(native["summary"]["bound_exceeding_pixel_endpoints"], 0)
        total = result["prefixes"]["full_timeline"]
        self.assertAlmostEqual(total["endpoint_residual_bound"], .4)
        self.assertEqual(total["bound_exceeding_pixels"], 1)
        self.assertAlmostEqual(total["bound_exceeding_fraction"], .25)
        self.assertAlmostEqual(total["excess_l1"], .6)
        self.assertAlmostEqual(total["excess_relative_to_observed_l1"], .6)
        self.assertFalse(result["approved_for_export"])

    def test_subthreshold_no_event_changes_have_no_excess_but_are_not_validated(self):
        predicted = np.array([[[.15, -.15]], [[.05, -.05]]])
        counts = np.zeros((2, 2, 1, 2), dtype=np.uint32)
        result = event_consistency_diagnostics(predicted, counts)
        total = result["prefixes"]["full_timeline"]
        self.assertEqual(total["excess_l1"], 0)
        self.assertEqual(total["no_event_pixels"], 2)
        self.assertIsNone(total["excess_relative_to_observed_l1"])
        self.assertEqual(total["existing_native_metric_diagnostics"]["status"], "abstain_no_events")
        self.assertFalse(total["existing_native_metric_diagnostics"]["passed"])
        self.assertFalse(result["motion_validated"])

    def test_cancelling_events_never_imply_motion_validated(self):
        counts = np.ones((4, 2, 2, 2), dtype=np.uint32)
        result = event_consistency_diagnostics(np.zeros((4, 2, 2)), counts)
        total = result["prefixes"]["full_timeline"]
        self.assertEqual(total["net_cancellation_pixels"], 4)
        self.assertEqual(total["excess_l1"], 0)
        self.assertEqual(total["existing_native_metric_diagnostics"]["status"], "abstain_signed_cancellation")
        self.assertFalse(total["motion_validated"])
        self.assertFalse(result["approved_for_export"])

    def test_prefix_reveals_error_hidden_by_final_signed_cancellation(self):
        counts = np.zeros((8, 2, 1, 1), dtype=np.uint32)
        counts[:4, 0] = 1
        counts[4:, 1] = 1
        result = event_consistency_diagnostics(np.zeros((8, 1, 1)), counts)
        self.assertEqual(result["prefixes"]["full_timeline"]["excess_l1"], 0)
        fourth = result["prefixes"]["intervals"][3]
        self.assertEqual(fourth["stop_index_exclusive"], 4)
        self.assertAlmostEqual(fourth["excess_l1"], .4)

    def test_asymmetric_contrasts_use_their_sum_not_double_one_threshold(self):
        predicted = np.array([[[.55, -.41]]])
        counts = np.zeros((1, 2, 1, 2), dtype=np.uint32)
        result = event_consistency_diagnostics(predicted, counts, .3, .1)
        total = result["prefixes"]["full_timeline"]
        self.assertAlmostEqual(total["endpoint_residual_bound"], .4)
        self.assertAlmostEqual(total["excess_l1"], .16)

    def test_temporal_sum_telescopes_known_endpoint_residuals(self):
        residual = np.array([-.1, .05, -.15, .12, -.08, .1])
        counts = np.zeros((5, 2, 1, 1), dtype=np.uint32)
        counts[:, 0] = 1
        predicted = (.2 + np.diff(residual))[:, None, None]
        result = event_consistency_diagnostics(predicted, counts)
        for scale in result["temporal_scales"]:
            self.assertEqual(scale["summary"]["excess_l1"], 0)
        self.assertEqual(result["prefixes"]["summary"]["excess_l1"], 0)

    def test_last_short_block_and_maximum_sixteen_prefixes_include_timeline_end(self):
        result = event_consistency_diagnostics(np.zeros((37, 1, 1)), np.zeros((37, 2, 1, 1)), temporal_scales=(4,))
        scale = result["temporal_scales"][0]
        self.assertTrue(scale["includes_short_final_block"])
        self.assertEqual(scale["intervals"][-1]["window_count"], 1)
        self.assertEqual(scale["intervals"][-1]["stop_index_exclusive"], 37)
        self.assertEqual(len(result["prefixes"]["intervals"]), 16)
        self.assertEqual(result["prefixes"]["stop_indices_exclusive"][0], 1)
        self.assertEqual(result["prefixes"]["stop_indices_exclusive"][-1], 37)
        one = event_consistency_diagnostics(np.zeros((3, 1, 1)), np.zeros((3, 2, 1, 1)), max_prefixes=1)
        self.assertEqual(one["prefixes"]["stop_indices_exclusive"], [3])

    def test_exact_prediction_leaves_existing_strict_audit_unchanged(self):
        counts = np.zeros((5, 2, 2, 2), dtype=np.uint32)
        counts[:, 0, 0, 0] = 1
        predicted = .2 * counts[:, 0]
        kwargs = {"fit_split": "all"}
        original = audit_event_prediction(predicted, counts, **kwargs)
        limits = dict(EVENT_ONLY_LIMITS)
        with patch("builtins.open", side_effect=AssertionError("No external inputs may be read")):
            result = event_consistency_diagnostics(predicted, counts)
        self.assertFalse(result["native_strict_gates_changed"])
        self.assertEqual(dict(EVENT_ONLY_LIMITS), limits)
        self.assertEqual(audit_event_prediction(predicted, counts, **kwargs), original)
        self.assertEqual(result["prefixes"]["full_timeline"]["excess_l1"], 0)
        self.assertFalse(result["approved_for_export"])
        json.dumps(result, allow_nan=False)

    def test_input_arrays_are_not_modified_and_integer_sums_do_not_wrap(self):
        counts = np.full((4, 2, 1, 1), 50000, dtype=np.uint16)
        counts[:, 1] = 0
        predicted = np.zeros((4, 1, 1), dtype=np.float32)
        original = counts.copy()
        result = event_consistency_diagnostics(predicted, counts)
        np.testing.assert_array_equal(counts, original)
        self.assertEqual(result["prefixes"]["full_timeline"]["existing_native_metric_diagnostics"]["event_mass"], 200000)
        np.testing.assert_array_equal(predicted, 0)

    def test_invalid_arrays_and_noninteger_counts_fail_closed(self):
        valid_counts = np.zeros((2, 2, 1, 1))
        bad = [(np.zeros((0, 1, 1)), np.zeros((0, 2, 1, 1))),
               (np.zeros((2, 1, 1)), np.zeros((2, 2, 2, 1))),
               (np.full((2, 1, 1), np.nan), valid_counts),
               (np.zeros((2, 1, 1)), -np.ones_like(valid_counts)),
               (np.zeros((2, 1, 1)), np.ones_like(valid_counts) * .5),
               (np.zeros((2, 1, 1), dtype=complex), valid_counts)]
        for predicted, counts in bad:
            with self.subTest(shape=predicted.shape), self.assertRaises(ValueError):
                event_consistency_diagnostics(predicted, counts)

    def test_invalid_thresholds_scales_and_prefix_counts_fail_closed(self):
        predicted, counts = np.zeros((2, 1, 1)), np.zeros((2, 2, 1, 1))
        for kwargs in ({"threshold_positive": 0}, {"threshold_negative": float("inf")},
                       {"temporal_scales": ()}, {"temporal_scales": (True,)},
                       {"temporal_scales": (1.5,)}, {"max_prefixes": 0}, {"max_prefixes": 17}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                event_consistency_diagnostics(predicted, counts, **kwargs)


if __name__ == "__main__":
    unittest.main()
