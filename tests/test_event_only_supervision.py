import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from utils.event_only_supervision import (EVENT_ONLY_LIMITS, audit_event_prediction,
                                         build_event_windows, metrics, pooled_counts, training_spans)


class EventOnlyWindowsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, values, **kwargs):
        path = self.root / "raw.npy"
        np.save(path, np.asarray(values), allow_pickle=False)
        options = dict(g0_time=0.0, time_scale=1.0)
        options.update(kwargs)
        return build_event_windows(path, 2, 2, **options)

    def test_right_closed_boundaries_final_short_window_and_anchor(self):
        result = self.build([[0, 0, 0., 1], [0, 0, .025, 1], [1, 0, .0250001, 0],
                             [0, 1, .05, 1], [1, 1, .065, 0]])
        np.testing.assert_allclose(result["times"], [0, .025, .05, .065], rtol=0, atol=1e-14)
        self.assertEqual(result["counts"].dtype, np.uint32)
        self.assertEqual(result["counts"][0, 0, 0, 0], 1)
        self.assertEqual(result["counts"][1, 1, 0, 1], 1)
        self.assertEqual(result["counts"][1, 0, 1, 0], 1)
        self.assertEqual(result["counts"][2, 1, 1, 1], 1)
        self.assertEqual(result["counts"].sum(), 4)
        self.assertEqual(result["metadata"]["ignored_at_or_before_g0"], 1)

    def test_g0_time_is_seconds_not_raw_clock_units(self):
        result = self.build([[0, 0, 900000, 1], [0, 0, 1000000, 1], [0, 0, 1025000, -1],
                             [1, 0, 1050000, 1]], g0_time=1.0, time_scale=1e-6)
        np.testing.assert_allclose(result["times"], [0, .025, .05], rtol=0, atol=1e-12)
        self.assertEqual(result["metadata"]["ignored_at_or_before_g0"], 2)
        self.assertEqual(result["metadata"]["polarity_encoding"], "-1/+1")
        self.assertEqual(result["counts"][0, 1, 0, 0], 1)
        self.assertEqual(result["counts"][1, 0, 0, 1], 1)

    def test_exact_665_second_endpoint_is_not_dropped_or_extended(self):
        result = self.build([[0, 0, 0., 1], [1, 1, 6.65, 1]])
        self.assertEqual(len(result["counts"]), 266)
        self.assertEqual(result["times"][-1], 6.65)
        self.assertEqual(result["counts"][-1, 0, 1, 1], 1)
        self.assertEqual(result["counts"].sum(), 1)

    def test_structured_input_and_both_polarity_encodings(self):
        dtype = [("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i1")]
        first = self.build(np.array([(0, 0, .01, 1), (1, 0, .02, -1)], dtype=dtype))
        second = self.build([[0, 0, .01, 1], [1, 0, .02, 0]])
        np.testing.assert_array_equal(first["counts"], second["counts"])

    def test_validation_is_every_fifth_actual_micro_window(self):
        result = self.build([[0, 0, .25, 1]])
        np.testing.assert_array_equal(result["train_mask"], [True, True, True, True, False] * 2)
        self.assertEqual(result["metadata"]["effective_step_seconds"], .025)

    def test_chunk_boundary_events_and_uint16_overflow(self):
        values = np.tile([0, 0, .01, 1], (70000, 1))
        with patch("utils.event_only_supervision._CHUNK_SIZE", 10000):
            result = self.build(values)
        self.assertEqual(result["counts"][0, 0, 0, 0], 70000)

    def test_loads_only_explicit_raw_input_with_hash(self):
        path = self.root / "raw.npy"
        np.save(path, [[0, 0, .01, 1]], allow_pickle=False)
        with patch("utils.event_only_supervision.np.load", wraps=np.load) as loader:
            result = build_event_windows(path, 2, 2, g0_time=0, time_scale=1)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args, (path.resolve(),))
        self.assertEqual(loader.call_args.kwargs, {"mmap_mode": "r", "allow_pickle": False})
        self.assertEqual(result["metadata"]["input_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(result["metadata"]["external_images_loaded"])
        self.assertFalse(result["metadata"]["inherited_supervision_loaded"])

    def test_extensionless_raw_npy_accepted_by_magic(self):
        path = self.root / "Lego_event"
        with path.open("wb") as handle:
            np.save(handle, np.array([[0, 0, .025, 1], [1, 1, .05, 0]]), allow_pickle=False)
        result = build_event_windows(path, 2, 2, g0_time=0, time_scale=1)
        np.testing.assert_allclose(result["times"], [0, .025, .05])
        self.assertEqual(result["counts"].sum(), 2)
        self.assertEqual(result["counts"][0, 0, 0, 0], 1)
        self.assertEqual(result["counts"][1, 1, 1, 1], 1)
        self.assertEqual(result["metadata"]["input_path"], str(path.resolve()))
        self.assertEqual(result["metadata"]["input_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_renamed_archive_rejected_before_numpy_load(self):
        for name in ("Lego_event", "raw.npy"):
            with self.subTest(name=name):
                path = self.root / name
                with path.open("wb") as handle:
                    np.savez(handle, events=np.array([[0, 0, .01, 1]]))
                with patch("utils.event_only_supervision.np.load") as loader, self.assertRaises(ValueError):
                    build_event_windows(path, 2, 2, g0_time=0, time_scale=1)
                loader.assert_not_called()

    def test_malformed_data_rejected(self):
        bad = [np.empty((0, 4)), [[0, 0, .1, 1, 0]], [[0, 0, .1, 2]], [[0, 0, .1, -1], [0, 0, .2, 0]],
               [[2, 0, .1, 1]], [[0, -1, .1, 1]], [[.5, 0, .1, 1]], [[0, 0, np.nan, 1]],
               [[0, 0, .2, 1], [0, 0, .1, 1]], [[0, 0, 0, 1]]]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.build(values)

    def test_parameters_and_non_raw_archive_rejected(self):
        for kwargs in ({"time_scale": 0}, {"step_seconds": -1}, {"g0_time": float("nan")},
                       {"substeps": 0}, {"validation_stride": 1}, {"substeps": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build([[0, 0, .1, 1]], **kwargs)
        path = self.root / "not_raw.npz"
        np.savez(path, events=np.array([[0, 0, .1, 1]]))
        with self.assertRaises(ValueError):
            build_event_windows(path, 2, 2, g0_time=0, time_scale=1)

    def test_structured_extra_fields_and_unsorted_across_chunks_rejected(self):
        dtype = [("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i1"), ("image", "f8")]
        with self.assertRaises(ValueError):
            self.build(np.array([(0, 0, .1, 1, 0)], dtype=dtype))
        with patch("utils.event_only_supervision._CHUNK_SIZE", 1), self.assertRaises(ValueError):
            self.build([[0, 0, .2, 1], [0, 0, .1, 1]])


class EventOnlyTrainingHelpersTests(unittest.TestCase):
    def test_area_mean_not_sum_preserves_log_difference_units(self):
        counts = np.zeros((3, 2, 4, 6), dtype=np.uint32)
        counts[:, 0] = 8
        counts[:, 1, 0, 0] = 4
        pooled = pooled_counts(counts, 3)
        self.assertEqual(pooled.shape, (3, 2, 2, 3))
        np.testing.assert_array_equal(pooled[:, 0], 8)
        np.testing.assert_array_equal(pooled[:, 1, 0, 0], 1)
        np.testing.assert_allclose(pooled.sum() * 4, counts.sum())

    def test_invalid_pooling_and_negative_counts(self):
        for counts, width in [(np.zeros((2, 3, 6)), 3), (np.zeros((2, 4, 6)), 4),
                              (-np.ones((2, 4, 4)), 2), (np.zeros((3, 4, 4)), 2)]:
            with self.subTest(shape=counts.shape, width=width), self.assertRaises(ValueError):
                pooled_counts(counts, width)

    def test_training_spans_never_cross_or_include_heldout(self):
        mask = np.array([True, True, True, True, False, True, False, True, True, False])
        spans = training_spans(mask, 4)
        self.assertIn((0, 4), spans)
        self.assertIn((5, 6), spans)
        self.assertIn((7, 9), spans)
        self.assertTrue(all(1 <= stop - start <= 4 and mask[start:stop].all() for start, stop in spans))
        self.assertNotIn((3, 6), spans)
        self.assertEqual(training_spans(np.zeros(5, dtype=bool)), [])

    def test_bad_training_masks_rejected(self):
        for mask in ([], [1, 0, 1], np.ones((2, 2), dtype=bool)):
            with self.assertRaises(ValueError):
                training_spans(mask)


class EventOnlyMetricsTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.zeros((2, 2, 2), dtype=np.uint32)
        self.counts[0, 0, 0] = 1
        self.counts[1, 1, 1] = 1
        self.expected = .2 * self.counts[0].astype(float) - .2 * self.counts[1]

    def test_perfect_signed_prediction_passes_fixed_metrics(self):
        result = metrics(self.expected, self.counts)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["cosine"], 1)
        self.assertEqual(result["polarity_agreement"], 1)
        self.assertEqual(result["relative_l1"], 0)
        with self.assertRaises(TypeError):
            EVENT_ONLY_LIMITS["minimum_cosine"] = 0

    def test_static_and_reversed_predictions_fail_active_events(self):
        for predicted in (np.zeros((2, 2)), -self.expected):
            result = metrics(predicted, self.counts)
            self.assertFalse(result["passed"])
            self.assertIn("cosine", result["reasons"])
            self.assertIn("polarity_agreement", result["reasons"])
        self.assertAlmostEqual(metrics(-self.expected, self.counts)["cosine"], -1)

    def test_zero_events_abstain_even_when_prediction_is_static(self):
        result = metrics(np.zeros((2, 2)), np.zeros_like(self.counts))
        self.assertEqual(result["status"], "abstain_no_events")
        self.assertFalse(result["passed"])
        self.assertIsNone(result["cosine"])
        moving = metrics(np.ones((2, 2)), np.zeros_like(self.counts))
        self.assertEqual(moving["unsupported_change_fraction"], 1)

    def test_balanced_polarity_window_abstains(self):
        result = metrics(np.zeros((2, 2)), np.ones_like(self.counts))
        self.assertEqual(result["status"], "abstain_signed_cancellation")
        self.assertFalse(result["passed"])

    def test_motion_outside_event_support_fails(self):
        predicted = self.expected.copy()
        predicted[0, 1] = 1
        result = metrics(predicted, self.counts)
        self.assertIn("unsupported_change_fraction", result["reasons"])

    def test_invalid_metrics_input_fails_closed(self):
        for predicted, counts, kwargs in [(self.expected, -np.ones_like(self.counts, dtype=float), {}),
                                           (np.full((2, 2), np.nan), self.counts, {}),
                                           (self.expected, self.counts, {"threshold_positive": 0}),
                                           (self.expected, self.counts, {"threshold_negative": float("inf")})]:
            with self.assertRaises(ValueError):
                metrics(predicted, counts, **kwargs)

    def test_uncalibrated_or_full_fit_never_claims_strict_approval(self):
        predicted = np.repeat(self.expected[None], 5, axis=0)
        counts = np.repeat(self.counts[None], 5, axis=0)
        mask = np.array([True, True, True, True, False])
        candidate = audit_event_prediction(predicted, counts, train_mask=mask)
        self.assertEqual(candidate["summary"]["all"]["passed"], 5)
        self.assertFalse(candidate["approved_for_export"])
        self.assertIn("contrast_thresholds_unconfirmed", candidate["approval_blockers"])
        fitted_all = audit_event_prediction(predicted, counts, train_mask=mask, fit_split="all", thresholds_confirmed=True)
        self.assertFalse(fitted_all["approved_for_export"])
        self.assertEqual(fitted_all["evaluation_scope"], "in_sample_consistency_only")
        self.assertEqual(fitted_all["validation_indices"], [])
        self.assertEqual(fitted_all["summary"]["validation_events"]["intervals"], 0)

    def test_confirmed_thresholds_need_heldout_and_all_observable_passing(self):
        predicted = np.repeat(self.expected[None], 5, axis=0)
        counts = np.repeat(self.counts[None], 5, axis=0)
        mask = np.array([True, True, True, True, False])
        result = audit_event_prediction(predicted, counts, train_mask=mask, thresholds_confirmed=True)
        self.assertTrue(result["approved_for_export"])
        self.assertEqual(result["validation_indices"], [4])
        counts[4] = 0
        result = audit_event_prediction(predicted, counts, train_mask=mask, thresholds_confirmed=True)
        self.assertFalse(result["approved_for_export"])
        self.assertIn("unobservable_intervals", result["approval_blockers"])

    def test_train_audit_requires_actual_mask(self):
        with self.assertRaises(ValueError):
            audit_event_prediction(self.expected[None], self.counts[None])
        with self.assertRaises(ValueError):
            audit_event_prediction(self.expected[None], self.counts[None], train_mask=[1])


if __name__ == "__main__":
    unittest.main()
