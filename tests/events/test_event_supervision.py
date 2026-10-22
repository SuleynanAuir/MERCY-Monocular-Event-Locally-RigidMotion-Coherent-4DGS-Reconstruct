from pathlib import Path
import tempfile
import unittest
import json
import numpy as np

from utils.event_supervision import (EventStream, log_intensity, fit_thresholds,
                                     signed_event_change, consistency_metrics, acceptance_gate,
                                     require_event_export_approval)


class EventSupervisionTests(unittest.TestCase):
    def test_interval_boundaries_and_polarities(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'events.npy'
            a = np.array([[1, 1, 0, 1], [2, 1, .1, 1], [2, 1, .1, -1], [3, 2, .2, -1]])
            np.save(path, a)
            stream = EventStream(path, 4, 3)
            count = stream.counts(0, .1)
            self.assertEqual(count.sum(), 2)
            np.testing.assert_array_equal(count[:, 1, 2], [1, 1])
            self.assertEqual(stream.counts(.1, .2).sum(), 1)
            for bad in (a[::-1], a * np.array([1, 1, 1, 2]), a + np.array([10, 0, 0, 0])):
                np.save(path, bad)
                with self.assertRaises(ValueError):
                    EventStream(path, 4, 3)

    def test_threshold_calibration_and_log_floor(self):
        counts = np.random.default_rng(7).integers(0, 4, size=(4, 2, 8, 8))
        delta = counts[:, 0] * .2 - counts[:, 1] * .3
        np.testing.assert_allclose(fit_thresholds(counts, delta), [.2, .3], atol=1e-6)
        np.testing.assert_allclose(signed_event_change(counts[0], [.2, .3]), delta[0])
        self.assertTrue(np.isfinite(log_intensity(np.array([0, .5, 1]))).all())

    def test_reversed_frozen_and_unrelated_motion_fail(self):
        observed = np.zeros((16, 16))
        observed[4:7, 4:7] = .4
        observed[8:11, 8:11] = -.4
        reference = consistency_metrics(observed, observed)
        self.assertTrue(acceptance_gate(reference, reference)['passed'])
        for predicted in (-observed, np.zeros_like(observed), np.roll(observed, 7, axis=0)):
            score = consistency_metrics(predicted, observed)
            self.assertFalse(acceptance_gate(score, reference)['passed'])

    def test_no_evidence_does_not_prove_correct_motion(self):
        zero = np.zeros((4, 4))
        result = consistency_metrics(zero, zero)
        self.assertFalse(result['observable'])
        self.assertFalse(acceptance_gate(result, result)['passed'])

    def test_additional_unsupported_motion_is_penalized(self):
        observed = np.zeros((32, 32)); observed[2:4, 2:4] = 1
        predicted = observed.copy(); predicted[20:30, 20:30] = 1
        score = consistency_metrics(predicted, observed)
        self.assertGreater(score['unsupported_change_fraction'], .9)
        self.assertFalse(acceptance_gate(score, consistency_metrics(observed, observed))['passed'])

    def test_export_gate_requires_complete_corrected_audit(self):
        score = consistency_metrics(np.ones((4, 4)), np.ones((4, 4)))
        record = {'interval': 0, 'passed': True, 'metrics': score, 'reference_metrics': score}
        report = {'format': 'lego_event_motion_audit_v1', 'approved_for_export': True,
                  'event_optimization_performed': True, 'frame_count': 2,
                  'failed_intervals': [], 'intervals': [record]}
        require_event_export_approval(report)
        for change in ({'approved_for_export': False}, {'intervals': []},
                       {'event_optimization_performed': False}, {'failed_intervals': [0]},
                       {'frame_count': 3}):
            with self.assertRaises(ValueError):
                require_event_export_approval({**report, **change})
        score['cosine'] = -1
        with self.assertRaises(ValueError):
            require_event_export_approval(report)

    def test_audit_rejects_missing_or_empty_windows(self):
        from scripts.events.audit_lego_event_motion import audit
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.npz'
            for frames, intervals in ((2, 0), (1, 0)):
                np.savez(path, images=np.zeros((frames, 8, 8), dtype=np.uint8),
                         counts=np.zeros((intervals, 2, 8, 8)),
                         metadata_json=np.asarray(json.dumps({'fps': 60, 'frame_count': frames})))
                with self.assertRaises(ValueError):
                    audit(np.zeros((frames, 8, 8)), path)


if __name__ == '__main__':
    unittest.main()
