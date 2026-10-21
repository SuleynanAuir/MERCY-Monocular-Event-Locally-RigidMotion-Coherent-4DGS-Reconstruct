import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from utils.event_only_supervision import audit_event_prediction


SPEC = importlib.util.spec_from_file_location(
    "diagnose_event_only_run", Path(__file__).resolve().parents[1] / "scripts" / "diagnose_event_only_run.py")
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSIS)


class EventOnlyRunDiagnosisTests(unittest.TestCase):
    def reports(self, *, fit_split="all", confirmed=False, static=False):
        counts = np.zeros((5, 2, 2, 2), dtype=np.uint32)
        counts[:, 0, 0, 0] = 1
        counts[:, 1, 1, 1] = 1
        predicted = .2 * (counts[:, 0].astype(float) - counts[:, 1])
        if static:
            predicted[:] = 0
        audit = audit_event_prediction(predicted, counts, train_mask=np.array([1, 1, 1, 1, 0], dtype=bool),
                                       fit_split=fit_split, thresholds_confirmed=confirmed)
        hashes = {key: str(index) * 64 for index, key in enumerate(DIAGNOSIS.INPUT_KEYS)}
        audit.update(input_sha256=hashes, whole_timeline_fitted=True, identity_and_time_confirmed=confirmed)
        training = {"training_performed": True, "whole_timeline_fitted": True, "fit_split": fit_split,
                    "explicitly_optimized_window_count": 5 if fit_split == "all" else 4,
                    "unoptimized_fitting_windows": [], "thresholds": [.2, .2],
                    "thresholds_confirmed": confirmed, "identity_and_time_confirmed": confirmed}
        provenance = {"input_sha256": hashes, "identity_and_time_confirmed": confirmed, "legacy_data_used": False}
        return audit, training, provenance

    def test_whole_fit_is_not_approval_for_all_fit_or_assumed_thresholds(self):
        result = DIAGNOSIS.diagnose_reports(*self.reports())
        self.assertTrue(result["coverage"]["whole_fitting_split_optimized"])
        self.assertTrue(result["native_event_check_passed"])
        self.assertFalse(result["approved_for_export"])
        self.assertEqual(result["evaluation_scope"], "in_sample_consistency_only")
        self.assertEqual(result["coverage"]["heldout_windows"], 0)
        for blocker in ("in_sample_consistency_only", "contrast_thresholds_unconfirmed", "g0_identity_or_time_unconfirmed"):
            self.assertIn(blocker, result["approval_blockers"])

    def test_analytic_static_baseline_and_signed_differences(self):
        result = DIAGNOSIS.diagnose_reports(*self.reports())
        baseline = result["baseline"]
        self.assertFalse(result["new_render_performed"])
        self.assertFalse(result["coarse_metrics_available"])
        self.assertEqual(baseline["metrics"]["metric_means"],
                         {"cosine": 0., "polarity_agreement": 0., "relative_l1": 1., "unsupported_change_fraction": 0.})
        self.assertAlmostEqual(baseline["candidate_minus_baseline"]["cosine"], 1.)
        self.assertEqual(baseline["candidate_minus_baseline"]["relative_l1"], -1.)
        self.assertEqual(baseline["metrics"]["passed"], 0)

    def test_static_candidate_cannot_pass_active_event_windows(self):
        result = DIAGNOSIS.diagnose_reports(*self.reports(static=True))
        self.assertFalse(result["native_event_check_passed"])
        self.assertEqual(result["failure_reason_counts"]["cosine"], 5)
        self.assertTrue(all(value == 0 for value in result["baseline"]["candidate_minus_baseline"].values()))

    def test_valid_confirmed_heldout_report_can_pass_narrow_event_check(self):
        result = DIAGNOSIS.diagnose_reports(*self.reports(fit_split="train", confirmed=True))
        self.assertTrue(result["approved_for_export"])
        self.assertEqual(result["state"], "strict_event_check_passed")
        self.assertFalse(result["coverage"]["all_event_windows_used_for_training"])

    def test_unknown_identity_blocks_even_if_source_audit_says_approved(self):
        audit, training, provenance = self.reports(fit_split="train", confirmed=True)
        provenance["identity_and_time_confirmed"] = False
        result = DIAGNOSIS.diagnose_reports(audit, training, provenance)
        self.assertFalse(result["approved_for_export"])
        self.assertIn("g0_identity_or_time_unconfirmed", result["approval_blockers"])

    def test_missing_audit_is_running_or_incomplete_not_passed(self):
        _, training, provenance = self.reports()
        result = DIAGNOSIS.diagnose_reports(None, training, provenance)
        self.assertEqual(result["state"], "running_or_incomplete")
        self.assertFalse(result["approved_for_export"])
        self.assertFalse(result["baseline"]["available"])
        self.assertIn("event-audit.json", result["missing_reports"])

    def test_false_whole_fit_claim_cannot_hide_incomplete_optimization(self):
        audit, training, provenance = self.reports(fit_split="train", confirmed=True)
        training.update(explicitly_optimized_window_count=3, unoptimized_fitting_windows=[3])
        result = DIAGNOSIS.diagnose_reports(audit, training, provenance)
        self.assertFalse(result["coverage"]["whole_fitting_split_optimized"])
        self.assertFalse(result["approved_for_export"])

    def test_missing_or_mismatched_input_hashes_block_approval(self):
        audit, training, provenance = self.reports(fit_split="train", confirmed=True)
        provenance = copy.deepcopy(provenance)
        provenance["input_sha256"]["events"] = "f" * 64
        result = DIAGNOSIS.diagnose_reports(audit, training, provenance)
        self.assertIn("input_provenance_hashes_missing_or_inconsistent", result["approval_blockers"])

    def test_declared_pass_cannot_override_native_fixed_metrics(self):
        audit, training, provenance = self.reports(fit_split="train", confirmed=True)
        audit["intervals"][0]["cosine"] = .4
        result = DIAGNOSIS.diagnose_reports(audit, training, provenance)
        self.assertFalse(result["approved_for_export"])
        self.assertEqual(result["failed_intervals"], [0])
        self.assertIn("interval_0_reported_pass_disagrees_with_fixed_limits", result["report_consistency_issues"])

    def test_no_interval_rows_is_invalid_not_approved(self):
        audit, training, provenance = self.reports()
        del audit["intervals"]
        result = DIAGNOSIS.diagnose_reports(audit, training, provenance)
        self.assertEqual(result["state"], "invalid_reports")
        self.assertFalse(result["approved_for_export"])

    def test_zero_events_and_signed_cancellation_abstain_in_both_comparisons(self):
        for event_mass in (0., 2.):
            with self.subTest(event_mass=event_mass):
                audit, training, provenance = self.reports()
                row = audit["intervals"][0]
                row.update(event_mass=event_mass, observed_change_l1=0., observable=False,
                           passed=False, cosine=None, polarity_agreement=None, relative_l1=None)
                result = DIAGNOSIS.diagnose_reports(audit, training, provenance)
                self.assertEqual(result["metrics"]["all"]["abstained"], 1)
                self.assertEqual(result["baseline"]["metrics"]["abstained"], 1)
                self.assertFalse(result["native_event_check_passed"])
                self.assertIn("unobservable_intervals", result["approval_blockers"])

    def test_non_object_reports_fail_closed(self):
        audit, training, provenance = self.reports()
        for reports in (([], training, provenance), (audit, 3, provenance), (audit, training, "invalid")):
            result = DIAGNOSIS.diagnose_reports(*reports)
            self.assertEqual(result["state"], "invalid_reports")
            self.assertFalse(result["approved_for_export"])

    def test_only_three_explicit_json_files_are_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, report in zip(DIAGNOSIS.REPORT_NAMES, self.reports()):
                report["external_path_do_not_follow"] = "/must/not/open/any/model-or-video"
                (root / name).write_text(json.dumps(report))
            opened = []
            original = Path.open

            def record_open(path, *args, **kwargs):
                opened.append(path)
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", record_open):
                result = DIAGNOSIS.diagnose_run(root)
            self.assertEqual({path.name for path in opened}, set(DIAGNOSIS.REPORT_NAMES))
            self.assertEqual(len(opened), 3)
            self.assertEqual(result["report_read_errors"], {})

    def test_partial_report_is_incomplete_and_symlink_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "event-audit.json").write_text('{"not_finished":')
            (root / "training-summary.json").symlink_to(root / "event-audit.json")
            result = DIAGNOSIS.diagnose_run(root)
            self.assertEqual(result["state"], "running_or_incomplete")
            self.assertFalse(result["approved_for_export"])
            self.assertIn("symlinks", result["report_read_errors"]["training-summary.json"])


if __name__ == "__main__":
    unittest.main()
