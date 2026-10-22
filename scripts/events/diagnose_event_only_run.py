"""CPU-only diagnosis of three explicit reports from one new event-only run.

No model, event array, image, video, or path embedded in a report is opened.
The static baseline is an analytic zero-log-change prediction derived from
each audit row's event statistics, NOT a new render or a coarse-scale audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.event_only_supervision import EVENT_ONLY_LIMITS


REPORT_NAMES = ("event-audit.json", "training-summary.json", "input-provenance.json")
METRICS = ("cosine", "polarity_agreement", "relative_l1", "unsupported_change_fraction")
INPUT_KEYS = ("g0", "events", "intrinsics", "extrinsics")
MAX_REPORT_BYTES = 64 * 1024**2


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def average(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    return {"intervals": len(rows), "passed": sum(row["passed"] for row in rows),
            "abstained": sum(not row["observable"] for row in rows),
            "metric_means": {key: average([row[key] for row in rows if row[key] is not None])
                             for key in METRICS}}


def checked_rows(audit):
    """Recheck native audit rows against the fixed policy; never trust a pass flag."""
    source = audit.get("intervals")
    if not isinstance(source, list) or not source:
        raise ValueError("Audit has no nonempty per-interval native metrics")
    rows, issues = [], []
    for index, row in enumerate(source):
        if not isinstance(row, dict) or row.get("index") != index:
            raise ValueError("Audit interval indices are not complete and sequential")
        if row.get("split") not in ("train", "validation") or type(row.get("observable")) is not bool:
            raise ValueError("Invalid interval split or observability")
        mass, observed = row.get("event_mass"), row.get("observed_change_l1")
        if not all(finite_number(value) and value >= 0 for value in (mass, observed)):
            raise ValueError("Missing or invalid per-interval event statistics")
        observable = observed > 1e-12
        if row["observable"] != observable or (observable and mass == 0):
            raise ValueError("Interval observability contradicts event statistics")
        values = {key: row.get(key) for key in METRICS}
        unsupported = values["unsupported_change_fraction"]
        if not finite_number(unsupported) or not 0 <= unsupported <= 1:
            raise ValueError("Invalid unsupported-change metric")
        if observable:
            if not (all(finite_number(values[key]) for key in METRICS)
                    and -1 <= values["cosine"] <= 1 and 0 <= values["polarity_agreement"] <= 1
                    and values["relative_l1"] >= 0):
                raise ValueError("Invalid observable-interval metrics")
            reasons = [name for name, failed in (
                ("cosine", values["cosine"] < EVENT_ONLY_LIMITS["minimum_cosine"]),
                ("polarity_agreement", values["polarity_agreement"] < EVENT_ONLY_LIMITS["minimum_polarity_agreement"]),
                ("relative_l1", values["relative_l1"] > EVENT_ONLY_LIMITS["maximum_relative_l1"])) if failed]
        else:
            if any(values[key] is not None for key in METRICS[:3]):
                raise ValueError("Unobservable intervals must abstain, not report fitted metrics")
            reasons = ["abstain_no_events" if mass == 0 else "abstain_signed_cancellation"]
        if unsupported > EVENT_ONLY_LIMITS["maximum_unsupported_change_fraction"]:
            reasons.append("unsupported_change_fraction")
        passed = not reasons
        if type(row.get("passed")) is not bool or row["passed"] != passed:
            issues.append(f"interval_{index}_reported_pass_disagrees_with_fixed_limits")
        rows.append({"index": index, "split": row["split"], "observable": observable,
                     "passed": passed, "reasons": reasons, **values})
    return rows, issues


def diagnose_reports(audit=None, training_summary=None, input_provenance=None):
    """Pure report-dict API; callers can test without accessing any input data."""
    records = (audit, training_summary, input_provenance)
    missing = [name for name, record in zip(REPORT_NAMES, records) if record is None]
    result = {"format": "event_only_run_diagnosis_v1", "state": "running_or_incomplete",
              "approved_for_export": False, "native_event_check_passed": False,
              "missing_reports": missing, "approval_blockers": list(missing),
              "scope": "Native-resolution event image-space consistency only, not unique or complete 3D reconstruction.",
              "process_liveness_checked": False, "new_render_performed": False,
              "coarse_metrics_available": False,
              "baseline": {"available": False, "kind": "analytic_zero_log_change"}}
    if isinstance(training_summary, dict):
        result["training"] = {key: training_summary.get(key) for key in (
            "training_performed", "whole_timeline_fitted", "duration_seconds", "fit_split",
            "explicitly_optimized_window_count", "unoptimized_fitting_windows")}
    if missing:
        return result
    if not all(isinstance(record, dict) for record in records):
        result.update(state="invalid_reports", approval_blockers=["reports_must_be_objects"])
        return result
    try:
        rows, issues = checked_rows(audit)
    except ValueError as error:
        result.update(state="invalid_reports", approval_blockers=[str(error)])
        return result
    result["state"] = "completed_candidate"
    split = audit.get("fit_split")
    if split not in ("all", "train") or training_summary.get("fit_split") != split:
        issues.append("fit_split_missing_or_inconsistent")
    train = [row["index"] for row in rows if row["split"] == "train"]
    validation = [row["index"] for row in rows if row["split"] == "validation"]
    if audit.get("training_indices") != train or audit.get("validation_indices") != validation:
        issues.append("declared_splits_disagree_with_interval_rows")
    if split == "all" and validation:
        issues.append("all_fit_cannot_claim_heldout_intervals")
    limits_match = audit.get("absolute_limits") == dict(EVENT_ONLY_LIMITS)
    if not limits_match:
        issues.append("reported_limits_do_not_match_fixed_policy")
    thresholds = audit.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    threshold_pair = [thresholds.get("positive"), thresholds.get("negative")]
    thresholds_consistent = (all(finite_number(value) and value > 0 for value in threshold_pair)
                             and training_summary.get("thresholds") == threshold_pair)
    thresholds_confirmed = (thresholds.get("confirmed") is True
                            and training_summary.get("thresholds_confirmed") is True)
    identity_confirmed = all(record.get("identity_and_time_confirmed") is True for record in records)
    if not thresholds_consistent:
        issues.append("contrast_threshold_values_missing_or_inconsistent")
    hashes = input_provenance.get("input_sha256", {})
    hashes_valid = isinstance(hashes, dict) and all(
        isinstance(hashes.get(key), str) and len(hashes[key]) == 64
        and all(character in "0123456789abcdef" for character in hashes[key].lower()) for key in INPUT_KEYS)
    if not hashes_valid or audit.get("input_sha256") != hashes:
        issues.append("input_provenance_hashes_missing_or_inconsistent")
    if input_provenance.get("legacy_data_used") is not False:
        issues.append("legacy_data_exclusion_not_declared")
    fitted = training_summary.get("explicitly_optimized_window_count")
    remaining = training_summary.get("unoptimized_fitting_windows")
    coverage_valid = (type(fitted) is int and 0 <= fitted <= len(train)
                      and isinstance(remaining, list) and all(type(item) is int and item in train for item in remaining)
                      and len(set(remaining)) == len(remaining) and fitted + len(remaining) == len(train))
    if not coverage_valid:
        issues.append("optimization_coverage_missing_or_inconsistent")
    whole_fit = (coverage_valid and bool(train) and fitted == len(train)
                 and training_summary.get("training_performed") is True
                 and training_summary.get("whole_timeline_fitted") is True
                 and audit.get("whole_timeline_fitted") is True)
    native_pass = all(row["passed"] for row in rows)
    blockers = list(issues)
    for reason, condition in (
        ("in_sample_consistency_only", split == "all"),
        ("no_training_intervals", not train), ("no_heldout_intervals", not validation),
        ("contrast_thresholds_unconfirmed", not thresholds_confirmed),
        ("g0_identity_or_time_unconfirmed", not identity_confirmed),
        ("not_all_fitting_windows_optimized", not whole_fit),
        ("unobservable_intervals", any(not row["observable"] for row in rows)),
        ("event_intervals_failed", not native_pass)):
        if condition:
            blockers.append(reason)
    declared_blockers = audit.get("approval_blockers")
    if isinstance(declared_blockers, list) and all(isinstance(item, str) for item in declared_blockers):
        blockers.extend(declared_blockers)
    else:
        blockers.append("missing_audit_approval_blockers")
    if audit.get("approved_for_export") is not True:
        blockers.append("source_audit_not_approved")
    elif audit.get("status") != "strict_event_check_passed":
        blockers.append("source_audit_approval_status_inconsistent")
    blockers = list(dict.fromkeys(blockers))
    static_rows = [{"observable": row["observable"], "passed": False,
                    "cosine": 0.0 if row["observable"] else None,
                    "polarity_agreement": 0.0 if row["observable"] else None,
                    "relative_l1": 1.0 if row["observable"] else None,
                    "unsupported_change_fraction": 0.0} for row in rows]
    fitted_stats, static_stats = summarize(rows), summarize(static_rows)
    differences = {key: fitted_stats["metric_means"][key] - static_stats["metric_means"][key]
                   if fitted_stats["metric_means"][key] is not None and static_stats["metric_means"][key] is not None else None
                   for key in METRICS}
    result.update({"approved_for_export": not blockers,
                   "native_event_check_passed": native_pass, "approval_blockers": blockers,
                   "state": "strict_event_check_passed" if not blockers else "completed_candidate",
                   "report_consistency_issues": issues,
                   "evaluation_scope": "in_sample_consistency_only" if split == "all" else "new_motion_stage_event_holdout",
                   "thresholds": {"positive": threshold_pair[0], "negative": threshold_pair[1],
                                  "confirmed": thresholds_confirmed, "consistent": thresholds_consistent},
                   "identity_and_time_confirmed": identity_confirmed,
                   "input_sha256": hashes, "raw_input_hashes_recomputed": False,
                   "coverage": {"total_event_windows": len(rows), "fitting_windows": len(train),
                                "heldout_windows": len(validation), "optimized_fitting_windows": fitted,
                                "whole_fitting_split_optimized": whole_fit,
                                "all_event_windows_used_for_training": whole_fit and split == "all"},
                   "metrics": {"all": fitted_stats,
                               "train": summarize([row for row in rows if row["split"] == "train"]),
                               "heldout": summarize([row for row in rows if row["split"] == "validation"])},
                   "failure_reason_counts": dict(Counter(reason for row in rows for reason in row["reasons"])),
                   "failed_intervals": [row["index"] for row in rows if not row["passed"]],
                   "baseline": {"available": True, "kind": "analytic_zero_log_change",
                                "explanation": "For every observable window, a zero prediction has cosine=0, polarity=0, relativeL1=1 and unsupported=0. Unobservable windows abstain. No rendering was performed.",
                                "metrics": static_stats, "candidate_minus_baseline": differences,
                                "difference_interpretation": "Higher cosine/polarity is better; lower relativeL1/unsupported is better. This comparison alone does not establish strict approval."}})
    return result


def diagnose_run(run_dir):
    directory = Path(run_dir).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Run directory must be an existing directory")
    reports, sources, errors = [], {}, {}
    for name in REPORT_NAMES:
        path = directory / name
        try:
            if path.is_symlink():
                raise ValueError("Report symlinks are not followed")
            with path.open("rb") as handle:
                payload = handle.read(MAX_REPORT_BYTES + 1)
            if len(payload) > MAX_REPORT_BYTES:
                raise ValueError("Report exceeds the bounded JSON size")
            record = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON value: {value}")))
            if not isinstance(record, dict):
                raise ValueError("Report must be a JSON object")
            reports.append(record)
            sources[name] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
        except (OSError, ValueError) as error:
            reports.append(None)
            errors[name] = str(error)
    result = diagnose_reports(*reports)
    result.update(run_directory=str(directory), report_sources=sources, report_read_errors=errors)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    result = diagnose_run(args.run_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
