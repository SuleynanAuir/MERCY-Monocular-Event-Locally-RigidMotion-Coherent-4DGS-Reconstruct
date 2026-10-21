"""Array-only, necessary-condition diagnostics for accumulated event changes.

These diagnostics do not replace or relax the native strict event audit.
Absence of an endpoint residual-bound violation is not proof of correct
motion, calibrated thresholds, complete event data, or valid 3D geometry.
"""
from __future__ import annotations

import math

import numpy as np

from utils.event_only_supervision import EVENT_ONLY_LIMITS, metrics


def _positive_number(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f"{name} must be a positive finite real number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _interval_diagnostics(predicted, counts, positive, negative, start, stop):
    observed = positive * counts[0] - negative * counts[1]
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Accumulated values or assumed event contrasts overflowed")
    residual = np.abs(predicted - observed)
    if not np.isfinite(residual).all():
        raise ValueError("Endpoint residual overflowed")
    bound = positive + negative
    # Only numerical roundoff is suppressed; the physical bound is never
    # increased with the interval length or fitted against these predictions.
    tolerance = 1e-12 * max(1., bound)
    excess = np.maximum(residual - bound, 0.)
    excess[excess <= tolerance] = 0.
    violated = excess > 0
    active = (counts[0] > 0) | (counts[1] > 0)
    net_zero = np.abs(observed) <= 1e-12
    observed_mass = float(np.abs(observed).sum())
    residual_mass = float(residual.sum())
    excess_mass = float(excess.sum())
    if not all(math.isfinite(value) for value in (observed_mass, residual_mass, excess_mass)):
        raise ValueError("Accumulated diagnostic mass overflowed")
    active_count = int(active.sum())
    strict_metrics = metrics(predicted, counts, positive, negative)
    return {
        "start_index": start, "stop_index_exclusive": stop, "window_count": stop - start,
        "pixel_count": int(predicted.size), "event_active_pixels": active_count,
        "no_event_pixels": int((~active).sum()),
        "net_cancellation_pixels": int((active & net_zero).sum()),
        "signed_change_observable_pixels": int((active & ~net_zero).sum()),
        "endpoint_residual_bound": bound, "excess_roundoff_tolerance": tolerance,
        "bound_exceeding_pixels": int(violated.sum()),
        "bound_exceeding_fraction": float(violated.mean()),
        "bound_exceeding_event_active_fraction": float(violated[active].mean()) if active_count else None,
        "bound_exceeding_no_event_pixels": int((violated & ~active).sum()),
        "absolute_residual_l1": residual_mass, "absolute_residual_mean": float(residual.mean()),
        "excess_l1": excess_mass, "excess_mean": float(excess.mean()), "excess_max": float(excess.max()),
        "excess_fraction_of_residual_l1": excess_mass / residual_mass if residual_mass > 1e-12 else None,
        "excess_relative_to_observed_l1": excess_mass / observed_mass if observed_mass > 1e-12 else None,
        "existing_native_metric_diagnostics": strict_metrics,
        "motion_validated": False,
    }


def _summary(rows):
    pixel_count = sum(row["pixel_count"] for row in rows)
    exceeding = sum(row["bound_exceeding_pixels"] for row in rows)
    excess = sum(row["excess_l1"] for row in rows)
    return {"intervals": len(rows), "intervals_with_bound_excess": sum(row["bound_exceeding_pixels"] > 0 for row in rows),
            "evaluated_pixel_endpoints": pixel_count, "bound_exceeding_pixel_endpoints": exceeding,
            "bound_exceeding_fraction": exceeding / pixel_count,
            "excess_l1": excess, "excess_mean": excess / pixel_count,
            "excess_max": max(row["excess_max"] for row in rows),
            "native_metric_means": {
                key: float(np.mean(values)) if values else None
                for key in ("cosine", "polarity_agreement", "relative_l1", "unsupported_change_fraction")
                for values in [[row["existing_native_metric_diagnostics"][key] for row in rows
                                if row["existing_native_metric_diagnostics"][key] is not None]]}}


def event_consistency_diagnostics(predicted_logdiff, counts, threshold_positive=.2, threshold_negative=.2, *,
                                  temporal_scales=(1, 2, 4, 8, 16), max_prefixes=16):
    """Diagnose native-spatial-resolution sums over time; do not issue approval.

    Inputs are predicted_logdiff[T,H,W] and raw integer counts[T,2,H,W]
    (positive, negative) over exactly the same consecutive (t0,t1] windows.
    T/H/W must be nonzero. Float arrays containing exact integer counts are
    accepted. No spatial pooling, external files, model loading or GPU is used.

    For the ideal reset event model, deltaL=C+*Np-C-*Nm+r_end-r_start,
    with r in (-C-,C+). Thus |deltaL-(C+*Np-C-*Nm)| < C+ + C- is necessary.
    Intermediate residuals telescope: summing N windows does NOT multiply
    this endpoint bound by N. We report positive excess above the bound,
    conservatively ignoring numerical ties; zero excess is not a pass.

    Each temporal scale forms disjoint blocks, retaining a shorter final
    block. At most 16 uniformly selected prefix endpoints include the final
    window. Prefix log-intensity errors are not geometric displacement errors.
    Prefixes overlap, so their totals must not be treated as independent data.
    Extra array memory is O(HW), not a materialized T-long cumulative stack.

    These assumptions may fail with uncalibrated thresholds, missed/noisy
    events, non-instantaneous sensors, or a clamped/nonphysical render model.
    All supplied windows are diagnostic inputs; this does not certify a
    train/held-out split or permit their use as held-out training targets.
    """
    positive = _positive_number(threshold_positive, "threshold_positive")
    negative = _positive_number(threshold_negative, "threshold_negative")
    if not math.isfinite(positive + negative):
        raise ValueError("The total event residual bound must be finite")
    max_prefixes = _positive_integer(max_prefixes, "max_prefixes")
    if max_prefixes > 16:
        raise ValueError("max_prefixes must be at most 16")
    try:
        scales = sorted(set(_positive_integer(value, "temporal_scale") for value in temporal_scales))
    except TypeError:
        raise ValueError("temporal_scales must be an iterable of positive integers") from None
    if not scales:
        raise ValueError("At least one temporal scale is required")
    predicted = np.asarray(predicted_logdiff)
    counts = np.asarray(counts)
    if (predicted.ndim != 3 or not predicted.size or counts.shape != (len(predicted), 2, *predicted.shape[1:])
            or predicted.dtype.kind not in "iuf" or counts.dtype.kind not in "iuf"):
        raise ValueError("Expected numeric nonempty predicted[T,H,W] and raw counts[T,2,H,W]")
    # Validate in slices instead of allocating a full-volume validity mask.
    for prediction, raw_counts in zip(predicted, counts):
        if not np.isfinite(prediction).all() or not np.isfinite(raw_counts).all() or np.any(raw_counts < 0):
            raise ValueError("Predictions/counts must be finite and raw counts nonnegative")
        if raw_counts.dtype.kind == "f" and np.any(raw_counts != np.floor(raw_counts)):
            raise ValueError("Counts must be raw integer event counts, not spatially pooled masses")
    temporal = []
    for scale in scales:
        rows = []
        for start in range(0, len(predicted), scale):
            stop = min(start + scale, len(predicted))
            delta = predicted[start:stop].sum(axis=0, dtype=np.float64)
            events = counts[start:stop].sum(axis=0, dtype=np.float64)
            rows.append(_interval_diagnostics(delta, events, positive, negative, start, stop))
        temporal.append({"scale_windows": scale, "overlapping": False,
                         "includes_short_final_block": len(predicted) % scale != 0,
                         "summary": _summary(rows), "intervals": rows})
    number = min(max_prefixes, len(predicted))
    stops = ([len(predicted)] if number == 1
             else sorted(set(np.linspace(1, len(predicted), number, dtype=int).tolist())))
    selected = set(stops)
    prefix_rows = []
    accumulated_prediction = np.zeros(predicted.shape[1:], dtype=np.float64)
    accumulated_counts = np.zeros(counts.shape[1:], dtype=np.float64)
    for index in range(len(predicted)):
        accumulated_prediction += predicted[index]
        accumulated_counts += counts[index]
        if index + 1 in selected:
            prefix_rows.append(_interval_diagnostics(accumulated_prediction, accumulated_counts,
                                                      positive, negative, 0, index + 1))
    return {
        "format": "event_only_endpoint_consistency_diagnostics_v1",
        "diagnostic_only": True, "approved_for_export": False, "motion_validated": False,
        "native_strict_gates_changed": False, "existing_native_strict_limits": dict(EVENT_ONLY_LIMITS),
        "thresholds": {"positive": positive, "negative": negative, "confirmed_by_this_diagnostic": False},
        "input_shape": list(predicted.shape), "spatial_pooling_applied": False,
        "endpoint_model": {"equation": "deltaL = C_positive*N_positive - C_negative*N_negative + r_end - r_start",
                           "residual_range": "(-C_negative, C_positive)",
                           "necessary_absolute_error_bound": positive + negative,
                           "bound_multiplied_by_window_count": False,
                           "sufficient_condition": False},
        "temporal_scales": temporal,
        "prefixes": {"overlapping": True, "stop_indices_exclusive": stops, "summary": _summary(prefix_rows),
                     "intervals": prefix_rows, "full_timeline": prefix_rows[-1]},
        "limitations": [
            "No bound excess is only a necessary endpoint condition, never strict acceptance or proof of correct motion.",
            "Zero-event pixels may move without crossing a contrast threshold; no-event predictions do not establish static geometry.",
            "Opposite-polarity events may cancel in an interval; zero net log change does not validate absence or presence of motion.",
            "Prefix endpoints do not identify a unique path between them, depth, physical motion or unseen geometry.",
            "The ideal bound assumes correct contrasts, timing, complete events and a comparable unclamped log-intensity model.",
            "All supplied events contribute; this helper neither proves held-out independence nor permits validation leakage into training.",
        ],
    }
