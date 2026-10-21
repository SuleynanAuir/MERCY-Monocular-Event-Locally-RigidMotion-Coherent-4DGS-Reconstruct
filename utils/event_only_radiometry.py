"""Array-only background hypotheses from G0 and event-derived log changes.

This is not radiometric calibration. In particular, dimming does not prove
that an object vacated a pixel rather than changing illumination or occluding
another surface. No files, images from a recording, models, or GPU are used.
"""
from __future__ import annotations

import math

import numpy as np


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def _finite_parameter(value, name, lower, upper=None, lower_inclusive=True):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if (value < lower if lower_inclusive else value <= lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def infer_background_from_events(reference_gray, reference_alpha, cumulative_log_changes_iter, *,
                                 minimum_alpha=.95, minimum_reference_gray=.1,
                                 minimum_log_decrease=math.log(2), minimum_support=64,
                                 minimum_support_fraction=.02, estimate_quantile=.25):
    """Return ``{'estimate': float | None, 'diagnostics': dict}`` without I/O.

    ``reference_gray`` is a nonnegative HxW linear G0 render on black, before
    a log floor; output uses the same intensity units. ``reference_alpha`` is HxW
    (or rasterizer-shaped 1xHxW) in [0,1]. Each yielded HxW array is an
    *unclipped cumulative* event log change from the same G0 anchor, not one
    window's increment and not a rendered or recorded image.

    For initially bright, near-opaque pixels, retain the negative temporal
    minimum and require at least a twofold intensity decrease by default.
    For r=exp(minimum_log_change), each background candidate solves
    B=C0*r/(1-(1-A0)*r), accounting for initial alpha compositing over B.
    The lower quartile is a background hypothesis when a sufficient
    population of foreground pixels vacates. Only one global scalar is
    returned; individual pixels are not assigned fitted intensity offsets.
    No temporal stack is materialized: memory is O(HW), independent of time.

    Insufficient evidence returns estimate=None; invalid inputs raise
    ValueError. The iterator cannot establish whether the selected pixels
    truly vacated or whether supplied events contain held-out intervals.
    Using all events here makes the estimate in-sample; train-only evaluation
    must not pass held-out cumulative events to this initialization.
    """
    gray = _real_array(reference_gray, "reference_gray")
    if gray.ndim != 2 or not gray.size or np.any(gray < 0):
        raise ValueError("reference_gray must be a nonempty nonnegative HxW array")
    alpha = _real_array(reference_alpha, "reference_alpha")
    if alpha.shape == (1, *gray.shape):
        alpha = alpha[0]
    if alpha.shape != gray.shape or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("reference_alpha must match HxW and lie in [0,1]")
    minimum_alpha = _finite_parameter(minimum_alpha, "minimum_alpha", 0, 1, False)
    minimum_reference_gray = _finite_parameter(minimum_reference_gray, "minimum_reference_gray", 0, lower_inclusive=False)
    minimum_log_decrease = _finite_parameter(minimum_log_decrease, "minimum_log_decrease", 0, lower_inclusive=False)
    minimum_support_fraction = _finite_parameter(minimum_support_fraction, "minimum_support_fraction", 0, 1)
    estimate_quantile = _finite_parameter(estimate_quantile, "estimate_quantile", 0, .5, False)
    if (isinstance(minimum_support, (bool, np.bool_)) or not isinstance(minimum_support, (int, np.integer))
            or minimum_support < 1):
        raise ValueError("minimum_support must be a positive integer")
    minimum_support = int(minimum_support)
    eligible = (alpha >= minimum_alpha) & (gray >= minimum_reference_gray)
    initial = np.asarray(gray[eligible], dtype=np.float64)
    initial_alpha = np.asarray(alpha[eligible], dtype=np.float64)
    minima = np.zeros(len(initial), dtype=np.float64)
    frames = 0
    try:
        iterator = iter(cumulative_log_changes_iter)
    except TypeError:
        raise ValueError("cumulative_log_changes_iter must yield HxW arrays") from None
    for delta in iterator:
        delta = _real_array(delta, "cumulative_log_change")
        if delta.shape != gray.shape:
            raise ValueError("Every cumulative log-change array must match the G0 HxW shape")
        np.minimum(minima, delta[eligible], out=minima)
        frames += 1
    required = max(minimum_support, int(math.ceil(minimum_support_fraction * len(initial))))
    diagnostics = {
        "method": "alpha_aware_g0_event_temporal_minimum_lower_quantile_v1",
        "status": "insufficient_evidence", "reason": None,
        "cumulative_windows_seen": frames, "eligible_foreground_pixels": len(initial),
        "dimming_support_pixels": int(np.count_nonzero(minima <= -minimum_log_decrease)),
        "required_support_pixels": required, "usable_support_pixels": 0,
        "support_fraction_of_eligible": 0., "candidate_quantiles": None,
        "estimate_quantile": estimate_quantile,
        "selection": {"minimum_alpha": minimum_alpha, "minimum_reference_gray": minimum_reference_gray,
                      "minimum_log_decrease": minimum_log_decrease, "minimum_support": minimum_support,
                      "minimum_support_fraction": minimum_support_fraction},
        "calibrated": False, "contrast_thresholds_confirmed": False,
        "identity_and_time_confirmed": False, "motion_confirmed": False,
        "estimated_parameter": "one_global_static_background_scalar",
        "per_pixel_correction_fitted": False, "background_contrast_coupled": True,
        "temporal_plateau_verified": False,
        "external_inputs_used": False, "heldout_independence_established": False,
        "warnings": [],
        "limitations": [
            "A dimming pixel may reveal another surface or change illumination; background vacancy is not observed directly.",
            "The estimate inherits G0 appearance, supplied event contrast, time alignment and accumulated-event drift assumptions.",
            "Alpha compositing is accounted for, but G0 intensity need not match the event sensor's radiometric response.",
            "A spatially uniform, static background is a hypothesis; this estimator does not validate it.",
            "All supplied cumulative events influence initialization; including held-out events invalidates an independent holdout claim.",
        ],
    }
    result = {"estimate": None, "diagnostics": diagnostics}
    if not frames:
        diagnostics["reason"] = "no_event_windows"
        return result
    if len(initial) < minimum_support:
        diagnostics["reason"] = "insufficient_bright_opaque_foreground"
        return result
    moving = minima <= -minimum_log_decrease
    if not np.any(moving):
        diagnostics["reason"] = "no_supported_negative_motion_change"
        return result
    # Multiplying exp(minima) by I0 can underflow before the multiplication;
    # combine in the log domain and explicitly reject unrepresentable zeros.
    with np.errstate(under="ignore"):
        relative_intensity = np.exp(minima[moving])
        denominator = 1 - (1 - initial_alpha[moving]) * relative_intensity
        candidates = np.exp(np.log(initial[moving]) + minima[moving]) / denominator
    representable = np.isfinite(candidates) & (candidates > 0)
    rejected = int(np.count_nonzero(~representable))
    candidates = candidates[representable]
    diagnostics.update(usable_support_pixels=len(candidates),
                       support_fraction_of_eligible=float(len(candidates) / len(initial)),
                       numerically_unrepresentable_candidates=rejected)
    if rejected:
        diagnostics["warnings"].append("Extreme accumulated dimming produced unrepresentable intensities; those pixels were excluded.")
    if len(candidates) < required:
        diagnostics["reason"] = "insufficient_dimming_support"
        return result
    quantiles = np.quantile(candidates, [.1, .25, .5, .75, .9])
    diagnostics["candidate_quantiles"] = dict(zip(("q10", "q25", "q50", "q75", "q90"), map(float, quantiles)))
    if math.log(float(quantiles[3])) - math.log(float(quantiles[1])) > math.log(3):
        diagnostics["warnings"].append("The interquartile candidate intensity range exceeds 3x; a single background estimate is weakly supported.")
    diagnostics.update(status="heuristic_candidate", reason="supported_foreground_dimming_hypothesis")
    result["estimate"] = float(np.quantile(candidates, estimate_quantile))
    return result
