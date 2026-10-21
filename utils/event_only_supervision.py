"""Fresh supervision from one explicitly supplied raw-event NPY, never images.

Only x/y/t/p are accepted. Times and contrast assumptions are explicit; this
module does not discover calibration, motion, binding, or supervision files.
Its absolute event checks do not establish complete or unique 3D geometry.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np


# Fixed in code before fitting, not relaxed against a video reference or result.
EVENT_ONLY_LIMITS = MappingProxyType({
    "minimum_cosine": 0.80,
    "minimum_polarity_agreement": 0.90,
    "maximum_relative_l1": 0.50,
    "maximum_unsupported_change_fraction": 0.05,
})
_CHUNK_SIZE = 262144
_TIME_TOLERANCE = 1e-12


def _positive_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _columns(array):
    if array.dtype.names is not None:
        if array.ndim != 1 or set(array.dtype.names) != {"x", "y", "t", "p"}:
            raise ValueError("Structured events must contain only x, y, t, p")
        columns = tuple(array[name] for name in ("x", "y", "t", "p"))
    elif array.ndim == 2 and array.shape[1] == 4:
        columns = tuple(array[:, index] for index in range(4))
    else:
        raise ValueError("Expected N x 4 [x,y,t,p] or a structured x/y/t/p event array")
    if not len(array) or any(column.dtype.kind not in "iufb" for column in columns):
        raise ValueError("Events must be nonempty real numeric data")
    return columns


def _relative_seconds(raw, time_scale, g0_time):
    # Work in extended precision before subtraction when absolute clocks are large.
    seconds = np.asarray(raw, dtype=np.longdouble) * np.longdouble(time_scale) - np.longdouble(g0_time)
    result = np.asarray(seconds, dtype=np.float64)
    result[np.abs(result) <= _TIME_TOLERANCE] = 0.0
    return result


def build_event_windows(events_path, width, height, *, g0_time, time_scale,
                        step_seconds=0.05, validation_stride=5, substeps=2):
    """Bin raw events into (t0,t1] windows relative to the supplied G0 time.

    ``g0_time`` is in seconds, while ``event_seconds = raw_t * time_scale``.
    The effective interval is ``step_seconds/substeps``. The final boundary
    equals the last event's relative time, including a shorter final window.
    Events at/before G0 are explicitly excluded and counted. A 1e-12 s roundoff
    tolerance assigns numerical boundary ties to the earlier window.

    Returns times[T+1] float64, counts[T,2,H,W] uint32 (positive,negative),
    train_mask[T] bool (every validation_stride-th window held out), metadata.
    Large N is handled through a memory map and bounded-size input chunks;
    only the requested dense output and one 2*H*W histogram are allocated.
    """
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    substeps = _positive_integer(substeps, "substeps")
    validation_stride = _positive_integer(validation_stride, "validation_stride", 2)
    time_scale = _positive_number(time_scale, "time_scale")
    step_seconds = _positive_number(step_seconds, "step_seconds")
    if isinstance(g0_time, (bool, np.bool_)) or not np.isscalar(g0_time):
        raise ValueError("g0_time must be finite seconds")
    try:
        g0_time = float(g0_time)
    except (ValueError, TypeError):
        raise ValueError("g0_time must be finite seconds") from None
    if not math.isfinite(g0_time):
        raise ValueError("g0_time must be finite seconds")
    step = step_seconds / substeps
    if step <= 2 * _TIME_TOLERANCE:
        raise ValueError("Event windows must exceed timestamp roundoff tolerance")
    source = Path(events_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Supply a raw x/y/t/p NPY file, not a directory or supervision cache")
    # Some event exports (for example Lego_event) are valid NPY files without
    # an extension. Validate the file format, never infer it from its name.
    with source.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("Supply raw x/y/t/p NPY data, not an archive or supervision cache")
    initial_hash = _sha256(source)
    array = np.load(source, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.ndarray):
        if hasattr(array, "close"):
            array.close()
        raise ValueError("Raw event input must be a single NPY array")
    x, y, timestamps, polarities = _columns(array)
    observed_polarities = set()
    previous_t = None
    excluded = 0
    for start in range(0, len(array), _CHUNK_SIZE):
        stop = min(len(array), start + _CHUNK_SIZE)
        cx, cy, ct, cp = (column[start:stop] for column in (x, y, timestamps, polarities))
        if any(not np.isfinite(column).all() for column in (cx, cy, ct, cp)):
            raise ValueError("Raw events contain non-finite values")
        if (np.any(cx < 0) or np.any(cx >= width) or np.any(cy < 0) or np.any(cy >= height)
                or np.any(cx != np.floor(cx)) or np.any(cy != np.floor(cy))):
            raise ValueError("Event coordinates must be integer sensor pixels within the supplied dimensions")
        if np.any(ct[1:] < ct[:-1]) or (previous_t is not None and ct[0] < previous_t):
            raise ValueError("Raw events must be sorted by nondecreasing timestamp")
        previous_t = ct[-1]
        values = np.unique(cp)
        if np.any(~np.isin(values, (-1, 0, 1))):
            raise ValueError("Event polarities must use {0,1} or {-1,+1}")
        observed_polarities.update(float(value) for value in values)
        excluded += int(np.count_nonzero(_relative_seconds(ct, time_scale, g0_time) <= 0))
    if -1 in observed_polarities and 0 in observed_polarities:
        raise ValueError("Mixed {0,1} and {-1,+1} polarity encodings are ambiguous")
    duration = float(_relative_seconds(timestamps[-1:], time_scale, g0_time)[0])
    if not math.isfinite(duration) or duration <= _TIME_TOLERANCE:
        raise ValueError("No events occur after the supplied G0 anchor time")
    full = int(math.floor(duration / step))
    times = np.arange(full + 1, dtype=np.float64) * step
    if duration - times[-1] > _TIME_TOLERANCE:
        times = np.r_[times, duration]
    else:
        times[-1] = duration
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid event-window boundaries")
    counts = np.zeros((len(times) - 1, 2, height, width), dtype=np.uint32)
    for start in range(0, len(array), _CHUNK_SIZE):
        stop = min(len(array), start + _CHUNK_SIZE)
        relative = _relative_seconds(timestamps[start:stop], time_scale, g0_time)
        keep = relative > 0
        if not np.any(keep):
            continue
        relative = relative[keep]
        window = np.searchsorted(times[1:], relative - _TIME_TOLERANCE, side="left")
        pixels = ((polarities[start:stop][keep] != 1).astype(np.int64) * (height * width)
                  + y[start:stop][keep].astype(np.int64) * width + x[start:stop][keep].astype(np.int64))
        cuts = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
        for first, last in zip(cuts[:-1], cuts[1:]):
            index = int(window[first])
            histogram = np.bincount(pixels[first:last], minlength=2 * height * width).astype(np.uint64)
            total = counts[index].reshape(-1).astype(np.uint64) + histogram
            if total.max() > np.iinfo(np.uint32).max:
                raise OverflowError("An event bin exceeds uint32 capacity")
            counts[index] = total.reshape(2, height, width).astype(np.uint32)
    final_hash = _sha256(source)
    if final_hash != initial_hash:
        raise ValueError("Raw event input changed while supervision was being built")
    train_mask = np.arange(len(counts)) % validation_stride != validation_stride - 1
    return {"times": times, "counts": counts, "train_mask": train_mask, "metadata": {
        "format": "raw_event_only_windows_v1", "input_path": str(source), "input_sha256": initial_hash,
        "input_event_count": len(array), "included_event_count": len(array) - excluded,
        "ignored_at_or_before_g0": excluded, "width": width, "height": height,
        "g0_time_seconds": g0_time, "time_scale": time_scale, "times_relative_to_g0": True,
        "step_seconds": step_seconds, "substeps": substeps, "effective_step_seconds": step,
        "duration_seconds": duration, "window_convention": "(t0,t1]", "time_roundoff_tolerance_seconds": _TIME_TOLERANCE,
        "polarity_encoding": "-1/+1" if -1 in observed_polarities else "0/1",
        "polarity_channels": ["positive", "negative"], "validation_stride": validation_stride,
        "contrast_thresholds_loaded": False, "external_images_loaded": False,
        "inherited_supervision_loaded": False,
        "g0_anchor_scope": "Explicit caller-supplied anchor; not inferred from a video or prior motion."}}


def pooled_counts(counts, target_width):
    """Area-average event counts, preserving per-pixel log-change units.

    Accepts [...,2,H,W]; aspect ratio and integer block alignment are preserved.
    These are average event masses, not raw integer event totals.
    """
    counts = np.asarray(counts)
    target_width = _positive_integer(target_width, "target_width")
    if counts.ndim < 3 or counts.shape[-3] != 2 or counts.dtype.kind not in "iuf":
        raise ValueError("Expected numeric [...,2,H,W] counts")
    height, width = counts.shape[-2:]
    if min(height, width) < 1 or target_width > width or width % target_width:
        raise ValueError("Target width must divide the original width")
    factor = width // target_width
    if height % factor:
        raise ValueError("The same pooling factor must divide height and width")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Event counts must be finite and nonnegative")
    shape = (*counts.shape[:-2], height // factor, factor, target_width, factor)
    return counts.reshape(shape).mean(axis=(-3, -1), dtype=np.float64)


def training_spans(train_mask, max_length=4):
    """Return [start,stop) spans of lengths 1..max_length containing training only.

    Sum only each returned span's event bins; an accumulated target reaching
    across a validation bin leaks held-out events even if its endpoint is train.
    """
    train_mask = np.asarray(train_mask)
    max_length = _positive_integer(max_length, "max_length")
    if train_mask.ndim != 1 or train_mask.dtype != np.bool_ or not len(train_mask):
        raise ValueError("train_mask must be a nonempty one-dimensional boolean array")
    spans = []
    for start in np.flatnonzero(train_mask):
        for stop in range(int(start) + 1, min(len(train_mask), int(start) + max_length) + 1):
            if not train_mask[stop - 1]:
                break
            spans.append((int(start), stop))
    return spans


def metrics(predicted_logdiff, counts, threshold_positive=0.2, threshold_negative=0.2):
    """Absolute signed event consistency, independent of any RGB reference.

    Zero-event and fully cancelling windows abstain, never pass by being blank.
    Cosine is signed (not abs(cosine)); reversed polarity must fail.
    """
    positive = _positive_number(threshold_positive, "threshold_positive")
    negative = _positive_number(threshold_negative, "threshold_negative")
    predicted = np.asarray(predicted_logdiff, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if predicted.ndim != 2 or not predicted.size or counts.shape != (2, *predicted.shape):
        raise ValueError("Expected predicted[H,W] and counts[2,H,W]")
    if not np.isfinite(predicted).all() or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Prediction and counts must be finite; counts must be nonnegative")
    observed = positive * counts[0] - negative * counts[1]
    activity = counts.sum(0) > 0
    observed_mass = float(np.abs(observed).sum())
    predicted_mass = float(np.abs(predicted).sum())
    event_mass = float(counts.sum())
    observable = observed_mass > 1e-12
    status = "observable" if observable else "abstain_no_events" if event_mass == 0 else "abstain_signed_cancellation"
    if observable:
        denominator = float(np.linalg.norm(predicted) * np.linalg.norm(observed))
        cosine = float(np.clip((predicted * observed).sum() / denominator, -1., 1.)) if denominator > 1e-12 else 0.0
        polarity = float(np.abs(observed)[np.sign(predicted) == np.sign(observed)].sum() / observed_mass)
        relative_l1 = float(np.abs(predicted - observed).sum() / observed_mass)
    else:
        cosine, polarity, relative_l1 = None, None, None
    unsupported = float(np.abs(predicted)[~activity].sum() / predicted_mass) if predicted_mass > 1e-12 else 0.0
    reasons = []
    if not observable:
        reasons.append(status)
    else:
        for name, failed in (("cosine", cosine < EVENT_ONLY_LIMITS["minimum_cosine"]),
                             ("polarity_agreement", polarity < EVENT_ONLY_LIMITS["minimum_polarity_agreement"]),
                             ("relative_l1", relative_l1 > EVENT_ONLY_LIMITS["maximum_relative_l1"])):
            if failed:
                reasons.append(name)
    if unsupported > EVENT_ONLY_LIMITS["maximum_unsupported_change_fraction"]:
        reasons.append("unsupported_change_fraction")
    return {"status": status, "observable": observable, "passed": not reasons, "reasons": reasons,
            "cosine": cosine, "polarity_agreement": polarity, "relative_l1": relative_l1,
            "unsupported_change_fraction": unsupported, "event_mass": event_mass,
            "active_pixels": int(activity.sum()), "signed_active_pixels": int(np.count_nonzero(observed)),
            "observed_change_l1": observed_mass, "predicted_change_l1": predicted_mass}


def audit_event_prediction(predicted_logdiffs, counts, *, train_mask=None, fit_split="train",
                           threshold_positive=0.2, threshold_negative=0.2, thresholds_confirmed=False):
    """Audit using fixed absolute checks; uncalibrated/in-sample fits stay candidates.

    ``fit_split='all'`` has no held-out validation even if a nominal mask exists.
    ``fit_split='train'`` requires the exact boolean mask used by optimization.
    Strict approval requires confirmed contrast thresholds, nonempty train and
    validation subsets, and every interval observable and passing. Approval is
    only an event image-space check, not proof of unique/complete 3D motion.
    """
    predicted = np.asarray(predicted_logdiffs)
    counts = np.asarray(counts)
    if predicted.ndim != 3 or not len(predicted) or counts.shape != (len(predicted), 2, *predicted.shape[1:]):
        raise ValueError("Expected predicted[T,H,W] and counts[T,2,H,W]")
    if fit_split not in ("train", "all") or type(thresholds_confirmed) is not bool:
        raise ValueError("Invalid fit split or threshold confirmation")
    if train_mask is not None:
        train_mask = np.asarray(train_mask)
        if train_mask.dtype != np.bool_ or train_mask.shape != (len(predicted),):
            raise ValueError("train_mask must be boolean and match the interval count")
    if fit_split == "train" and train_mask is None:
        raise ValueError("The actual training mask is required for held-out claims")
    fitted = np.ones(len(predicted), dtype=bool) if fit_split == "all" else train_mask
    rows = [dict(index=index, split="train" if fitted[index] else "validation",
                 **metrics(predicted[index], counts[index], threshold_positive, threshold_negative))
            for index in range(len(predicted))]

    def summary(selected):
        return {"intervals": len(selected), "passed": sum(row["passed"] for row in selected),
                "abstained": sum(not row["observable"] for row in selected),
                **{key: float(np.mean(values)) if values else None
                   for key in ("cosine", "polarity_agreement", "relative_l1", "unsupported_change_fraction")
                   for values in [[row[key] for row in selected if row[key] is not None]]}}

    failures = []
    if not thresholds_confirmed:
        failures.append("contrast_thresholds_unconfirmed")
    if fit_split == "all":
        failures.append("in_sample_consistency_only")
    if not np.any(fitted):
        failures.append("no_training_intervals")
    if np.all(fitted):
        failures.append("no_heldout_intervals")
    if any(not row["observable"] for row in rows):
        failures.append("unobservable_intervals")
    if any(not row["passed"] for row in rows):
        failures.append("event_intervals_failed")
    return {"format": "event_only_absolute_audit_v1", "approved_for_export": not failures,
            "status": "strict_event_check_passed" if not failures else "candidate_unapproved",
            "fit_split": fit_split,
            "evaluation_scope": "in_sample_consistency_only" if fit_split == "all" else "new_motion_stage_event_holdout",
            "thresholds": {"positive": float(threshold_positive), "negative": float(threshold_negative),
                           "confirmed": thresholds_confirmed},
            "absolute_limits": dict(EVENT_ONLY_LIMITS), "approval_blockers": failures,
            "training_indices": np.flatnonzero(fitted).tolist(), "validation_indices": np.flatnonzero(~fitted).tolist(),
            "failed_intervals": [row["index"] for row in rows if not row["passed"]],
            "summary": {"all": summary(rows), "train_events": summary([row for row in rows if row["split"] == "train"]),
                        "validation_events": summary([row for row in rows if row["split"] == "validation"])},
            "intervals": rows,
            "scope": "Fixed absolute raw-event image-space consistency only; not unique depth, unseen geometry, or complete physical motion. No RGB reference or inherited event gate is used."}
