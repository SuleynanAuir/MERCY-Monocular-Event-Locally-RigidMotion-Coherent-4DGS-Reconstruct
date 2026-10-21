#!/usr/bin/env python3
"""Read-only full-timeline diagnostics for one new event-only checkpoint.

Only the three explicit checkpoint/G0/raw-events inputs are opened as data.
Recorded provenance paths are never followed. This creates a separate JSON;
it neither overwrites an existing audit nor changes acceptance thresholds.
"""

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def output_schema():
    """A dependency-free JSON schema for the diagnostic envelope."""
    return {
        "title": "Event-only checkpoint consistency diagnostics",
        "type": "object",
        "required": ["format", "diagnostic_only", "approval_changed", "checkpoint_sha256",
                     "input_sha256", "radiometry", "timeline", "diagnostics"],
        "properties": {
            "format": {"const": "event_only_checkpoint_consistency_v1"},
            "diagnostic_only": {"const": True},
            "approval_changed": {"const": False},
            "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "input_sha256": {"type": "object"},
            "radiometry": {"type": "object"},
            "timeline": {"type": "object", "required": ["windows", "endpoint_times_seconds",
                "checkpoint_start_seconds", "checkpoint_end_seconds", "exact_final_endpoint_rendered"]},
            "diagnostics": {"type": "object"},
        },
    }


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def checkpoint_observation_settings(payload):
    """Read only explicit metadata; never recover settings from embedded paths.

    Old final checkpoints omitted B/floor. Their documented defaults are B=0
    and floor=.01 and are recorded as assumptions, not inferred calibration.
    Contrast thresholds and the clock cannot safely be guessed and are required.
    """
    provenance = payload.get("input_provenance")
    summary = payload.get("training_summary")
    if not isinstance(provenance, dict) or not isinstance(summary, dict):
        raise ValueError("Checkpoint must contain input_provenance and training_summary dictionaries")
    g0_time = _finite_number(provenance.get("g0_time"), "provenance.g0_time")
    time_scale = _finite_number(provenance.get("time_scale"), "provenance.time_scale")
    if time_scale <= 0:
        raise ValueError("Checkpoint time_scale must be positive")
    thresholds = summary.get("thresholds")
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 2:
        raise ValueError("Checkpoint must record [positive, negative] contrast thresholds; no implicit threshold fallback")
    positive, negative = (_finite_number(value, "training_summary.thresholds") for value in thresholds)
    if min(positive, negative) <= 0:
        raise ValueError("Contrast thresholds must be positive")
    defaults = []
    if "event_background" not in summary:
        defaults.append({"field": "event_background", "value": 0.,
                         "reason": "Field absent: explicit legacy-run black-background assumption"})
    if "log_intensity_floor" not in summary:
        defaults.append({"field": "log_intensity_floor", "value": .01,
                         "reason": "Field absent: explicit legacy-run log-floor assumption"})
    background = _finite_number(summary.get("event_background", 0.), "training_summary.event_background")
    floor = _finite_number(summary.get("log_intensity_floor", .01), "training_summary.log_intensity_floor")
    if not 0 <= background <= 1 or not 0 < floor < 1:
        raise ValueError("Observation background must be in [0,1] and log floor in (0,1)")
    return {
        "g0_time": g0_time, "time_scale": time_scale,
        "threshold_positive": positive, "threshold_negative": negative,
        "event_background": background, "log_intensity_floor": floor,
        "thresholds_confirmed": summary.get("thresholds_confirmed", False) is True,
        "event_radiometry": summary.get("event_radiometry"), "explicit_legacy_fallbacks": defaults,
        "observation_equation": "log(clamp(gray_black + (1 - alpha) * B, log_intensity_floor, 1))",
        "observation_function": "scripts.reconstruct_lego_event_only.observation_log",
    }


def validate_timeline(endpoint_times, knot_times):
    """Require the entire raw-event interval, including the exact last knot."""
    endpoints, knots = [float(value) for value in endpoint_times], [float(value) for value in knot_times]
    for name, values in (("Event endpoints", endpoints), ("Checkpoint knots", knots)):
        if (len(values) < 2 or any(not math.isfinite(value) for value in values) or
                values[0] != 0 or any(left >= right for left, right in zip(values, values[1:]))):
            raise ValueError(f"{name} must be finite, strictly increasing, and begin at zero")
    if endpoints[-1] > knots[-1]:
        raise ValueError("Raw-event timeline exceeds the checkpoint range; extrapolation is forbidden")
    if endpoints[-1] < knots[-1]:
        raise ValueError("Raw events do not cover the checkpoint's exact final endpoint; refusing a partial whole-sequence audit")
    return endpoints, knots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "g0", "events", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--step-seconds", type=float, default=.05,
                        help="Event bin interval before substeps (default .05 seconds)")
    parser.add_argument("--substeps", type=int, default=2,
                        help="Subdivide each interval (default 2, effective interval .025s)")
    parser.add_argument("--schema", action="store_true", help="Print output schema without loading numerical dependencies")
    args = parser.parse_args(argv)
    if args.schema:
        print(json.dumps(output_schema(), indent=2))
        return
    if any(getattr(args, name) is None for name in ("checkpoint", "g0", "events", "output")):
        parser.error("--checkpoint, --g0, --events, and --output are required unless --schema is used")
    if args.output.exists():
        raise FileExistsError(f"Diagnostic output already exists: {args.output}")
    if not math.isfinite(args.step_seconds) or args.step_seconds <= 0 or args.substeps < 1:
        parser.error("--step-seconds must be positive/finite and --substeps must be positive")

    import numpy as np
    import torch
    from utils.event_only_checkpoint import load_event_only_checkpoint
    from utils.event_only_consistency import event_consistency_diagnostics
    from utils.event_only_inputs import sha256_file
    from utils.event_only_renderer import EventOnlyRenderer
    from utils.event_only_supervision import build_event_windows
    from scripts.reconstruct_lego_event_only import observation_log

    started = time.perf_counter()
    checkpoint_hash = sha256_file(args.checkpoint)
    # Validate identity/clock and the complete event time range before acquiring
    # GPU memory. No settings.json or other stored path is ever opened.
    motion, camera, payload = load_event_only_checkpoint(args.checkpoint, args.g0, device="cpu")
    radiometry = checkpoint_observation_settings(payload)
    if sha256_file(args.events) != payload["input_sha256"]["events"]:
        raise ValueError("Explicit raw-event SHA256 does not match checkpoint input")
    windows = build_event_windows(
        args.events, camera["width"], camera["height"], g0_time=radiometry["g0_time"],
        time_scale=radiometry["time_scale"], step_seconds=args.step_seconds, substeps=args.substeps)
    if windows["metadata"]["input_sha256"] != payload["input_sha256"]["events"]:
        raise ValueError("Raw events changed between identity checking and window construction")
    times, knots = validate_timeline(windows["times"], payload["times"].tolist())
    counts = windows["counts"]
    if len(times) != len(counts) + 1:
        raise ValueError("Event endpoints and actual event-window count disagree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a true full-G0 checkpoint-render audit")
    motion = motion.to("cuda")
    renderer = EventOnlyRenderer(args.g0, camera)
    predictions = np.empty((len(counts), camera["height"], camera["width"]), dtype=np.float32)
    rendered_endpoints = 0
    with torch.no_grad():
        def observed(timestamp):
            xyz, rotation = motion.deform(renderer.g0.xyz, renderer.g0.rotation, timestamp)
            result = renderer.render(xyz, rotation)
            return observation_log(result, radiometry["event_background"], radiometry["log_intensity_floor"])

        previous = observed(times[0])
        if not bool(torch.isfinite(previous).all()):
            raise ValueError("Non-finite checkpoint observation at the initial endpoint")
        rendered_endpoints += 1
        for index, timestamp in enumerate(times[1:]):
            current = observed(timestamp)
            if not bool(torch.isfinite(current).all()):
                raise ValueError(f"Non-finite checkpoint observation at endpoint {index + 1}")
            predictions[index] = (current - previous).cpu().numpy()
            previous = current
            rendered_endpoints += 1
            if index % 40 == 0 or index + 1 == len(counts):
                print(json.dumps({"phase": "reloaded_checkpoint_audit", "windows_rendered": index + 1,
                                  "windows": len(counts), "endpoint_time_seconds": timestamp}), flush=True)
    diagnostics = event_consistency_diagnostics(
        predictions, counts, radiometry["threshold_positive"], radiometry["threshold_negative"],
        max_prefixes=16)
    prefixes = diagnostics["prefixes"]["intervals"]
    if len(prefixes) > 16:
        raise ValueError("Consistency helper returned more than 16 prefix diagnostics")
    for interval in prefixes:
        interval["start_time_seconds"] = times[interval["start_index"]]
        interval["end_time_seconds"] = times[interval["stop_index_exclusive"]]
    for scale in diagnostics["temporal_scales"]:
        for interval in scale["intervals"]:
            interval["start_time_seconds"] = times[interval["start_index"]]
            interval["end_time_seconds"] = times[interval["stop_index_exclusive"]]
    # Changes during the run invalidate its identity rather than borrowing data
    # from an earlier model or silently accepting a mixed snapshot.
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while diagnostics were running")
    for name, path in (("g0", args.g0), ("events", args.events)):
        if sha256_file(path) != payload["input_sha256"][name]:
            raise ValueError(f"Explicit {name} changed while diagnostics were running")
    report = {
        "format": "event_only_checkpoint_consistency_v1", "diagnostic_only": True,
        "approval_changed": False, "acceptance_gate_modified": False,
        "status": "diagnostic_only_not_an_approval", "checkpoint_sha256": checkpoint_hash,
        "input_sha256": payload["input_sha256"], "radiometry": radiometry,
        "checkpoint_format": payload["format"], "source_gaussians": renderer.g0.count,
        "explicit_inputs": {"checkpoint": str(args.checkpoint.resolve()), "g0": str(args.g0.resolve()),
                            "events": str(args.events.resolve())},
        "camera_source": "Numeric camera whitelist from the hash-bound checkpoint; no camera file reopened",
        "timeline": {
            "windows": len(counts), "rendered_endpoints": rendered_endpoints,
            "endpoint_times_seconds": times, "checkpoint_start_seconds": knots[0],
            "checkpoint_end_seconds": knots[-1], "exact_final_endpoint_rendered": times[-1] == knots[-1],
            "step_seconds": args.step_seconds, "substeps": args.substeps,
            "effective_step_seconds": args.step_seconds / args.substeps,
            "event_windows_metadata": windows["metadata"],
        },
        "fit_split_recorded_by_training": payload["training_summary"].get("fit_split", "not_recorded"),
        "evaluation_scope": "Full-sequence post-training consistency diagnostics; not independent validation",
        "diagnostics": diagnostics, "prefix_point_limit": 16,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "No acceptance thresholds, existing audits, model parameters, or approval flags are changed.",
            "Estimated or default radiometry is not an independently calibrated event observation model.",
            "Single-view consistency does not establish unique 3D motion or unseen geometry.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(json.dumps({"report": str(args.output), "windows": len(counts),
                      "diagnostic_only": True, "approval_changed": False}), flush=True)


if __name__ == "__main__":
    main()
