"""Array-only G0/event-proxy photometric scores for existing rigid hypotheses.

This diagnostic does not bind Gaussians, create observations, interpolate
poses, infer hidden surfaces, or verify occlusion. Its world-space anchors
are backprojections of the *rendered G0 depth hypothesis*, not measurements.
Only NumPy is required. There is no file access or renderer dependency.
"""

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real

import numpy as np


def _real(value, name, lower, upper=None, inclusive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
            not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite and real")
    value = float(value)
    if (value < lower if inclusive else value <= lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(value)


def _image(value, shape, name, finite=True):
    array = np.asarray(value)
    if array.shape == (1,) + shape:
        array = array[0]
    if array.shape != shape or array.dtype.kind not in "fiu" or (finite and not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a real {'finite ' if finite else ''}H-by-W image")
    return array.astype(np.float64, copy=False)


def _camera(camera, shape):
    if not isinstance(camera, Mapping) or camera.get("pixel_center") != "boundary":
        raise ValueError("camera must explicitly use boundary pixel coordinates")
    height, width = shape
    for key, expected in (("width", width), ("height", height)):
        if isinstance(camera.get(key), (bool, np.bool_)) or not isinstance(camera.get(key), Integral) or camera[key] != expected:
            raise ValueError("Camera width/height must match the reference image")
    fx = _real(camera.get("fx"), "fx", 0)
    fy = _real(camera.get("fy"), "fy", 0)
    cx, cy = camera.get("cx"), camera.get("cy")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
           not math.isfinite(float(value)) for value in (cx, cy)):
        raise ValueError("Camera principal point must be finite")
    transform = np.asarray(camera.get("w2c"), dtype=np.float64)
    if not _rigid(transform):
        raise ValueError("camera w2c must be a proper finite rigid transform")
    return fx, fy, float(cx), float(cy), transform


def _rigid(matrix):
    return (matrix.shape == (4, 4) and np.isfinite(matrix).all() and
            np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0) and
            np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-5, rtol=0) and
            abs(np.linalg.det(matrix[:3, :3]) - 1) <= 1e-5)


def _bilinear(image, xy):
    """Array-centre coordinates, with out-of-image taps marked invalid."""
    height, width = image.shape
    finite = np.isfinite(xy).all(axis=-1)
    valid = finite & (xy[..., 0] >= 0) & (xy[..., 0] <= width - 1) & (xy[..., 1] >= 0) & (xy[..., 1] <= height - 1)
    safe = np.where(finite[..., None], xy, 0)
    x = np.clip(safe[..., 0], 0, width - 1)
    y = np.clip(safe[..., 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    sample = ((1 - dx) * (1 - dy) * image[y0, x0] + dx * (1 - dy) * image[y0, x1] +
              (1 - dx) * dy * image[y1, x0] + dx * dy * image[y1, x1])
    return sample, valid


def _huber_absolute(residual, delta=.2):
    value = np.abs(residual)
    return np.where(value <= delta, value * value / (2 * delta), value - delta / 2)


def _hash_array(array):
    array = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256(np.asarray(array.shape, dtype="<i8").tobytes())
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def score_binding_event_unaries(reference_log, reference_alpha, normalized_depth,
                                cumulative_at_knots, camera, times,
                                candidate_world_poses, pose_valid, *, background,
                                log_floor=1e-4, max_anchors=4096, max_times=12,
                                min_common_times=4, minimum_alpha=.95,
                                maximum_relative_depth_range=.05,
                                minimum_background_log_contrast=.3,
                                minimum_patch_log_contrast=.12,
                                minimum_event_log_change=.1,
                                maximum_absolute_log_error=.3,
                                minimum_cost_gap=.05, minimum_relative_margin=.15,
                                minimum_winning_fraction=.7):
    """Compare fixed SE3 hypotheses using the same visible samples and times.

    ``reference_log`` must already use the trainer's background-composited
    observation model: log(clamp(gray + (1-alpha)*B, log_floor, 1)). Depth
    must be camera-z depth *normalized by alpha*, not the rasterizer sum.
    ``cumulative_at_knots`` is an array or one-shot iterable of T finite HxW
    maps C+N+ - C-N-, including exactly zero at t0. It is streamed once; the
    complete sequence is not stacked. All inputs must come from the same run.

    World poses are T-by-G-by-4-by-4, mapping G0 world to current world.
    Only original true pose_valid entries are used. Invalid non-t0 matrices
    may contain NaN; they are never filled or evaluated. Every t0 pose must
    be identity regardless of its valid flag, and t0 never counts as evidence.

    Anchors lie on a deterministic regular pixel grid. All nine source patch
    taps require high alpha, positive finite locally consistent depth and a
    non-saturated reference. Background-like low-texture patches abstain;
    dark material distinct from background is not automatically rejected.
    Each patch tap is backprojected and SE3-warped separately, not translated
    in image space. Residuals retain the absolute original G0 log intensity.

    At most twelve times are sampled from the *all-candidate* valid-time
    intersection. Out-of-frame, behind-camera or saturated proxy patches
    remove that anchor/time for EVERY candidate. No candidate may discard
    just its own bad frames. A winner needs >=4 common times with >=2 in each
    half of the full supplied timeline, >=2 times with local net event
    changes, low absolute residual, a score margin and >=70% consistent wins.
    No target visibility is measured: unexplained darkness/high residual or
    temporally conflicting fits remain unknown. Real occlusion, repeated
    texture, SH/view dependence and event-integral drift can still confound
    a hypothesis; returned winners are diagnostic suggestions, not bindings.

    All top-level non-array values are finite JSON data. Missing scores may
    be inf in arrays. winner=-1 means unknown; candidate indices are NOT
    semantic part labels. No Gaussian, pose-valid flag or input is modified.
    """
    max_anchors = _integer(max_anchors, "max_anchors", 1, 4096)
    max_times = _integer(max_times, "max_times", 4, 12)
    min_common_times = _integer(min_common_times, "min_common_times", 4, max_times)
    minimum_alpha = _real(minimum_alpha, "minimum_alpha", .95, 1, inclusive=True)
    depth_range = _real(maximum_relative_depth_range, "maximum_relative_depth_range", 0, .1)
    background_contrast = _real(minimum_background_log_contrast, "minimum_background_log_contrast", 0)
    patch_contrast = _real(minimum_patch_log_contrast, "minimum_patch_log_contrast", 0)
    event_change = _real(minimum_event_log_change, "minimum_event_log_change", 0)
    absolute_limit = _real(maximum_absolute_log_error, "maximum_absolute_log_error", 0)
    cost_gap_limit = _real(minimum_cost_gap, "minimum_cost_gap", 0)
    relative_limit = _real(minimum_relative_margin, "minimum_relative_margin", 0, 1, inclusive=True)
    winning_fraction = _real(minimum_winning_fraction, "minimum_winning_fraction", .5, 1)
    floor = _real(log_floor, "log_floor", 0, .01)
    background = _real(background, "background", floor, 1, inclusive=True)
    lower_log = math.log(floor)

    reference = np.asarray(reference_log)
    if reference.ndim == 3 and reference.shape[0] == 1:
        reference = reference[0]
    if reference.ndim != 2 or min(reference.shape) < 3:
        raise ValueError("reference_log must be an H-by-W image at least 3-by-3")
    height, width = reference.shape
    reference = _image(reference, (height, width), "reference_log")
    if np.any(reference < lower_log - 1e-6) or np.any(reference > 1e-6):
        raise ValueError("reference_log must use the stated floor and unit intensity ceiling")
    alpha = _image(reference_alpha, reference.shape, "reference_alpha")
    depth = _image(normalized_depth, reference.shape, "normalized_depth", finite=False)
    if np.any(alpha < 0) or np.any(alpha > 1 + 1e-6):
        raise ValueError("reference_alpha must be in [0,1]")
    fx, fy, cx, cy, w2c = _camera(camera, reference.shape)
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("times must start at zero and increase strictly")
    poses, valid = np.asarray(candidate_world_poses, dtype=np.float64), np.asarray(pose_valid)
    if (poses.ndim != 4 or poses.shape[0] != len(times) or poses.shape[2:] != (4, 4) or
            not 1 <= poses.shape[1] <= 32 or valid.dtype != np.bool_ or valid.shape != poses.shape[:2]):
        raise ValueError("Candidate poses and boolean validity must share T-by-G, with 1<=G<=32")
    groups = poses.shape[1]
    if any(not _rigid(pose) for pose in poses[valid]):
        raise ValueError("Every valid candidate pose must be a finite proper SE3")
    if not np.allclose(poses[0], np.eye(4), atol=1e-8, rtol=0):
        raise ValueError("Every candidate must retain exact G0 identity at t0")

    common = np.flatnonzero(valid.all(axis=1) & (times > 0))
    if len(common) > max_times:
        common = common[np.rint(np.linspace(0, len(common) - 1, max_times)).astype(np.int64)]
    selected_lookup = {int(index): offset for offset, index in enumerate(common)}
    stride = max(1, math.ceil(math.sqrt((height - 2) * (width - 2) / max_anchors)))
    grid_y, grid_x = np.meshgrid(np.arange(1, height - 1, stride), np.arange(1, width - 1, stride), indexing="ij")
    grid = np.column_stack((grid_y.ravel(), grid_x.ravel()))
    if len(grid) > max_anchors:
        grid = grid[np.rint(np.linspace(0, len(grid) - 1, max_anchors)).astype(np.int64)]
    offsets = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    pixels = grid[:, None, :] + offsets[None]
    rows, columns = pixels[..., 0], pixels[..., 1]
    source_logs, patch_depth = reference[rows, columns], depth[rows, columns]
    alpha_ok = (alpha[rows, columns] >= minimum_alpha).all(axis=1)
    finite_depth = (np.isfinite(patch_depth) & (patch_depth > 0)).all(axis=1)
    safe_depth = np.where(np.isfinite(patch_depth) & (patch_depth > 0), patch_depth, 1.)
    depth_ok = finite_depth & (np.ptp(safe_depth, axis=1) <= depth_range * np.maximum(safe_depth[:, 4], 1e-12))
    radiometric_ok = ((source_logs > lower_log + .1) & (source_logs < -.01)).all(axis=1)
    texture = np.ptp(source_logs, axis=1)
    source_signal = np.abs(np.median(source_logs, axis=1) - math.log(background))
    signal_ok = (texture >= patch_contrast) | (source_signal >= background_contrast)
    source_keep = alpha_ok & depth_ok & radiometric_ok & signal_ok
    source_grid, source_pixels = grid[source_keep], pixels[source_keep]
    source_logs, source_depth = source_logs[source_keep], patch_depth[source_keep]
    boundary_xy = source_pixels[..., ::-1].astype(np.float64) + .5
    camera_patch = np.stack(((boundary_xy[..., 0] - cx) * source_depth / fx,
                             (boundary_xy[..., 1] - cy) * source_depth / fy, source_depth), axis=-1)
    world_patch = (camera_patch - w2c[:3, 3]) @ w2c[:3, :3]
    anchors, selected_count = len(source_grid), len(common)
    costs_per_time = np.full((anchors, selected_count, groups), np.inf)
    absolute_per_time = np.full_like(costs_per_time, np.inf)
    no_event_per_time = np.full_like(costs_per_time, np.inf)
    shared_mask = np.zeros((anchors, selected_count), dtype=bool)
    event_mask = np.zeros_like(shared_mask)
    source_rows, source_columns = source_pixels[..., 0], source_pixels[..., 1]
    rejected_projection = rejected_saturation = 0
    cumulative_hash = hashlib.sha256(np.asarray((len(times), height, width), dtype="<i8").tobytes())
    try:
        cumulative = iter(cumulative_at_knots)
    except TypeError as error:
        raise ValueError("cumulative_at_knots must be an array or iterable of maps") from error
    for time_index in range(len(times)):
        try:
            delta = _image(next(cumulative), reference.shape, "cumulative event map")
        except StopIteration as error:
            raise ValueError("Too few cumulative event maps") from error
        cumulative_hash.update(memoryview(np.ascontiguousarray(delta, dtype="<f8")).cast("B"))
        if time_index == 0 and not np.allclose(delta, 0, atol=1e-8, rtol=0):
            raise ValueError("The t0 cumulative event map must be zero")
        if time_index not in selected_lookup or not anchors:
            continue
        slot = selected_lookup[time_index]
        proxy = reference + delta
        joint_projection, joint_radiometry = np.ones(anchors, bool), np.ones(anchors, bool)
        local_event = np.mean(np.abs(delta[source_rows, source_columns]), axis=1)
        for group in range(groups):
            pose = poses[time_index, group]
            moved_world = world_patch @ pose[:3, :3].T + pose[:3, 3]
            moved_camera = moved_world @ w2c[:3, :3].T + w2c[:3, 3]
            z = moved_camera[..., 2]
            safe_z = np.maximum(z, 1e-8)
            xy = np.stack((fx * moved_camera[..., 0] / safe_z + cx - .5,
                           fy * moved_camera[..., 1] / safe_z + cy - .5), axis=-1)
            sampled, inside = _bilinear(proxy, xy)
            initial, _ = _bilinear(reference, xy)
            projection_ok = (inside & (z > 1e-8)).all(axis=1)
            radiometry_ok = ((sampled > lower_log + .1) & (sampled < -.01)).all(axis=1)
            joint_projection &= projection_ok
            joint_radiometry &= radiometry_ok
            residual = sampled - source_logs
            costs_per_time[:, slot, group] = np.mean(_huber_absolute(residual), axis=1)
            absolute_per_time[:, slot, group] = np.mean(np.abs(residual), axis=1)
            no_event_per_time[:, slot, group] = np.mean(_huber_absolute(initial - source_logs), axis=1)
            local_event = np.maximum(local_event, np.mean(np.abs(sampled - initial), axis=1))
        shared_mask[:, slot] = joint_projection & joint_radiometry
        event_mask[:, slot] = local_event >= event_change
        rejected_projection += int((~joint_projection).sum())
        rejected_saturation += int((joint_projection & ~joint_radiometry).sum())

    try:
        next(cumulative)
    except StopIteration:
        pass
    else:
        raise ValueError("Too many cumulative event maps")
    costs_per_time[~shared_mask] = np.inf
    absolute_per_time[~shared_mask] = np.inf
    no_event_per_time[~shared_mask] = np.inf
    count = shared_mask.sum(axis=1)
    denominator = np.maximum(count, 1)[:, None]
    cost = np.where(shared_mask[..., None], costs_per_time, 0).sum(axis=1) / denominator
    mean_absolute = np.where(shared_mask[..., None], absolute_per_time, 0).sum(axis=1) / denominator
    no_event_cost = np.where(shared_mask[..., None], no_event_per_time, 0).sum(axis=1) / denominator
    cost[count == 0] = mean_absolute[count == 0] = no_event_cost[count == 0] = np.inf
    winner = np.full(anchors, -1, np.int32)
    margin, gap, win_fraction = np.zeros(anchors), np.zeros(anchors), np.zeros(anchors)
    reasons = np.full(anchors, "insufficient_common_times", dtype="<U40")
    early = times[common] <= times[-1] / 2
    span_ok = (shared_mask[:, early].sum(axis=1) >= 2) & (shared_mask[:, ~early].sum(axis=1) >= 2)
    event_count = (shared_mask & event_mask).sum(axis=1)
    for anchor in range(anchors):
        if count[anchor] < min_common_times:
            continue
        if not span_ok[anchor]:
            reasons[anchor] = "insufficient_timeline_span"
            continue
        if event_count[anchor] < 2:
            reasons[anchor] = "insufficient_net_event_change"
            continue
        if groups < 2:
            reasons[anchor] = "no_competing_candidate"
            continue
        ranking = np.argsort(cost[anchor], kind="stable")
        best, runner_up = int(ranking[0]), int(ranking[1])
        gap[anchor] = cost[anchor, runner_up] - cost[anchor, best]
        margin[anchor] = gap[anchor] / max(cost[anchor, runner_up], 1e-12)
        used = costs_per_time[anchor, shared_mask[anchor]]
        competitors = np.delete(used, best, axis=1).min(axis=1)
        win_fraction[anchor] = np.mean(competitors - used[:, best] >= cost_gap_limit)
        if mean_absolute[anchor, best] > absolute_limit:
            reasons[anchor] = "high_residual_or_occlusion"
        elif gap[anchor] < cost_gap_limit or margin[anchor] < relative_limit:
            reasons[anchor] = "ambiguous_candidate_scores"
        elif win_fraction[anchor] < winning_fraction:
            reasons[anchor] = "temporally_inconsistent_winner"
        else:
            winner[anchor], reasons[anchor] = best, "diagnostic_winner"

    reason_counts = {str(reason): int((reasons == reason).sum()) for reason in np.unique(reasons)}
    finite_best = cost.min(axis=1) if groups else np.empty(anchors)
    finite_best = finite_best[np.isfinite(finite_best)]
    diagnostics = {
        "reference_grid_stride_pixels": stride, "grid_anchor_count": len(grid), "anchor_count": anchors,
        "grid_rejected_low_alpha": int((~alpha_ok).sum()),
        "grid_rejected_invalid_or_discontinuous_depth": int((~depth_ok).sum()),
        "grid_rejected_reference_floor_or_saturation": int((~radiometric_ok).sum()),
        "grid_rejected_background_like_low_texture": int((~signal_ok).sum()),
        "all_candidate_valid_non_t0_count": int((valid[1:].all(axis=1)).sum()),
        "sampled_common_time_count": selected_count,
        "shared_anchor_times_rejected_projection": rejected_projection,
        "shared_anchor_times_rejected_proxy_floor_or_saturation": rejected_saturation,
        "winner_counts_by_candidate": [int((winner == group).sum()) for group in range(groups)],
        "unknown_anchor_count": int((winner < 0).sum()), "rejection_and_acceptance_counts": reason_counts,
        "mean_best_cost_over_finite_anchors": float(finite_best.mean()) if len(finite_best) else None,
        "binding_modified": False, "pose_observations_added": 0, "gaussians_removed": 0,
    }
    metadata = {
        "kind": "G0_event_proxy_binding_photometric_diagnostic_v1", "version": 1,
        "array_inputs_only": True, "raw_input_provenance_verified_here": False,
        "source_anchor_kind": "G0_rendered_camera_z_depth_backprojection_not_measurement",
        "anchor_pixel_coordinates": "boundary", "patch_shape": [3, 3],
        "depth_is_normalized_by_alpha": True, "reference_log_includes_background": True,
        "background": background, "log_floor": floor, "minimum_alpha": minimum_alpha,
        "maximum_relative_depth_range": depth_range,
        "minimum_background_log_contrast": background_contrast,
        "minimum_patch_log_contrast": patch_contrast, "minimum_event_log_change": event_change,
        "maximum_absolute_log_error": absolute_limit, "minimum_cost_gap": cost_gap_limit,
        "minimum_relative_margin": relative_limit, "minimum_winning_fraction": winning_fraction,
        "minimum_common_times": min_common_times, "minimum_times_per_timeline_half": 2,
        "maximum_anchors": max_anchors, "maximum_scored_times": max_times,
        "loss": "mean_absolute_reference_log_huber_delta_0.2_no_patch_mean_subtraction",
        "time_selection": "all_candidate_true_valid_intersection_then_even_index_subsample",
        "shared_mask": "all_candidate_patch_in_frame_positive_depth_unsaturated_proxy",
        "event_support": "net_cumulative_log_change_at_source_or_candidate_targets_not_raw_activity",
        "reference_log_sha256": _hash_array(reference), "times_sha256": _hash_array(times),
        "cumulative_log_maps_sha256": cumulative_hash.hexdigest(),
        "pose_valid_sha256": _hash_array(valid),
        "valid_candidate_poses_sha256": _hash_array(np.where(valid[..., None, None], poses, 0)),
        "input_array_hash_encoding": "int64_le_shape_then_float64_le_values",
        "poses_interpolated": False, "t0_counts_as_observation": False,
        "target_occlusion_verified": False, "occlusion_can_remain_confounded": True,
        "SH_view_dependence_modelled": False, "proxy_is_measured_RGB": False,
        "source_points_are_raw_Gaussian_rows": False, "geometry_reach_applied": False,
        "physical_motion_validated": False, "backside_observations_added": 0,
        "pose_observations_added": 0, "binding_modified": False, "approved_for_export": False,
        "winner_is_unverified_hypothesis": True,
    }
    return {
        "anchor_points": world_patch[:, 4].copy(), "anchor_pixels": boundary_xy[:, 4].copy(),
        "anchor_pixel_indices_yx": source_grid.copy(), "cost": cost,
        "mean_absolute_log_error": mean_absolute, "counterfactual_no_event_cost": no_event_cost,
        "winner": winner, "margin": margin, "cost_gap": gap,
        "winning_time_fraction": win_fraction, "informative": winner >= 0,
        "common_time_indices": common.copy(), "common_times": times[common].copy(),
        "shared_time_mask": shared_mask, "net_event_supported_mask": event_mask & shared_mask,
        "common_time_count": count, "net_event_supported_time_count": event_count,
        "per_time_cost": costs_per_time, "reason": reasons,
        "metadata": metadata, "diagnostics": diagnostics,
    }
