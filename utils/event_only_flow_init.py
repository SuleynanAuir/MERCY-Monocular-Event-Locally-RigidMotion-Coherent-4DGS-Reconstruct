"""Classical, array-only event-proxy initialization; not a 3D observation.

Only fit-all callers may supply cumulative events from the whole sequence.
This module does not enforce a training/validation split: callers must never
use held-out future events to initialize a training-only experiment.
"""

from __future__ import annotations

import math


def estimate_local_kabsch_rotations(graph_centers_np, translations_np, tracked_mask_np,
                                    previous_rotation_vectors_np=None, neighbors=12):
    """Fit local proper rotations using only spatial neighbors from G0.

    This estimates a rotation compatible with the provisional translations,
    not an independently observed 3D rotation. At least four supported points
    spanning two spatial directions are required. Unsupported or nonrigid
    neighborhoods retain their preceding reliable rotation (initially zero).
    """
    import cv2
    import numpy as np
    from scipy.spatial import cKDTree

    centers = np.asarray(graph_centers_np, dtype=np.float64)
    translations = np.asarray(translations_np, dtype=np.float64)
    tracked = np.asarray(tracked_mask_np)
    if (centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or
            translations.shape != centers.shape or tracked.shape != (len(centers),) or
            tracked.dtype != np.bool_ or not np.isfinite(centers).all() or not np.isfinite(translations).all()):
        raise ValueError("Kabsch requires finite N-by-3 centers/translations and a boolean tracked mask")
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 4:
        raise ValueError("Kabsch neighbors must be an integer of at least four")
    previous = np.zeros_like(centers) if previous_rotation_vectors_np is None else np.asarray(
        previous_rotation_vectors_np, dtype=np.float64)
    if previous.shape != centers.shape or not np.isfinite(previous).all():
        raise ValueError("Previous rotations must be finite N-by-3 axis-angle vectors")
    rotations = previous.copy()
    reliable = np.zeros(len(centers), dtype=bool)
    relative_rmse = np.zeros(len(centers), dtype=np.float64)
    if len(centers) < 4 or tracked.sum() < 4:
        return {"rotation_vectors": rotations.astype(np.float32), "reliable": reliable,
                "fit_relative_rmse": relative_rmse}
    distances, neighbor_ids = cKDTree(centers).query(centers, k=min(neighbors, len(centers)))
    moved = centers + translations
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-12)
    for node in np.flatnonzero(tracked):
        supported = tracked[neighbor_ids[node]]
        ids, distance = neighbor_ids[node][supported], distances[node][supported]
        if len(ids) < 4:
            continue
        radius = max(float(distance.max()), scale * 1e-8)
        weights = np.exp(-.5 * (distance / radius) ** 2)
        weights /= weights.sum()
        source = centers[ids] - (centers[ids] * weights[:, None]).sum(axis=0)
        target = moved[ids] - (moved[ids] * weights[:, None]).sum(axis=0)
        source_covariance = (source * weights[:, None]).T @ source
        target_covariance = (target * weights[:, None]).T @ target
        source_eigenvalues = np.linalg.eigvalsh(source_covariance)
        target_eigenvalues = np.linalg.eigvalsh(target_covariance)
        # Planar patches are supported; collinear/collapsed patches cannot
        # identify rotation around the missing spatial direction reliably.
        if (source_eigenvalues[-2] <= max(source_eigenvalues[-1] * 1e-4, scale**2 * 1e-12) or
                target_eigenvalues[-2] <= max(target_eigenvalues[-1] * 1e-4, scale**2 * 1e-12)):
            continue
        covariance = (source * weights[:, None]).T @ target
        left, _, right_t = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = 1 if np.linalg.det(right_t.T @ left.T) >= 0 else -1
        rotation = right_t.T @ correction @ left.T
        # Never emit a reflection; incompatible reflected/nonrigid motion is
        # rejected by the proper-rotation residual, not called a valid fit.
        residual = source @ rotation.T - target
        error = float(np.sqrt((weights * (residual * residual).sum(axis=1)).sum()))
        spread = float(np.sqrt((weights * (source * source).sum(axis=1)).sum()))
        relative_error = error / max(spread, scale * 1e-8)
        if not math.isfinite(relative_error) or relative_error > .15:
            continue
        vector = cv2.Rodrigues(rotation)[0].reshape(3)
        angle = float(np.linalg.norm(vector))
        # Rodrigues returns the principal angle. Pick its nearest equivalent
        # lift so crossing pi does not interpolate through a spurious reversal.
        candidates = [vector]
        if angle > 1e-7:
            axis = vector / angle
            turns = int(round(float(np.dot(previous[node] - vector, axis)) / (2 * math.pi)))
            candidates.extend(vector + axis * (2 * math.pi * turn) for turn in (turns - 1, turns, turns + 1))
        else:
            old_angle = float(np.linalg.norm(previous[node]))
            if old_angle > 1e-7:
                candidates.append(previous[node] / old_angle * (2 * math.pi * round(old_angle / (2 * math.pi))))
        rotations[node] = min(candidates, key=lambda candidate: np.linalg.norm(candidate - previous[node]))
        reliable[node] = True
        relative_rmse[node] = relative_error
    return {"rotation_vectors": rotations.astype(np.float32), "reliable": reliable,
            "fit_relative_rmse": relative_rmse}


def infer_event_flow_initialization(reference_gray_np, cumulative_log_changes_at_knots_np,
                                    graph_centers_np, camera_whitelist, knot_times_np,
                                    log_floor=.01, reference_depth_np=None, reference_alpha_np=None):
    """Track G0 projections through event-derived proxies and lift at G0 depth.

    Inputs are arrays (or a one-pass iterable of cumulative H-by-W log changes),
    never paths. Cumulative changes and times include the zero-change t=0 knot.
    Returned translations/rotation_vectors omit the exactly fixed first knot.
    The grayscale reference must be a G0 render, not an input RGB/video frame.
    This function cannot establish the caller's claimed source provenance.
    Optional depth/alpha must come from that same G0 render. Depth is the
    alpha-weighted camera-z sum and is normalized by alpha before visibility
    checks. These are initial visibility checks, not future occlusion labels.
    A node is accepted only while every preceding G0-to-current pair was
    accepted. Rejection or no observation permanently breaks that chain;
    retaining the last pose is an unverified prior, never reacquisition.
    """
    import cv2
    import numpy as np

    if isinstance(log_floor, bool) or not math.isfinite(float(log_floor)) or not 0 < log_floor < 1:
        raise ValueError("log_floor must lie strictly between zero and one")
    reference = np.asarray(reference_gray_np)
    if reference.dtype == np.uint8:
        reference = reference.astype(np.float32) / 255
    else:
        reference = np.asarray(reference, dtype=np.float32)
    if (reference.ndim != 2 or min(reference.shape) < 16 or not np.isfinite(reference).all() or
            reference.min() < 0 or reference.max() > 1):
        raise ValueError("G0 reference must be a finite H-by-W grayscale array in [0,1]")
    height, width = reference.shape
    depth_image, alpha_image = None, None
    if (reference_depth_np is None) != (reference_alpha_np is None):
        raise ValueError("G0 depth and alpha must be supplied together")
    if reference_depth_np is not None:
        depth_image = np.asarray(reference_depth_np, dtype=np.float32)
        alpha_image = np.asarray(reference_alpha_np, dtype=np.float32)
        if (depth_image.shape != reference.shape or alpha_image.shape != reference.shape or
                not np.isfinite(depth_image).all() or not np.isfinite(alpha_image).all() or
                np.any(depth_image < 0) or np.any(alpha_image < 0) or np.any(alpha_image > 1 + 1e-5)):
            raise ValueError("G0 depth/alpha must be finite H-by-W arrays with nonnegative depth and alpha in [0,1]")
    if camera_whitelist.get("width") != width or camera_whitelist.get("height") != height:
        raise ValueError("Camera dimensions must match the G0 reference")
    if camera_whitelist.get("pixel_center") != "boundary":
        raise ValueError("Only the explicit boundary-coordinate camera convention is supported")
    fx, fy, cx, cy = [float(camera_whitelist[name]) for name in ("fx", "fy", "cx", "cy")]
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("Camera intrinsics are invalid")
    w2c = np.asarray(camera_whitelist["w2c"], dtype=np.float64)
    if (w2c.shape != (4, 4) or not np.isfinite(w2c).all() or
            not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-5, rtol=0) or
            abs(np.linalg.det(w2c[:3, :3]) - 1) > 1e-5):
        raise ValueError("w2c must be a finite proper rigid camera transform")
    centers = np.asarray(graph_centers_np, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("Graph centers must be finite N-by-3 G0 positions")
    times = np.asarray(knot_times_np, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Knot times must increase strictly from zero")
    cumulative = iter(cumulative_log_changes_at_knots_np)

    def next_change():
        try:
            value = np.asarray(next(cumulative), dtype=np.float32)
        except StopIteration as exc:
            raise ValueError("Too few cumulative event maps for the knot times") from exc
        if value.shape != reference.shape or not np.isfinite(value).all():
            raise ValueError("Each cumulative event map must be finite and match the reference shape")
        return value

    previous_change = next_change()
    if np.max(np.abs(previous_change)) > 1e-6:
        raise ValueError("The first knot must have zero cumulative log change at G0")
    log_reference = np.log(np.maximum(reference, log_floor))
    previous_proxy = np.maximum(reference, log_floor)
    camera_positions = centers @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera_positions[:, 2].copy()
    front = depth > 1e-8
    safe_depth = np.where(front, depth, 1)
    # Calibrated boundary coordinates put the first pixel center at .5;
    # OpenCV samples that same center at array coordinate zero.
    tracked = np.column_stack((fx * camera_positions[:, 0] / safe_depth + cx - .5,
                               fy * camera_positions[:, 1] / safe_depth + cy - .5)).astype(np.float32)
    initial_projection = tracked.copy()
    ever_active = np.zeros(reference.shape, dtype=bool)
    accepted_masks = np.zeros((len(times) - 1, len(centers)), dtype=bool)
    translation = np.zeros_like(centers)
    translations = np.zeros((len(times) - 1, len(centers), 3), dtype=np.float32)
    rotation = np.zeros_like(centers, dtype=np.float32)
    rotations = np.zeros_like(translations)
    stats = []

    def sample(array, coordinates):
        # Offscreen and behind-camera nodes use safe dummy map coordinates;
        # their samples are never accepted by the visibility mask.
        coordinates = np.nan_to_num(coordinates, nan=-1, posinf=-1, neginf=-1)
        return cv2.remap(array, coordinates[:, 0].reshape(-1, 1).astype(np.float32),
                         coordinates[:, 1].reshape(-1, 1).astype(np.float32),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0).reshape((len(centers),) + array.shape[2:])

    def inside(coordinates):
        return (np.isfinite(coordinates).all(axis=1) & (coordinates[:, 0] >= 1) &
                (coordinates[:, 0] <= width - 2) & (coordinates[:, 1] >= 1) &
                (coordinates[:, 1] <= height - 2))

    def contrast(image):
        mean = cv2.blur(image, (9, 9))
        return np.sqrt(np.maximum(cv2.blur(image * image, (9, 9)) - mean * mean, 0))

    initially_in_frame = front & inside(tracked)
    initially_visible = initially_in_frame.copy()
    depth_tolerance = None
    if depth_image is not None:
        from scipy.spatial import cKDTree
        scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-8)
        if len(centers) > 1:
            local_distances = cKDTree(centers).query(centers, k=min(4, len(centers)))[0]
            local_spacing = np.median(local_distances[:, 1:], axis=1)
        else:
            local_spacing = np.zeros(1)
        # Sparse graph centers and Gaussian depth averaging require a tolerance
        # wider than a pixel's precision, but unrelated hidden nodes must not
        # inherit foreground motion merely because their projections overlap.
        depth_tolerance = np.maximum(.015 * scale, .5 * local_spacing)
        sampled_alpha = sample(alpha_image, tracked)
        sampled_depth = sample(depth_image, tracked) / np.maximum(sampled_alpha, 1e-8)
        initially_visible &= ((sampled_alpha >= .10) & np.isfinite(sampled_depth) &
                              (sampled_depth > 0) & (np.abs(sampled_depth - depth) <= depth_tolerance))

    chain_valid = initially_visible.copy()
    first_chain_break_knot = np.full(len(centers), -1, dtype=np.int64)
    for knot in range(1, len(times)):
        change = next_change()
        proxy_log = log_reference + change
        # Deliberately bounded display/intensity proxy, not recovered observed RGB.
        proxy = np.exp(np.clip(proxy_log, -20, 0)).astype(np.float32)
        delta = np.abs(change - previous_change)
        active = cv2.dilate((delta > .005).astype(np.float32), np.ones((9, 9), np.uint8))
        ever_active |= active > 0
        # Confidence is measured only over informative pixels, so the ordinary
        # black background does not make an entire view look saturated.
        informative = ((previous_proxy > log_floor * 1.5) | (proxy > log_floor * 1.5))
        exposed = ((previous_proxy < .985) & (proxy < .985) &
                   (previous_proxy > log_floor * .5) & (proxy > log_floor * .5))
        informative_density = cv2.blur(informative.astype(np.float32), (9, 9))
        exposure_confidence = cv2.blur((informative & exposed).astype(np.float32), (9, 9)) / np.maximum(informative_density, 1e-8)
        visible = initially_visible & inside(tracked)
        chain_valid_before = int(chain_valid.sum())
        local_accepted = np.zeros(len(centers), dtype=bool)
        accepted = np.zeros(len(centers), dtype=bool)
        consistent = np.zeros(len(centers), dtype=bool)
        magnitude = np.zeros(len(centers), dtype=np.float32)
        if np.any(active):
            previous_image = np.rint(np.clip(previous_proxy, 0, 1) * 255).astype(np.uint8)
            current_image = np.rint(proxy * 255).astype(np.uint8)
            parameters = dict(pyr_scale=.5, levels=4, winsize=21, iterations=4,
                              poly_n=7, poly_sigma=1.5, flags=0)
            forward = cv2.calcOpticalFlowFarneback(previous_image, current_image, None, **parameters)
            backward = cv2.calcOpticalFlowFarneback(current_image, previous_image, None, **parameters)
            displacement = sample(forward, tracked)
            candidate = tracked + displacement
            inverse_displacement = sample(backward, candidate)
            magnitude = np.linalg.norm(displacement, axis=1)
            error = np.linalg.norm(displacement + inverse_displacement, axis=1)
            consistent = np.isfinite(error) & (error <= .75 + .05 * magnitude)
            texture = (sample(contrast(previous_proxy), tracked) >= .008) & (sample(contrast(proxy), candidate) >= .008)
            activity = (sample(active, tracked) > .1) | (sample(active, candidate) > .1)
            exposed_support = ((sample(exposure_confidence, tracked) >= .25) &
                               (sample(exposure_confidence, candidate) >= .25))
            local_accepted = (visible & inside(candidate) & activity & texture & consistent & exposed_support &
                              np.isfinite(displacement).all(axis=1) & (magnitude <= .35 * math.hypot(width, height)))
            # After a missed pair, tracked belongs to an older frame. A good
            # adjacent-pair flow at that stale pixel cannot recover G0 identity.
            accepted = chain_valid & local_accepted
            tracked[accepted] = candidate[accepted]
            camera_lifted = np.column_stack(((tracked[:, 0] + .5 - cx) * depth / fx,
                                             (tracked[:, 1] + .5 - cy) * depth / fy, depth))
            lifted_world = (camera_lifted - w2c[:3, 3]) @ w2c[:3, :3]
            translation[accepted] = lifted_world[accepted] - centers[accepted]
        blocked_unanchored = local_accepted & ~chain_valid
        newly_broken = chain_valid & ~accepted
        first_chain_break_knot[newly_broken] = knot
        # This is intentionally outside the active-event branch: no event
        # support is not evidence that the same physical point stayed still.
        chain_valid &= accepted
        accepted_masks[knot - 1] = accepted
        translations[knot - 1] = translation
        local_fit = estimate_local_kabsch_rotations(centers, translation, accepted, rotation)
        rotation = local_fit["rotation_vectors"]
        rotations[knot - 1] = rotation
        reliable_rotation = local_fit["reliable"]
        stats.append({"time": float(times[knot]), "active_fraction": float(np.mean(active > 0)),
                      "visible_nodes": int(visible.sum()), "accepted_nodes": int(accepted.sum()),
                      "chain_valid_nodes_before": chain_valid_before,
                      "chain_valid_nodes": int(chain_valid.sum()),
                      "newly_broken_chain_nodes": int(newly_broken.sum()),
                      "local_pair_gate_passed_nodes": int(local_accepted.sum()),
                      "blocked_unanchored_pair_candidates": int(blocked_unanchored.sum()),
                      "retained_unverified_prior_nodes": int((initially_visible & ~chain_valid).sum()),
                      "flow_consistent_nodes": int((consistent & visible).sum()),
                      "mean_accepted_flow_px": float(magnitude[accepted].mean()) if accepted.any() else 0.,
                      "max_accepted_flow_px": float(magnitude[accepted].max()) if accepted.any() else 0.,
                      "reliable_rotation_nodes": int(reliable_rotation.sum()),
                      "mean_rotation_fit_relative_rmse": float(local_fit["fit_relative_rmse"][reliable_rotation].mean()) if reliable_rotation.any() else 0.,
                      "informative_proxy_fraction": float(informative.mean()),
                      "proxy_exposed_informative_ratio": float((informative & exposed).sum() / informative.sum()) if informative.any() else 1.,
                      "clipped_proxy_fraction": float(np.mean((proxy_log < -20) | (proxy_log > 0)))})
        previous_proxy, previous_change = proxy, change
    try:
        next(cumulative)
    except StopIteration:
        pass
    else:
        raise ValueError("Too many cumulative event maps for the knot times")
    # A conservative static anchor hypothesis for optional 3D graph
    # propagation. This is a prior, not proof that a no-event point is static.
    union_support = cv2.dilate(ever_active.astype(np.float32), np.ones((17, 17), np.uint8))
    static_anchor_mask = initially_visible & (sample(union_support, initial_projection) <= .01)
    return {"translations": translations, "rotation_vectors": rotations,
            "initially_visible_mask": initially_visible.copy(), "accepted_masks": accepted_masks,
            "final_chain_valid_mask": chain_valid.copy(), "first_chain_break_knot": first_chain_break_knot,
            "static_anchor_mask": static_anchor_mask,
            "debug": {"pairs": stats, "invalid_initial_depth_nodes": int((~front).sum()),
                      "initial_visibility_checked": depth_image is not None,
                      "initial_visible_nodes": int(initially_visible.sum()),
                      "final_chain_valid_nodes": int(chain_valid.sum()),
                      "total_broken_chain_nodes": int((first_chain_break_knot >= 0).sum()),
                      "total_blocked_unanchored_pair_candidates": sum(item["blocked_unanchored_pair_candidates"] for item in stats),
                      "initial_occluded_or_low_alpha_nodes": int((initially_in_frame & ~initially_visible).sum()),
                      "depth_tolerance_min": float(depth_tolerance.min()) if depth_tolerance is not None else None,
                      "depth_tolerance_max": float(depth_tolerance.max()) if depth_tolerance is not None else None,
                      "total_accepted_node_steps": sum(item["accepted_nodes"] for item in stats)},
            "metadata": {"kind": "event_derived_proxy_farneback_initialization", "version": 3,
                         "proxy_is_observed_rgb": False, "pretrained_models": False,
                         "trajectory_identity_policy": "unbroken_G0_chain_only_no_automatic_reacquisition",
                         "unobserved_step_policy": "break_chain_even_if_no_event_activity",
                         "unsupported_translation": "retain_previous_translation_or_initial_zero_as_unverified_prior",
                         "retained_pose_is_observation": False,
                         "depth_prior": "constant_initial_G0_camera_depth_per_node",
                         "rotation_prior": "approximate_G0_local_weighted_Kabsch_from_provisional_translations",
                         "rotation_neighbors": 12, "rotation_min_supported_neighbors": 4,
                         "rotation_max_relative_fit_error": .15,
                         "unsupported_rotation": "retain_previous_reliable_rotation_or_initial_zero",
                         "rotation_is_independent_3d_measurement": False,
                         "visibility": "initial_G0_depth_alpha_checked" if depth_image is not None else "visibility_not_occlusion_checked",
                         "depth_render_semantics": "alpha_weighted_camera_z_normalized_by_alpha" if depth_image is not None else None,
                         "future_occlusion_checked": False,
                         "physical_uniqueness_guaranteed": False, "hidden_motion_validated": False,
                         "first_knot": "fixed_G0_identity_omitted_from_parameters",
                         "allowed_scope": "fit_all_only_if_full_sequence_events_are_used",
                         "log_floor": float(log_floor), "quaternion_order": "wxyz"}}
