"""Experimental multi-time rigid grouping from G0 and event-flow tracks only.

Groups are inferred motion-consistency hypotheses, not ground-truth parts.
The module accepts arrays only; it has no access to RGB, meshes, old labels,
old trajectories, or pretrained models. Unsupported poses remain explicitly
invalid, rather than being invented by temporal interpolation.
"""

from __future__ import annotations

import math


def discover_rigid_groups(graph_centers_np, tracked_pixels_np, accepted_masks_np,
                          camera_whitelist, static_anchor_mask_np=None, *,
                          max_groups=3, min_group_nodes=8, consensus_timestamps=3,
                          min_consensus_timestamps=2, reprojection_threshold_px=2.5,
                          seed=0, local_seed_count=8, local_seed_size=32,
                          max_refinement_candidates=4, motion_model="se3",
                          plane_ransac_iterations=192):
    """Infer disjoint rigid groups with cross-time robust motion consensus.

    ``tracked_pixels_np`` is T-by-N-by-2 in calibrated *boundary* coordinates.
    Unaccepted tracks may contain NaNs. Group label -1 means unknown, label 0
    means the caller's explicit static mask, and positive labels are inferred
    dynamic groups. Returned world-space part poses map original G0 positions
    to their deformed world positions. Poses have shape [T,G,4,4], including
    static group 0; missing group/time poses contain NaNs and pose_valid=False.
    Input times are not inferred: callers retain the fixed G0/t0 identity.

    Global proposals are attempted first. When they fail, at most
    ``local_seed_count`` spatially dispersed G0 neighborhoods provide further
    hypotheses; these neighborhoods are seeds, never asserted semantic parts.
    Each scope selects overlapping, time-separated observations independently.
    Temporal spacing uses array indices because this API receives no timestamps.
    Diagnostics distinguish insufficient observations, degeneracy, solver
    failures, and cross-time rejection; they do not certify track identities.
    Before taking ownership of points, a candidate is pruned against all input
    times twice and audited again. A fitted motion smaller than 2.5 pixels is
    treated as unresolved at this precision, not as proof of a static point.
    At most ``max_refinement_candidates`` distinct candidates per search tier
    receive this expensive all-time audit. Missing poses are never filled.

    ``motion_model='se3'`` retains the general PnP solver. The optional
    ``'camera_plane'`` model assumes camera-XY rigid motion (no scaling) and
    each point's original camera depth. This is an unconfirmed motion prior,
    not an extra observation or proof of physical motion. Both solvers face
    identical perspective-pixel and cross-time acceptance gates. Plane-mode
    results additionally expose unwrapped camera-plane angles and world-axis
    rotation vectors at valid poses only; their nearest-branch unwrapping is
    a representation prior and does not fill missing poses.
    """
    import cv2
    import numpy as np

    for value, name, minimum in ((max_groups, "max_groups", 1), (min_group_nodes, "min_group_nodes", 8),
                                 (consensus_timestamps, "consensus_timestamps", 2),
                                 (min_consensus_timestamps, "min_consensus_timestamps", 2),
                                 (local_seed_count, "local_seed_count", 0),
                                 (local_seed_size, "local_seed_size", 8),
                                 (max_refinement_candidates, "max_refinement_candidates", 1),
                                 (plane_ransac_iterations, "plane_ransac_iterations", 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if min_consensus_timestamps > consensus_timestamps:
        raise ValueError("Consensus minimum exceeds the timestamp budget")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(float(reprojection_threshold_px)) or not 0 < reprojection_threshold_px <= 2.5:
        raise ValueError("Reprojection threshold must be finite, positive and <= 2.5 pixels")
    if local_seed_count > 16 or local_seed_size < min_group_nodes:
        raise ValueError("Use at most 16 local seeds, each large enough for the minimum support")
    if motion_model not in ("se3", "camera_plane"):
        raise ValueError("motion_model must be 'se3' or 'camera_plane'")
    if plane_ransac_iterations > 512:
        raise ValueError("plane_ransac_iterations must not exceed 512")
    centers = np.asarray(graph_centers_np, dtype=np.float64)
    tracks = np.asarray(tracked_pixels_np, dtype=np.float64)
    accepted = np.asarray(accepted_masks_np)
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("G0 centers must be finite N-by-3 positions")
    if tracks.ndim != 3 or tracks.shape[1:] != (len(centers), 2) or not len(tracks):
        raise ValueError("Tracks must be T-by-N-by-2 boundary pixel coordinates")
    if accepted.dtype != np.bool_ or accepted.shape != tracks.shape[:2] or not np.isfinite(tracks[accepted]).all():
        raise ValueError("Accepted masks must be boolean with finite accepted tracks")
    static = np.zeros(len(centers), dtype=bool) if static_anchor_mask_np is None else np.asarray(static_anchor_mask_np)
    if static.dtype != np.bool_ or static.shape != (len(centers),):
        raise ValueError("Static mask must be a boolean N-vector")
    if camera_whitelist.get("pixel_center") != "boundary":
        raise ValueError("An explicit boundary-coordinate camera is required")
    fx, fy, cx, cy = [float(camera_whitelist[key]) for key in ("fx", "fy", "cx", "cy")]
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("Invalid camera intrinsics")
    w2c = np.asarray(camera_whitelist["w2c"], dtype=np.float64)
    if (w2c.shape != (4, 4) or not np.isfinite(w2c).all() or
            not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-5, rtol=0) or
            abs(np.linalg.det(w2c[:3, :3]) - 1) > 1e-5):
        raise ValueError("w2c must be a proper finite rigid transform")
    # OpenCV's array pixel center (0,0) corresponds to boundary coordinate (.5,.5).
    intrinsic = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1]], dtype=np.float64)
    image_points = tracks - .5
    original_camera = centers @ w2c[:3, :3].T + w2c[:3, 3]
    source_front = original_camera[:, 2] > 1e-8
    reference_pixels = np.column_stack((fx * original_camera[:, 0] / np.maximum(original_camera[:, 2], 1e-8) + cx,
                                        fy * original_camera[:, 1] / np.maximum(original_camera[:, 2], 1e-8) + cy))
    valid_tracks = accepted & source_front[None]
    displacement = np.linalg.norm(tracks - reference_pixels[None], axis=2)
    moving_tracks = valid_tracks & (displacement >= .75)
    scores = [float(np.median(displacement[time, valid_tracks[time] & ~static]))
              if int((valid_tracks[time] & ~static).sum()) >= min_group_nodes else 0.
              for time in range(len(tracks))]
    labels = np.full(len(centers), -1, dtype=np.int32)
    labels[static] = 0
    groups = []
    rejected_proposals = 0
    proposal_reports = []
    selection_reports = []
    minimum_fitted_motion = 2.5
    minimum_identity_improvement = max(.75, .5 * reprojection_threshold_px)
    resolvable_motion_tracks = valid_tracks & (displacement >= minimum_fitted_motion)
    motion_seed_mask = resolvable_motion_tracks.sum(axis=0) >= min_consensus_timestamps

    def overlap_report(ids, times):
        masks = valid_tracks[np.asarray(times, dtype=np.int64)][:, ids]
        counts = masks.sum(axis=0)
        integer_masks = masks.astype(np.int64)
        return {"valid_counts": masks.sum(axis=1).tolist(),
                "pairwise_common_valid_nodes": (integer_masks @ integer_masks.T).tolist(),
                "common_all_selected_nodes": int((counts == len(times)).sum()) if times else 0,
                "nodes_observed_at_minimum_times": int((counts >= min_consensus_timestamps).sum())}

    def choose_times(ids, scope, anchor, round_index):
        masks = valid_tracks[:, ids]
        moving = moving_tracks[:, ids]
        counts = masks.sum(axis=1)
        moving_counts = moving.sum(axis=1)
        resolvable_counts = (masks & (displacement[:, ids] >= minimum_fitted_motion)).sum(axis=1)
        # A large stationary background must not hide a smaller moving part.
        # This only proposes timestamps; no point is accepted without PnP and
        # the unchanged multi-time support/reprojection tests below.
        motion_scores = [float(np.median(displacement[time, ids[moving[time]]]))
                         if moving_counts[time] >= min_group_nodes else 0.
                         for time in range(len(tracks))]
        eligible = np.flatnonzero((counts >= min_group_nodes) &
                                 (moving_counts >= min_group_nodes))
        resolvable_times = np.flatnonzero(resolvable_counts >= min_group_nodes)
        prefer_resolvable = len(resolvable_times) >= min_consensus_timestamps
        if prefer_resolvable:
            # Widely separated but near-identity endpoints cannot establish
            # dynamic evidence. Prefer resolvable observations when available;
            # the fitted-model gate still has to independently verify them.
            eligible = resolvable_times
        chosen = []
        pairs = []
        if len(eligible) >= 2:
            # Float64 dot products have exact integer counts at practical array
            # sizes and avoid uint8 overflow for dense (>255 point) inputs.
            observed = masks[eligible].astype(np.float64)
            overlaps = observed @ observed.T
            max_motion = max(motion_scores) or 1.
            span = max(1, len(tracks) - 1)
            for first in range(len(eligible)):
                for second in range(first + 1, len(eligible)):
                    common = int(overlaps[first, second])
                    if common < min_group_nodes:
                        continue
                    left, right = int(eligible[first]), int(eligible[second])
                    ratio = common / max(1, min(int(counts[left]), int(counts[right])))
                    separation = (right - left) / span
                    motion = min(motion_scores[left], motion_scores[right]) / max_motion
                    quality = .55 * ratio + .30 * separation + .15 * motion
                    pairs.append((quality, common, separation, -left, -right, left, right))
            if pairs:
                best = max(pairs)
                chosen = [best[-2], best[-1]]
                while len(chosen) < consensus_timestamps:
                    extras = []
                    for time in eligible:
                        time = int(time)
                        if time in chosen:
                            continue
                        common = max(int((masks[time] & masks[other]).sum()) for other in chosen)
                        if common < min_group_nodes:
                            continue
                        ratio = max(int((masks[time] & masks[other]).sum()) /
                                    max(1, min(int(counts[time]), int(counts[other]))) for other in chosen)
                        separation = min(abs(time - other) for other in chosen) / span
                        quality = .55 * ratio + .30 * separation + .15 * motion_scores[time] / max_motion
                        extras.append((quality, common, separation, -time, time))
                    if not extras:
                        break
                    chosen.append(max(extras)[-1])
        chosen.sort()
        shared = overlap_report(ids, chosen)
        reason = "selected"
        if len(eligible) < min_consensus_timestamps:
            reason = "insufficient_motion_timestamps"
        elif not pairs:
            reason = "insufficient_cross_time_common_nodes"
        elif len(chosen) < min_consensus_timestamps or shared["nodes_observed_at_minimum_times"] < min_group_nodes:
            reason = "insufficient_repeated_node_support"
        selection_reports.append({"round": round_index, "scope": scope, "anchor_node": anchor,
                                  "scope_nodes": len(ids), "eligible_times": eligible.tolist(),
                                  "selected_times": chosen, "valid_nodes_by_time": counts.tolist(),
                                  "moving_nodes_by_time": moving_counts.tolist(),
                                  "resolvable_motion_nodes_by_time": resolvable_counts.tolist(),
                                  "preferred_resolvable_motion_times": prefer_resolvable,
                                  "selection_motion_scores_px": motion_scores,
                                  "reason": reason, **shared})
        return chosen if reason == "selected" else [], len(selection_reports) - 1

    def sufficiently_spread(ids):
        if len(ids) < min_group_nodes:
            return False
        singular = np.linalg.svd(centers[ids] - centers[ids].mean(axis=0), compute_uv=False)
        return len(singular) >= 2 and singular[1] > max(singular[0] * 1e-4, 1e-8)

    def estimate_camera_plane(ids, time, report):
        """Fit fixed-depth SE2 hypotheses; accept only true pixel residuals."""
        source_xy = original_camera[ids, :2]
        depth = original_camera[ids, 2]
        target_xy = np.column_stack(((tracks[time, ids, 0] - cx) * depth / fx,
                                     (tracks[time, ids, 1] - cy) * depth / fy))
        # Camera-XY squared errors approximately scale as 1/Z^2 in pixels.
        # This scalar weighting is exact for square pixels; different fx/fy
        # still use the actual perspective residual for all acceptance gates.
        weights = (float(depth.min()) / depth) ** 2
        report.update(solver="fixed_G0_camera_depth_SE2_weighted_Procrustes_RANSAC",
                      ransac_sample_size=3, ransac_iteration_limit=plane_ransac_iterations,
                      fixed_depth_is_unconfirmed_prior=True)

        def fit_subset(indices):
            weight = weights[indices]
            weight = weight / weight.sum()
            source = source_xy[indices]
            target = target_xy[indices]
            source_mean = (weight[:, None] * source).sum(axis=0)
            target_mean = (weight[:, None] * target).sum(axis=0)
            centered_source = source - source_mean
            centered_target = target - target_mean
            if float((weight[:, None] * centered_source ** 2).sum()) <= 1e-16:
                return None
            cross = centered_source.T @ (weight[:, None] * centered_target)
            try:
                left, singular, right_t = np.linalg.svd(cross)
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(singular).all() or float(singular.sum()) <= 1e-16:
                return None
            correction = np.diag([1., 1. if np.linalg.det(right_t.T @ left.T) >= 0. else -1.])
            rotation_xy = right_t.T @ correction @ left.T
            translation_xy = target_mean - rotation_xy @ source_mean
            rotation_camera = np.eye(3)
            rotation_camera[:2, :2] = rotation_xy
            translation_camera = np.array([translation_xy[0], translation_xy[1], 0.])
            # Model convention is absolute G0-world -> current-camera, like
            # PnP. The shared final conversion produces world-relative SE3.
            rotation = rotation_camera @ w2c[:3, :3]
            translation = rotation_camera @ w2c[:3, 3] + translation_camera
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                return None
            return rotation, translation, ids[indices]

        random = np.random.default_rng(int((seed + time * 104729 + len(ids) * 17 +
                                            int(ids.sum()) * 31) % (2**31 - 1)))
        best_model, best_indices, best_key = None, None, (-1, -math.inf)
        attempted, required = 0, plane_ransac_iterations
        while attempted < min(required, plane_ransac_iterations):
            attempted += 1
            model = fit_subset(random.choice(len(ids), size=3, replace=False))
            if model is None:
                continue
            residual = errors(model, time, ids)
            inlier_indices = np.flatnonzero(residual <= reprojection_threshold_px)
            if len(inlier_indices) < min_group_nodes or not sufficiently_spread(ids[inlier_indices]):
                continue
            key = len(inlier_indices), -float(residual[inlier_indices].mean())
            if key <= best_key:
                continue
            best_model, best_indices, best_key = model, inlier_indices, key
            inlier_ratio = len(inlier_indices) / len(ids)
            if inlier_ratio == 1.:
                required = attempted
            else:
                # Bounded .999-confidence sampling; no evidence threshold is
                # changed if the iteration budget cannot find a good model.
                required = min(required, max(1, math.ceil(math.log(.001) /
                                                          math.log1p(-inlier_ratio ** 3))))
        report.update(ransac_iterations=attempted,
                      ransac_inliers=0 if best_indices is None else len(best_indices))
        def reject(reason):
            report.update(valid=False, reason=reason)
            return None
        if best_model is None:
            return reject("camera_plane_ransac_failed")
        indices = best_indices
        model = best_model
        for refinement in range(3):
            model = fit_subset(indices)
            if model is None:
                return reject("camera_plane_refinement_failed")
            residual = errors(model, time, ids)
            next_indices = np.flatnonzero(residual <= reprojection_threshold_px)
            report.update(refinement_iterations=refinement + 1, refined_inliers=len(next_indices))
            if len(next_indices) < min_group_nodes:
                return reject("insufficient_refined_support")
            if not sufficiently_spread(ids[next_indices]):
                return reject("degenerate_refined_support")
            unchanged = np.array_equal(indices, next_indices)
            indices = next_indices
            if unchanged:
                break
        report.update(valid=True, reason="accepted",
                      mean_inlier_reprojection_px=float(residual[indices].mean()))
        return model[0], model[1], ids[indices]

    def estimate(ids, time, report):
        ids = np.asarray(ids, dtype=np.int64)
        report.update(time_index=int(time), candidate_nodes=len(ids))
        ids = ids[valid_tracks[time, ids]]
        report["valid_nodes"] = len(ids)
        def reject(reason):
            report.update(valid=False, reason=reason)
            return None
        if len(ids) < min_group_nodes:
            return reject("insufficient_valid_tracks")
        if not sufficiently_spread(ids):
            return reject("degenerate_3d_support")
        report["source_singular_values"] = np.linalg.svd(
            centers[ids] - centers[ids].mean(axis=0), compute_uv=False).tolist()
        image_spread = np.linalg.svd(image_points[time, ids] - image_points[time, ids].mean(axis=0), compute_uv=False)
        report["image_singular_values"] = image_spread.tolist()
        if image_spread[1] <= max(image_spread[0] * 1e-4, .25):
            return reject("degenerate_2d_support")
        if motion_model == "camera_plane":
            return estimate_camera_plane(ids, time, report)
        cv2.setRNGSeed(int((seed + time * 104729 + len(ids) * 17 + int(ids.sum()) * 31) % (2**31 - 1)))
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.ascontiguousarray(centers[ids]), np.ascontiguousarray(image_points[time, ids]),
                intrinsic, None, iterationsCount=400, reprojectionError=float(reprojection_threshold_px),
                confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            return reject("pnp_ransac_exception")
        if not success or inliers is None:
            return reject("pnp_ransac_failed")
        support = ids[inliers.reshape(-1)]
        report["ransac_inliers"] = len(support)
        if len(support) < min_group_nodes:
            return reject("insufficient_ransac_support")
        if not sufficiently_spread(support):
            return reject("degenerate_ransac_support")
        try:
            rvec, tvec = cv2.solvePnPRefineLM(np.ascontiguousarray(centers[support]),
                                           np.ascontiguousarray(image_points[time, support]),
                                           intrinsic, None, rvec, tvec)
        except cv2.error:
            return reject("pnp_refinement_exception")
        rotation = cv2.Rodrigues(rvec)[0]
        if not np.isfinite(rotation).all() or not np.isfinite(tvec).all():
            return reject("nonfinite_refined_pose")
        model = rotation, tvec.reshape(3), support
        residual = errors(model, time, ids)
        support = ids[residual <= reprojection_threshold_px]
        report["refined_inliers"] = len(support)
        if len(support) < min_group_nodes:
            return reject("insufficient_refined_support")
        if not sufficiently_spread(support):
            return reject("degenerate_refined_support")
        report.update(valid=True, reason="accepted", mean_inlier_reprojection_px=float(
            residual[residual <= reprojection_threshold_px].mean()))
        return rotation, tvec.reshape(3), support

    def projected_positions(model, ids):
        camera_positions = centers[ids] @ model[0].T + model[1]
        front = camera_positions[:, 2] > 1e-8
        projected = np.column_stack((fx * camera_positions[:, 0] / np.maximum(camera_positions[:, 2], 1e-8) + cx,
                                      fy * camera_positions[:, 1] / np.maximum(camera_positions[:, 2], 1e-8) + cy))
        return projected, front

    def errors(model, time, ids):
        projected, front = projected_positions(model, ids)
        error = np.linalg.norm(projected - tracks[time, ids], axis=1)
        return np.where(front & valid_tracks[time, ids] & np.isfinite(error), error, np.inf)

    def motion_evidence(model, time, ids, return_support=False):
        ids = np.asarray(ids, dtype=np.int64)
        projected, front = projected_positions(model, ids)
        residual = errors(model, time, ids)
        observed_motion = displacement[time, ids]
        fitted_motion = np.linalg.norm(projected - reference_pixels[ids], axis=1)
        improvement = observed_motion - residual
        inliers = front & valid_tracks[time, ids] & (residual <= reprojection_threshold_px)
        substantial = (inliers & (fitted_motion >= minimum_fitted_motion) &
                       (observed_motion >= minimum_fitted_motion) &
                       (improvement >= minimum_identity_improvement) &
                       (improvement >= .5 * observed_motion))
        def median(values):
            return float(np.median(values[inliers])) if inliers.any() else None
        report = {"time_index": int(time), "dynamic": int(substantial.sum()) >= min_group_nodes,
                  "inliers": int(inliers.sum()), "substantial_motion_inliers": int(substantial.sum()),
                  "median_fitted_motion_px": median(fitted_motion),
                  "median_observed_motion_px": median(observed_motion),
                  "median_identity_residual_improvement_px": median(improvement),
                  "identity_inliers": int((valid_tracks[time, ids] &
                                           (observed_motion <= reprojection_threshold_px)).sum())}
        return (report, substantial) if return_support else report

    def consensus(seed_ids, remaining, selected_times, report):
        members = seed_ids
        agreement = None
        report["seed_overlap"] = overlap_report(seed_ids, selected_times)
        report["consensus_rounds"] = []
        for iteration in range(2):
            observed = np.zeros(len(remaining), dtype=np.int32)
            agreeing = np.zeros(len(remaining), dtype=np.int32)
            reliable_times = []
            models = {}
            fit_reports = []
            for time in selected_times:
                fit = {}
                model = estimate(members, time, fit)
                fit_reports.append(fit)
                if model is None:
                    continue
                reliable_times.append(time)
                models[time] = model
                observed += valid_tracks[time, remaining]
                agreeing += errors(model, time, remaining) <= reprojection_threshold_px
            step = {"iteration": iteration, "seed_nodes": len(members), "fits": fit_reports,
                    "reliable_times": reliable_times,
                    "nodes_observed_at_minimum_times": int((observed >= min_consensus_timestamps).sum())}
            report["consensus_rounds"].append(step)
            if len(reliable_times) < min_consensus_timestamps:
                report.update(failure_stage="cross_time_consensus", reason="insufficient_reliable_time_models")
                return None
            agreement = ((observed >= min_consensus_timestamps) &
                         (agreeing >= np.ceil(.8 * observed)) & motion_seed_mask[remaining])
            members = remaining[agreement]
            step["consensus_nodes"] = len(members)
            if not sufficiently_spread(members):
                report.update(failure_stage="cross_time_consensus", reason=(
                    "insufficient_cross_time_inliers" if len(members) < min_group_nodes else "degenerate_consensus_support"))
                return None
        dynamic_reports = [motion_evidence(models[time], time, members) for time in reliable_times]
        motion_times = [entry["time_index"] for entry in dynamic_reports if entry["dynamic"]]
        report["fitted_motion_evidence"] = dynamic_reports
        report["supported_motion_times"] = motion_times
        if len(motion_times) < min_consensus_timestamps:
            report.update(failure_stage="motion_support", reason="insufficient_dynamic_support")
            return None
        report.update(reason="provisional_consensus_pending_all_time_audit", failure_stage=None,
                      consensus_nodes=len(members))
        return members, reliable_times, int(agreeing[agreement].sum())

    def choose_temporal_branch(model, time, members, previous, previous_time, fit):
        """Compare two observation-supported PnP branches, never fill a pose."""
        if motion_model == "camera_plane":
            fit["temporal_branch_selection"] = {
                "selected": "camera_plane_unique_fit", "reason": "SE2_no_PnP_branch_search",
                "previous_reliable_time": previous_time, "candidates": [],
                "regularization_only_not_observation": False}
            return model
        branch_report = {"regularization_only_not_observation": True,
                         "previous_reliable_time": previous_time, "selected": "ransac",
                         "candidates": []}
        fit["temporal_branch_selection"] = branch_report
        if previous is None:
            branch_report["reason"] = "no_earlier_reliable_pose"
            return model
        audited = model[2]
        ids = members[valid_tracks[time, members]]
        center = centers[members].mean(axis=0)
        scale = max(float(np.sqrt(np.mean(np.sum((centers[members] - center) ** 2, axis=1)))), 1e-8)
        def validate(candidate, name):
            residual = errors(candidate, time, ids)
            support = ids[residual <= reprojection_threshold_px]
            audited_inliers = int((errors(candidate, time, audited) <= reprojection_threshold_px).sum())
            entry = {"source": name, "inliers": len(support), "audited_nodes": len(audited),
                     "audited_inliers": audited_inliers,
                     "audited_fraction": audited_inliers / len(audited)}
            branch_report["candidates"].append(entry)
            if (len(support) < min_group_nodes or audited_inliers < min_group_nodes or
                    audited_inliers < math.ceil(.8 * len(audited)) or not sufficiently_spread(support)):
                entry.update(accepted=False, reason="insufficient_same_audited_inlier_support")
                return None
            angle = float(np.arccos(np.clip((np.trace(candidate[0] @ previous[0].T) - 1.) / 2., -1., 1.)))
            center_delta = float(np.linalg.norm((candidate[0] @ center + candidate[1]) -
                                                (previous[0] @ center + previous[1]))) / scale
            score = angle + center_delta
            entry.update(accepted=True, reason="same_observation_gates_passed",
                         continuity_score=score, rotation_step_radians=angle,
                         normalized_center_step=center_delta,
                         mean_inlier_reprojection_px=float(residual[residual <= reprojection_threshold_px].mean()))
            return score, (candidate[0], candidate[1], support), entry
        candidates = []
        checked = validate(model, "ransac")
        if checked is not None:
            candidates.append(checked)
        try:
            success, rvec, tvec = cv2.solvePnP(
                np.ascontiguousarray(centers[audited]), np.ascontiguousarray(image_points[time, audited]),
                intrinsic, None, rvec=cv2.Rodrigues(previous[0])[0].copy(),
                tvec=np.ascontiguousarray(previous[1].reshape(3, 1)).copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if success and np.isfinite(rvec).all() and np.isfinite(tvec).all():
                candidate = cv2.Rodrigues(rvec)[0], tvec.reshape(3), audited
                checked = validate(candidate, "previous_pose_initialized_iterative")
                if checked is not None:
                    candidates.append(checked)
            else:
                branch_report["candidates"].append({"source": "previous_pose_initialized_iterative",
                                                    "accepted": False, "reason": "iterative_solver_failed_or_nonfinite"})
        except cv2.error:
            branch_report["candidates"].append({"source": "previous_pose_initialized_iterative",
                                                "accepted": False, "reason": "iterative_solver_exception"})
        if not candidates:
            # The already-audited original is retained for conservative
            # pruning; this path cannot create new evidence or valid poses.
            branch_report["reason"] = "no_alternative_supported_branch"
            return model
        _, selected_model, entry = min(candidates, key=lambda item: (item[0], item[2]["mean_inlier_reprojection_px"]))
        branch_report.update(selected=entry["source"], reason="minimum_supported_pose_step")
        fit.update(refined_inliers=len(selected_model[2]),
                   mean_inlier_reprojection_px=entry["mean_inlier_reprojection_px"])
        return selected_model

    def timeline_pass(members, iteration):
        observed = np.zeros(len(members), dtype=np.int32)
        inlier_counts = np.zeros(len(members), dtype=np.int32)
        informative_observed = np.zeros(len(members), dtype=np.int32)
        informative_inliers = np.zeros(len(members), dtype=np.int32)
        informative_motion_support = np.zeros(len(members), dtype=np.int32)
        models, fits = {}, []
        eligible_times, reliable_times = [], []
        informative_times = []
        previous, previous_time = None, None
        for time in range(len(tracks)):
            fit = {}
            if int(valid_tracks[time, members].sum()) >= min_group_nodes:
                eligible_times.append(time)
            model = estimate(members, time, fit)
            fits.append(fit)
            if model is None:
                continue
            model = choose_temporal_branch(model, time, members, previous, previous_time, fit)
            models[time] = model
            reliable_times.append(time)
            observed += valid_tracks[time, members]
            inliers = errors(model, time, members) <= reprojection_threshold_px
            inlier_counts += inliers
            current_observed = int(valid_tracks[time, members].sum())
            if int(inliers.sum()) >= min_group_nodes and inliers.sum() >= math.ceil(.8 * current_observed):
                previous, previous_time = model, time
            evidence, supported_motion = motion_evidence(model, time, members, return_support=True)
            fit["motion_evidence"] = evidence
            if evidence["dynamic"]:
                informative_times.append(time)
                informative_observed += valid_tracks[time, members]
                informative_inliers += inliers
                informative_motion_support += supported_motion
        all_time_keep = ((observed >= min_consensus_timestamps) &
                         (inlier_counts >= min_consensus_timestamps) &
                         (inlier_counts >= np.ceil(.8 * observed)))
        informative_keep = ((informative_observed >= min_consensus_timestamps) &
                            (informative_inliers >= np.ceil(.8 * informative_observed)) &
                            (informative_motion_support >= min_consensus_timestamps))
        keep = all_time_keep & informative_keep
        report = {"iteration": iteration, "before_nodes": len(members), "after_nodes": int(keep.sum()),
                  "node_ids": members.tolist(),
                  "actual_observed_counts": valid_tracks[:, members].sum(axis=0).tolist(),
                  "reliable_model_observed_counts": observed.tolist(),
                  "inlier_counts": inlier_counts.tolist(),
                  "motion_informative_times": informative_times,
                  "motion_informative_observed_counts": informative_observed.tolist(),
                  "motion_informative_inlier_counts": informative_inliers.tolist(),
                  "motion_informative_supported_counts": informative_motion_support.tolist(),
                  "all_time_consistent_nodes": int(all_time_keep.sum()),
                  "motion_informative_consistent_nodes": int(informative_keep.sum()),
                  "eligible_times": eligible_times, "reliable_model_times": reliable_times,
                  "eligible_time_count": len(eligible_times), "reliable_model_time_count": len(reliable_times),
                  "reliable_model_all_timeline_coverage": len(reliable_times) / len(tracks),
                  "reliable_model_eligible_coverage": len(reliable_times) / max(1, len(eligible_times)),
                  "fits": fits}
        return keep, models, report

    def refine_candidate(candidate, report):
        members, _, _ = candidate
        timeline = {"before_nodes": len(members), "rounds": []}
        report["timeline_refinement"] = timeline
        def reject(reason):
            report.update(failure_stage="all_time_refinement", reason=reason)
            timeline["after_nodes"] = len(members)
            return None
        for iteration in range(2):
            keep, _, audit = timeline_pass(members, iteration)
            timeline["rounds"].append(audit)
            if audit["reliable_model_time_count"] < min_consensus_timestamps:
                return reject("insufficient_all_time_reliable_models")
            members = members[keep]
            if not sufficiently_spread(members):
                return reject("insufficient_motion_informative_consistent_nodes"
                              if audit["motion_informative_consistent_nodes"] < min_group_nodes
                              else "insufficient_all_time_consistent_nodes")
        # Audit the final membership, rather than reusing models fitted before
        # pruning. An unstable last pruning step is not silently accepted.
        keep, models, audit = timeline_pass(members, "final_audit")
        timeline["final_audit"] = audit
        if not keep.all():
            return reject("unstable_final_node_consensus")
        valid_times, dynamic_times = [], []
        final_models = {}
        for time, fit in enumerate(audit["fits"]):
            model = models.get(time)
            if model is None:
                continue
            ids = members[valid_tracks[time, members]]
            residual = errors(model, time, ids)
            inliers = residual <= reprojection_threshold_px
            fit["inliers"] = int(inliers.sum())
            if inliers.sum() < min_group_nodes or inliers.mean() < .8:
                fit.update(valid=False, reason="group_reprojection_consensus_failed")
                continue
            evidence = motion_evidence(model, time, members)
            fit["motion_evidence"] = evidence
            valid_times.append(time)
            final_models[time] = model
            if evidence["dynamic"]:
                dynamic_times.append(time)
        eligible_coverage = len(valid_times) / max(1, audit["eligible_time_count"])
        audit.update(pose_valid_times=valid_times, dynamic_times=dynamic_times,
                     pose_valid_time_count=len(valid_times), dynamic_time_count=len(dynamic_times),
                     pose_valid_all_timeline_coverage=len(valid_times) / len(tracks),
                     pose_valid_eligible_coverage=eligible_coverage)
        if len(valid_times) < min_consensus_timestamps:
            return reject("insufficient_final_supported_times")
        if len(dynamic_times) < min_consensus_timestamps:
            return reject("insufficient_final_dynamic_times")
        # This is an initialization usability rule, not an event-window score.
        # A solver failure must not make most observed frames disappear from
        # the denominator and turn an episodic fit into a full-motion group.
        if eligible_coverage < .8:
            return reject("insufficient_final_eligible_time_coverage")
        timeline["after_nodes"] = len(members)
        report.update(reason="accepted", failure_stage=None, consensus_nodes=len(members))
        return {"members": members, "consensus_times": valid_times,
                "consensus_observations": int(sum(audit["inlier_counts"])),
                "models": final_models, "fits": audit["fits"], "dynamic_times": dynamic_times,
                "timeline_refinement": timeline}

    def local_scopes(remaining):
        eligible = remaining[motion_seed_mask[remaining]]
        if len(eligible) < min_group_nodes or not local_seed_count:
            return
        # Farthest-point coverage in G0 3D, not a ranking of the largest flow
        # magnitudes. Nearby points are only a hypothesis; PnP must verify it.
        offset = centers[eligible] - centers[eligible].mean(axis=0)
        current = int(np.argmax(np.einsum("ij,ij->i", offset, offset)))
        nearest = np.full(len(eligible), np.inf)
        anchors = []
        yielded = set()
        pool = eligible
        for _ in range(min(local_seed_count, len(eligible))):
            anchor = int(eligible[current])
            anchors.append(current)
            distance = np.sum((centers[pool] - centers[anchor]) ** 2, axis=1)
            size = min(local_seed_size, len(pool))
            ids = np.sort(pool[np.argsort(distance, kind="stable")[:size]])
            key = tuple(ids.tolist())
            if key not in yielded:
                yielded.add(key)
                yield ids, anchor
            nearest = np.minimum(nearest, np.sum((centers[eligible] - centers[anchor]) ** 2, axis=1))
            nearest[anchors] = -1.
            current = int(np.argmax(nearest))

    selected = []
    for round_index in range(max_groups):
        remaining = np.flatnonzero((labels < 0) & source_front)
        if len(remaining) < min_group_nodes:
            break
        motion_seeds = remaining[motion_seed_mask[remaining]]
        if len(motion_seeds) < min_group_nodes:
            # Preserve the observation/overlap diagnosis even when no motion-
            # informative seed set exists; no quiet point becomes dynamic.
            _, selection_index = choose_times(remaining, "global", None, round_index)
            selection_reports[selection_index]["search_rejected_reason"] = "insufficient_motion_seed_nodes"
            selection_reports[selection_index]["motion_seed_nodes"] = len(motion_seeds)
            break
        accepted_candidate = None
        # Bounded fallback: do not spend local RANSAC calls when global seeds
        # already establish a supported part; remove it and re-evaluate first.
        scopes = [(motion_seeds, None, "global")]
        for tier in range(2):
            if tier:
                if accepted_candidate is not None:
                    break
                scopes = [(ids, anchor, "local_geometry") for ids, anchor in local_scopes(remaining)]
            proposals = []
            for ids, anchor, scope in scopes:
                chosen, selection_index = choose_times(ids, scope, anchor, round_index)
                if round_index == 0 and scope == "global":
                    selected = chosen
                for time in chosen:
                    report = {"round": round_index, "scope": scope, "anchor_node": anchor,
                              "selection_index": selection_index, "selected_times": chosen,
                              "time_index": time, "candidate_nodes": len(ids), "initial_fit": {}}
                    proposal_reports.append(report)
                    current_motion_seeds = ids[resolvable_motion_tracks[time, ids]]
                    report["current_motion_seed_nodes"] = len(current_motion_seeds)
                    initial = estimate(current_motion_seeds, time, report["initial_fit"])
                    if initial is None:
                        rejected_proposals += 1
                        report.update(failure_stage=("initial_pnp" if motion_model == "se3" else
                                                    "initial_camera_plane"),
                                      reason=report["initial_fit"]["reason"])
                        continue
                    candidate = consensus(initial[2], remaining, chosen, report)
                    if candidate is None:
                        rejected_proposals += 1
                        continue
                    proposals.append((candidate, report))
            # Expensive full-timeline checks are bounded and deduplicated.
            # A provisional global match must pass them before local searches
            # can be skipped or any node is removed from the unknown pool.
            seen_memberships = set()
            refined_count = 0
            for candidate, report in sorted(proposals, key=lambda item: (-len(item[0][0]), -item[0][2])):
                membership = tuple(candidate[0].tolist())
                if membership in seen_memberships:
                    report["reason"] = "duplicate_membership_not_refined"
                    continue
                seen_memberships.add(membership)
                if refined_count >= max_refinement_candidates:
                    report["reason"] = "refinement_candidate_budget_exhausted"
                    continue
                refined_count += 1
                refined = refine_candidate(candidate, report)
                if refined is None:
                    rejected_proposals += 1
                    continue
                accepted_candidate = refined
                break
            for _, report in proposals:
                if report["reason"] == "provisional_consensus_pending_all_time_audit":
                    report["reason"] = "not_refined_after_supported_candidate"
        if accepted_candidate is None:
            break
        label = len(groups) + 1
        labels[accepted_candidate["members"]] = label
        groups.append({"label": label, **accepted_candidate})
    poses = np.full((len(tracks), len(groups) + 1, 4, 4), np.nan, dtype=np.float64)
    pose_valid = np.zeros((len(tracks), len(groups) + 1), dtype=bool)
    poses[:, 0] = np.eye(4)
    pose_valid[:, 0] = True
    group_reports = []
    for group in groups:
        label, members = group["label"], group["members"]
        time_reports = []
        for time in range(len(tracks)):
            fit = group["fits"][time]
            model = group["models"].get(time)
            if model is None:
                time_reports.append(fit)
                continue
            pose = np.eye(4)
            pose[:3, :3] = w2c[:3, :3].T @ model[0]
            pose[:3, 3] = w2c[:3, :3].T @ (model[1] - w2c[:3, 3])
            poses[time, label] = pose
            pose_valid[time, label] = True
            time_reports.append(fit)
        group_reports.append({"label": label, "nodes": len(members),
                              "consensus_times": group["consensus_times"], "times": time_reports,
                              "dynamic_times": group["dynamic_times"],
                              "timeline_refinement": group["timeline_refinement"]})
    # Final group refits must also retain multi-time support. A provisional
    # consensus is not permission to publish a one-frame or invalid model.
    retained = [label for label in range(1, len(groups) + 1)
                if pose_valid[:, label].sum() >= min_consensus_timestamps and
                len(groups[label - 1]["dynamic_times"]) >= min_consensus_timestamps]
    remapped = np.full_like(labels, -1)
    remapped[static] = 0
    final_reports = []
    for new_label, old_label in enumerate(retained, 1):
        remapped[labels == old_label] = new_label
        report = dict(group_reports[old_label - 1])
        report["label"] = new_label
        final_reports.append(report)
    discarded_final_groups = len(groups) - len(retained)
    labels = remapped
    poses = poses[:, [0] + retained]
    pose_valid = pose_valid[:, [0] + retained]
    failure_stage_counts = {}
    rejection_reason_counts = {}
    for report in proposal_reports:
        stage = report.get("failure_stage")
        if stage is not None:
            failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1
            reason = report["reason"]
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    result = {"labels": labels, "poses": poses, "pose_valid": pose_valid,
            "diagnostics": {"selected_high_motion_times": selected, "motion_scores_px": scores,
                            "time_selections": selection_reports, "proposals": proposal_reports,
                            "failure_stage_counts": failure_stage_counts,
                            "rejection_reason_counts": rejection_reason_counts,
                            "valid_nonstatic_nodes_by_time": (valid_tracks & ~static[None]).sum(axis=1).tolist(),
                            "motion_seed_nodes": int((motion_seed_mask & ~static & source_front).sum()),
                            "nonstatic_nodes_without_repeated_resolvable_motion": int((~motion_seed_mask & ~static & source_front).sum()),
                            "dynamic_groups": len(retained), "explicit_static_nodes": int(static.sum()),
                            "unknown_nodes": int((labels < 0).sum()), "rejected_proposals": rejected_proposals,
                            "discarded_after_final_refit": discarded_final_groups,
                            "groups": final_reports},
            "metadata": {"kind": "multi_timestamp_event_track_PnP_rigid_hypotheses", "version": 5,
                         "motion_model": motion_model, "physical_motion_validated": False,
                         "labels_are_ground_truth_parts": False, "physical_uniqueness_guaranteed": False,
                         "uses_G0_and_event_tracks_only": True, "uses_semantic_or_blender_parts": False,
                         "static_label": 0, "unknown_label": -1, "pose_space": "G0_world_to_deformed_world",
                         "missing_pose_policy": "NaN_and_pose_valid_false_no_invented_motion",
                         "single_frame_hypotheses_accepted": False, "source_authenticity_verified": False,
                         "track_identity_certified": False, "geometric_seeds_are_parts": False,
                         "time_selection": "shared_valid_support_and_index_separation_with_motion_proposals",
                         "local_seed_policy": "G0_farthest_point_neighborhoods_only_after_global_rejection",
                         "local_seed_count": local_seed_count, "local_seed_size": local_seed_size,
                         "max_refinement_candidates_per_tier": max_refinement_candidates,
                         "all_time_prune_refit_rounds": 2,
                         "minimum_fitted_motion_px": minimum_fitted_motion,
                         "minimum_identity_residual_improvement_px": minimum_identity_improvement,
                         "minimum_identity_residual_improvement_fraction": .5,
                         "motion_informative_point_inlier_fraction": .8,
                         "minimum_motion_informative_point_support_times": min_consensus_timestamps,
                         "quiet_observations_cannot_dilute_informative_time_failures": True,
                         "temporal_branch_policy": "same_audited_inliers_then_minimum_SE3_step_prior",
                         "temporal_branch_continuity_is_regularization_not_observation": True,
                         "temporal_branch_hard_speed_limit": None,
                         "minimum_pose_valid_eligible_time_coverage": .8,
                         "coverage_rule_scope": "candidate_initialization_usability_not_event_window_acceptance",
                         "subthreshold_motion_policy": "unresolved_not_automatically_static",
                         "minimum_group_nodes": min_group_nodes,
                         "minimum_consensus_timestamps": min_consensus_timestamps,
                         "minimum_consensus_fraction": .8,
                         "reprojection_threshold_px": float(reprojection_threshold_px)}}
    if motion_model == "camera_plane":
        axis_world = w2c[:3, :3].T @ np.array([0., 0., 1.])
        angles = np.full(pose_valid.shape, np.nan, dtype=np.float64)
        angles[:, 0] = 0.
        for label in range(1, poses.shape[1]):
            valid_times = np.flatnonzero(pose_valid[:, label])
            relative_camera_rotations = (w2c[:3, :3][None] @
                                         poses[valid_times, label, :3, :3] @ w2c[:3, :3].T)
            principal = np.arctan2(relative_camera_rotations[:, 1, 0],
                                   relative_camera_rotations[:, 0, 0])
            angles[valid_times, label] = np.unwrap(principal)
        result["camera_plane_angles_unwrapped"] = angles
        result["camera_plane_rotation_vectors"] = angles[:, :, None] * axis_world[None, None]
        result["metadata"].update(
            kind="multi_timestamp_event_track_fixed_depth_camera_plane_hypotheses",
            camera_plane_prior={
                "confirmed": False, "extra_observations_used": False,
                "rotation_axis_camera": [0., 0., 1.], "rotation_axis_world": axis_world.tolist(),
                "translation_axes": "camera_XY", "scale": "fixed_one",
                "depth_policy": "each_point_original_G0_camera_Z_preserved",
                "world_axis_is_camera_derived_not_inferred_physical_axis": True,
                "nonplanar_G0_does_not_validate_motion_prior": True},
            plane_ransac_iterations=plane_ransac_iterations,
            plane_ransac_sample_size=3,
            temporal_branch_policy="not_applicable_SE2_unique_rotation_no_PnP_prior",
            temporal_branch_continuity_is_regularization_not_observation=False,
            rotation_vector_branch_policy="nearest_2pi_unwrap_across_valid_times_only",
            rotation_vector_unwrap_is_representation_prior_not_observation=True)
    return result
