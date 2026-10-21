"""G0-surface feature hypotheses tracked through event-derived image proxies.

Only supplied arrays are used. G0-rendered depth is a geometric hypothesis,
not a new measurement. Accepted tracks are 2D proxy correspondences with
event support, never confirmation of physical 3D motion or hidden surfaces.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _number(value, name, lower, upper=None, inclusive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if (value < lower if inclusive else value <= lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _sift_detections(detector, image, feature_mask, limit, min_distance, margin):
    """Bounded, spatially unique SIFT centres; descriptors never create depth."""
    keypoints, descriptors = detector.detectAndCompute(
        image, None if feature_mask is None else feature_mask.astype(np.uint8) * 255)
    if descriptors is None or not keypoints:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    height, width = image.shape
    selected, coordinates = [], []
    separation = max(1., float(min_distance))
    for index in sorted(range(len(keypoints)), key=lambda row: (-keypoints[row].response, row)):
        point = keypoints[index]
        xy = np.asarray(point.pt, dtype=np.float32)
        # Keep the local keypoint footprint inside the image. This does not
        # claim that an entire descriptor is confined to one physical part.
        border = max(margin, 1.5 * point.size)
        if (not np.isfinite(xy).all() or not math.isfinite(point.size) or point.size <= 0
                or np.any(xy < border) or xy[0] > width - border - 1 or xy[1] > height - border - 1):
            continue
        if feature_mask is not None:
            x0, y0 = np.floor(xy).astype(int)
            x1, y1 = np.ceil(xy).astype(int)
            # All bilinear depth taps must have the same alpha/depth support
            # as the reference corner path, not just a rounded mask centre.
            if not feature_mask[y0:y1 + 1, x0:x1 + 1].all():
                continue
        if coordinates and np.any(np.sum((np.asarray(coordinates) - xy) ** 2, axis=1) < separation ** 2):
            continue
        if not np.isfinite(descriptors[index]).all():
            continue
        selected.append(index)
        coordinates.append(xy)
        if len(selected) == limit:
            break
    return (np.asarray(coordinates, dtype=np.float32).reshape(-1, 2),
            np.asarray(descriptors[selected], dtype=np.float32).reshape(-1, 128))


def _mutual_sift_matches(reference_descriptors, current_descriptors, ratio=.7):
    """Both directions require a distinct 2nd neighbour; equal ties abstain."""
    if min(len(reference_descriptors), len(current_descriptors)) < 2:
        return np.empty((0, 2), dtype=np.int64)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def qualified(first, second):
        matches = matcher.knnMatch(first, second, k=2)
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matches
                if len(pair) == 2 and np.isfinite([pair[0].distance, pair[1].distance]).all()
                and pair[1].distance > 0 and pair[0].distance < ratio * pair[1].distance}

    forward = qualified(reference_descriptors, current_descriptors)
    reverse = qualified(current_descriptors, reference_descriptors)
    return np.asarray([(source, target) for source, target in sorted(forward.items())
                       if reverse.get(target) == source], dtype=np.int64).reshape(-1, 2)


def infer_event_feature_tracks(reference_gray_np, reference_alpha_np, reference_depth_np,
                               cumulative_log_changes_at_knots_np, camera_whitelist, knot_times_np, *,
                               max_corners=2048, quality_level=.005, min_distance=3., minimum_alpha=.95,
                               log_floor=1e-4, fb_max_error=.75, patch_max_error=.08,
                               allow_reference_reacquisition=False, maximum_relative_depth_range=.05,
                               background_intensity=None, foreground_survival_ratio=.25,
                               feature_method='corners'):
    """Extract G0 surface points and track KLT pairs without reading any files.

    reference_gray is the linear G0 render *already composited* over the
    estimated background. Alpha and camera-z depth are HxW (or 1xHxW); depth
    must already be normalized by alpha, NOT the rasterizer's weighted sum.
    Cumulative log changes and increasing knot times include G0 at t=0.

    Returns points[N,3] world coordinates, tracks[T,N,2] pixel-boundary
    coordinates (pixel zero centre is .5), accepted[T,N], tracking_continuous,
    reassociated, event_supported and tracking_source. Code 0 is a frozen
    unverified prior, 1 the G0 anchor, 2 a continuous image-pair hypothesis,
    3 an explicit G0-to-current KLT re-association, and 4 a direct original-G0
    SIFT descriptor match. Missing frames remain
    unaccepted; a later direct reference match never fills a preceding gap.

    Quiet image pairs may continue as image hypotheses, but event_supported
    and accepted remain false without local current-window event activity.
    A failed KLT/FB/patch pair breaks the image chain permanently unless
    explicit reference re-association is enabled and independently passes
    G0-to-current and current-to-G0 KLT plus photometric patch checks.
    Frozen old coordinates are never sampled as a valid adjacent-pair track.

    If background_intensity is supplied, a heuristic long-term foreground
    gate compares a radius-5 circular patch at each current candidate against
    that point's FIXED original G0 patch. Meaningful positive-background
    brightness and contrast must each retain foreground_survival_ratio of
    their G0 value. Dark material is not required to have positive foreground
    brightness: its meaningful contrast can be used alone. These statistics
    are less orientation-sensitive than a direct pixelwise G0 patch match.
    None explicitly disables this extra protection for backward compatibility.

    feature_method='sift' is a separate original-G0 matching backend. SIFT
    seeds and descriptors are extracted once from G0 and never refreshed.
    Each proxy independently requires mutual, unique, bidirectional ratio-.7
    matching; duplicate orientations at one location are spatially deduplicated.
    It does not apply rotation-sensitive pixelwise KLT patch tests. Fixed-G0
    foreground survival still applies. A quiet, uninterrupted match is only
    an image prior, and recovery after a gap requires local current events.
    After any gap tracking_continuous stays false forever for that point,
    even if direct G0 matching later restores accepted event support.

    The caller is responsible for input provenance and split discipline:
    whole-sequence cumulative events cannot initialize independent holdout
    training. These image proxies are not observed video or new RGB inputs.
    """
    if (isinstance(max_corners, (bool, np.bool_)) or not isinstance(max_corners, (int, np.integer))
            or not 1 <= max_corners <= 2048):
        raise ValueError("max_corners must be an integer from 1 to 2048")
    if type(allow_reference_reacquisition) is not bool:
        raise ValueError("allow_reference_reacquisition must be boolean")
    if not isinstance(feature_method, str) or feature_method not in ('corners', 'sift'):
        raise ValueError("feature_method must be corners or sift")
    quality_level = _number(quality_level, "quality_level", 0, 1)
    min_distance = _number(min_distance, "min_distance", 0, inclusive=True)
    minimum_alpha = _number(minimum_alpha, "minimum_alpha", 0, 1)
    log_floor = _number(log_floor, "log_floor", 0, 1)
    if log_floor == 1:
        raise ValueError("log_floor must be smaller than one")
    fb_max_error = _number(fb_max_error, "fb_max_error", 0)
    patch_max_error = _number(patch_max_error, "patch_max_error", 0, 1)
    maximum_relative_depth_range = _number(maximum_relative_depth_range, "maximum_relative_depth_range", 0)
    foreground_survival_ratio = _number(foreground_survival_ratio, "foreground_survival_ratio", 0, 1)
    if background_intensity is not None:
        background_intensity = _number(background_intensity, "background_intensity", 0, 1, inclusive=True)
    survival_enabled = background_intensity is not None
    reference = np.asarray(reference_gray_np)
    if reference.dtype.kind not in "iuf":
        raise ValueError("reference_gray must be real numeric data")
    reference = reference.astype(np.float32) / 255 if reference.dtype == np.uint8 else reference.astype(np.float32)
    if (reference.ndim != 2 or min(reference.shape) < 16 or not np.isfinite(reference).all()
            or np.any(reference < 0) or np.any(reference > 1)):
        raise ValueError("reference_gray must be finite HxW linear intensities in [0,1], at least 16x16")
    height, width = reference.shape

    def render_plane(value, name):
        plane = np.asarray(value)
        if plane.shape == (1, height, width):
            plane = plane[0]
        if plane.shape != reference.shape or plane.dtype.kind not in "iuf" or not np.isfinite(plane).all():
            raise ValueError(f"{name} must be a finite real HxW render array")
        with np.errstate(over="ignore"):
            converted = plane.astype(np.float32)
        if not np.isfinite(converted).all():
            raise ValueError(f"{name} exceeds float32 range")
        return converted

    alpha = render_plane(reference_alpha_np, "reference_alpha")
    depth = render_plane(reference_depth_np, "reference_depth")
    if np.any(alpha < 0) or np.any(alpha > 1) or np.any(depth < 0):
        raise ValueError("Alpha must lie in [0,1] and normalized camera-z depth must be nonnegative")
    if (camera_whitelist.get("width") != width or camera_whitelist.get("height") != height
            or camera_whitelist.get("pixel_center") != "boundary"):
        raise ValueError("Camera dimensions and explicit boundary-coordinate convention must match")
    fx, fy, cx, cy = (float(camera_whitelist[key]) for key in ("fx", "fy", "cx", "cy"))
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("Invalid camera intrinsics")
    w2c = np.asarray(camera_whitelist["w2c"], dtype=np.float64)
    if (w2c.shape != (4, 4) or not np.isfinite(w2c).all()
            or not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(w2c[:3, :3]) - 1) > 1e-5):
        raise ValueError("w2c must be a finite proper rigid transform")
    times = np.asarray(knot_times_np, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Knot times must increase strictly from zero")
    try:
        cumulative = iter(cumulative_log_changes_at_knots_np)
    except TypeError:
        raise ValueError("Cumulative log changes must be an iterable of HxW arrays") from None

    def next_change():
        try:
            raw = np.asarray(next(cumulative))
        except StopIteration:
            raise ValueError("Too few cumulative event maps") from None
        if raw.shape != reference.shape or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("Every cumulative event map must be finite real HxW data")
        value = raw.astype(np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("Cumulative log changes exceed float32 range")
        return value

    previous_change = next_change()
    if np.max(np.abs(previous_change)) > 1e-6:
        raise ValueError("The first event map must be zero at G0")
    initial_proxy = np.maximum(reference, log_floor)
    reference_u8 = np.rint(initial_proxy * 255).astype(np.uint8)
    log_reference = np.log(initial_proxy)
    survival_radius = 5
    margin, patch_radius = (survival_radius + 1 if survival_enabled else 4), 3
    kernel = np.ones((3, 3), dtype=np.uint8)
    opaque = cv2.erode(((alpha >= minimum_alpha) & (depth > 0)).astype(np.uint8), kernel).astype(bool)
    relative_range = (cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)) / np.maximum(depth, 1e-8)
    depth_supported = relative_range <= maximum_relative_depth_range
    feature_mask = opaque & depth_supported
    feature_mask[:margin] = feature_mask[-margin:] = False
    feature_mask[:, :margin] = feature_mask[:, -margin:] = False
    if feature_method == 'sift':
        if not hasattr(cv2, 'SIFT_create'):
            raise RuntimeError('SIFT was explicitly requested but is unavailable; no silent fallback')
        # Detect an overcomplete set before spatial deduplication, then keep
        # at most max_corners reference and current locations.
        sift = cv2.SIFT_create(nfeatures=int(max_corners) * 4)
        initial_uv, reference_descriptors = _sift_detections(
            sift, reference_u8, feature_mask, int(max_corners), min_distance, margin)
    else:
        corners = cv2.goodFeaturesToTrack(reference_u8, maxCorners=int(max_corners), qualityLevel=quality_level,
                                         minDistance=min_distance, mask=feature_mask.astype(np.uint8) * 255,
                                         blockSize=5, useHarrisDetector=False)
        initial_uv = np.empty((0, 2), dtype=np.float32) if corners is None else corners.reshape(-1, 2).astype(np.float32)

    def sample(plane, coordinates):
        if not len(coordinates):
            return np.empty(0, dtype=np.float32)
        return cv2.remap(plane, coordinates[:, 0, None].astype(np.float32), coordinates[:, 1, None].astype(np.float32),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1)

    z = sample(depth, initial_uv).astype(np.float64)
    camera_points = np.column_stack(((initial_uv[:, 0] + .5 - cx) * z / fx,
                                     (initial_uv[:, 1] + .5 - cy) * z / fy, z))
    points = (camera_points - w2c[:3, 3]) @ w2c[:3, :3]
    if not np.isfinite(points).all() or np.any(np.abs(points) > np.finfo(np.float32).max):
        raise ValueError("Backprojected world points exceed finite float32 range")
    count = len(points)
    tracks = np.zeros((len(times), count, 2), dtype=np.float32)
    accepted = np.zeros((len(times), count), dtype=bool)
    continuous = np.zeros_like(accepted)
    reassociated = np.zeros_like(accepted)
    event_supported = np.zeros_like(accepted)
    sources = np.zeros((len(times), count), dtype=np.uint8)
    tracked = initial_uv.copy()
    tracks[0] = initial_uv + .5
    continuous[0] = True
    sources[0] = 1
    anchor_valid = np.ones(count, dtype=bool)
    sift_chain_intact = anchor_valid.copy()
    previous_proxy, previous_u8 = initial_proxy, reference_u8
    offsets_y, offsets_x = np.mgrid[-patch_radius:patch_radius + 1, -patch_radius:patch_radius + 1]
    offsets_x, offsets_y = offsets_x.reshape(1, -1), offsets_y.reshape(1, -1)
    circular_y, circular_x = np.mgrid[-survival_radius:survival_radius + 1, -survival_radius:survival_radius + 1]
    circular_mask = circular_x ** 2 + circular_y ** 2 <= survival_radius ** 2
    circular_x, circular_y = circular_x[circular_mask][None], circular_y[circular_mask][None]

    def foreground_statistics(plane, coordinates):
        if not len(coordinates):
            return np.empty(0), np.empty(0)
        values = cv2.remap(plane, (coordinates[:, :1] + circular_x).astype(np.float32),
                           (coordinates[:, 1:] + circular_y).astype(np.float32), interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_intensity))
        brightness = np.quantile(np.maximum(values - background_intensity, 0), .8, axis=1)
        quantiles = np.quantile(values, [.1, .9], axis=1)
        return brightness, quantiles[1] - quantiles[0]

    reference_brightness = reference_contrast = np.zeros(count, dtype=np.float64)
    brightness_applicable = contrast_applicable = np.zeros(count, dtype=bool)
    if survival_enabled:
        # These arrays are never refreshed from later frames: adapting the
        # reference would let gradual background ghosts survive indefinitely.
        reference_brightness, reference_contrast = foreground_statistics(initial_proxy, initial_uv)
        brightness_applicable = reference_brightness >= .02
        contrast_applicable = reference_contrast >= .01
    survival_applicable = brightness_applicable | contrast_applicable

    def foreground_survival(plane, coordinates, original_ids):
        if not survival_enabled or not len(original_ids):
            return np.ones(len(original_ids), dtype=bool)
        brightness, contrast = foreground_statistics(plane, coordinates)
        meaningful_brightness = brightness_applicable[original_ids]
        meaningful_contrast = contrast_applicable[original_ids]
        survived_brightness = (~meaningful_brightness
                               | (brightness >= foreground_survival_ratio * reference_brightness[original_ids]))
        survived_contrast = (~meaningful_contrast
                             | (contrast >= foreground_survival_ratio * reference_contrast[original_ids]))
        return survived_brightness & survived_contrast

    def inside(coordinates):
        return (np.isfinite(coordinates).all(axis=1) & (coordinates[:, 0] >= margin)
                & (coordinates[:, 0] <= width - margin - 1) & (coordinates[:, 1] >= margin)
                & (coordinates[:, 1] <= height - margin - 1))

    def patches(plane, coordinates):
        return cv2.remap(plane, (coordinates[:, :1] + offsets_x).astype(np.float32),
                         (coordinates[:, 1:] + offsets_y).astype(np.float32), interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def pair(source_image, target_image, source_gray, target_gray, source_uv):
        if not len(source_uv):
            return source_uv.copy(), np.zeros(0, bool), np.zeros(0)
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                      flags=0, minEigThreshold=1e-4)
        candidate, forward_ok, _ = cv2.calcOpticalFlowPyrLK(source_image, target_image, source_uv[:, None], None, **params)
        if candidate is None or forward_ok is None:
            return source_uv.copy(), np.zeros(len(source_uv), bool), np.full(len(source_uv), np.inf)
        candidate = candidate.reshape(-1, 2)
        finite_candidate = np.isfinite(candidate).all(axis=1)
        safe_candidate = np.nan_to_num(candidate, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
        reverse, backward_ok, _ = cv2.calcOpticalFlowPyrLK(target_image, source_image, safe_candidate[:, None], None, **params)
        if reverse is None or backward_ok is None:
            return source_uv.copy(), np.zeros(len(source_uv), bool), np.full(len(source_uv), np.inf)
        fb_error = np.linalg.norm(reverse.reshape(-1, 2) - source_uv, axis=1)
        first_patch, second_patch = patches(source_gray, source_uv), patches(target_gray, safe_candidate)
        patch_error = np.mean(np.abs(first_patch - second_patch), axis=1)
        first_zero = first_patch - first_patch.mean(axis=1, keepdims=True)
        second_zero = second_patch - second_patch.mean(axis=1, keepdims=True)
        first_std, second_std = first_zero.std(axis=1), second_zero.std(axis=1)
        correlation = np.mean(first_zero * second_zero, axis=1) / np.maximum(first_std * second_std, 1e-12)
        good = (forward_ok.reshape(-1).astype(bool) & backward_ok.reshape(-1).astype(bool)
                & finite_candidate & inside(source_uv) & inside(safe_candidate) & np.isfinite(fb_error)
                & (fb_error <= fb_max_error) & (patch_error <= patch_max_error) & (correlation >= .8)
                & (first_std >= .005) & (second_std >= .005))
        return safe_candidate, good, fb_error

    diagnostics = []
    for knot in range(1, len(times)):
        change = next_change()
        proxy_log = log_reference + change
        proxy = np.exp(np.clip(proxy_log, math.log(log_floor), 0)).astype(np.float32)
        current_u8 = np.rint(proxy * 255).astype(np.uint8)
        activity = cv2.dilate((np.abs(change - previous_change) > .005).astype(np.float32), np.ones((7, 7), np.uint8))
        prior_uv = tracked.copy()
        previous_valid = anchor_valid.copy()
        direct_count = 0
        direct_survival_rejected = direct_survival_checked = 0
        if feature_method == 'sift':
            current_uv, current_descriptors = _sift_detections(
                sift, current_u8, None, int(max_corners), min_distance, margin)
            matches = _mutual_sift_matches(reference_descriptors, current_descriptors)
            ids, candidate = matches[:, 0], current_uv[matches[:, 1]]
            survival = foreground_survival(proxy, candidate, ids)
            direct_survival_rejected = int((~survival).sum())
            direct_survival_checked = int(survival_applicable[ids].sum())
            ids, candidate = ids[survival], candidate[survival]
            supported = ((sample(activity, initial_uv[ids]) > .1)
                         | (sample(activity, candidate) > .1))
            # Without events only an uninterrupted descriptor image prior
            # may continue; quiet periods never resurrect a failed identity.
            usable = previous_valid[ids] | supported
            ids, candidate, supported = ids[usable], candidate[usable], supported[usable]
            anchor_valid[:] = False
            anchor_valid[ids] = True
            tracked[ids] = candidate
            sources[knot, ids] = 4
            event_supported[knot, ids] = supported
            restored = ids[~previous_valid[ids]]
            reassociated[knot, restored] = True
            direct_count = len(restored)
            chain_before = sift_chain_intact.copy()
            sift_chain_intact &= anchor_valid
            continuous[knot] = sift_chain_intact
            newly_broken = int((chain_before & ~sift_chain_intact).sum())
            survival_rejected = survival_checked = 0
            errors, good = np.empty(0), np.empty(0, dtype=bool)
        else:
            ids = np.flatnonzero(previous_valid)
            candidate, good, errors = pair(previous_u8, current_u8, previous_proxy, proxy, tracked[ids])
            survival = foreground_survival(proxy, candidate, ids)
            survival_rejected = int((good & ~survival).sum())
            survival_checked = int((good & survival_applicable[ids]).sum())
            good &= survival
            local_ids = ids[good]
            tracked[local_ids] = candidate[good]
            continuous[knot, local_ids] = True
            sources[knot, local_ids] = 2
            anchor_valid[:] = False
            anchor_valid[local_ids] = True
            if allow_reference_reacquisition and np.any(activity):
                lost_ids = np.flatnonzero(~anchor_valid)
                direct, direct_good, _ = pair(reference_u8, current_u8, initial_proxy, proxy, initial_uv[lost_ids])
                direct_survival = foreground_survival(proxy, direct, lost_ids)
                direct_survival_rejected = int((direct_good & ~direct_survival).sum())
                direct_survival_checked = int((direct_good & survival_applicable[lost_ids]).sum())
                direct_good &= direct_survival
                direct_ids = lost_ids[direct_good]
                # Re-association uses exactly the original G0 pixels, never the
                # stale retained positions. A quiet frame cannot claim reacquisition.
                if len(direct_ids):
                    supported = (sample(activity, initial_uv[direct_ids]) > .1) | (sample(activity, direct[direct_good]) > .1)
                    direct_ids, direct = direct_ids[supported], direct[direct_good][supported]
                    tracked[direct_ids] = direct
                    reassociated[knot, direct_ids] = True
                    anchor_valid[direct_ids] = True
                    sources[knot, direct_ids] = 3
                    direct_count = len(direct_ids)
            newly_broken = int((previous_valid & ~continuous[knot]).sum())
        if count:
            if feature_method == 'corners':
                event_supported[knot] = ((sample(activity, prior_uv) > .1) | (sample(activity, tracked) > .1))
            # For restored tracks, use the actual reference used by the match,
            # not the stale retained coordinate, when testing event support.
            restored = reassociated[knot]
            event_supported[knot, restored] = ((sample(activity, initial_uv[restored]) > .1)
                                                | (sample(activity, tracked[restored]) > .1))
        accepted[knot] = anchor_valid & event_supported[knot]
        tracks[knot] = tracked + .5
        diagnostics.append({"knot": knot, "time_seconds": float(times[knot]),
                            "continuous_image_correspondences": int(continuous[knot].sum()),
                            "event_supported_correspondences": int(accepted[knot].sum()),
                            "image_only_priors": int((anchor_valid & ~event_supported[knot]).sum()),
                            "newly_broken_image_chains": newly_broken,
                            "frozen_unverified_priors": int((~anchor_valid).sum()),
                            "explicit_reference_reassociations": direct_count,
                            "original_G0_sift_descriptor_matches": int((sources[knot] == 4).sum()),
                            "foreground_survival_checked_continuations": survival_checked,
                            "foreground_survival_rejected_continuations": survival_rejected,
                            "foreground_survival_checked_reference_matches": direct_survival_checked,
                            "foreground_survival_rejected_reference_matches": direct_survival_rejected,
                            "mean_accepted_pair_fb_error": float(errors[good].mean()) if good.any() else None,
                            "event_active_pixel_fraction": float(np.mean(activity > 0)),
                            "clipped_proxy_fraction": float(np.mean((proxy_log < math.log(log_floor)) | (proxy_log > 0)))})
        previous_proxy, previous_u8, previous_change = proxy, current_u8, change
    try:
        next(cumulative)
    except StopIteration:
        pass
    else:
        raise ValueError("Too many cumulative event maps")
    return {"points": points.astype(np.float32), "tracks": tracks, "accepted": accepted,
            "tracking_continuous": continuous, "reassociated": reassociated,
            "event_supported": event_supported, "tracking_source": sources,
            "metadata": {
                "kind": ("g0_surface_event_proxy_klt_features" if feature_method == 'corners'
                         else "g0_surface_event_proxy_sift_reference_features"), "version": 3,
                "feature_method": feature_method,
                "point_source": "G0_render_normalized_depth_backprojection_hypothesis",
                "new_geometry_measurement": False, "proxy_is_observed_rgb": False,
                "pretrained_models": False, "points_are_world_coordinates": True,
                "track_pixel_center": "boundary", "depth_semantics": "normalized_camera_z_not_alpha_weighted_sum",
                "minimum_alpha": minimum_alpha, "maximum_relative_3x3_depth_range": maximum_relative_depth_range,
                "maximum_corners": int(max_corners), "reference_background_already_composited": True,
                "fb_max_error_pixels": fb_max_error if feature_method == 'corners' else None,
                "patch_mean_absolute_error_limit": patch_max_error if feature_method == 'corners' else None,
                "patch_minimum_correlation": .8 if feature_method == 'corners' else None,
                "patch_minimum_std": .005 if feature_method == 'corners' else None,
                "reference_reacquisition_enabled": feature_method == 'sift' or allow_reference_reacquisition,
                "sift_matching": {
                    "enabled": feature_method == 'sift', "ratio_both_directions": .7,
                    "mutual_unique_matches_required": True, "equal_distance_ties": "reject",
                    "reference_descriptor_refresh": "never_original_G0_only",
                    "spatial_deduplication_distance": max(1., min_distance),
                    "duplicate_orientation_policy": "one_strongest_response_keypoint_per_spatial_location",
                    "image_border_exclusion": "max(existing_margin,1.5*keypoint_size)",
                    "subpixel_depth_taps_must_all_be_supported": True,
                    "pixelwise_patch_correlation_gate_used": False,
                    "no_KLT_or_descriptor_fallback": True,
                    "independent_backend": True,
                    "reference_reacquisition_flag_scope": "KLT_only; SIFT_always_matches_original_G0",
                },
                "foreground_survival": {
                    "enabled": survival_enabled, "background_intensity": background_intensity,
                    "background_source": "explicit_caller_G0_event_derived_estimate" if survival_enabled else "not_supplied",
                    "long_term_photometric_protection": "heuristic_G0_fixed_statistics" if survival_enabled else "disabled",
                    "reference_refresh": "never_original_G0_only", "circular_patch_radius": survival_radius,
                    "positive_brightness_statistic": "q80(max(I-background,0))",
                    "contrast_statistic": "q90(I)-q10(I)",
                    "minimum_meaningful_reference_brightness": .02,
                    "minimum_meaningful_reference_contrast": .01,
                    "minimum_retained_ratio": foreground_survival_ratio,
                    "meaningful_terms_must_all_survive": True,
                    "dark_material_requires_positive_brightness": False,
                    "heuristic_not_photometric_calibration": True,
                    "rotation_invariant_guaranteed": False,
                    "unprotected_features_policy": ("retain_original_pair_gates_without_claiming_survival_protection"
                                                     if feature_method == 'corners' else "retain_mutual_descriptor_gates_without_claiming_survival_protection"),
                },
                "reacquisition_source": ("original_G0_pixels_with_direct_bidirectional_KLT_and_patch_checks"
                                         if feature_method == 'corners' else "original_G0_fixed_SIFT_mutual_bidirectional_ratio_matches"),
                "source_codes": {"0": "frozen_unverified_prior", "1": "G0_anchor", "2": "continuous_image_pair_hypothesis", "3": "explicit_G0_reference_match", "4": "explicit_SIFT_original_G0_match"},
                "accepted_semantics": "current_anchored_2D_proxy_correspondence_AND_local_current_window_event_support",
                "continuous_semantics": ("valid_local_continuation_from_previous_anchor; false_at_reassociation; never_fills_prior_gaps"
                                         if feature_method == 'corners' else "uninterrupted_original_G0_descriptor_matches; permanently_false_after_first_gap"),
                "quiet_interval_policy": "continue_checked_image_correspondence_as_prior_only; accepted_false_without_events",
                "t0_is_event_observation": False, "physical_motion_validated": False,
                "hidden_surface_geometry_validated": False, "heldout_independence_established": False,
                "limitations": ["G0 rendered depth and proxy photometry are assumptions, not new observations.",
                                "FB and patch agreement cannot disambiguate repeated texture or prove physical point identity.",
                                "Direct reference KLT is not rotation invariant and may reject large displacement or appearance change.",
                                "SIFT matching may help image-plane rotation but does not guarantee identity under repeated Lego textures, occlusion or perspective changes.",
                                "Fixed-G0 foreground survival is heuristic; lighting, foreshortening and background error may reject real correspondences.",
                                "Whole-sequence cumulative event initialization cannot be called independent holdout training."],
            },
            "diagnostics": {"feature_count": count, "initial_feature_mask_pixels": int(feature_mask.sum()),
                            "opaque_pixels_rejected_by_depth_range": int((opaque & ~depth_supported).sum()),
                            "foreground_survival_brightness_supported_features": int(brightness_applicable.sum()),
                            "foreground_survival_contrast_supported_features": int(contrast_applicable.sum()),
                            "foreground_survival_unprotected_features": int((~survival_applicable).sum()),
                            "foreground_survival_rejected_point_steps": sum(
                                row["foreground_survival_rejected_continuations"]
                                + row["foreground_survival_rejected_reference_matches"] for row in diagnostics),
                            "total_event_supported_point_steps": int(accepted.sum()),
                            "total_reference_reassociations": int(reassociated.sum()), "pairs": diagnostics}}
