"""Array-only orchestration for G0/event-derived rigid feature hypotheses."""


def add_binding_arguments(parser):
    parser.add_argument('--binding-radius-fraction', type=float, default=.08)
    parser.add_argument('--geodesic-binding', action='store_true',
                        help='Bounded occupied-G0 surface propagation; a spatial prior, not part truth')
    parser.add_argument('--binding-cell-fraction', type=float, default=.004)
    parser.add_argument('--binding-path-fraction', type=float, default=.2)
    parser.add_argument('--binding-max-nodes', type=int, default=100000)


def validate_binding_arguments(args):
    import math
    from numbers import Integral, Real
    for name in ('binding_radius_fraction', 'binding_cell_fraction', 'binding_path_fraction'):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(name + ' must be between zero and one')
    if (isinstance(args.binding_max_nodes, bool) or not isinstance(args.binding_max_nodes, Integral)
            or not 1 <= args.binding_max_nodes <= 100000):
        raise ValueError('binding_max_nodes must be in [1, 100000]')


def bind_feature_groups(xyz, features, discovery, knot_times, input_sha256, args):
    """Build an explicit same-run prior; never load an old binding cache."""
    import numpy as np
    from utils.event_only_hard_binding import build_hard_binding
    validate_binding_arguments(args)
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Nondegenerate finite G0 extent required')
    all_poses = np.concatenate((np.broadcast_to(np.eye(4), (1,) + discovery['poses'].shape[1:]),
                                discovery['poses']))
    all_valid = np.concatenate((np.zeros((1, discovery['pose_valid'].shape[1]), dtype=bool),
                                discovery['pose_valid']))
    labels = np.asarray(discovery['labels']).copy()
    dropped = []
    for group in np.unique(labels[labels >= 0]):
        distinct = len(np.unique(features['points'][labels == group], axis=0))
        if distinct < 8 or (group != 0 and int(discovery['pose_valid'][:, group].sum()) < 2):
            labels[labels == group] = -1
            dropped.append(int(group))
    options = {}
    if args.geodesic_binding:
        from utils.event_only_geodesic_binding import build_geodesic_binding
        options['assignment_prior'] = build_geodesic_binding(
            xyz, features['points'], labels,
            cell_size=args.binding_cell_fraction * scale,
            max_euclidean_distance=args.binding_radius_fraction * scale,
            max_path_length=args.binding_path_fraction * scale,
            max_edge_length=np.sqrt(3) * args.binding_cell_fraction * scale,
            max_nodes=args.binding_max_nodes)
    result = build_hard_binding(xyz, features['points'], labels, knot_times,
                                all_poses, all_valid, input_sha256,
                                max_binding_distance=args.binding_radius_fraction * scale, **options)
    result['diagnostics']['prebinding_unsupported_source_labels'] = dropped
    return result


def static_feature_prior(tracks, continuous, raw_event_counts):
    """Event-quiet and image-stationary hypothesis, never static-motion proof."""
    import cv2
    import numpy as np
    tracks, continuous = np.asarray(tracks), np.asarray(continuous)
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or continuous.shape != tracks.shape[:2]:
        raise ValueError('Tracks and continuity masks must share T-by-N shape')
    if not len(tracks) or continuous.dtype != np.bool_ or not np.isfinite(tracks).all():
        raise ValueError('Finite tracks and boolean continuity are required')
    activity = None
    for counts in raw_event_counts:
        counts = np.asarray(counts)
        if counts.ndim != 3 or counts.shape[0] != 2 or counts.dtype.kind not in 'ui' or np.any(counts < 0):
            raise ValueError('Raw event windows must be nonnegative integer 2-by-H-by-W counts')
        active = (counts[0] > 0) | (counts[1] > 0)
        if activity is None:
            activity = np.zeros(active.shape, dtype=bool)
        if activity.shape != active.shape:
            raise ValueError('Event window shapes must agree')
        activity |= active
    if activity is None:
        raise ValueError('At least one event map is required')
    activity = cv2.dilate(activity.astype(np.uint8), np.ones((9, 9), np.uint8))
    pixels = tracks[0] - .5
    quiet = cv2.remap(activity.astype(np.float32), pixels[:, 0].astype(np.float32)[:, None],
                      pixels[:, 1].astype(np.float32)[:, None], cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=1).reshape(-1) <= .01
    displacement = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    return quiet & continuous.all(axis=0) & (displacement.max(axis=0) < .5)


def infer_feature_groups(reference, alpha, normalized_depth, cumulative, camera, window_times,
                         knot_indices, *, raw_event_counts, max_corners=2048, reference_reacquisition=False,
                         max_groups=6, event_background=None, feature_method='corners', motion_model='se3'):
    import numpy as np
    from utils.event_only_feature_tracks import infer_event_feature_tracks
    from utils.event_only_rigid_discovery import discover_rigid_groups
    indices = np.asarray(knot_indices, dtype=np.int64)
    times = np.asarray(window_times)
    features = infer_event_feature_tracks(
        reference, alpha, normalized_depth,
        (np.zeros_like(reference) if index == 0 else cumulative[index - 1] for index in indices),
        camera, times[indices], max_corners=max_corners,
        allow_reference_reacquisition=reference_reacquisition, background_intensity=event_background,
        **({'feature_method': feature_method} if feature_method != 'corners' else {}))
    if not len(features['points']):
        raise ValueError('G0 rendered surface has no supported feature seeds')
    static = static_feature_prior(features['tracks'], features['tracking_continuous'], iter(raw_event_counts))
    discovery = discover_rigid_groups(features['points'], features['tracks'][1:], features['accepted'][1:], camera,
                                      static_anchor_mask_np=static, max_groups=max_groups,
                                      **({'motion_model': motion_model} if motion_model != 'se3' else {}))
    return {'features': features, 'discovery': discovery, 'static_prior_mask': static,
            'static_hypothesis': 'both_polarities_raw_event_quiet_and_image_stationary_G0_corners_not_static_motion_proof'}
