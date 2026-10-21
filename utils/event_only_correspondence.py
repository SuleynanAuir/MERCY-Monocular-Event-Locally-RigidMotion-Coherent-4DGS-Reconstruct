"""Small, explicitly event-derived 2D correspondence prior for rigid fitting.

This does not replace raw-event rendering losses and is not an independent
measurement. Inputs come only from G0/event proxy tracks in the same run.
"""
import torch

from utils.event_only_motion import _rotate


def feature_reprojection_loss(motion, timestamp, evidence):
    """Robust pixel residual / 5, using only supported dynamic seed tracks.

At exact knots, use that knot's accepted mask; between knots, both endpoint
identities must be accepted. Interpolated targets are explicitly a prior.
"""
    zero = (motion.translations.sum() + motion.rotation_vectors.sum()) * 0
    if evidence is None or not len(evidence['points']):
        return zero
    time = torch.as_tensor(timestamp, dtype=motion.times.dtype, device=motion.times.device)
    if time.ndim != 0 or not torch.isfinite(time) or time < motion.times[0] or time > motion.times[-1]:
        raise ValueError('Correspondence time is outside the motion interval')
    right = torch.searchsorted(motion.times, time, right=True).clamp(1, len(motion.times) - 1)
    left = right - 1
    blend = ((time - motion.times[left]) / (motion.times[right] - motion.times[left])).to(motion.translations.dtype)
    if float(blend) <= 1e-6:
        mask = evidence['accepted'][left]
    elif float(blend) >= 1 - 1e-6:
        mask = evidence['accepted'][right]
    else:
        mask = evidence['accepted'][left] & evidence['accepted'][right]
    if not bool(mask.any()):
        return zero
    groups = evidence['groups'][mask]
    target = evidence['tracks'][left, mask] * (1 - blend) + evidence['tracks'][right, mask] * blend
    translations, rotations = motion.at_time(time)
    centers = motion.centers[groups]
    world = _rotate(rotations[groups], evidence['points'][mask] - centers) + centers + translations[groups]
    w2c = evidence['w2c']
    camera_xyz = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera_xyz[:, 2]
    intrinsic = evidence['intrinsic']
    predicted = camera_xyz[:, :2] / z[:, None].clamp_min(.2) * intrinsic[:2] + intrinsic[2:]
    residual = predicted - target
    robust = (residual.square() + .01 ** 2).sqrt() - .01
    return robust.mean() / 5 + torch.relu(.2 - z).mean() / motion.scene_scale


def make_feature_evidence(features, seed_group_ids, frozen_groups, camera, knot_times, device):
    import numpy as np
    points, tracks = np.asarray(features['points']), np.asarray(features['tracks'])
    accepted, groups, frozen = np.asarray(features['accepted']), np.asarray(seed_group_ids), np.asarray(frozen_groups)
    if (points.ndim != 2 or points.shape[1] != 3 or tracks.shape != (len(knot_times), len(points), 2)
            or accepted.shape != tracks.shape[:2] or accepted.dtype != np.bool_
            or groups.shape != (len(points),) or groups.dtype.kind not in 'iu'
            or frozen.ndim != 1 or frozen.dtype != np.bool_ or np.any(groups >= len(frozen)) or np.any(groups < -1)
            or not np.isfinite(points).all() or not np.isfinite(tracks).all()):
        raise ValueError('Malformed G0/event feature evidence')
    supported = groups >= 0
    supported[supported] &= ~frozen[groups[supported]]
    result = {
        'points': torch.as_tensor(points[supported], dtype=torch.float32, device=device),
        'tracks': torch.as_tensor(tracks[:, supported], dtype=torch.float32, device=device),
        'accepted': torch.as_tensor(accepted[:, supported], dtype=torch.bool, device=device),
        'groups': torch.as_tensor(groups[supported], dtype=torch.long, device=device),
        'w2c': torch.as_tensor(camera['w2c'], dtype=torch.float32, device=device),
        'intrinsic': torch.tensor([camera[name] for name in ('fx', 'fy', 'cx', 'cy')], dtype=torch.float32, device=device),
    }
    return result
