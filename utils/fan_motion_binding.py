"""Conservative correction of static support mistakenly bound to Fan's rotor.

Only changes moving -> static membership; no Gaussian is removed or recolored.
Even video frames select candidates. Odd frames report support independently of
that selection, although the input rotation trajectory was fitted to all frames.
"""
import numpy as np


def stationary_reassignments(static_support, rotating_support):
    static = np.asarray(static_support, dtype=np.float64)
    rotating = np.asarray(rotating_support, dtype=np.float64)
    if (static.ndim != 1 or static.shape != rotating.shape
            or not np.isfinite(static).all() or not np.isfinite(rotating).all()
            or np.any((static < 0) | (static > 1))
            or np.any((rotating < 0) | (rotating > 1))):
        raise ValueError('Support must be finite equally sized probability vectors')
    # Ambiguous points supported by both hypotheses retain their original motion.
    return (static > .96) & (static - rotating > .20)


def refine_static_membership(xyz, moving, center, angles, frames, camera):
    import cv2
    xyz = np.asarray(xyz)
    moving = np.asarray(moving)
    center = np.asarray(center)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or moving.shape != (len(xyz),)
            or moving.dtype != np.bool_ or center.shape != (3,)
            or len(angles) != len(frames) or len(frames) < 4):
        raise ValueError('Invalid Fan motion binding inputs')
    ids = np.flatnonzero(moving)
    points = xyz[ids]
    rotation = np.asarray(camera['R']).T
    translation = np.asarray(camera['T'])
    width, height = camera['width'], camera['height']

    def project(p):
        v = p @ rotation.T + translation
        return np.column_stack((v[:, 0]/v[:, 2]*camera['fx']+camera['cx'],
                                v[:, 1]/v[:, 2]*camera['fy']+camera['cy']))

    def sample(mask, uv):
        ij = np.floor(uv).astype(np.int64)
        valid = ((ij[:, 0] >= 0) & (ij[:, 0] < width)
                 & (ij[:, 1] >= 0) & (ij[:, 1] < height))
        support = np.zeros(len(uv), dtype=np.float64)
        support[valid] = mask[ij[valid, 1], ij[valid, 0]]
        return support

    initial_uv = project(points)
    static = np.zeros((2, len(ids)), np.float64)
    rotating = np.zeros_like(static)
    counts = [0, 0]
    for index, (angle, gray) in enumerate(zip(angles, frames)):
        gray = np.asarray(gray)
        if gray.shape != (height, width):
            raise ValueError('Video dimensions disagree with the calibrated camera')
        # Two-pixel tolerance protects antialiasing and the fitted shaft's small
        # projection errors. Background mismatches must persist across time.
        threshold = 8 if gray.dtype == np.uint8 else 8/255
        mask = cv2.dilate((gray > threshold).astype(np.uint8), np.ones((5, 5), np.uint8))
        co, si = np.cos(angle), np.sin(angle)
        r = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
        transformed = (points-center) @ r.T + center
        split = index % 2
        static[split] += sample(mask, initial_uv)
        rotating[split] += sample(mask, project(transformed))
        counts[split] += 1
    static /= np.asarray(counts)[:, None]
    rotating /= np.asarray(counts)[:, None]
    selected = stationary_reassignments(static[0], rotating[0])
    changed = ids[selected]
    corrected = moving.copy()
    corrected[changed] = False
    report = {
        'method': 'temporal silhouette support, moving-to-static correction only',
        'reassigned_to_static': int(len(changed)), 'deleted_gaussians': 0,
        'selection_frames': list(range(0, len(frames), 2)),
        'check_frames': list(range(1, len(frames), 2)),
        'scope': 'Odd frames check membership; rotation angles used the full video. Not event validation.',
        'selection_static_support': float(static[0, selected].mean()) if len(changed) else None,
        'selection_rotating_support': float(rotating[0, selected].mean()) if len(changed) else None,
        'check_static_support': float(static[1, selected].mean()) if len(changed) else None,
        'check_rotating_support': float(rotating[1, selected].mean()) if len(changed) else None,
    }
    return corrected, changed, report
