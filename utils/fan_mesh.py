"""Identity and motion partitions for Fan's before/after binding comparison."""
import numpy as np


def comparison_groups(count, before, after):
    """Separate the rotor, static support, and the support whose binding changed."""
    before, after = np.asarray(before), np.asarray(after)
    for rows in (before, after):
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                or np.any(rows < 0) or np.any(rows >= count)
                or np.any(np.diff(rows.astype(np.int64)) <= 0)):
            raise ValueError('Fan comparison needs sorted unique dynamic rows')
    if not np.isin(after, before).all():
        raise ValueError('Fan repair must only restore moving support to static')
    changed = np.setdiff1d(before, after)
    if not len(changed):
        raise ValueError('Fan comparison needs distinct motion versions')
    groups = np.zeros(count, np.int32)
    groups[after] = 1
    groups[changed] = 2
    return groups


def validate_source_pair(before, after):
    # Mesh geometry may be shared only because the canonical Gaussian rows are
    # byte-identical. Each side still embeds its own measured source motion.
    for key in ('base_vertices_sha256', 'frame_count', 'fps', 'preview_camera', 'input_sha256'):
        if before.get(key) is None or before[key] != after.get(key):
            raise ValueError(f'Fan comparison source mismatch: {key}')
    if ([f['time_seconds'] for f in before['frames']]
            != [f['time_seconds'] for f in after['frames']]):
        raise ValueError('Fan comparison timestamps differ')
