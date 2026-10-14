"""Deterministic, CPU-only sampling of an explicitly supplied triangle prior.

These samples are *prior geometry*, not observations of an unseen surface.
Every retained location is an actual triangle sample.  Voxel deduplication
selects an existing sample; it never averages across faces, holes or parts.
"""

import math

import numpy as np


_CHUNK = 262144
_INT32_MAX = np.iinfo(np.int32).max


def _integer(value, name, minimum=0, maximum=_INT32_MAX):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return int(value)


def _points(value, name):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in 'fiu'
            or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be finite numeric [N, 3]')
    return array


def _radical_inverse(indices):
    """Base-2 radical inverse, including a half-last-bit interior offset."""
    bits = np.asarray(indices, np.uint32)
    bits = ((bits >> 1) & np.uint32(0x55555555)) | ((bits & np.uint32(0x55555555)) << 1)
    bits = ((bits >> 2) & np.uint32(0x33333333)) | ((bits & np.uint32(0x33333333)) << 2)
    bits = ((bits >> 4) & np.uint32(0x0f0f0f0f)) | ((bits & np.uint32(0x0f0f0f0f)) << 4)
    bits = ((bits >> 8) & np.uint32(0x00ff00ff)) | ((bits & np.uint32(0x00ff00ff)) << 8)
    bits = (bits >> 16) | (bits << 16)
    return (bits.astype(np.float64) + .5) / 4294967296.


def sample_mesh(vertices, triangles, part_ids, material_gray, excluded_parts,
                spacing, max_samples=10000000):
    """Sample then deduplicate a mesh without creating off-surface positions.

    For each nonzero, nonexcluded triangle, generate ``ceil(area / spacing**2)``
    samples (at least one): its centroid plus a uniform-area Hammersley sequence.
    Generation is vectorized in bounded chunks.  The safety cap is checked
    *before* sample allocation; exceeding it raises instead of truncating.

    Deduplication keys are (part, floor(xyz / spacing), oriented dominant-normal
    axis).  Six normal bins keep oppositely facing thin walls separate.  The
    first actual sample in each key is retained in input order.  Thus every
    eligible triangle has a centroid before deduplication, but adjacent faces
    can share one retained sample.  No minimum triangle size is silently used.
    """
    vertices = _points(vertices, 'vertices')
    triangles, part_ids, material_gray = map(np.asarray, (triangles, part_ids, material_gray))
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu'
            or len(triangles) > _INT32_MAX or len(vertices) > _INT32_MAX
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))):
        raise ValueError('triangles must be bounded integer vertex indices [M, 3]')
    triangle_count = len(triangles)
    if (part_ids.shape != (triangle_count,) or part_ids.dtype.kind not in 'iu'
            or np.any(part_ids < 0) or np.any(part_ids > _INT32_MAX)):
        raise ValueError('part_ids must be nonnegative int32-compatible [M]')
    if (material_gray.shape != (triangle_count,) or material_gray.dtype.kind not in 'fiu'
            or not np.isfinite(material_gray).all()
            or np.any(material_gray < 0) or np.any(material_gray > 1)):
        raise ValueError('material_gray must be finite [M] in [0, 1]')
    excluded_parts = np.asarray(excluded_parts)
    if (excluded_parts.ndim != 1 or (excluded_parts.size and excluded_parts.dtype.kind not in 'iu')
            or np.any(excluded_parts < 0) or np.any(excluded_parts > _INT32_MAX)):
        raise ValueError('excluded_parts must be a list of nonnegative integer part IDs')
    if (isinstance(spacing, (bool, np.bool_)) or not isinstance(spacing, (float, int, np.floating, np.integer))
            or not np.isfinite(spacing) or spacing <= 0):
        raise ValueError('spacing must be finite and positive')
    spacing = float(spacing)
    cell_area = spacing * spacing
    if not math.isfinite(cell_area) or cell_area <= 0:
        raise ValueError('spacing squared must be finite and positive')
    max_samples = _integer(max_samples, 'max_samples', 1)
    excluded = np.isin(part_ids, excluded_parts)
    counts = np.zeros(triangle_count, np.int64)
    face_normals = np.zeros((triangle_count, 3), np.float32)
    degenerate_count, eligible_count, total, surface_area = 0, 0, 0, 0.
    for start in range(0, triangle_count, _CHUNK):
        end = min(start + _CHUNK, triangle_count)
        corners = vertices[triangles[start:end]].astype(np.float64)
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        if not np.isfinite(twice_area).all():
            raise ValueError('Triangle areas overflow; rescale the supplied mesh')
        nonzero = twice_area > 0
        eligible = nonzero & ~excluded[start:end]
        degenerate_count += int(np.sum(~nonzero))
        eligible_count += int(eligible.sum())
        surface_area += float(np.sum(twice_area[eligible] * .5))
        face_normals[start:end][nonzero] = cross[nonzero] / twice_area[nonzero, None]
        with np.errstate(over='ignore'):
            planned = np.maximum(1., np.ceil((twice_area[eligible] * .5) / cell_area))
        if np.any(~np.isfinite(planned)) or np.any(planned > max_samples):
            raise ValueError(f'Mesh sampling exceeds max_samples={max_samples}; no silent truncation')
        block_counts = counts[start:end]
        block_counts[eligible] = planned.astype(np.int64)
        total += int(block_counts.sum())
        if total > max_samples:
            raise ValueError(f'Mesh sampling needs more than max_samples={max_samples}; no silent truncation')

    stats = {'input_vertices': int(len(vertices)), 'input_triangles': int(triangle_count),
             'excluded_triangles': int(excluded.sum()), 'degenerate_triangles': degenerate_count,
             'eligible_triangles': eligible_count, 'eligible_surface_area': surface_area,
             'spacing': spacing, 'max_samples': max_samples, 'generated_samples': total,
             'sampling': 'area_weighted_hammersley_with_centroid',
             'voxel_merge': 'part_floor_xyz_oriented_normal_6bin_first_actual_sample',
             'normal_bins': ['+x', '-x', '+y', '-y', '+z', '-z'],
             'geometry_provenance': 'supplied_triangle_prior_not_unseen_image_evidence'}
    generated_xyz = np.empty((total, 3), np.float32)
    generated_faces = np.empty(total, np.int32)
    ends = np.cumsum(counts)
    for start in range(0, total, _CHUNK):
        end = min(start + _CHUNK, total)
        global_index = np.arange(start, end, dtype=np.int64)
        face = np.searchsorted(ends, global_index, side='right')
        local = global_index - (ends[face] - counts[face])
        # The centroid has a dedicated index. Remaining samples have equal-area
        # barycentric coordinates via sqrt(u); triangle winding is preserved.
        extra_count = np.maximum(counts[face] - 1, 1)
        extra_index = np.maximum(local - 1, 0)
        u = (extra_index + .5) / extra_count
        v = _radical_inverse(extra_index)
        root = np.sqrt(u)
        weights = np.stack((1-root, root*(1-v), root*v), axis=1)
        weights[local == 0] = 1. / 3.
        corners = vertices[triangles[face]].astype(np.float64)
        generated_xyz[start:end] = np.sum(corners * weights[:, :, None], axis=1)
        generated_faces[start:end] = face
    if not np.isfinite(generated_xyz).all():
        raise ValueError('Sample positions cannot be represented as finite float32')

    # Structured keys use no lossy hash. Chunk construction bounds temporary RAM
    # even for millions of input triangles; the cap bounds the global sort.
    key_dtype = [('part', '<i4'), ('x', '<i8'), ('y', '<i8'), ('z', '<i8'), ('normal', 'i1')]
    keys = np.empty(total, dtype=key_dtype)
    for start in range(0, total, _CHUNK):
        end = min(start + _CHUNK, total)
        face = generated_faces[start:end]
        cells = np.floor(generated_xyz[start:end].astype(np.float64) / spacing)
        if not np.isfinite(cells).all() or np.any(np.abs(cells) >= 2.**62):
            raise ValueError('Voxel indices exceed a safe int64 range; rescale the mesh or spacing')
        normals = face_normals[face]
        dominant = np.argmax(np.abs(normals), axis=1)
        sign = normals[np.arange(len(normals)), dominant] < 0
        keys['part'][start:end] = part_ids[face]
        for axis, name in enumerate(('x', 'y', 'z')):
            keys[name][start:end] = cells[:, axis].astype(np.int64)
        keys['normal'][start:end] = 2*dominant + sign
    _, first = np.unique(keys, return_index=True)
    selected = np.sort(first)
    kept_faces = generated_faces[selected]
    stats['merged_samples'] = int(len(selected))
    stats['removed_duplicate_samples'] = total - int(len(selected))
    return {'xyz': generated_xyz[selected], 'normals': face_normals[kept_faces],
            'part_ids': np.asarray(part_ids[kept_faces], np.int32),
            'material_gray': np.asarray(material_gray[kept_faces], np.float32),
            'triangle_ids': kept_faces, 'stats': stats}


def normals_to_wxyz(normals):
    """Unit wxyz quaternions mapping local +z to each supplied nonzero normal."""
    normals = _points(normals, 'normals').astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0) or not np.isfinite(lengths).all():
        raise ValueError('Normals must have nonzero finite length')
    unit = normals / lengths[:, None]
    transverse = np.hypot(unit[:, 0], unit[:, 1])
    half_angle = .5 * np.arctan2(transverse, unit[:, 2])
    q = np.zeros((len(normals), 4), np.float64)
    q[:, 0] = np.cos(half_angle)
    off_axis = transverse > 0
    scale = np.sin(half_angle[off_axis]) / transverse[off_axis]
    q[off_axis, 1] = -unit[off_axis, 1] * scale
    q[off_axis, 2] = unit[off_axis, 0] * scale
    # At -z either transverse axis is valid; choose +x deterministically.
    q[~off_axis & (unit[:, 2] < 0)] = [0., 1., 0., 0.]
    return q.astype(np.float32)


def projection(points, camera_pose, width, height, fovy_degrees):
    """OpenGL camera-to-world pose -> pixel-center xy and positive forward depth.

    The camera looks along local -z; image y increases downwards. Pixel (0, 0)
    denotes the top-left pixel *center*, so the optical center is
    ((width - 1)/2, (height - 1)/2). At or behind the camera, xy is NaN and depth
    is zero/negative. The inverse agrees with rays through (x+.5, y+.5).
    """
    points = _points(points, 'points')
    width, height = _integer(width, 'width', 1, 16384), _integer(height, 'height', 1, 16384)
    pose = np.asarray(camera_pose)
    if (pose.shape != (4, 4) or pose.dtype.kind not in 'fiu' or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0., 0., 0., 1.], rtol=0, atol=1e-7)):
        raise ValueError('camera_pose must be a finite affine camera-to-world 4x4 matrix')
    if (isinstance(fovy_degrees, (bool, np.bool_)) or not isinstance(fovy_degrees, (float, int, np.floating, np.integer))
            or not np.isfinite(fovy_degrees) or not 0 < fovy_degrees < 180):
        raise ValueError('fovy_degrees must lie strictly between 0 and 180')
    try:
        inverse = np.linalg.inv(pose.astype(np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError('camera_pose must be invertible') from exc
    if not np.isfinite(inverse).all():
        raise ValueError('camera_pose inverse must be finite')
    local = points.astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    depth = -local[:, 2]
    xy = np.full((len(points), 2), np.nan, np.float64)
    front = depth > 0
    focal = height / (2 * np.tan(np.deg2rad(float(fovy_degrees)) / 2))
    with np.errstate(over='ignore', invalid='ignore'):
        xy[front, 0] = focal * local[front, 0] / depth[front] + (width - 1)/2
        xy[front, 1] = -focal * local[front, 1] / depth[front] + (height - 1)/2
    return xy, depth


def depth_buffer(xyz, camera_pose, width, height, fovy_degrees, radius=1):
    """Conservative sampled-point z-min buffer; empty pixels remain +infinity.

    ``radius=0`` keeps exactly the containing pixel. ``radius=1`` expands its
    z-min footprint by one pixel (3x3), not an unbounded hole-filling operation.
    This sampled-prior buffer is not a triangle visibility oracle.
    """
    radius = _integer(radius, 'radius', 0, 1)
    xy, depth = projection(xyz, camera_pose, width, height, fovy_degrees)
    width, height = int(width), int(height)
    if width * height > 67108864:
        raise ValueError('Depth buffer exceeds the 64-megapixel safety limit')
    image = np.full((height, width), np.inf, np.float64)
    inside = ((depth > 0) & np.isfinite(depth) & np.isfinite(xy).all(axis=1)
              & (xy[:, 0] >= -.5) & (xy[:, 0] < width-.5)
              & (xy[:, 1] >= -.5) & (xy[:, 1] < height-.5))
    pixels = np.floor(xy[inside] + .5).astype(np.int64)
    np.minimum.at(image, (pixels[:, 1], pixels[:, 0]), depth[inside])
    if radius:
        # Exactly scipy.ndimage.minimum_filter(size=3, mode='constant', cval=inf),
        # written in NumPy so a platform-specific SciPy build is not required.
        padded = np.pad(image, 1, constant_values=np.inf)
        for dy in range(3):
            for dx in range(3):
                np.minimum(image, padded[dy:dy+height, dx:dx+width], out=image)
    return image
