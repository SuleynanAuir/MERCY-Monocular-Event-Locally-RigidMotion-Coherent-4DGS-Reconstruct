"""CPU-only roof surface replacement PROPOSAL; all input models stay read-only.

Only explicit Blender roof parts 11..15 are sampled. Original static retirement
uses conservative world geometry, never old screen-space binding labels. This
writes a small NPZ, not a PLY, motion cache, GPU render, or acceptance decision.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.diagnose_lego_sparse_points import load_inputs, sha256, summarize
from scripts.prepare_lego_surface_colors import C0, part_color_table, shaded_gray
from utils.mesh_surface_sampling import normals_to_wxyz, projection, sample_mesh


ROOF_PARTS = np.arange(11, 16, dtype=np.int32)


def geometry_retirement(xyz, roof_distance, other_distance, maximum_scale, bounds,
                        padding=.01, max_distance=.01, margin=.002, footprint_margin=.001):
    """Conservative geometry-only decision, on already protected STATIC rows."""
    xyz, roof_distance, other_distance, maximum_scale, bounds = map(np.asarray,
        (xyz, roof_distance, other_distance, maximum_scale, bounds))
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or bounds.shape != (2, 3)
            or np.any(bounds[0] > bounds[1]) or not np.isfinite(bounds).all()
            or not np.isfinite(xyz).all() or any(a.shape != (len(xyz),)
                or not np.isfinite(a).all() or np.any(a < 0)
                for a in (roof_distance, other_distance, maximum_scale))
            or any(not np.isfinite(x) or x < 0
                   for x in (padding, max_distance, margin, footprint_margin))):
        raise ValueError('Invalid geometric retirement metrics or bounds')
    inside = np.all((xyz >= bounds[0]-padding) & (xyz <= bounds[1]+padding), axis=1)
    close = roof_distance <= max_distance
    unambiguous = roof_distance+margin < other_distance
    isolated = other_distance > 3*maximum_scale+footprint_margin
    chosen = inside & close & unambiguous & isolated
    return chosen, {'tested_static_rows': len(xyz), 'inside_expanded_roof_bounds': int(inside.sum()),
        'inside_and_close_to_roof': int((inside & close).sum()),
        'inside_close_unambiguous': int((inside & close & unambiguous).sum()),
        'selected_original_static': int(chosen.sum()),
        'protected_ambiguous_or_nonroof_footprint': int((inside & close & ~chosen).sum())}


def assemble_retired(original_indices, eligible_indices, suffix_parts, prefix, total, dynamic_indices):
    """Protect every dynamic row and every nonroof completion-suffix row."""
    arrays = [np.asarray(x) for x in (original_indices, eligible_indices, suffix_parts, dynamic_indices)]
    original_indices, eligible_indices, suffix_parts, dynamic_indices = arrays
    if (type(prefix) is not int or type(total) is not int or not 0 < prefix < total
            or total > np.iinfo(np.int32).max
            or any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in arrays)
            or suffix_parts.shape != (total-prefix,)
            or np.any(original_indices < 0) or np.any(original_indices >= prefix)
            or np.any(original_indices[1:] <= original_indices[:-1])
            or not np.isin(original_indices, eligible_indices).all()
            or np.any(dynamic_indices < 0) or np.any(dynamic_indices >= total)):
        raise ValueError('Invalid protected prefix/static/suffix partitions')
    suffix = (prefix+np.flatnonzero(np.isin(suffix_parts, ROOF_PARTS))).astype(np.int32)
    retired = np.concatenate((original_indices.astype(np.int32), suffix))
    if np.intersect1d(retired, dynamic_indices).size:
        raise ValueError('A proposed retirement touches baked dynamic motion')
    return retired, suffix


def gaussian_attributes(normals, spacing=.0015, tangent_factor=.70, normal_factor=.10, opacity=.95):
    if (not np.isfinite([spacing, tangent_factor, normal_factor, opacity]).all()
            or spacing <= 0 or not 0 < normal_factor < tangent_factor <= 1 or not 0 < opacity < 1):
        raise ValueError('Invalid bounded Gaussian surface parameters')
    q = normals_to_wxyz(normals)
    scales = np.tile(np.log(np.asarray([tangent_factor, tangent_factor, normal_factor])*spacing),
                     (len(q), 1)).astype(np.float32)
    logits = np.full(len(q), np.log(opacity/(1-opacity)), np.float32)
    return q, scales, logits


def observed_gray(xyz, normals, part_ids, masks, depths, frames, images, camera,
                  depth_tolerance=.0009, minimum_support=3, elastic_boxes=None):
    """Original-camera nearest-pixel color, supported by same-part first-hit depth.

    Four surrounding mask pixels must have the same part. Depth is compared to
    the triangle tangent-plane intersection at the nearest pixel, avoiding a
    slanted-face center/pixel mismatch. No brightness threshold discards black
    plastic. Multiple static frames are temporal repeats, NOT new viewpoints.
    """
    xyz, normals, part_ids, masks, depths, frames, images = map(np.asarray,
        (xyz, normals, part_ids, masks, depths, frames, images))
    height, width = camera['height'], camera['width']
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape
            or not np.isfinite(xyz).all() or not np.isfinite(normals).all()
            or part_ids.shape != (len(xyz),) or part_ids.dtype.kind not in 'iu'
            or masks.shape != (len(frames), height, width) or depths.shape != masks.shape
            or masks.dtype.kind not in 'iu' or frames.ndim != 1 or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or np.any(frames >= len(images))
            or np.any(frames[1:] <= frames[:-1])
            or images.shape[1:] != (height, width) or images.dtype != np.uint8
            or not np.isfinite(depth_tolerance) or depth_tolerance <= 0
            or type(minimum_support) is not int or not 1 <= minimum_support <= len(frames)):
        raise ValueError('Invalid observed-color camera, images, masks or frames')
    if elastic_boxes is None:
        elastic_boxes = [[] for _ in frames]
    if len(elastic_boxes) != len(frames):
        raise ValueError('Elastic screen exclusions must match sampled frames')
    pose = np.asarray(camera['pose'], np.float64)
    xy, point_depth = projection(xyz, pose, width, height, camera['fovy_degrees'])
    finite = np.isfinite(xy).all(1) & (point_depth > 0)
    safe_xy = np.where(np.isfinite(xy), xy, -1000)
    lower = np.floor(safe_xy).astype(np.int32)
    nearest = np.floor(safe_xy+.5).astype(np.int32)
    in_image = finite & (lower[:, 0] >= 0) & (lower[:, 0]+1 < width)
    in_image &= (lower[:, 1] >= 0) & (lower[:, 1]+1 < height)
    towards_camera = pose[:3, 3]-xyz
    towards_camera /= np.maximum(np.linalg.norm(towards_camera, axis=1, keepdims=True), 1e-20)
    facing = np.sum(normals*towards_camera, axis=1) >= .2
    selected = np.flatnonzero(in_image & facing)
    values = np.full((len(frames), len(xyz)), np.nan, np.float32)
    tangent = np.tan(np.deg2rad(camera['fovy_degrees'])/2)
    local_point = (xyz-pose[:3, 3]) @ pose[:3, :3]
    local_normal = normals @ pose[:3, :3]
    nx, ny = nearest[selected].T
    rays = np.column_stack(((2*(nx+.5)/width-1)*tangent*width/height,
                            (1-2*(ny+.5)/height)*tangent, -np.ones(len(selected))))
    denominator = np.sum(local_normal[selected]*rays, axis=1)
    plane_depth = np.divide(np.sum(local_normal[selected]*local_point[selected], axis=1),
                            denominator, out=np.full(len(selected), np.nan), where=np.abs(denominator)>1e-10)
    x0, y0 = lower[selected].T
    frame_counts, elastic_rejected = [], 0
    for sample, frame in enumerate(frames):
        same = np.ones(len(selected), bool)
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            same &= masks[sample, y0+dy, x0+dx] == part_ids[selected]
        front = depths[sample, ny, nx]
        supported = same & np.isfinite(front) & (front > 0) & np.isfinite(plane_depth)
        supported &= (plane_depth > 0) & (np.abs(front-plane_depth) <= depth_tolerance)
        blocked = np.zeros(len(selected), bool)
        for box in elastic_boxes[sample]:
            box = np.asarray(box)
            if box.shape != (2, 2) or not np.isfinite(box).all():
                raise ValueError('Invalid projected elastic exclusion bounds')
            blocked |= np.all((xy[selected] >= box[0]) & (xy[selected] <= box[1]), axis=1)
        elastic_rejected += int((supported & blocked).sum())
        supported &= ~blocked
        take = selected[supported]
        values[sample, take] = images[int(frame), nearest[take, 1], nearest[take, 0]]/np.float32(255)
        frame_counts.append({'frame': int(frame), 'supported_points': int(len(take))})
    support = np.isfinite(values).sum(0).astype(np.int32)
    measured = support >= minimum_support
    result = np.full(len(xyz), np.nan, np.float32)
    if measured.any():
        result[measured] = np.nanmedian(values[:, measured], axis=0)
    return result, support, {'observed_points': int(measured.sum()), 'per_frame': frame_counts,
        'elastic_bbox_rejected_sample_observations': elastic_rejected,
        'front_facing_in_image_points': int(len(selected))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'parts', 'binding', 'front-masks', 'supervision', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--spacing', type=float, default=.0015)
    parser.add_argument('--tangent-factor', type=float, default=.70)
    parser.add_argument('--normal-factor', type=float, default=.10)
    parser.add_argument('--opacity', type=float, default=.95)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.output.suffix != '.npz' or args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new NPZ/JSON output pair')
    if not .0015 <= args.spacing <= .0025 or not 1 <= args.workers <= 8:
        raise ValueError('Roof experiment spacing must be in [.0015,.0025], workers in [1,8]')
    args.output = args.output.resolve()
    for name in ('baseline', 'candidate', 'parts', 'binding', 'front_masks', 'supervision'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    started = time.monotonic()
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from scripts.render_lego_multiview import verify_inputs
    cache, record, mask_meta, _, checked_paths, checked_hashes, motion_error, _ = verify_inputs(
        args.baseline, args.candidate, args.front_masks)
    inputs, eligible, old_xyz, old_labels, reference, reference_parts, old_scales, _ = load_inputs(
        args.baseline, args.binding, args.parts)
    paths, hashes = dict(inputs['paths']), dict(inputs['hashes'])
    paths.update(checked_paths)
    hashes.update(checked_hashes)
    paths['supervision'] = args.supervision
    hashes['supervision'] = sha256(args.supervision)
    audit = json.loads((args.candidate/'event-audit.json').read_text())
    if audit.get('artifact_sha256', {}).get('supervision') != hashes['supervision']:
        raise ValueError('RGB supervision differs from the current event audit')
    if (mask_meta.get('front_only_stride') is None or mask_meta.get('camera') != cache['camera']
            or any(v['yaw'] != 0 or v['pitch'] != 0 or v['zoom'] != 1 or v['pan_x'] != 0 or v['pan_y'] != 0
                   for v in mask_meta['views'])):
        raise ValueError('Colors require only the original calibrated camera, not orbit views')
    with np.load(paths['motion'], allow_pickle=False) as archive:
        group_ids, q, t = archive['group_ids'], archive['quaternions'], archive['translations']
    root_groups = np.flatnonzero(np.asarray(record['joint_structure']['parents']) == -1)
    if (not np.isin(group_ids[ROOF_PARTS], root_groups).all()
            or not np.allclose(q[:, group_ids[ROOF_PARTS]], [1, 0, 0, 0], rtol=0, atol=1e-7)
            or not np.allclose(t[:, group_ids[ROOF_PARTS]], 0, rtol=0, atol=1e-7)):
        raise ValueError('Requested roof parts are not identity-root static for all 400 frames')
    with np.load(paths['mesh'], allow_pickle=False) as archive:
        vertices, triangles, parts = archive['vertices'], archive['triangles'], archive['part_ids']
        material = archive['material_gray']
        mesh_meta = json.loads(str(archive['metadata_json']))
    if (mesh_meta['objects'] != inputs['objects'] or mesh_meta['parts_sha256'] != hashes['parts']
            or mesh_meta['binding_sha256'] != hashes['binding']
            or not np.allclose(mesh_meta['registration'], inputs['registration'], rtol=0, atol=1e-12)):
        raise ValueError('Roof mesh source identity/registration differs from parent binding')
    chosen_triangles = np.flatnonzero(np.isin(parts, ROOF_PARTS)).astype(np.int32)
    if not np.array_equal(np.unique(parts[chosen_triangles]), ROOF_PARTS):
        raise ValueError('The requested roof part set is incomplete')
    roof_vertices = vertices[np.unique(triangles[chosen_triangles])]
    bounds = np.stack((roof_vertices.min(0), roof_vertices.max(0))).astype(np.float64)
    samples = sample_mesh(vertices, triangles[chosen_triangles], parts[chosen_triangles],
                          material[chosen_triangles], [], args.spacing, max_samples=1000000)
    samples['triangle_ids'] = chosen_triangles[samples['triangle_ids']]
    n = len(samples['xyz'])
    if not 0 < n <= 100000:
        raise ValueError('Roof sampling exceeds the explicit 100k output cap; never truncate')
    new_q, new_scale, new_opacity = gaussian_attributes(samples['normals'], args.spacing,
        args.tangent_factor, args.normal_factor, args.opacity)
    roi = np.all((old_xyz >= bounds[0]-.01) & (old_xyz <= bounds[1]+.01), axis=1)
    roi_positions = np.flatnonzero(roi)
    roof_ref = np.isin(reference_parts, ROOF_PARTS)
    roof_tree, other_tree = cKDTree(reference[roof_ref]), cKDTree(reference[~roof_ref])
    d_roof, _ = roof_tree.query(old_xyz[roi], workers=args.workers)
    d_other, _ = other_tree.query(old_xyz[roi], workers=args.workers)
    selected, retirement_stats = geometry_retirement(old_xyz[roi], d_roof, d_other,
        old_scales[roi].max(1), bounds)
    old_retired = eligible[roi_positions[selected]]
    retirement_stats['eligible_static_rows'] = len(eligible)
    retirement_stats['old_screen_label_is_roof_among_selected'] = int(np.isin(old_labels[roi][selected], ROOF_PARTS).sum())
    del roof_tree, other_tree, reference, reference_parts, old_xyz, old_scales
    completion = inputs['completion']
    with np.load(paths['completion_extension'], allow_pickle=False) as archive:
        suffix_parts, suffix_groups = archive['part_ids'], archive['group_ids']
    suffix_roof = np.isin(suffix_parts, ROOF_PARTS)
    if not np.isin(suffix_groups[suffix_roof], root_groups).all():
        raise ValueError('Roof completion suffix unexpectedly uses moving groups')
    dynamic = np.load(args.baseline/'dynamic_indices.npy', allow_pickle=False)
    retired, retired_suffix = assemble_retired(old_retired, eligible, suffix_parts,
        completion['prefix_count'], cache['gaussian_count'], dynamic)
    with np.load(args.front_masks, allow_pickle=False) as archive:
        masks, depths, frames = archive['part_ids'], archive['depth'], archive['frame_indices']
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
        supervision_meta = json.loads(str(archive['metadata_json']))
    if (images.shape != (cache['frame_count'], 640, 640) or supervision_meta.get('fps') != cache['fps']
            or supervision_meta.get('format') != 'lego_raw_event_supervision_v1'):
        raise ValueError('RGB supervision timeline/calibration differs')
    # Front masks explicitly omit elastic bands. Conservatively exclude their
    # projected bounding boxes using real deformed source samples at each frame.
    elastic_boxes = [[] for _ in frames]
    with np.load(args.parts, allow_pickle=False) as archive:
        registration = inputs['registration']
        for part in inputs['excluded_deforming_parts']:
            deformed = archive[f'deformed_{part}']
            for index, frame in enumerate(frames):
                world = deformed[int(frame)] @ registration[:3, :3].T+registration[:3, 3]
                xy, z = projection(world, np.asarray(cache['camera']['pose']), 640, 640,
                                   cache['camera']['fovy_degrees'])
                valid = np.isfinite(xy).all(1) & (z > 0)
                if valid.any():
                    elastic_boxes[index].append(np.stack((xy[valid].min(0)-2, xy[valid].max(0)+2)).tolist())
    measured, support, color_stats = observed_gray(samples['xyz'], samples['normals'], samples['part_ids'],
        masks, depths, frames, images, cache['camera'], elastic_boxes=elastic_boxes)
    old_vertices = PlyData.read(paths['source_ply'], mmap='r')['vertex'].data
    with np.load(args.binding, allow_pickle=False) as archive:
        parent_labels = archive['labels']
    base_table, color_table = part_color_table(old_vertices['f_dc_0'][:completion['prefix_count']]*C0+.5,
        parent_labels, parts, material, ROOF_PARTS)
    gray = shaded_gray(base_table[samples['part_ids']], samples['normals'])
    observed = np.isfinite(measured)
    gray[observed] = measured[observed]
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError('Input changed during roof preparation: '+name)
    metadata = {'format': 'lego_static_roof_rebuild_v1', 'source': str(paths['source_ply']),
        'source_sha256': hashes['source_ply'], 'source_ply_sha256': hashes['source_ply'],
        'input_paths': {k: str(v) for k, v in paths.items()}, 'input_sha256': hashes,
        'target_part_ids': ROOF_PARTS.tolist(), 'allowed_part_ids': ROOF_PARTS.tolist(),
        'target_part_names': [inputs['objects'][i] for i in ROOF_PARTS],
        'new_count': n, 'retired_count': len(retired), 'retired_original_static_count': len(old_retired),
        'retired_compact_roof_count': len(retired_suffix), 'source_gaussian_count': cache['gaussian_count'],
        'original_prefix_count': completion['prefix_count'], 'dynamic_rows_protected': len(dynamic),
        'new_geometry_bounds': bounds.tolist(), 'retirement_roi_bounds': (bounds+np.array([[-.01],[.01]])).tolist(),
        'retirement_bounds': (bounds+np.array([[-.01],[.01]])).tolist(),
        'retirement_statistics': retirement_stats,
        'retirement_rule': {'roi_padding': .01, 'maximum_roof_distance': .01, 'nearest_part_margin': .002,
            'nonroof_clearance': '3*maximum_original_scale + .001', 'old_binding_labels_used_to_retire': False,
            'distance_reference': 'Every rigid registered rest sample, including static platform and moving rigid parts; elastic rows separately protected. Sample distances are conservative heuristics, not exact triangle distances.',
            'compact_suffix_rule': 'Retire ALL and ONLY existing completion rows for exact roof parts 11..15.'},
        'retired_roof_distance': summarize(d_roof[selected]), 'retired_other_distance': summarize(d_other[selected]),
        'sampling': samples['stats'], 'spacing': args.spacing, 'tangent_factor': args.tangent_factor,
        'normal_factor': args.normal_factor, 'opacity': args.opacity,
        'triangle_ids_convention': 'Global indices into input full-mesh triangles, never roof-local row numbers.',
        'quaternion_convention': 'Normalized wxyz, local +Z maps to actual oriented triangle normal.',
        'dc_gray_convention': 'Grayscale image intensity in [0,1], NOT SH coefficients; (gray-.5)/C0 on all 3 DC channels.',
        'color_observation': {'frames': frames.tolist(), 'minimum_temporal_support': 3, 'depth_tolerance': .0009,
            'same_part_mask_footprint': 'All four pixels enclosing projected sample', 'pixel_color': 'Nearest native pixel, median across supported frames; no spatial smoothing or brightness cutoff.',
            'depth_comparison': 'First-hit positive camera z against the sample triangle plane at nearest pixel ray.',
            'minimum_oriented_facing_cosine': .2, 'elastic_projected_bbox_margin_pixels': 2,
            'limitation': 'One fixed view only; repeated static frames are not new viewpoints. Visibility uses sampled rigid-mesh prior plus conservative elastic bbox exclusion.',
            **color_stats},
        'color_prior_table': color_table, 'observed_color_count': int(observed.sum()),
        'prior_color_count': int((~observed).sum()), 'new_gray_statistics': summarize(gray),
        'per_part': {str(int(p)): {'new_count': int((samples['part_ids']==p).sum()),
            'observed_color_count': int(((samples['part_ids']==p)&observed).sum())} for p in ROOF_PARTS},
        'maximum_verified_motion_error': motion_error, 'input_identity_rechecked': True,
        'all_motion_frozen': True, 'original_files_modified': False, 'cpu_only': True,
        'unobserved_geometry_is_prior': True, 'unobserved_appearance_is_prior': True,
        'approved_for_export': False, 'seconds': time.monotonic()-started,
        'risk': 'Ambiguous boundary splats are preserved, so some old fragments can remain. Finite Gaussian footprints can cross tiny mesh holes despite exact on-triangle centers. Only actual Gaussian multi-view and full-timeline RGB/event tests can assess this experiment.',
        'scope': 'Roof-only diagnostic proposal, not published. No pillars, cab connectors, platform, bands or motion replaced. Hidden geometry/appearance remain explicit Blender/material/same-part priors, not monocular reconstructed truth.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, retired_indices=retired.astype(np.int32),
        new_xyz=samples['xyz'].astype(np.float32), new_quaternions=new_q, new_log_scale=new_scale,
        new_opacity_logit=new_opacity, new_dc_gray=gray.astype(np.float32), part_ids=samples['part_ids'],
        triangle_ids=samples['triangle_ids'], new_normals=samples['normals'],
        observed_color=observed, observation_support=support, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'new_count': n, 'retired_count': len(retired),
        'retired_original_static_count': len(old_retired), 'retired_compact_roof_count': len(retired_suffix),
        'observed_color_count': int(observed.sum()), 'seconds': metadata['seconds'], 'source_modified': False}), flush=True)


if __name__ == '__main__':
    main()
