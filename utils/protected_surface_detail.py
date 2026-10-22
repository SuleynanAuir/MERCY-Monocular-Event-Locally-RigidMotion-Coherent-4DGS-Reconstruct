"""Strict runtime overlay for raster-protected static mesh-prior surface detail.

This appends actual triangle samples and suppresses only explicitly guarded
original/static rows. Parent geometry, color residual and motion are immutable.
Hidden surfaces remain priors; a preview guard is never fresh event approval.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.gaussian_cleanup import _absolute, _checksum, _motion_identity, _sha256, cleanup_eligible_mask


FORMAT = 'lego_protected_surface_detail_v1'
MAX_ADDED = 500000
C0 = .28209479177387814
CONTRACT_FIELDS = {'format', 'report', 'report_sha256', 'extension', 'extension_sha256'}
GEOMETRY_FIELDS = ('new_xyz', 'new_quaternions', 'new_log_scale', 'new_opacity_logit',
                   'new_dc_gray', 'part_ids', 'triangle_ids')
ARRAY_FIELDS = set(GEOMETRY_FIELDS) | {'retired_indices', 'metadata_json'}
INPUT_FIELDS = ('source_ply', 'baseline_cache_metadata', 'correction', 'event_audit',
                'baseline_poses', 'dynamic_indices', 'color_residual', 'color_residual_report',
                'selection', 'preview_json', 'preview_npz', 'mesh', 'parts', 'binding')
SUPPORT_FORMAT = 'lego_global_static_support_guard_v1'
SUPPORT_ROOF_PARTS = np.arange(11, 16, dtype=np.int32)


def static_support_guard(proposed_indices, source_xyz, source_log_scale,
                         source_quaternions, mesh, target_part_ids, active_part_ids,
                         *, prefix_count, suffix_part_ids):
    """Conservatively preserve physical support independently of any camera.

    Return protected source indices and deterministic metadata. A rotated 4-sigma
    ellipsoid is bounded by its world-axis AABB; overlap with the merged roof
    mesh AABB plus .01 protects the row. Every target with no surviving new
    samples gets the same protection using its own mesh AABB. Exact completion
    part identities add a second veto. Old screen-space labels are NOT ownership.
    This is a conservative AABB test, not an exact ellipsoid/mesh intersection.
    """
    proposed, xyz, log_scale, quaternion, active, suffix = map(np.asarray,
        (proposed_indices, source_xyz, source_log_scale, source_quaternions,
         active_part_ids, suffix_part_ids))
    requested = np.asarray(target_part_ids)
    if requested.ndim != 1 or requested.dtype.kind not in 'iu':
        raise ValueError('Invalid global static-support target IDs')
    allowed = target_parts(requested.tolist())
    total = len(xyz)
    if (proposed.dtype != np.int32 or proposed.ndim != 1
            or np.any(proposed < 0) or np.any(proposed >= total)
            or np.any(proposed[1:] <= proposed[:-1])
            or xyz.shape != (total, 3) or log_scale.shape != xyz.shape
            or quaternion.shape != (total, 4)
            or any(a.dtype.kind != 'f' or not np.isfinite(a).all()
                   for a in (xyz, log_scale, quaternion))
            or type(prefix_count) is not int or not 0 < prefix_count < total
            or suffix.shape != (total-prefix_count,) or suffix.dtype.kind not in 'iu'
            or active.ndim != 1 or active.dtype.kind not in 'iu'
            or not np.isin(active, allowed).all()):
        raise ValueError('Invalid global static-support source, proposal or active part identities')
    vertices, triangles, parts = (np.asarray(mesh[k]) for k in ('vertices', 'triangles', 'part_ids'))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu'
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))
            or parts.shape != (len(triangles),) or parts.dtype.kind not in 'iu'
            or not np.isin(np.union1d(allowed, SUPPORT_ROOF_PARTS), parts).all()):
        raise ValueError('Global static-support guard requires the complete target and roof mesh')
    counts = {str(int(p)): int((active == p).sum()) for p in allowed}
    zero = allowed[np.asarray([counts[str(int(p))] == 0 for p in allowed])]

    def mesh_bounds(ids):
        points = vertices[np.unique(triangles[np.isin(parts, ids)])].astype(np.float64)
        return np.stack((points.min(0), points.max(0)))

    roof_bounds = mesh_bounds(SUPPORT_ROOF_PARTS)
    zero_bounds = [{'part_id': int(p), 'bounds': mesh_bounds([p]).tolist()} for p in zero]
    centers = xyz[proposed].astype(np.float64)
    q = quaternion[proposed].astype(np.float64)
    norms = np.linalg.norm(q, axis=1)
    with np.errstate(over='ignore', under='ignore'):
        scale = np.exp(log_scale[proposed].astype(np.float64))
    if np.any(norms < 1e-8) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Global static-support covariance is nonfinite or degenerate')
    w, x, y, z = (q/norms[:, None]).T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)
    extent = 4*np.sqrt(np.sum(rotation**2*scale[:, None, :]**2, axis=2))
    if not np.isfinite(extent).all():
        raise ValueError('Global static-support covariance overflow')

    def overlap(bounds):
        return np.all((centers+extent >= bounds[0]-.01)
                      & (centers-extent <= bounds[1]+.01), axis=1)

    protected = overlap(roof_bounds)
    for item in zero_bounds:
        protected |= overlap(np.asarray(item['bounds']))
    suffix_rows = proposed >= prefix_count
    protected[suffix_rows] |= np.isin(suffix[proposed[suffix_rows]-prefix_count],
                                      np.union1d(SUPPORT_ROOF_PARTS, zero))
    rows = proposed[protected].copy()
    digest = lambda a: hashlib.sha256(np.asarray(a, dtype='<i4').tobytes()).hexdigest()
    metadata = {'format': SUPPORT_FORMAT, 'roof_part_ids': SUPPORT_ROOF_PARTS.tolist(),
        'target_part_ids': allowed.tolist(), 'zero_added_part_ids': zero.tolist(),
        'active_added_counts': counts, 'sigma': 4., 'padding': .01,
        'geometry_rule': 'rotated_gaussian_support_aabb_intersects_mesh_aabb',
        'suffix_identity_rule': 'exact_part_ids_union_roof_and_zero_added',
        'legacy_screen_labels_used': False, 'source_count': total, 'prefix_count': prefix_count,
        'proposed_retired_count': len(proposed), 'proposed_retired_sha256': digest(proposed),
        'protected_count': len(rows), 'protected_prefix_count': int((rows < prefix_count).sum()),
        'protected_suffix_count': int((rows >= prefix_count).sum()), 'protected_sha256': digest(rows),
        'roof_mesh_bounds': roof_bounds.tolist(), 'zero_added_mesh_bounds': zero_bounds}
    return rows, metadata


def verify_static_support_guard(metadata, proposed_indices, retired_indices, *args, **kwargs):
    """Recompute support from immutable source/mesh and final active sample IDs."""
    protected, expected = static_support_guard(proposed_indices, *args, **kwargs)
    if metadata != expected:
        raise ValueError('Global static-support metadata differs from actual source, mesh or final active parts')
    if np.intersect1d(retired_indices, protected).size:
        raise ValueError('Final retirement removes globally protected roof or zero-addition support')
    return protected


def protected_surface_contract(*records):
    values = [r.get('protected_surface_detail') for r in records]
    if not values or all(v is None for v in values):
        return None
    first = values[0]
    if (not isinstance(first, dict) or set(first) != CONTRACT_FIELDS or first.get('format') != FORMAT
            or any(v != first for v in values)):
        raise ValueError('Protected surface detail contract mismatch')
    for key in ('report', 'extension'):
        if not _absolute(first[key]) or not _checksum(first[key+'_sha256']):
            raise ValueError('Invalid protected surface provenance: '+key)
    return dict(first)


def require_no_protected_surface_detail(metadata, context):
    if protected_surface_contract(metadata) is not None:
        raise ValueError(context+' does not support protected surface detail; refusing to omit geometry')


def target_parts(value):
    if (not isinstance(value, list) or not value or any(type(v) is not int for v in value)
            or value != sorted(set(value)) or value[0] < 0 or value[-1] > 141):
        raise ValueError('Protected surface targets must be sorted unique explicit parts in 0..141')
    return np.asarray(value, np.int32)


def additional_guard_views(metadata, *, required=False):
    """Optional exact 24-view construction guard; no longer independent QA."""
    views = metadata.get('additional_guard_views')
    path = metadata.get('additional_guard_masks')
    checksum = metadata.get('additional_guard_masks_sha256')
    artifact = metadata.get('artifact_sha256', {}).get('additional_guard_masks')
    if views in (None, []) and path is None and checksum is None and artifact is None:
        if required:
            raise ValueError('This package requires the additional 24-view construction guard')
        return []
    plans = [(f, y, 0) for f in (50, 150) for y in (0, 90, 180, 270)]
    plans += [(150, y, p) for p in (-60, 60) for y in range(0, 360, 45)]
    if (not isinstance(views, list) or len(views) != 24
            or not _absolute(path) or not _checksum(checksum) or artifact != checksum
            or any(not isinstance(v, dict) or v.get('view_index') != i
                   or (v.get('frame'), v.get('yaw'), v.get('pitch')) != plans[i]
                   or v.get('zoom') != 1 or v.get('pan_x') != 0 or v.get('pan_y') != 0
                   for i, v in enumerate(views))):
        raise ValueError('Additional construction guard requires exact canonical 24 views and mask fingerprint')
    return views


def validate_report_shape(report):
    names = ('source_gaussian_count', 'retained_gaussian_count', 'retired_gaussian_count',
             'added_gaussian_count', 'rendered_gaussian_count', 'frame_count')
    if (report.get('format') != FORMAT or any(type(report.get(k)) is not int for k in names)
            or not 0 < report['retained_gaussian_count'] < report['source_gaussian_count']
            or not 0 <= report['retired_gaussian_count'] <= report['source_gaussian_count']
            or not 1 <= report['added_gaussian_count'] <= MAX_ADDED
            or report['rendered_gaussian_count'] != report['source_gaussian_count']+report['added_gaussian_count']
            or report['frame_count'] != 400 or report.get('approved_for_export') is not False
            or report.get('unobserved_geometry_is_prior') is not True
            or report.get('unobserved_appearance_is_prior') is not True
            or report.get('variant') != 'guarded_remove_add'
            or not isinstance(report.get('scope'), str) or not report['scope']):
        raise ValueError('Invalid protected surface counts, preview scope or approval')
    target_parts(report.get('allowed_part_ids'))
    for key in ('source', 'parent_baseline', 'parent_candidate', 'child_baseline', 'child_candidate', 'extension'):
        if not _absolute(report.get(key)):
            raise ValueError('Invalid protected surface path: '+key)
    if not _checksum(report.get('source_sha256')) or not _checksum(report.get('extension_sha256')):
        raise ValueError('Invalid protected surface source/extension checksum')
    paths, hashes = report.get('input_paths', {}), report.get('input_sha256', {})
    extras = set(paths)-set(INPUT_FIELDS)
    if (set(paths) != set(hashes) or not set(INPUT_FIELDS).issubset(paths)
            or extras not in (set(), {'additional_guard_masks'})):
        raise ValueError('Incomplete protected surface immutable input identities')
    if extras and (report.get('additional_guard_views_used_for_selection') is not True
                   or report.get('additional_guard_view_count') != 24
                   or report.get('construction_guard_view_count') != 60):
        raise ValueError('Additional guard views must be disclosed as construction/selection data')
    for key in paths:
        if not _absolute(paths[key]) or not _checksum(hashes[key]):
            raise ValueError('Invalid protected surface input provenance: '+key)
    bounds = np.asarray(report.get('retirement_bounds'), np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('Invalid protected surface retirement bounds')


def validate_preview(selection, preview, metadata, frame_count, variant='guarded_remove_add'):
    """Validate final masks/maxima, not the pre-restoration retirement set."""
    selected_meta = json.loads(str(selection['metadata_json']))
    allowed = target_parts(selected_meta.get('target_part_ids'))
    count = len(selection['new_xyz'])
    views = metadata.get('views')
    expected_views = [(f, y, 0) for f in (0, 100, 200, 300) for y in (0, 90, 180, 270)]
    expected_views += [(100, y, 0) for y in (45, 135, 225, 315)]
    expected_views += [(100, y, p) for p in (-35, 35) for y in range(0, 360, 45)]
    if (not isinstance(views, list) or len(views) != 36
            or any(not isinstance(v, dict) or v.get('view_index') != i
                   or (v.get('frame'), v.get('yaw'), v.get('pitch')) != expected_views[i]
                   or v.get('zoom') != 1 or v.get('pan_x') != 0 or v.get('pan_y') != 0
                   for i, v in enumerate(views))):
        raise ValueError('Protected surface requires the exact canonical 36-view guard plan')
    extra_views = additional_guard_views(metadata)
    if (frame_count != 400 or variant != 'guarded_remove_add'
            or selected_meta.get('format') not in ('lego_static_roof_rebuild_v1', 'lego_static_back_detail_proposal_v1')
            or not 0 < count <= MAX_ADDED
            or metadata.get('format') != 'lego_guarded_roof_detail_preview_v1'
            or metadata.get('full_timeline_exposure_scan') is not True
            or metadata.get('front_probe_frames') != list(range(frame_count))
            or metadata.get('all_motion_frozen') is not True
            or metadata.get('source_artifacts_changed') is not False
            or metadata.get('target_part_ids') != allowed.tolist()
            or metadata.get('proposed_new_count') != count):
        raise ValueError('Protected surface requires exact full-400 immutable preview identities')
    state = metadata.get('variants', {}).get(variant, {})
    threshold = metadata.get('threshold')
    if (type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 < threshold <= 1e-4
            or state.get('probe_converged') is not True
            or state.get('interior_restore_converged') is not True):
        raise ValueError('Protected surface exposure/interior guard has not converged')
    history = state.get('interior_restore_history')
    if (not isinstance(history, list) or not history
            or not isinstance(history[-1].get('new_hole_pixels_by_view'), list)
            or len(history[-1]['new_hole_pixels_by_view']) != 36+len(extra_views)
            or not metadata.get('views')
            or any(type(v) is not int or v != 0 for v in history[-1]['new_hole_pixels_by_view'])):
        raise ValueError('Final target-interior guard must show zero new holes at every preview view')
    active, retired, maximum = (preview[variant+suffix] for suffix in
                                ('_active', '_retired_indices', '_maximum_probe_exposure'))
    proposed, original_maximum = preview['proposed_retired_indices'], preview['retirement_maximum_probe_exposure']
    if (active.dtype != bool or active.shape != (count,)
            or maximum.shape != (count,) or maximum.dtype.kind != 'f'
            or not np.isfinite(maximum).all() or np.any(maximum < 0)
            or not active.any() or np.any(maximum[active] > threshold)
            or retired.dtype != np.int32 or retired.ndim != 1
            or np.any(retired < 0) or np.any(retired[1:] <= retired[:-1])
            or proposed.dtype != np.int32 or proposed.ndim != 1
            or np.any(proposed < 0) or np.any(proposed[1:] <= proposed[:-1])
            or not np.array_equal(proposed, selection['retired_indices'])
            or original_maximum.shape != proposed.shape or original_maximum.dtype.kind != 'f'
            or not np.isfinite(original_maximum).all() or np.any(original_maximum < 0)
            or not np.isin(retired, proposed).all()
            or not np.isin(retired, proposed[original_maximum <= threshold]).all()
            or state.get('active_added_count') != int(active.sum())
            or state.get('retired_count') != len(retired)
            or state.get('maximum_final_active_exposure') != float(maximum[active].max())):
        raise ValueError('Final protected surface active/retired rows or actual exposure maxima differ')
    if 'global_support_guard' in state:
        support = state['global_support_guard']
        active_parts = selection['part_ids'][active]
        counts = {str(int(p)): int((active_parts == p).sum()) for p in allowed}
        zero = [int(p) for p in allowed if counts[str(int(p))] == 0]
        if (not isinstance(support, dict) or support.get('format') != SUPPORT_FORMAT
                or support.get('sigma') != 4 or support.get('padding') != .01
                or support.get('roof_part_ids') != SUPPORT_ROOF_PARTS.tolist()
                or support.get('target_part_ids') != allowed.tolist()
                or support.get('zero_added_part_ids') != zero
                or support.get('active_added_counts') != counts
                or support.get('legacy_screen_labels_used') is not False):
            raise ValueError('Global static-support guard must use exact geometry rules and final active part counts')
    # Inactive proposals are omitted, never retained as floating transparent rows.
    arrays = {key: selection[key][active].copy() for key in GEOMETRY_FIELDS}
    arrays['retired_indices'] = retired.copy()
    return arrays, selected_meta


def validate_mesh_support(arrays, mesh, allowed, bounds, tolerance=2e-6):
    vertices, triangles, parts = (mesh[k] for k in ('vertices', 'triangles', 'part_ids'))
    ids, xyz, q = (arrays[k] for k in ('triangle_ids', 'new_xyz', 'new_quaternions'))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu'
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))
            or parts.shape != (len(triangles),) or parts.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(triangles))
            or not np.array_equal(parts[ids], arrays['part_ids'])
            or not np.isin(arrays['part_ids'], allowed).all()):
        raise ValueError('Protected surface triangle/part identities differ')
    target_faces = triangles[np.isin(parts, allowed)]
    if not len(target_faces):
        raise ValueError('Protected surface target mesh is empty')
    all_points = vertices[np.unique(target_faces)].astype(np.float64)
    expected_bounds = np.stack((all_points.min(0)-.01, all_points.max(0)+.01))
    if not np.allclose(bounds, expected_bounds, rtol=0, atol=tolerance):
        raise ValueError('Protected surface bounds differ from true target mesh plus .01')
    corners = vertices[triangles[ids]].astype(np.float64)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    cross = np.cross(b-a, c-a); area = np.linalg.norm(cross, axis=1)
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError('Protected surface uses a degenerate triangle')
    normals = cross/area[:, None]
    distance = np.sum((xyz-a)*normals, axis=1)
    point = xyz-distance[:, None]*normals
    bary = np.column_stack((np.sum(np.cross(b-point, c-point)*normals, axis=1),
                            np.sum(np.cross(c-point, a-point)*normals, axis=1),
                            np.sum(np.cross(a-point, b-point)*normals, axis=1)))/area[:, None]
    edges = np.column_stack((np.linalg.norm(b-c, axis=1), np.linalg.norm(c-a, axis=1), np.linalg.norm(a-b, axis=1)))
    if (np.any(np.abs(distance) > tolerance) or not np.isfinite(bary).all()
            or np.any(bary < -tolerance*edges/area[:, None]) or np.any(np.abs(bary.sum(1)-1) > 1e-6)):
        raise ValueError('Protected surface center lies off its actual triangle')
    w, x, y, z = (q.astype(np.float64)/np.linalg.norm(q, axis=1)[:, None]).T
    normal = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    if np.any(np.sum(normal*normals, axis=1) < .999):
        raise ValueError('Protected surface oriented normal differs from its triangle')


def validate_arrays(arrays, report, source_xyz, eligible, mesh):
    if set(arrays) != ARRAY_FIELDS:
        raise ValueError('Unexpected protected surface array schema')
    n, r, total = (report[k] for k in ('added_gaussian_count', 'retired_gaussian_count', 'source_gaussian_count'))
    schema = {'new_xyz': (np.float32, (n, 3)), 'new_quaternions': (np.float32, (n, 4)),
        'new_log_scale': (np.float32, (n, 3)), 'new_opacity_logit': (np.float32, (n,)),
        'new_dc_gray': (np.float32, (n,)), 'part_ids': (np.int32, (n,)),
        'triangle_ids': (np.int32, (n,)), 'retired_indices': (np.int32, (r,))}
    for key, (dtype, shape) in schema.items():
        value = arrays[key]
        if value.dtype != np.dtype(dtype) or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid protected surface array: '+key)
    retired = arrays['retired_indices']
    if (source_xyz.shape != (total, 3) or not np.isfinite(source_xyz).all()
            or eligible.dtype != bool or eligible.shape != (total,)
            or np.any(retired < 0) or np.any(retired >= total)
            or np.any(retired[1:] <= retired[:-1]) or not eligible[retired].all()):
        raise ValueError('Protected surface retirement touches ineligible source rows')
    if (np.any(np.abs(np.linalg.norm(arrays['new_quaternions'], axis=1)-1) > 1e-4)
            or np.any(arrays['new_log_scale'] < np.float32(np.log(1e-6)))
            or np.any(arrays['new_log_scale'] > np.float32(np.log(.01)))
            or np.any(arrays['new_log_scale'][:, 2] >= np.minimum(arrays['new_log_scale'][:, 0], arrays['new_log_scale'][:, 1]))
            or np.any(arrays['new_dc_gray'] < 0) or np.any(arrays['new_dc_gray'] > 1)
            or np.any(arrays['new_opacity_logit'] < -30) or np.any(arrays['new_opacity_logit'] > 20)):
        raise ValueError('Protected surface geometry/appearance exceeds bounded attributes')
    bounds = np.asarray(report['retirement_bounds'])
    if np.any(source_xyz[retired] < bounds[0]-2e-6) or np.any(source_xyz[retired] > bounds[1]+2e-6):
        raise ValueError('Protected surface retirement escaped its target ROI')
    validate_mesh_support(arrays, mesh, target_parts(report['allowed_part_ids']), bounds)
    meta = arrays['metadata_json']
    if meta.ndim != 0 or meta.dtype.kind not in 'US':
        raise ValueError('Protected surface metadata must be scalar JSON')
    value = json.loads(str(meta))
    if (value.get('format') != FORMAT or value.get('source_sha256') != report['source_sha256']
            or value.get('selection_sha256') != report['input_sha256']['selection']
            or value.get('preview_npz_sha256') != report['input_sha256']['preview_npz']):
        raise ValueError('Protected surface extension metadata/source differs')


def verify_protected_surface_detail(cache, correction=None, audit=None, *, baseline=None):
    records = [cache]+([correction] if correction is not None else [])+([audit] if audit is not None else [])
    contract = protected_surface_contract(*records)
    if contract is None:
        return None
    if any(r.get(k) is not None for r in records for k in ('gaussian_cleanup', 'static_surface_rebuild')):
        raise ValueError('Protected surface cannot stack with legacy geometry/cleanup overlays')
    for key in ('report', 'extension'):
        if _sha256(contract[key]) != contract[key+'_sha256']:
            raise ValueError('Protected surface checksum mismatch: '+key)
    report = json.loads(Path(contract['report']).read_text())
    validate_report_shape(report)
    paths, hashes = report['input_paths'], report['input_sha256']
    for key in paths:
        if _sha256(paths[key]) != hashes[key]:
            raise ValueError('Protected surface input checksum mismatch: '+key)
    parent_baseline, parent_candidate = Path(report['parent_baseline']), Path(report['parent_candidate'])
    expected_paths = {'source_ply': report['source'], 'baseline_cache_metadata': parent_baseline/'cache.json',
        'correction': parent_candidate/'correction.json', 'event_audit': parent_candidate/'event-audit.json',
        'baseline_poses': parent_baseline/'poses.npy', 'dynamic_indices': parent_baseline/'dynamic_indices.npy'}
    if any(Path(paths[k]) != Path(v) for k, v in expected_paths.items()):
        raise ValueError('Protected surface parent paths do not match report')
    from scripts.events.export_lego_event_preview import verify_candidate
    parent_cache, parent_record, parent_audit = verify_candidate(parent_baseline, parent_candidate, allow_unapproved=True)
    parents = [parent_cache, parent_record, parent_audit]
    if any(r.get(k) is not None for r in parents for k in
           ('protected_surface_detail', 'gaussian_cleanup', 'static_surface_rebuild')):
        raise ValueError('Protected surface parent must be the unmodified color-residual stage')
    from utils.color_residual import color_residual_contract
    color = color_residual_contract(*records, *parents)
    if color is None or paths['color_residual'] != color['residual'] or paths['color_residual_report'] != color['report']:
        raise ValueError('Protected surface requires the exact current color residual')
    if hashes['color_residual'] != color['residual_sha256'] or hashes['color_residual_report'] != color['report_sha256']:
        raise ValueError('Protected surface color residual checksums differ')
    from utils.multiview_completion import completion_contract
    completion = completion_contract(*records, *parents)
    if completion is None:
        raise ValueError('Protected surface requires inherited compact completion')
    completed = json.loads(Path(completion['report']).read_text())
    rebuild = json.loads(Path(completed['parent_report']).read_text())
    for key, expected, checksum in (('mesh', completed['mesh_prior'], completed['mesh_prior_sha256']),
                                  ('parts', completed['parts'], completed['parts_sha256']),
                                  ('binding', rebuild['binding'], rebuild['binding_sha256'])):
        if paths[key] != expected or hashes[key] != checksum:
            raise ValueError('Protected surface changed parent geometry identity: '+key)
    if (report['source'] != cache.get('source') or report['source_sha256'] != cache.get('source_sha256')
            or report['source_sha256'] != hashes['source_ply']
            or report['source_gaussian_count'] != cache.get('gaussian_count')
            or report['rendered_gaussian_count'] != cache.get('rendered_gaussian_count')
            or report['retained_gaussian_count'] != rebuild['retained_gaussian_count']
            or report['frame_count'] != cache.get('frame_count')
            or report['extension'] != contract['extension'] or report['extension_sha256'] != contract['extension_sha256']):
        raise ValueError('Protected surface child/source/extension identities differ')
    for key in ('source', 'source_sha256', 'gaussian_count', 'dynamic_count', 'rendered_dynamic_count', 'frame_count', 'fps', 'camera'):
        if cache.get(key) != parent_cache.get(key):
            raise ValueError('Protected surface altered inherited cache: '+key)
    from utils.render_sampling import require_same_sampling
    require_same_sampling(*records, *parents)
    if correction is not None:
        stripped = copy.deepcopy(correction); stripped.pop('protected_surface_detail', None)
        if _motion_identity(stripped) != _motion_identity(parent_record):
            raise ValueError('Protected surface changed inherited motion or color record')
    if baseline is not None:
        baseline = Path(baseline)
        if baseline.resolve() != Path(report['child_baseline']).resolve():
            raise ValueError('Protected surface loaded from a different child baseline')
        if correction is not None and correction['joint_structure']['input_sha256'].get('baseline_cache_metadata') != _sha256(baseline/'cache.json'):
            raise ValueError('Protected surface correction cache fingerprint differs')
        for filename, key in (('poses.npy', 'baseline_poses'), ('dynamic_indices.npy', 'dynamic_indices')):
            child = baseline/filename
            if child.resolve() != Path(paths[key]).resolve() and _sha256(child) != hashes[key]:
                raise ValueError('Protected surface changed inherited trajectory: '+filename)
    allowed = target_parts(report['allowed_part_ids'])
    structure = parent_record['joint_structure']
    root_ids = np.flatnonzero(np.asarray(structure['parents']) == -1)
    mapping = {p: g['group'] for g in structure['source_groups'] for p in g['objects']}
    groups = np.asarray([mapping.get(int(p), -1) for p in allowed])
    if not np.isin(groups, root_ids).all():
        raise ValueError('Protected surface target is not an identity-root source part')
    from utils.event_refinement import correction_from_record
    import torch
    motion = correction_from_record(parent_record, 'cpu')
    with torch.no_grad():
        for frame in range(report['frame_count']):
            q, t = motion.group_transforms(frame)
            if (not np.allclose(q.numpy()[groups], [1, 0, 0, 0], rtol=0, atol=1e-7)
                    or not np.allclose(t.numpy()[groups], 0, rtol=0, atol=1e-7)):
                raise ValueError('Protected surface target moves in inherited trajectory')
    with np.load(paths['binding'], allow_pickle=False) as data:
        labels, bound_indices = data['labels'], data['dynamic_indices']
    dynamic = np.load(paths['dynamic_indices'], allow_pickle=False)
    if not np.array_equal(dynamic, bound_indices):
        raise ValueError('Protected surface baked/binding identities differ')
    with np.load(paths['parts'], allow_pickle=False) as data:
        part_meta = json.loads(str(data['metadata_json']))
    if np.isin(allowed, part_meta.get('deforming_objects', [])).any():
        raise ValueError('Protected surface includes elastic parts')
    eligible = cleanup_eligible_mask(cache['gaussian_count'], rebuild['retained_gaussian_count'],
        labels, structure, dynamic, part_meta.get('deforming_objects', []))
    with np.load(completion['extension'], allow_pickle=False) as data:
        suffix_parts, suffix_groups = data['part_ids'], data['group_ids']
    eligible[completed['prefix_count']:] = np.isin(suffix_parts, allowed) & np.isin(suffix_groups, root_ids)
    with np.load(paths['selection'], allow_pickle=False) as data:
        selection = {k: data[k] for k in data.files}
    preview_json = json.loads(Path(paths['preview_json']).read_text())
    needed = ('proposed_retired_indices', 'retirement_maximum_probe_exposure', 'metadata_json',
              report['variant']+'_active', report['variant']+'_retired_indices', report['variant']+'_maximum_probe_exposure')
    with np.load(paths['preview_npz'], allow_pickle=False) as data:
        preview = {k: data[k] for k in needed}
    if json.loads(str(preview['metadata_json'])) != preview_json:
        raise ValueError('Protected surface preview JSON and NPZ metadata disagree')
    expected, proposal_meta = validate_preview(selection, preview, preview_json, cache['frame_count'], report['variant'])
    extra_views = additional_guard_views(preview_json)
    has_extra = 'additional_guard_masks' in paths
    if bool(extra_views) != has_extra:
        raise ValueError('Protected surface report omits or invents additional construction guard inputs')
    if has_extra:
        if (paths['additional_guard_masks'] != preview_json['additional_guard_masks']
                or hashes['additional_guard_masks'] != preview_json['additional_guard_masks_sha256']):
            raise ValueError('Additional construction guard mask path/hash differs from exact preview')
        with np.load(paths['additional_guard_masks'], allow_pickle=False) as data:
            extra_meta = json.loads(str(data['metadata_json']))
            extra_parts, extra_depth, extra_frames = data['part_ids'], data['depth'], data['frame_indices']
        if (extra_meta.get('views') != extra_views or extra_meta.get('camera') != cache['camera']
                or extra_meta.get('mesh') != paths['mesh'] or extra_meta.get('mesh_sha256') != hashes['mesh']
                or extra_meta.get('objects') != part_meta['objects']):
            raise ValueError('Additional construction masks differ from inherited mesh, parts or exact cameras')
        from scripts.geometry.render_lego_multiview import validate_mask_arrays, validate_view_plan, validate_motion_metadata
        validate_view_plan(extra_meta, extra_frames, cache)
        validate_mask_arrays(extra_parts, extra_depth, extra_meta)
        del extra_parts, extra_depth
        motion_path = extra_meta.get('motion')
        if (not _absolute(motion_path) or not _checksum(extra_meta.get('motion_sha256'))
                or _sha256(motion_path) != extra_meta['motion_sha256']):
            raise ValueError('Additional guard motion artifact hash differs')
        with np.load(motion_path, allow_pickle=False) as data:
            motion_meta = json.loads(str(data['metadata_json']))
            motion_q, motion_t = data['quaternions'], data['translations']
            motion_groups = data['group_ids']
        validate_motion_metadata(extra_meta, motion_meta, parent_cache, parent_record,
            hashes['baseline_cache_metadata'], hashes['correction'], completed)
        if (motion_q.shape != (400, len(structure['parents']), 4)
                or motion_t.shape != (400, len(structure['parents']), 3)
                or motion_groups.shape != (len(part_meta['objects']),)
                or not np.array_equal(motion_groups, np.asarray([mapping.get(p, -1) for p in range(len(motion_groups))]))
                or not np.isfinite(motion_q).all() or not np.isfinite(motion_t).all()):
            raise ValueError('Additional guard motion arrays do not identify the inherited rigid groups')
        with torch.no_grad():
            for frame in sorted({view['frame'] for view in extra_views}):
                q, t = motion.group_transforms(frame)
                if (not np.allclose(motion_q[frame], q.numpy(), rtol=0, atol=1e-6)
                        or not np.allclose(motion_t[frame], t.numpy(), rtol=0, atol=1e-6)):
                    raise ValueError('Additional guard cameras used a different inherited motion')
    from utils.viewer_state import orbit_pose
    for view in preview_json['views']+extra_views:
        pose = np.asarray(view.get('camera_pose'), np.float64)
        actual = orbit_pose(cache['camera']['pose'], cache['camera']['target'], view)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose, actual, rtol=0, atol=1e-7):
            raise ValueError('Protected surface preview camera differs from inherited canonical orbit')
    if (proposal_meta.get('source_sha256') != hashes['source_ply']
            or proposal_meta.get('source_gaussian_count') != cache['gaussian_count']
            or proposal_meta.get('target_part_ids') != report['allowed_part_ids']
            or proposal_meta.get('input_sha256', {}).get('mesh') != hashes['mesh']
            or preview_json.get('baseline') != str(parent_baseline) or preview_json.get('candidate') != str(parent_candidate)):
        raise ValueError('Protected surface selection or preview identifies another parent')
    for key in ('source_ply', 'correction', 'baseline_cache_metadata', 'selection', 'color_residual', 'color_residual_report', 'mesh'):
        if preview_json.get('artifact_sha256', {}).get(key) != hashes[key]:
            raise ValueError('Protected surface preview input mismatch: '+key)
    with np.load(contract['extension'], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    for key, value in expected.items():
        if key not in arrays or arrays[key].dtype != value.dtype or not np.array_equal(arrays[key], value):
            raise ValueError('Protected surface extension differs from final guarded preview: '+key)
    from plyfile import PlyData
    source = PlyData.read(report['source'])['vertex'].data
    source_arrays = {key: np.column_stack([source[f] for f in fields]) for key, fields in (
        ('source_xyz', ('x', 'y', 'z')), ('source_quaternions', tuple(f'rot_{i}' for i in range(4))),
        ('source_log_scale', tuple(f'scale_{i}' for i in range(3))), ('source_dc', tuple(f'f_dc_{i}' for i in range(3))))}
    source_arrays['source_opacity'] = np.asarray(source['opacity']).copy()
    if np.isnan(source_arrays['source_opacity']).any() or any(not np.isfinite(v).all() for k, v in source_arrays.items() if k != 'source_opacity'):
        raise ValueError('Invalid protected surface immutable source attributes')
    with np.load(paths['mesh'], allow_pickle=False) as data:
        mesh = {k: data[k] for k in ('vertices', 'triangles', 'part_ids')}
    validate_arrays(arrays, report, source_arrays['source_xyz'], eligible, mesh)
    support = preview_json['variants'][report['variant']].get('global_support_guard')
    if support is not None:
        verify_static_support_guard(support, selection['retired_indices'], arrays['retired_indices'],
            source_arrays['source_xyz'], source_arrays['source_log_scale'],
            source_arrays['source_quaternions'], mesh, report['allowed_part_ids'], arrays['part_ids'],
            prefix_count=completed['prefix_count'], suffix_part_ids=suffix_parts)
    with np.load(paths['color_residual'], allow_pickle=False) as data:
        residual = {k: data[k] for k in ('indices', 'delta_gray')}
    if audit is not None:
        fingerprints = {'surface_detail_report': contract['report_sha256'], 'surface_detail_extension': contract['extension_sha256'],
            **{k: hashes[k] for k in ('source_ply', 'baseline_poses', 'dynamic_indices', 'color_residual', 'color_residual_report')}}
        child_cache_file = Path(report['child_baseline'])/'cache.json'
        child_correction_file = Path(report['child_candidate'])/'correction.json'
        actual_cache = json.loads(child_cache_file.read_text())
        actual_correction = json.loads(child_correction_file.read_text())
        child_motion = copy.deepcopy(actual_correction); child_motion.pop('protected_surface_detail', None)
        if (actual_cache != cache or protected_surface_contract(cache, actual_correction) != contract
                or (correction is not None and actual_correction != correction)
                or _motion_identity(child_motion) != _motion_identity(parent_record)):
            raise ValueError('Protected surface audit does not identify exact child cache/correction')
        fingerprints['baseline_cache_metadata'] = _sha256(child_cache_file)
        fingerprints['correction'] = _sha256(child_correction_file)
        for key, value in fingerprints.items():
            if audit.get('artifact_sha256', {}).get(key) != value:
                raise ValueError('Protected surface fresh audit fingerprint differs: '+key)
    return {'report': report, 'arrays': arrays, 'residual': residual, **source_arrays}


class ProtectedSurfaceDetail:
    """Append fixed static detail after inherited motion; preserve optimized prefix."""
    def __init__(self, verified, device='cpu'):
        import torch
        self.report, self.source = verified['report'], verified
        self.source_count = self.report['source_gaussian_count']
        self.new_count = self.report['added_gaussian_count']
        self.rendered_count = self.report['rendered_gaussian_count']
        arrays = verified['arrays']
        self.xyz = torch.as_tensor(arrays['new_xyz'], device=device).clone()
        self.quaternions = torch.as_tensor(arrays['new_quaternions'], device=device).clone()
        self.log_scale = torch.as_tensor(arrays['new_log_scale'], device=device).clone()
        self.opacity = torch.as_tensor(arrays['new_opacity_logit'], device=device).reshape(-1, 1).clone()
        gray = torch.as_tensor(arrays['new_dc_gray'], device=device)
        self.dc = ((gray-.5)/C0).reshape(-1, 1, 1).expand(-1, 1, 3).clone()
        self.retired = torch.as_tensor(arrays['retired_indices'].astype(np.int64), device=device)
        self._applied = False

    def apply_appearance(self, model):
        import torch
        if self._applied:
            raise ValueError('Protected surface must be applied exactly once after the parent color residual')
        for attribute, key, shape in (('_xyz', 'source_xyz', (-1, 3)), ('_rotation', 'source_quaternions', (-1, 4)),
                ('_scaling', 'source_log_scale', (-1, 3)), ('_opacity', 'source_opacity', (-1, 1)),
                ('_features_dc', 'source_dc', (-1, 1, 3))):
            actual = getattr(model, attribute).detach()
            expected = torch.as_tensor(self.source[key], dtype=actual.dtype, device=actual.device).reshape(shape)
            if attribute == '_features_dc':
                rows = torch.as_tensor(self.source['residual']['indices'].astype(np.int64), device=actual.device)
                delta = torch.as_tensor(self.source['residual']['delta_gray'], dtype=actual.dtype, device=actual.device)
                expected = expected.index_copy(0, rows, expected[rows]+delta[:, None, None]/C0)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError('Protected surface runtime differs from exact optimized parent: '+attribute)
        rest = model._features_rest.detach()
        if rest.ndim != 3 or rest.shape[0] != self.source_count or rest.shape[2] != 3 or torch.count_nonzero(rest).item():
            raise ValueError('Protected surface requires DC-only parent attributes')
        opacity = model._opacity.detach().clone()
        rows = self.retired.to(opacity.device)
        opacity[rows] = torch.minimum(opacity[rows], opacity.new_full((len(rows), 1), -30.))
        model._opacity = torch.cat((opacity, self.opacity.to(opacity)), 0)
        model._scaling = torch.cat((model._scaling.detach(), self.log_scale.to(model._scaling)), 0)
        model._features_dc = torch.cat((model._features_dc.detach(), self.dc.to(model._features_dc)), 0)
        model._features_rest = torch.cat((rest, rest.new_zeros((self.new_count, rest.shape[1], 3))), 0)
        self._applied, self.source = True, None

    def append_geometry(self, xyz, quaternion):
        import torch
        if (not self._applied or xyz.shape != (self.source_count, 3) or quaternion.shape != (self.source_count, 4)
                or xyz.device != quaternion.device or xyz.dtype != quaternion.dtype):
            raise ValueError('Protected surface requires inherited posed geometry once after appearance setup')
        return torch.cat((xyz, self.xyz.to(xyz)), 0), torch.cat((quaternion, self.quaternions.to(quaternion)), 0)
