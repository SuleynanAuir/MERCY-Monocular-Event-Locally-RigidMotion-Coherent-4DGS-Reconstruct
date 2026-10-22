"""Explicit, audited static roof replacement on an immutable Gaussian source.

Only known CAD roof parts 11..15 can supply new points. Retired source rows stay
in place with suppressed opacity; all inherited motion and other rows survive.
This is a geometry prior, not a claim of newly observed monocular back surfaces.
"""
import json
from pathlib import Path

import numpy as np

from utils.gaussian_cleanup import (_absolute, _checksum, _motion_identity, _sha256,
                                    cleanup_contract, cleanup_eligible_mask)


FORMAT = 'lego_static_surface_rebuild_v1'
PART_IDS = (11, 12, 13, 14, 15)
MAX_ADDED = 100000
C0 = 0.28209479177387814
CONTRACT_FIELDS = {'format', 'report', 'report_sha256', 'extension', 'extension_sha256'}
ARRAY_FIELDS = {'retired_indices', 'new_xyz', 'new_quaternions', 'new_log_scale',
                'new_opacity_logit', 'new_dc_gray', 'part_ids', 'triangle_ids', 'metadata_json'}


def static_surface_contract(*records):
    values = [record.get('static_surface_rebuild') for record in records]
    if not values or all(value is None for value in values):
        return None
    first = values[0]
    if (not isinstance(first, dict) or set(first) != CONTRACT_FIELDS or first.get('format') != FORMAT
            or any(value != first for value in values)):
        raise ValueError('Static surface rebuild contract mismatch')
    for key in ('report', 'extension'):
        if not _absolute(first[key]) or not _checksum(first[key + '_sha256']):
            raise ValueError('Invalid static surface rebuild provenance: ' + key)
    return dict(first)


def require_no_static_surface_rebuild(metadata, context):
    if static_surface_contract(metadata) is not None:
        raise ValueError(f'{context} does not support static surface rebuild; export would omit rebuilt geometry')


def validate_report_shape(report):
    counts = ('source_gaussian_count', 'retained_gaussian_count', 'retired_gaussian_count',
              'added_gaussian_count', 'rendered_gaussian_count')
    if (report.get('format') != FORMAT or any(type(report.get(key)) is not int for key in counts)
            or not 0 < report['retained_gaussian_count'] < report['source_gaussian_count']
            or not 0 < report['retired_gaussian_count'] <= report['source_gaussian_count']
            or not 1 <= report['added_gaussian_count'] <= MAX_ADDED
            or report['rendered_gaussian_count'] != report['source_gaussian_count'] + report['added_gaussian_count']
            or report.get('allowed_part_ids') != list(PART_IDS)
            or report.get('unobserved_geometry_is_prior') is not True
            or not isinstance(report.get('scope'), str) or not report['scope']):
        raise ValueError('Invalid static surface rebuild report counts or scope')
    for field in ('source', 'mesh_prior', 'parts', 'selection', 'extension'):
        if not _absolute(report.get(field)) or not _checksum(report.get(field + '_sha256')):
            raise ValueError('Invalid static surface report provenance: ' + field)
    for field in ('parent_baseline', 'parent_candidate'):
        if not _absolute(report.get(field)):
            raise ValueError('Invalid static surface parent path: ' + field)
    for key in ('parent_baseline_cache_sha256', 'parent_correction_sha256', 'parent_event_audit_sha256',
                'parent_poses_sha256', 'parent_dynamic_indices_sha256'):
        if not _checksum(report.get(key)):
            raise ValueError('Invalid static surface parent checksum: ' + key)
    bounds = np.asarray(report.get('retirement_bounds'), dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('Invalid static roof retirement bounds')


def retirement_eligible_mask(source_count, retained_count, labels, structure, dynamic_indices,
                             deforming_parts, completion_arrays, prefix_count):
    eligible = cleanup_eligible_mask(source_count, retained_count, labels, structure,
                                     dynamic_indices, deforming_parts)
    parts, groups = completion_arrays['part_ids'], completion_arrays['group_ids']
    parents = np.asarray(structure['parents'])
    if (type(prefix_count) is not int or prefix_count != len(labels)
            or parts.shape != (source_count-prefix_count,) or groups.shape != parts.shape
            or parts.dtype != np.dtype('int32') or groups.dtype != np.dtype('int32')):
        raise ValueError('Static roof completion suffix identities disagree')
    roots = np.flatnonzero(parents == -1)
    root_parts = [part for group in structure['source_groups'] if group['group'] in roots for part in group['objects']]
    if not set(PART_IDS).issubset(root_parts):
        raise ValueError('Every rebuilt roof part must belong to a static root group')
    roof = np.isin(parts, PART_IDS)
    if np.any(roof & ~np.isin(groups, roots)):
        raise ValueError('Roof completion rows are not static root rows')
    eligible[prefix_count:] = roof & np.isin(groups, roots)
    required_retired = (prefix_count + np.flatnonzero(eligible[prefix_count:])).astype(np.int32)
    return eligible, required_retired


def validate_mesh_support(xyz, quaternion, part_ids, triangle_ids, mesh, bounds, tolerance=2e-6):
    """Verify true roof-triangle support, barycentric containment and normals."""
    vertices, triangles, face_parts = mesh['vertices'], mesh['triangles'], mesh['part_ids']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu'
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))
            or face_parts.shape != (len(triangles),) or face_parts.dtype.kind not in 'iu'
            or np.any(triangle_ids < 0) or np.any(triangle_ids >= len(triangles))):
        raise ValueError('Invalid static roof mesh/triangle identity')
    if not np.array_equal(face_parts[triangle_ids], part_ids) or not np.isin(part_ids, PART_IDS).all():
        raise ValueError('Static roof points reference a nonroof or mismatched triangle part')
    roof_faces = triangles[np.isin(face_parts, PART_IDS)]
    if not len(roof_faces):
        raise ValueError('The source mesh has no roof triangles')
    roof_vertices = vertices[np.unique(roof_faces)].astype(np.float64)
    expected_bounds = np.stack((roof_vertices.min(0)-.01, roof_vertices.max(0)+.01))
    if not np.allclose(bounds, expected_bounds, rtol=0, atol=tolerance):
        raise ValueError('Roof retirement bounds must match actual roof mesh AABB plus 0.01')
    corners = vertices[triangles[triangle_ids]].astype(np.float64)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    cross = np.cross(b-a, c-a)
    area2 = np.linalg.norm(cross, axis=1)
    if np.any(area2 <= 0) or not np.isfinite(area2).all():
        raise ValueError('Static roof point references a degenerate triangle')
    normals = cross/area2[:, None]
    plane = np.sum((xyz.astype(np.float64)-a)*normals, axis=1)
    projected = xyz.astype(np.float64)-plane[:, None]*normals
    bary = np.column_stack((np.sum(np.cross(b-projected, c-projected)*normals, axis=1),
                            np.sum(np.cross(c-projected, a-projected)*normals, axis=1),
                            np.sum(np.cross(a-projected, b-projected)*normals, axis=1)))/area2[:, None]
    opposite_edges = np.column_stack((np.linalg.norm(b-c, axis=1), np.linalg.norm(c-a, axis=1),
                                      np.linalg.norm(a-b, axis=1)))
    bary_tolerance = tolerance*opposite_edges/area2[:, None]
    if (np.any(np.abs(plane) > tolerance) or not np.isfinite(bary).all()
            or np.any(bary < -bary_tolerance) or np.any(np.abs(bary.sum(1)-1) > 1e-6)):
        raise ValueError('New static Gaussian is not on its actual roof triangle')
    q = quaternion.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    local_z = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    if np.any(np.abs(np.sum(local_z*normals, axis=1)) <= .999):
        raise ValueError('New static Gaussian quaternion normal disagrees with roof triangle')


def validate_extension_arrays(arrays, report, source_xyz, source_opacity, eligible, required_retired, mesh):
    if set(arrays) != ARRAY_FIELDS:
        raise ValueError('Static roof extension has unexpected or missing arrays')
    count, retired_count = report['added_gaussian_count'], report['retired_gaussian_count']
    retired = arrays['retired_indices']
    if (retired.dtype != np.dtype('int32') or retired.shape != (retired_count,)
            or np.any(retired < 0) or np.any(retired >= report['source_gaussian_count'])
            or np.any(retired[1:] <= retired[:-1])):
        raise ValueError('Static roof retired rows must be strictly sorted unique int32')
    expected = {'new_xyz': (np.float32, (count, 3)), 'new_quaternions': (np.float32, (count, 4)),
                'new_log_scale': (np.float32, (count, 3)), 'new_opacity_logit': (np.float32, (count,)),
                'new_dc_gray': (np.float32, (count,)), 'part_ids': (np.int32, (count,)),
                'triangle_ids': (np.int32, (count,))}
    for key, (dtype, shape) in expected.items():
        value = arrays[key]
        if value.dtype != np.dtype(dtype) or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid static roof extension array: ' + key)
    if (source_xyz.shape != (report['source_gaussian_count'], 3) or not np.isfinite(source_xyz).all()
            or source_opacity.shape != (len(source_xyz),) or np.isnan(source_opacity).any()
            or eligible.dtype != np.dtype(bool) or eligible.shape != (len(source_xyz),)):
        raise ValueError('Invalid static roof source/eligibility arrays')
    if not eligible[retired].all() or not np.isin(required_retired, retired).all():
        raise ValueError('Retirement must protect dynamic/nonroof suffix rows and retire every old roof suffix row')
    if np.any(source_opacity[retired] < -30):
        raise ValueError('Retirement must not increase an already lower source opacity')
    bounds = np.asarray(report['retirement_bounds'], np.float64)
    for name, points in (('retired', source_xyz[retired]), ('new', arrays['new_xyz'])):
        if np.any(points < bounds[0]-2e-6) or np.any(points > bounds[1]+2e-6):
            raise ValueError(name + ' static roof rows lie outside the explicit retirement ROI')
    if np.any(np.abs(np.linalg.norm(arrays['new_quaternions'], axis=1)-1) > 1e-4):
        raise ValueError('Static roof new quaternions must have unit norm')
    scales = arrays['new_log_scale']
    if np.any(scales < np.float32(np.log(1e-6))) or np.any(scales > np.float32(np.log(.01))):
        raise ValueError('Static roof scale must be in [1e-6, .01] scene units')
    if np.any(arrays['new_dc_gray'] < 0) or np.any(arrays['new_dc_gray'] > 1):
        raise ValueError('Static roof DC gray must lie in [0,1]')
    validate_mesh_support(arrays['new_xyz'], arrays['new_quaternions'], arrays['part_ids'],
                          arrays['triangle_ids'], mesh, bounds)
    metadata = arrays['metadata_json']
    if metadata.ndim != 0 or metadata.dtype.kind not in 'US':
        raise ValueError('Static roof extension requires scalar JSON metadata')
    meta = json.loads(str(metadata))
    if not isinstance(meta, dict) or meta.get('format') != FORMAT or meta.get('source_sha256') != report['source_sha256']:
        raise ValueError('Static roof extension metadata/source mismatch')


def verify_static_surface_rebuild(cache, correction=None, audit=None, *, baseline=None):
    records = [cache]+([correction] if correction is not None else [])+([audit] if audit is not None else [])
    contract = static_surface_contract(*records)
    if contract is None:
        return None
    if any(record.get('gaussian_cleanup') is not None for record in records):
        raise ValueError('Static roof rebuild cannot be combined with a cleanup overlay')
    for key in ('report', 'extension'):
        if _sha256(contract[key]) != contract[key+'_sha256']:
            raise ValueError('Static roof checksum mismatch: '+key)
    report = json.loads(Path(contract['report']).read_text())
    validate_report_shape(report)
    if (report['source'] != cache.get('source') or report['source_sha256'] != cache.get('source_sha256')
            or report['source_gaussian_count'] != cache.get('gaussian_count')
            or report['rendered_gaussian_count'] != cache.get('rendered_gaussian_count')
            or report['extension'] != contract['extension'] or report['extension_sha256'] != contract['extension_sha256']):
        raise ValueError('Static roof report/cache/extension identity mismatch')
    parent_baseline, parent_candidate = Path(report['parent_baseline']), Path(report['parent_candidate'])
    files = {key+'_sha256': report[key] for key in ('source', 'mesh_prior', 'parts', 'selection')}
    files.update(parent_baseline_cache_sha256=parent_baseline/'cache.json',
                 parent_correction_sha256=parent_candidate/'correction.json',
                 parent_event_audit_sha256=parent_candidate/'event-audit.json',
                 parent_poses_sha256=parent_baseline/'poses.npy',
                 parent_dynamic_indices_sha256=parent_baseline/'dynamic_indices.npy')
    for key, path in files.items():
        if _sha256(path) != report[key]:
            raise ValueError('Static roof provenance checksum mismatch: '+key)
    parents = [json.loads(path.read_text()) for path in (parent_baseline/'cache.json',
                 parent_candidate/'correction.json', parent_candidate/'event-audit.json')]
    parent_cache, parent_record, parent_audit = parents
    if cleanup_contract(*parents) is not None or static_surface_contract(*parents) is not None:
        raise ValueError('Static roof parent must have neither cleanup nor prior static rebuild')
    from scripts.events.export_lego_event_preview import verify_candidate
    from utils.multiview_completion import completion_contract
    from utils.render_sampling import require_same_sampling
    verify_candidate(parent_baseline, parent_candidate, allow_unapproved=True)
    completion = completion_contract(cache, *parents, *records[1:])
    if completion is None:
        raise ValueError('Static roof requires the unmodified compact-completion parent')
    completed = json.loads(Path(completion['report']).read_text())
    rebuild = json.loads(Path(completed['parent_report']).read_text())
    if (report['retained_gaussian_count'] != rebuild['retained_gaussian_count']
            or any(report[key] != completed[key] or report[key+'_sha256'] != completed[key+'_sha256']
                   for key in ('mesh_prior', 'parts'))):
        raise ValueError('Static roof changed immutable parent geometry/part provenance')
    for key in ('source', 'source_sha256', 'gaussian_count', 'dynamic_count', 'rendered_dynamic_count',
                'frame_count', 'fps', 'camera'):
        if cache.get(key) != parent_cache.get(key):
            raise ValueError('Static roof changed inherited cache identity: '+key)
    require_same_sampling(*records, *parents)
    if correction is not None and _motion_identity(correction) != _motion_identity(parent_record):
        # The common identity helper only removes gaussian_cleanup, so strip
        # exactly this new contract before comparing the otherwise whole record.
        child = dict(correction); child.pop('static_surface_rebuild', None)
        if _motion_identity(child) != _motion_identity(parent_record):
            raise ValueError('Static roof changed inherited hard-joint motion')
    if baseline is not None:
        if correction is not None and correction.get('joint_structure', {}).get('input_sha256', {}).get(
                'baseline_cache_metadata') != _sha256(Path(baseline)/'cache.json'):
            raise ValueError('Static roof correction cache fingerprint mismatch')
        for filename, key in (('poses.npy', 'parent_poses_sha256'), ('dynamic_indices.npy', 'parent_dynamic_indices_sha256')):
            child = Path(baseline)/filename
            if child.resolve() != (parent_baseline/filename).resolve() and _sha256(child) != report[key]:
                raise ValueError('Static roof changed baked parent motion: '+filename)
    if not _absolute(rebuild.get('binding')) or not _checksum(rebuild.get('binding_sha256')):
        raise ValueError('Static roof requires a hashed parent binding')
    if _sha256(rebuild['binding']) != rebuild['binding_sha256']:
        raise ValueError('Static roof parent binding checksum mismatch')
    with np.load(rebuild['binding'], allow_pickle=False) as data:
        labels, bound_indices = data['labels'], data['dynamic_indices']
    dynamic = np.load(parent_baseline/'dynamic_indices.npy', allow_pickle=False)
    if not np.array_equal(dynamic, bound_indices):
        raise ValueError('Static roof binding and parent baked rows disagree')
    with np.load(completed['parts'], allow_pickle=False) as data:
        part_meta = json.loads(str(data['metadata_json']))
    if np.any(labels < 0) or np.any(labels >= len(part_meta['objects'])):
        raise ValueError('Static roof binding references unknown source parts')
    with np.load(completion['extension'], allow_pickle=False) as data:
        completion_arrays = {key: data[key] for key in ('part_ids', 'group_ids')}
    eligible, required_retired = retirement_eligible_mask(report['source_gaussian_count'], report['retained_gaussian_count'],
        labels, parent_record['joint_structure'], dynamic, part_meta.get('deforming_objects', []),
        completion_arrays, completed['prefix_count'])
    from plyfile import PlyData
    source = PlyData.read(report['source'])['vertex'].data
    if len(source) != report['source_gaussian_count']:
        raise ValueError('Static roof source PLY count differs')
    source_arrays = {key: np.column_stack([source[field] for field in fields]) for key, fields in (
        ('source_xyz', ('x', 'y', 'z')), ('source_quaternions', tuple(f'rot_{i}' for i in range(4))),
        ('source_log_scale', tuple(f'scale_{i}' for i in range(3))), ('source_dc', tuple(f'f_dc_{i}' for i in range(3))))}
    source_arrays['source_opacity'] = np.asarray(source['opacity']).copy()
    if any(not np.isfinite(value).all() for key, value in source_arrays.items() if key != 'source_opacity'):
        raise ValueError('Static roof source has nonfinite geometry or DC')
    with np.load(report['mesh_prior'], allow_pickle=False) as data:
        mesh = {key: data[key] for key in ('vertices', 'triangles', 'part_ids')}
    with np.load(contract['extension'], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_extension_arrays(arrays, report, source_arrays['source_xyz'], source_arrays['source_opacity'],
                              eligible, required_retired, mesh)
    with np.load(report['selection'], allow_pickle=False) as selected:
        for key in ARRAY_FIELDS-{'metadata_json'}:
            if key not in selected or selected[key].dtype != arrays[key].dtype or not np.array_equal(selected[key], arrays[key]):
                raise ValueError('Static roof extension differs from exact selection: '+key)
    if audit is not None:
        for key, expected in (('static_surface_report', contract['report_sha256']),
                              ('static_surface_extension', contract['extension_sha256']),
                              ('source_ply', report['source_sha256']), ('baseline_poses', report['parent_poses_sha256']),
                              ('dynamic_indices', report['parent_dynamic_indices_sha256'])):
            if audit.get('artifact_sha256', {}).get(key) != expected:
                raise ValueError('Static roof audit checksum mismatch: '+key)
    return {'report': report, 'arrays': arrays, **source_arrays}


class StaticSurfaceRebuild:
    """O(Nnew) fixed static tensors, appended after every inherited motion path."""
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
            raise ValueError('Static roof appearance must be applied once to a fresh source model')
        for attribute, key, shape in (('_xyz', 'source_xyz', (-1, 3)),
                ('_rotation', 'source_quaternions', (-1, 4)), ('_scaling', 'source_log_scale', (-1, 3)),
                ('_opacity', 'source_opacity', (-1, 1)), ('_features_dc', 'source_dc', (-1, 1, 3))):
            actual = getattr(model, attribute).detach()
            expected = torch.as_tensor(self.source[key], dtype=actual.dtype, device=actual.device).reshape(shape)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError('Static roof runtime differs from immutable source: '+attribute)
        rest = model._features_rest.detach()
        if rest.ndim != 3 or rest.shape[0] != self.source_count or rest.shape[2] != 3 or torch.count_nonzero(rest).item():
            raise ValueError('Static roof expects DC-only immutable source features')
        opacity = model._opacity.detach().clone()
        opacity[self.retired.to(opacity.device)] = -30.
        model._opacity = torch.cat((opacity, self.opacity.to(opacity)), 0)
        model._scaling = torch.cat((model._scaling.detach(), self.log_scale.to(model._scaling)), 0)
        model._features_dc = torch.cat((model._features_dc.detach(), self.dc.to(model._features_dc)), 0)
        model._features_rest = torch.cat((rest, rest.new_zeros((self.new_count, rest.shape[1], 3))), 0)
        self._applied = True
        # Whole-source CPU attributes are needed only for this fresh-load
        # identity check, never for per-frame posing or duplicate GPU caches.
        self.source = None

    def append_geometry(self, xyz, quaternion):
        import torch
        if (xyz.shape != (self.source_count, 3) or quaternion.shape != (self.source_count, 4)
                or xyz.device != quaternion.device or xyz.dtype != quaternion.dtype):
            raise ValueError('Static roof requires inherited baked+completion geometry exactly once')
        return (torch.cat((xyz, self.xyz.to(xyz)), 0),
                torch.cat((quaternion, self.quaternions.to(quaternion)), 0))
