"""Audited, sparse appearance changes of immutable retained static Gaussians.

The source PLY, row order, all geometry and motion remain unchanged. This is an
explicit runtime Gaussian attribute overlay, never an image-space filter. Old
event approval does not approve the resulting images.
"""
import copy
import json
from pathlib import Path
import re

import numpy as np


FORMAT = 'lego_gaussian_cleanup_overlay_v1'
CONTRACT_FIELDS = {'format', 'report', 'report_sha256', 'overlay', 'overlay_sha256'}
C0 = 0.28209479177387814


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _absolute(value):
    return isinstance(value, str) and Path(value).is_absolute()


def _checksum(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def cleanup_contract(*records):
    """Legacy absence is valid; partial presence and differing contracts fail."""
    contracts = [record.get('gaussian_cleanup') for record in records]
    if not contracts or all(value is None for value in contracts):
        return None
    first = contracts[0]
    if (not isinstance(first, dict) or set(first) != CONTRACT_FIELDS
            or first.get('format') != FORMAT or any(value != first for value in contracts)):
        raise ValueError('Gaussian cleanup contract mismatch')
    for field in ('report', 'overlay'):
        if not _absolute(first[field]) or not _checksum(first[field + '_sha256']):
            raise ValueError(f'Invalid Gaussian cleanup {field} provenance')
    return dict(first)


def require_no_cleanup(metadata, context):
    if cleanup_contract(metadata) is not None:
        raise ValueError(f'{context} does not support Gaussian cleanup appearance overlays; '
                         'export would silently restore uncleaned attributes')


def validate_report_shape(report):
    counts = ('gaussian_count', 'retained_gaussian_count', 'modified_gaussian_count')
    if (report.get('format') != FORMAT or report.get('attributes') not in (['opacity'], ['dc_gray'])
            or any(type(report.get(key)) is not int for key in counts)
            or not 0 < report['modified_gaussian_count'] <= report['retained_gaussian_count'] < report['gaussian_count']
            or not isinstance(report.get('scope'), str) or not report['scope']):
        raise ValueError('Invalid Gaussian cleanup report shape or scope')
    for path, checksum in (('source', 'source_sha256'), ('overlay', 'overlay_sha256')):
        if not _absolute(report.get(path)) or not _checksum(report.get(checksum)):
            raise ValueError(f'Invalid Gaussian cleanup report provenance: {path}')
    for path in ('parent_baseline', 'parent_candidate'):
        if not _absolute(report.get(path)):
            raise ValueError(f'Invalid Gaussian cleanup {path}')
    for key in ('parent_baseline_cache_sha256', 'parent_correction_sha256', 'parent_event_audit_sha256',
                'parent_poses_sha256', 'parent_dynamic_indices_sha256'):
        if not _checksum(report.get(key)):
            raise ValueError(f'Invalid Gaussian cleanup parent checksum: {key}')


def cleanup_eligible_mask(gaussian_count, retained_count, labels, structure,
                          dynamic_indices, deforming_parts):
    """Use audited row identities, never screen-assignment distances, as gates."""
    labels = np.asarray(labels)
    indices = np.asarray(dynamic_indices)
    parents = np.asarray(structure['parents'])
    if (type(gaussian_count) is not int or type(retained_count) is not int
            or not 0 < retained_count < gaussian_count or labels.ndim != 1
            or labels.dtype.kind not in 'iu' or not retained_count <= len(labels) < gaussian_count
            or indices.ndim != 1 or indices.dtype not in (np.dtype('int32'), np.dtype('int64'))
            or np.any(indices < 0) or np.any(indices >= len(labels))
            or np.any(indices[1:] <= indices[:-1]) or parents.ndim != 1
            or parents.dtype.kind not in 'iu' or not np.any(parents == -1)):
        raise ValueError('Invalid Gaussian cleanup parent row identities')
    mapping = {}
    for group in structure['source_groups']:
        number = group['group']
        if type(number) is not int or not 0 <= number < len(parents):
            raise ValueError('Invalid Gaussian cleanup source group')
        for part in group['objects']:
            if type(part) is not int or part < 0 or part in mapping:
                raise ValueError('Invalid or duplicated Gaussian cleanup source part')
            mapping[part] = number
    roots = set(np.flatnonzero(parents == -1).tolist())
    root_parts = [part for part, group in mapping.items() if group in roots]
    eligible = np.zeros(gaussian_count, dtype=bool)
    eligible[:retained_count] = (np.isin(labels[:retained_count], root_parts)
                                 & ~np.isin(labels[:retained_count], deforming_parts))
    eligible[indices] = False
    return eligible


def validate_overlay_arrays(arrays, report, source_opacity, eligible, source_dc=None):
    if report.get('attributes') not in (['opacity'], ['dc_gray']):
        raise ValueError('Gaussian cleanup requires one explicit opacity or dc_gray attribute mode')
    color_mode = report['attributes'] == ['dc_gray']
    value_key = 'dc_gray' if color_mode else 'opacity_logit'
    if set(arrays) != {'indices', value_key, 'metadata_json'}:
        raise ValueError(f'Gaussian cleanup supports only indices and {value_key} in this mode')
    indices, values = arrays['indices'], arrays[value_key]
    count = report['modified_gaussian_count']
    if (not isinstance(indices, np.ndarray) or indices.dtype not in (np.dtype('int32'), np.dtype('int64'))
            or indices.shape != (count,) or np.any(indices < 0)
            or np.any(indices >= report['gaussian_count']) or np.any(indices[1:] <= indices[:-1])):
        raise ValueError('Gaussian cleanup indices must be strictly sorted unique valid rows')
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float32')
            or values.shape != (count,) or not np.isfinite(values).all()):
        raise ValueError(f'Gaussian cleanup {value_key} must be finite float32[K]')
    if color_mode:
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError('Gaussian cleanup dc_gray must be in [0,1]')
        if (not isinstance(source_dc, np.ndarray) or source_dc.dtype != np.dtype('float32')
                or source_dc.shape != (report['gaussian_count'], 3) or not np.isfinite(source_dc).all()):
            raise ValueError('Gaussian cleanup requires finite canonical source_dc float32[N,3]')
    elif np.any(values < -30):
        raise ValueError('Gaussian cleanup opacity_logit must be >= -30')
    source_opacity, eligible = np.asarray(source_opacity), np.asarray(eligible)
    if (source_opacity.shape != (report['gaussian_count'],) or np.isnan(source_opacity).any()
            or eligible.shape != source_opacity.shape or eligible.dtype != np.dtype('bool')):
        raise ValueError('Invalid Gaussian cleanup source opacity or eligibility')
    if not eligible[indices].all():
        raise ValueError('Gaussian cleanup may affect only retained static root rows')
    # Original +/- infinity is a valid serialized opacity logit. Positive
    # infinity may be reduced; negative infinity cannot be made finite here.
    if not color_mode and np.any(values > source_opacity[indices]):
        raise ValueError('Gaussian cleanup cannot increase original opacity')
    metadata = arrays['metadata_json']
    if not isinstance(metadata, np.ndarray) or metadata.ndim != 0 or metadata.dtype.kind not in 'US':
        raise ValueError('Invalid Gaussian cleanup overlay metadata')
    meta = json.loads(str(metadata))
    if (not isinstance(meta, dict) or meta.get('format') != FORMAT
            or meta.get('source_sha256') != report['source_sha256']):
        raise ValueError('Gaussian cleanup overlay metadata/source mismatch')


def _motion_identity(record):
    if record.get('format') != 'event_supervised_hard_joint_alignment_v4':
        raise ValueError('Gaussian cleanup requires inherited hard-joint v4 motion')
    identity = copy.deepcopy(record)
    identity.pop('gaussian_cleanup', None)
    # The child cache contains the new overlay contract, so this one input
    # fingerprint necessarily changes. Its actual file is checked below when
    # the caller supplies baseline. No other correction field may differ.
    identity.get('joint_structure', {}).get('input_sha256', {}).pop('baseline_cache_metadata', None)
    return identity


def verify_gaussian_cleanup(cache, correction=None, audit=None, *, baseline=None):
    """Verify the original v2 chain and the independent appearance-only overlay.

    No audit is required to render a new candidate. Once supplied, its contract
    and artifact checksums must identify these exact changed attributes.
    """
    records = [cache] + ([correction] if correction is not None else []) + ([audit] if audit is not None else [])
    contract = cleanup_contract(*records)
    if contract is None:
        return None
    for field in ('report', 'overlay'):
        if _sha256(contract[field]) != contract[field + '_sha256']:
            raise ValueError(f'Gaussian cleanup checksum mismatch: {field}')
    report = json.loads(Path(contract['report']).read_text())
    validate_report_shape(report)
    if (report['source'] != cache.get('source') or report['source_sha256'] != cache.get('source_sha256')
            or report['gaussian_count'] != cache.get('gaussian_count')
            or report['overlay'] != contract['overlay'] or report['overlay_sha256'] != contract['overlay_sha256']):
        raise ValueError('Gaussian cleanup report/cache/overlay identity mismatch')
    parent_baseline, parent_candidate = Path(report['parent_baseline']), Path(report['parent_candidate'])
    files = {'source_sha256': report['source'], 'parent_baseline_cache_sha256': parent_baseline/'cache.json',
             'parent_correction_sha256': parent_candidate/'correction.json',
             'parent_event_audit_sha256': parent_candidate/'event-audit.json',
             'parent_poses_sha256': parent_baseline/'poses.npy',
             'parent_dynamic_indices_sha256': parent_baseline/'dynamic_indices.npy'}
    for key, path in files.items():
        if _sha256(path) != report[key]:
            raise ValueError(f'Gaussian cleanup parent/source checksum mismatch: {key}')
    parent_cache = json.loads((parent_baseline/'cache.json').read_text())
    parent_record = json.loads((parent_candidate/'correction.json').read_text())
    parent_audit = json.loads((parent_candidate/'event-audit.json').read_text())
    # Reject recursive overlays before invoking the shared candidate validator.
    if cleanup_contract(parent_cache, parent_record, parent_audit) is not None:
        raise ValueError('Gaussian cleanup parent must not already have a cleanup overlay')
    from scripts.events.export_lego_event_preview import verify_candidate
    from utils.multiview_completion import completion_contract
    from utils.render_sampling import require_same_sampling
    verify_candidate(parent_baseline, parent_candidate, allow_unapproved=True)
    completion = completion_contract(cache, parent_cache, parent_record, parent_audit)
    if completion is None:
        raise ValueError('Gaussian cleanup requires the verified compact-completion parent')
    completion_report = json.loads(Path(completion['report']).read_text())
    rebuild = json.loads(Path(completion_report['parent_report']).read_text())
    if (report['retained_gaussian_count'] != rebuild['retained_gaussian_count']
            or parent_cache.get('source') != report['source']
            or parent_cache.get('source_sha256') != report['source_sha256']):
        raise ValueError('Gaussian cleanup changed its immutable retained-source parent')
    for key in ('gaussian_count', 'dynamic_count', 'rendered_dynamic_count', 'frame_count', 'fps', 'camera'):
        if cache.get(key) != parent_cache.get(key):
            raise ValueError(f'Gaussian cleanup changed parent cache geometry/motion: {key}')
    require_same_sampling(cache, parent_cache, parent_record, parent_audit, *records[1:])
    if correction is not None:
        completion_contract(cache, correction)
        if _motion_identity(correction) != _motion_identity(parent_record):
            raise ValueError('Gaussian cleanup changed inherited hard-joint motion')
    if baseline is not None:
        if correction is not None and correction.get('joint_structure', {}).get('input_sha256', {}).get(
                'baseline_cache_metadata') != _sha256(Path(baseline)/'cache.json'):
            raise ValueError('Gaussian cleanup correction cache fingerprint mismatch')
        for filename, key in (('poses.npy', 'parent_poses_sha256'),
                              ('dynamic_indices.npy', 'parent_dynamic_indices_sha256')):
            path = Path(baseline)/filename
            if path.resolve() != (parent_baseline/filename).resolve() and _sha256(path) != report[key]:
                raise ValueError(f'Gaussian cleanup changed parent baked motion: {filename}')
    if not _absolute(rebuild.get('binding')) or not _checksum(rebuild.get('binding_sha256')):
        raise ValueError('Gaussian cleanup requires hashed parent binding identities')
    if _sha256(rebuild['binding']) != rebuild['binding_sha256']:
        raise ValueError('Gaussian cleanup parent binding checksum mismatch')
    with np.load(rebuild['binding'], allow_pickle=False) as archive:
        labels, bound_indices = archive['labels'], archive['dynamic_indices']
    indices = np.load(parent_baseline/'dynamic_indices.npy', allow_pickle=False)
    if len(labels) != completion_report['prefix_count'] or not np.array_equal(indices, bound_indices):
        raise ValueError('Gaussian cleanup binding and parent baked rows disagree')
    with np.load(completion_report['parts'], allow_pickle=False) as archive:
        part_meta = json.loads(str(archive['metadata_json']))
    if np.any(labels < 0) or np.any(labels >= len(part_meta['objects'])):
        raise ValueError('Gaussian cleanup binding references unknown source parts')
    eligible = cleanup_eligible_mask(report['gaussian_count'], report['retained_gaussian_count'], labels,
                                     parent_record['joint_structure'], indices, part_meta.get('deforming_objects', []))
    from plyfile import PlyData
    source = PlyData.read(report['source'])['vertex'].data
    if len(source) != report['gaussian_count']:
        raise ValueError('Gaussian cleanup PLY row count mismatch')
    opacity = np.asarray(source['opacity']).copy()
    source_dc = np.column_stack([source[f'f_dc_{channel}'] for channel in range(3)])
    with np.load(contract['overlay'], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_overlay_arrays(arrays, report, opacity, eligible, source_dc)
    if audit is not None:
        completion_contract(cache, audit)
        for artifact, expected in (('cleanup_report', contract['report_sha256']),
                                   ('cleanup_overlay', contract['overlay_sha256']),
                                   ('source_ply', report['source_sha256']),
                                   ('baseline_poses', report['parent_poses_sha256']),
                                   ('dynamic_indices', report['parent_dynamic_indices_sha256'])):
            if audit.get('artifact_sha256', {}).get(artifact) != expected:
                raise ValueError(f'Gaussian cleanup audit checksum mismatch: {artifact}')
    return {'report': report, 'arrays': arrays, 'source_opacity': opacity,
            'source_dc': source_dc, 'eligible': eligible}


def apply_gaussian_cleanup(model, verified):
    """Apply once to a freshly loaded source; leave its other tensors untouched."""
    if verified is None:
        return None
    import torch
    report, arrays = verified['report'], verified['arrays']
    validate_overlay_arrays(arrays, report, verified['source_opacity'], verified['eligible'], verified.get('source_dc'))
    old = model._opacity.detach()
    expected = torch.as_tensor(verified['source_opacity'], dtype=old.dtype, device=old.device).reshape(-1, 1)
    if old.shape != expected.shape or not torch.equal(old, expected):
        raise ValueError('Gaussian cleanup runtime opacity differs from immutable source')
    rows = torch.as_tensor(arrays['indices'].astype(np.int64), device=old.device)
    if report['attributes'] == ['dc_gray']:
        original_dc = model._features_dc.detach()
        expected_dc = torch.as_tensor(verified['source_dc'], dtype=original_dc.dtype,
                                      device=original_dc.device).reshape(-1, 1, 3)
        if original_dc.shape != expected_dc.shape or not torch.equal(original_dc, expected_dc):
            raise ValueError('Gaussian cleanup runtime DC differs from immutable source')
        gray = torch.as_tensor(arrays['dc_gray'], dtype=original_dc.dtype, device=original_dc.device)
        dc = ((gray - .5) / C0).reshape(-1, 1, 1).expand(-1, 1, 3)
        model._features_dc = original_dc.clone().index_copy(0, rows.to(original_dc.device), dc)
        return model._features_dc
    values = torch.as_tensor(arrays['opacity_logit'], dtype=old.dtype, device=old.device).reshape(-1, 1)
    model._opacity = old.clone().index_copy(0, rows, values)
    return model._opacity
