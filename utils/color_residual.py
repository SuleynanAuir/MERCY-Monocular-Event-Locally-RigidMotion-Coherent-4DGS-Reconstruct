"""Verified Gaussian-color residual overlay on an immutable audited parent."""
import copy
import json
from pathlib import Path
import numpy as np
from utils.gaussian_cleanup import _absolute, _checksum, _sha256, _motion_identity

FORMAT = 'lego_color_residual_overlay_v1'
FIELDS = {'format', 'report', 'report_sha256', 'residual', 'residual_sha256'}
C0 = .28209479177387814


def color_residual_contract(*records):
    contracts = [r.get('color_residual') for r in records]
    if not contracts or all(c is None for c in contracts):
        return None
    first = contracts[0]
    if (not isinstance(first, dict) or set(first) != FIELDS or first.get('format') != FORMAT
            or any(c != first for c in contracts)):
        raise ValueError('Color residual contract mismatch')
    for key in ('report', 'residual'):
        if not _absolute(first[key]) or not _checksum(first[key+'_sha256']):
            raise ValueError('Invalid color residual provenance: '+key)
    return dict(first)


def require_no_color_residual(metadata, context):
    if color_residual_contract(metadata) is not None:
        raise ValueError(context+' does not support color residuals; refusing to silently drop optimized colors')


def validate_residual_arrays(arrays, report, source_count, prefix_count, labels, deforming_parts):
    if set(arrays) != {'indices', 'delta_gray', 'maximum_training_visibility'}:
        raise ValueError('Unexpected residual array schema')
    indices, delta, visibility = (arrays[k] for k in ('indices', 'delta_gray', 'maximum_training_visibility'))
    count = report.get('optimized_count')
    if (type(count) is not int or not 0 < count <= prefix_count < source_count
            or report.get('gaussian_count') != source_count or report.get('prefix_count') != prefix_count
            or labels.shape != (prefix_count,) or labels.dtype.kind not in 'iu'
            or indices.dtype != np.int32 or indices.shape != (count,)
            or np.any(indices < 0) or np.any(indices >= prefix_count)
            or np.any(indices[1:] <= indices[:-1])):
        raise ValueError('Invalid residual source/row identities')
    maximum, threshold = report.get('maximum_delta'), report.get('visibility_threshold')
    if (type(maximum) not in (int, float) or not np.isfinite(maximum) or not 0 < maximum <= .15
            or type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 < threshold <= .1):
        raise ValueError('Invalid residual bounds')
    if any(v.dtype != np.float32 or v.shape != (count,) or not np.isfinite(v).all() for v in (delta, visibility)):
        raise ValueError('Residual delta/visibility must be finite float32 vectors')
    if np.any(np.abs(delta) > maximum+1e-7) or np.any(visibility < threshold):
        raise ValueError('Residual exceeds its measured visibility or color bound')
    if np.isin(labels[indices], deforming_parts).any():
        raise ValueError('Color residual touches protected elastic rows')


def validate_training_split(report, frame_count):
    from utils.residual_appearance import split_timeline
    allowed, held_event, held_rgb = split_timeline(frame_count)
    train, probe = report.get('training_intervals'), report.get('training_probe_intervals')
    if (not isinstance(train, list) or not train or any(type(i) is not int for i in train)
            or train != sorted(set(train)) or not set(train).issubset(allowed.tolist())
            or not isinstance(probe, list) or not probe or not set(probe).issubset(train)
            or report.get('rgb_heldout_frames') != held_rgb.tolist()
            or report.get('event_heldout_intervals') != held_event.tolist()):
        raise ValueError('Residual fitting split includes held-out data or has wrong identity')
    frames = report.get('visibility_probe_frames')
    if (not isinstance(frames, list) or not frames
            or any(type(i) is not int for i in frames)
            or not set(frames).issubset(set(train) | {i+1 for i in train})):
        raise ValueError('Residual confidence scan includes nontraining frames')


def verify_color_residual(cache, correction=None, audit=None, *, baseline=None):
    records = [cache]+([correction] if correction is not None else [])+([audit] if audit is not None else [])
    contract = color_residual_contract(*records)
    if contract is None:
        return None
    if any(r.get(k) is not None for r in records for k in ('gaussian_cleanup', 'static_surface_rebuild')):
        raise ValueError('Color residual cannot stack with experimental geometry/cleanup')
    for key in ('report', 'residual'):
        if _sha256(contract[key]) != contract[key+'_sha256']:
            raise ValueError('Color residual checksum mismatch: '+key)
    report = json.loads(Path(contract['report']).read_text())
    if (report.get('format') != 'lego_observed_color_residual_fit_v1'
            or report.get('source') != cache.get('source')
            or report.get('residual_sha256') != contract['residual_sha256']
            or any(report.get(k) is not True for k in ('positions_rotations_scales_opacity_motion_frozen',
                'elastic_and_compact_completion_rows_frozen', 'source_chroma_preserved',
                'heldout_data_not_used_for_fitting_or_selection'))):
        raise ValueError('Invalid color residual source/scope report')
    validate_training_split(report, cache['frame_count'])
    for key in ('parent_baseline', 'parent_candidate'):
        if not _absolute(report.get(key)):
            raise ValueError('Invalid residual parent path')
    parent_baseline, parent_candidate = Path(report['parent_baseline']), Path(report['parent_candidate'])
    expected_paths = {'source_ply': Path(cache['source']), 'correction': parent_candidate/'correction.json',
        'baseline_cache_metadata': parent_baseline/'cache.json', 'baseline_poses': parent_baseline/'poses.npy',
        'dynamic_indices': parent_baseline/'dynamic_indices.npy', 'event_audit': parent_candidate/'event-audit.json'}
    hashes, paths = report.get('input_sha256', {}), report.get('input_paths', {})
    for key in (*expected_paths, 'supervision', 'multiview_render'):
        if not _absolute(paths.get(key)) or not _checksum(hashes.get(key)):
            raise ValueError('Missing residual input provenance: '+key)
        if key in expected_paths and Path(paths[key]) != expected_paths[key]:
            raise ValueError('Residual parent input path differs: '+key)
        if _sha256(paths[key]) != hashes[key]:
            raise ValueError('Residual parent input hash differs: '+key)
    if hashes['source_ply'] != cache.get('source_sha256'):
        raise ValueError('Residual source checksum differs from child cache')
    parent_records = [json.loads(p.read_text()) for p in (parent_baseline/'cache.json',
                      parent_candidate/'correction.json', parent_candidate/'event-audit.json')]
    if any(r.get(k) is not None for r in parent_records
           for k in ('color_residual', 'gaussian_cleanup', 'static_surface_rebuild')):
        raise ValueError('Residual parent must be the original audited completion, not an overlay chain')
    from scripts.export_lego_event_preview import verify_candidate
    from utils.multiview_completion import completion_contract
    from utils.render_sampling import require_same_sampling
    verify_candidate(parent_baseline, parent_candidate, allow_unapproved=True)
    completion = completion_contract(*records, *parent_records)
    if completion is None:
        raise ValueError('Residual requires the original compact-completion geometry')
    require_same_sampling(*records, *parent_records)
    for key in ('source', 'source_sha256', 'gaussian_count', 'dynamic_count', 'rendered_dynamic_count',
                'frame_count', 'fps', 'camera'):
        if cache.get(key) != parent_records[0].get(key):
            raise ValueError('Color residual changed parent geometry/camera: '+key)
    if correction is not None:
        stripped = copy.deepcopy(correction); stripped.pop('color_residual', None)
        # This separately verified child layer does not alter inherited motion.
        # Its own verifier compares the full record against this residual parent.
        stripped.pop('protected_surface_detail', None)
        if _motion_identity(stripped) != _motion_identity(parent_records[1]):
            raise ValueError('Color residual changed inherited motion')
    if baseline is not None:
        baseline = Path(baseline)
        if correction is not None and correction['joint_structure']['input_sha256'].get(
                'baseline_cache_metadata') != _sha256(baseline/'cache.json'):
            raise ValueError('Color residual correction cache hash differs')
        for name, key in (('poses.npy', 'baseline_poses'), ('dynamic_indices.npy', 'dynamic_indices')):
            if (baseline/name).resolve() != Path(paths[key]).resolve() and _sha256(baseline/name) != hashes[key]:
                raise ValueError('Color residual changed baked motion: '+name)
    completed = json.loads(Path(completion['report']).read_text())
    rebuild = json.loads(Path(completed['parent_report']).read_text())
    if _sha256(rebuild['binding']) != rebuild['binding_sha256']:
        raise ValueError('Residual parent binding checksum differs')
    with np.load(rebuild['binding'], allow_pickle=False) as data:
        labels = data['labels']
    with np.load(completed['parts'], allow_pickle=False) as data:
        part_meta = json.loads(str(data['metadata_json']))
    with np.load(contract['residual'], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_residual_arrays(arrays, report, cache['gaussian_count'], completed['prefix_count'],
                             labels, part_meta['deforming_objects'])
    if audit is not None:
        for key, value in (('color_residual_report', contract['report_sha256']),
                           ('color_residual', contract['residual_sha256'])):
            if audit.get('artifact_sha256', {}).get(key) != value:
                raise ValueError('Fresh event audit does not identify color residual: '+key)
    from plyfile import PlyData
    vertices = PlyData.read(cache['source'])['vertex'].data
    dc = np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)])
    if dc.shape != (cache['gaussian_count'], 3) or not np.isfinite(dc).all():
        raise ValueError('Invalid immutable source colors')
    return {'report': report, 'arrays': arrays, 'source_dc': dc}


def apply_color_residual(model, verified):
    if verified is None:
        return
    import torch
    original = model._features_dc.detach()
    source = torch.as_tensor(verified['source_dc'], dtype=original.dtype, device=original.device)[:, None]
    if source.shape != original.shape or not torch.equal(source, original):
        raise ValueError('Color residual must be applied once to a fresh immutable source')
    rows = torch.as_tensor(verified['arrays']['indices'].astype(np.int64), device=original.device)
    delta = torch.as_tensor(verified['arrays']['delta_gray'], device=original.device, dtype=original.dtype)
    model._features_dc = original.index_copy(0, rows, original[rows]+delta[:, None, None]/C0)
