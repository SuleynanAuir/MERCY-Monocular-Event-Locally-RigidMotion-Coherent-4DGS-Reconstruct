"""Package appearance-only Lego updates without changing event-corrected motion.

The output is deliberately unaudited: it contains neither inherited rendered
frames nor an inherited event-audit.json. New renders and a new event audit are
required before a viewer can load it. All source inputs are opened read-only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.render_sampling import render_sampling_contract


APPEARANCE_FIELDS = ('f_dc_0', 'f_dc_1', 'f_dc_2', 'scale_0', 'scale_1', 'scale_2', 'opacity')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(vertices):
    digest = hashlib.sha256()
    for start in range(0, len(vertices), 65536):
        digest.update(np.ascontiguousarray(vertices[start:start + 65536]).view(np.uint8))
    return digest.hexdigest()


def apply_attributes(vertices, retained, attributes):
    """Change only tail DC/scale/opacity, preserving exact other vertex bytes."""
    if type(retained) is not int or not 0 < retained < len(vertices):
        raise ValueError('Invalid retained prefix count')
    if vertices.dtype.hasobject or any(name not in vertices.dtype.names for name in APPEARANCE_FIELDS):
        raise ValueError('Unsupported Gaussian vertex schema')
    count = len(vertices) - retained
    expected = {'dc_gray': (count,), 'log_scale': (count, 3), 'opacity_logit': (count,)}
    if set(attributes) != set(expected):
        raise ValueError('Appearance archive must contain exactly dc_gray, log_scale, opacity_logit')
    for key, shape in expected.items():
        value = np.asarray(attributes[key])
        if value.shape != shape or value.dtype != np.dtype(np.float32) or not np.isfinite(value).all():
            raise ValueError(f'Invalid appearance attribute: {key}')
    if np.any(attributes['dc_gray'] < 0) or np.any(attributes['dc_gray'] > 1):
        raise ValueError('dc_gray must contain linear grayscale RGB in [0,1], not SH coefficients')
    # Activated scale must remain representable; bounds used by the optimizer
    # are recorded separately, rather than silently clipping its chosen model.
    with np.errstate(over='ignore', under='ignore'):
        scales = np.exp(attributes['log_scale'])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Appearance contains invalid activated scale')
    result = vertices.copy()
    dc = (attributes['dc_gray'] - np.float32(.5)) / np.float32(.28209479177387814)
    for axis in range(3):
        result[f'f_dc_{axis}'][retained:] = dc
        result[f'scale_{axis}'][retained:] = attributes['log_scale'][:, axis]
    result['opacity'][retained:] = attributes['opacity_logit']
    verify_appearance_only(vertices, result, retained)
    return result


def verify_appearance_only(before, after, retained):
    if before.dtype != after.dtype or before.shape != after.shape:
        raise ValueError('Appearance update changed schema or vertex count')
    if rows_sha256(before[:retained]) != rows_sha256(after[:retained]):
        raise ValueError('Appearance update changed frozen retained rows')
    for name in before.dtype.names:
        if name not in APPEARANCE_FIELDS:
            # Field byte comparison, not numerical closeness, also checks any
            # NaN payload or signed zero without giving an accidental exception.
            if before[name].tobytes() != after[name].tobytes():
                raise ValueError(f'Appearance update changed protected field {name}')


def verify_structure_unchanged(before, after):
    old = {key: value for key, value in before.items() if key != 'input_sha256'}
    new = {key: value for key, value in after.items() if key != 'input_sha256'}
    if old != new:
        raise ValueError('Rebuilt hard-joint structure would change motion')


def share_motion_file(source, target, storage):
    """Share read-only-in-this-program; copied files get explicit read-only mode.

    Symlink storage does not claim OS-enforced immutability of the underlying
    existing input. No chmod or write is ever performed on a source/hardlink.
    """
    source, target = Path(source).resolve(strict=True), Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if storage == 'symlink':
        target.symlink_to(source)
    elif storage == 'copy':
        shutil.copyfile(source, target)
        target.chmod(0o444)
    else:
        raise ValueError('Unknown motion storage mode')


def verify_identical_motion(old_record, new_record, poses, progress=None):
    import torch
    from utils.event_refinement import correction_from_record
    if old_record['group_angles'] != new_record['group_angles']:
        raise ValueError('Copied correction changed event-learned group angles')
    old_model = correction_from_record(old_record, 'cpu')
    new_model = correction_from_record(new_record, 'cpu')
    old_state, new_state = old_model.state_dict(), new_model.state_dict()
    if old_state.keys() != new_state.keys() or any(not torch.equal(old_state[k], new_state[k]) for k in old_state):
        raise ValueError('Copied correction changed runtime motion tensors')
    rest = torch.from_numpy(np.asarray(poses[0]).copy())
    with torch.no_grad():
        for frame in range(len(poses)):
            baked = torch.from_numpy(np.asarray(poses[frame]).copy())
            old = old_model(rest, baked, frame)
            new = new_model(rest, baked, frame)
            if not torch.isfinite(old).all() or not torch.equal(old, new):
                raise ValueError(f'Appearance update changed actual XYZ/quaternions at frame {frame}')
            if progress and frame % 100 == 0:
                progress(frame)
    return {'frames_checked': len(poses), 'rows_checked_per_frame': len(rest),
            'xyz_and_quaternions_byte_identical': True, 'group_angles_exact': True,
            'motion_runtime_tensors_exact': True, 'device': 'cpu'}


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def package(args):
    from plyfile import PlyData, PlyElement
    import torch
    from scripts.export_lego_event_preview import verify_candidate
    from scripts.lego_joint_structure import build_structure
    from utils.comparison_renderer import verify_rebuild_provenance

    for name in ('base_rebuild', 'baseline', 'candidate', 'appearance', 'parts', 'joint_report'):
        setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    old_report_path = args.base_rebuild / 'rebuild-report.json'
    old_report = json.loads(old_report_path.read_text())
    if old_report.get('format') != 'lego_prior_surface_rebuild_v1':
        raise ValueError('Expected a prior rigid-surface rebuild report')
    old_source = args.base_rebuild / 'source.ply'
    old_binding = args.base_rebuild / 'initial-binding.npz'
    cache, correction, old_audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if (Path(cache['source']).resolve() != old_source or sha256(old_source) != old_report['rebuilt_sha256']
            or correction['format'] != 'event_supervised_hard_joint_alignment_v4'):
        raise ValueError('Prior baseline/candidate do not identify the provided rigid-surface model')
    verify_rebuild_provenance(old_report, old_report['source'], old_source)
    provenance = correction['joint_structure']['input_sha256']
    for key, path in (('parts', args.parts), ('binding', old_binding), ('joint_report', args.joint_report),
                      ('dynamic_indices', args.baseline / 'dynamic_indices.npy'),
                      ('baseline_cache_metadata', args.baseline / 'cache.json')):
        if sha256(path) != provenance.get(key):
            raise ValueError(f'Prior correction source checksum mismatch: {key}')
    attributes_path, optimization_path = args.appearance / 'attributes.npz', args.appearance / 'optimization.json'
    optimization = json.loads(optimization_path.read_text())
    sampling = render_sampling_contract(optimization)
    if (optimization.get('format') != 'lego_frozen_motion_surface_appearance_v1'
            or optimization.get('positions_rotations_motion_frozen') is not True
            or optimization.get('original_static_and_bands_frozen') is not True):
        raise ValueError('Expected explicitly frozen-motion rigid-seed appearance optimization')
    optimization_hashes = optimization.get('input_sha256', {})
    for key, path in (('source_ply', old_source), ('correction', args.candidate / 'correction.json'),
                      ('baseline_cache_metadata', args.baseline / 'cache.json'), ('rebuild_report', old_report_path)):
        if optimization_hashes.get(key) != sha256(path):
            raise ValueError(f'Appearance optimization source checksum mismatch: {key}')
    if optimization_hashes.get('supervision') != old_audit['artifact_sha256'].get('supervision'):
        raise ValueError('Appearance optimization supervision differs from prior event candidate')
    if optimization.get('attributes_sha256') != sha256(attributes_path):
        raise ValueError('Appearance optimization attributes checksum mismatch')
    with np.load(attributes_path, allow_pickle=False) as archive:
        attributes = {key: archive[key] for key in archive.files}
    old_ply = PlyData.read(old_source)
    before = old_ply['vertex'].data
    retained = old_report['retained_gaussian_count']
    after = apply_attributes(before, retained, attributes)
    if len(before) != old_report['gaussian_count'] or len(before) - retained != old_report['rebuilt_gaussians']:
        raise ValueError('Prior rebuild counts disagree with source vertices')
    if optimization.get('seed_count') != len(before) - retained or optimization.get('retained_count') != retained:
        raise ValueError('Appearance optimizer row counts disagree with source vertices')
    old_scale = np.column_stack([before[f'scale_{axis}'][retained:] for axis in range(3)])
    ratios = np.exp(attributes['log_scale'].astype(np.float64) - old_scale)
    lower, upper = optimization['radius_ratio_bounds']
    if np.any(ratios < lower - 1e-6) or np.any(ratios > upper + 1e-6):
        raise ValueError('Optimized radius exceeds recorded bounds')
    opacity = 1. / (1. + np.exp(-attributes['opacity_logit'].astype(np.float64)))
    lower, upper = optimization['opacity_bounds']
    if np.any(opacity < lower - 1e-6) or np.any(opacity > upper + 1e-6):
        raise ValueError('Optimized opacity exceeds recorded bounds')
    with np.load(old_binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    indices = np.load(args.baseline / 'dynamic_indices.npy', allow_pickle=False)
    poses = np.load(args.baseline / 'poses.npy', mmap_mode='r', allow_pickle=False)
    if (not np.array_equal(indices, binding['dynamic_indices'])
            or poses.shape != (cache['frame_count'], cache['dynamic_count'], 7)):
        raise ValueError('Prior binding and cached poses disagree')
    source_pose = np.column_stack([before[key][indices] for key in ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
    if not np.array_equal(source_pose, poses[0]):
        raise ValueError('Cached rest poses differ from immutable source XYZ/quaternions')

    # No mutation above this line: every source and appearance input has passed
    # validation before the new, independently named package is created.
    model, baseline, candidate = (output / name for name in ('model', 'baseline', 'candidate'))
    for path in (model, baseline, candidate):
        path.mkdir(parents=True, exist_ok=False)
    target = model / 'source.ply'
    PlyData([PlyElement.describe(after, 'vertex')], text=False, byte_order=old_ply.byte_order).write(target)
    readback = PlyData.read(target)['vertex'].data
    verify_appearance_only(before, readback, retained)
    if rows_sha256(after) != rows_sha256(readback):
        raise ValueError('Serialized appearance vertices differ from optimizer attributes')

    source_hash = sha256(target)
    appearance = {'format': 'lego_rigid_seed_appearance_refinement_v1',
        **sampling,
        'parent_source': str(old_source), 'parent_source_sha256': sha256(old_source),
        'parent_rebuild_report': str(old_report_path), 'parent_rebuild_report_sha256': sha256(old_report_path),
        'attributes': str(attributes_path), 'attributes_sha256': sha256(attributes_path),
        'optimization': str(optimization_path), 'optimization_sha256': sha256(optimization_path),
        'optimized_rows': len(before) - retained, 'row_start': retained,
        'allowed_fields': list(APPEARANCE_FIELDS),
        'dc_gray_convention': 'Input linear grayscale RGB in [0,1]; output SH DC=(gray-0.5)/0.28209479177387814 repeated on RGB',
        'xyz_and_quaternions_frozen': True, 'binding_arrays_frozen': True, 'motion_parameters_frozen': True,
        'retained_rows_byte_identical': True,
        'scope': 'Appearance-only refinement of rebuilt moving rigid seeds; no new motion optimization. '
                 'Prior static/elastic rows and all canonical XYZ/quaternions are byte-identical. '
                 'New event render audit is required; old event approval does not carry over.'}
    shutil.copyfile(optimization_path, output / 'optimization.json')
    binding_meta = json.loads(str(binding['metadata_json']))
    binding_meta.update(source=str(target), source_sha256=source_hash,
                        prior_binding=str(old_binding), prior_binding_sha256=sha256(old_binding),
                        geometry_rebuild_report=str(model / 'rebuild-report.json'), appearance_refinement=appearance,
                        event_approved=False, status='APPEARANCE_UPDATED_REQUIRES_NEW_EVENT_AUDIT')
    binding['metadata_json'] = np.asarray(json.dumps(binding_meta, allow_nan=False))
    new_binding = model / 'initial-binding.npz'
    np.savez_compressed(new_binding, **binding)
    with np.load(old_binding, allow_pickle=False) as old, np.load(new_binding, allow_pickle=False) as new:
        if old.files != new.files or any(old[key].tobytes() != new[key].tobytes()
                                        for key in old.files if key != 'metadata_json'):
            raise ValueError('Packaging changed binding arrays')
    write_json(model / 'initial-binding.json', binding_meta)
    report = copy.deepcopy(old_report)
    report.update(rebuilt_source=str(target), rebuilt_sha256=source_hash,
                  **sampling,
                  binding=str(new_binding), binding_sha256=sha256(new_binding), appearance_refinement=appearance,
                  event_approved=False, status='APPEARANCE_UPDATED_REQUIRES_NEW_EVENT_AUDIT')
    report['initial_radius_scene_units'] = copy.deepcopy(old_report.get('radius_scene_units'))
    current_scales = np.exp(attributes['log_scale'].astype(np.float64))
    report['radius_scene_units'] = {'minimum': float(current_scales.min()),
                                   'median': float(np.median(current_scales)),
                                   'maximum': float(current_scales.max())}
    # Keep original seed/mesh evidence and all retained-row mapping paths intact.
    # Original radius/opacity parameters describe initialization, not new values.
    report['parameters_scope'] = 'Surface-rebuild initialization; current optimized appearance is documented by appearance_refinement.'
    write_json(model / 'rebuild-report.json', report)
    verify_rebuild_provenance(report, report['source'], target)
    for name in ('poses.npy', 'dynamic_indices.npy'):
        share_motion_file(args.baseline / name, baseline / name, args.pose_storage)
    new_cache = copy.deepcopy(cache)
    new_cache.update(source=str(target), source_sha256=source_hash, binding=binding_meta,
                     **sampling,
                     appearance_refinement=appearance, shared_motion_parent=str(args.baseline),
                     shared_motion_storage=args.pose_storage, event_supervised=False,
                     status='INITIALIZATION_POSES_WITH_REFINED_APPEARANCE_NOT_EVENT_APPROVED',
                     rendered_gray_available=False)
    write_json(baseline / 'cache.json', new_cache)
    structure = build_structure(args.parts, new_binding, baseline, args.joint_report,
                                correction['joint_structure']['joint_tolerance'])
    verify_structure_unchanged(correction['joint_structure'], structure)
    result = copy.deepcopy(correction)
    result.update(joint_structure=structure,
                  **sampling,
                  group_input_sha256={key: structure['input_sha256'][key] for key in ('parts', 'binding', 'joint_report')},
                  appearance_refinement=appearance,
                  motion_optimization_inherited=True, motion_reoptimized_for_new_appearance=False,
                  motion_training_provenance={'candidate': str(args.candidate),
                      'correction_sha256': sha256(args.candidate / 'correction.json'),
                      'event_audit_sha256': sha256(args.candidate / 'event-audit.json'),
                      'source_ply_sha256': sha256(old_source), 'original_training_record': correction},
                  rgb_appearance_parameters_frozen=True,
                  rgb_appearance_parameters_frozen_scope='This motion correction only changes XYZ/quaternions of the NEW source; '
                      'the preceding appearance-only refinement is separately documented and is not claimed frozen relative to the old source.')
    result['training_fields_scope'] = 'Existing iteration/history/probe/calibration fields describe the inherited event-motion optimization on the parent source, not a fresh motion fit on this appearance-refined source.'
    torch.set_num_threads(2)
    checked = verify_identical_motion(correction, result, poses,
        progress=lambda frame: print(f'Checked identical event-corrected motion {frame}/{len(poses)}', flush=True))
    write_json(candidate / 'correction.json', result)
    package_report = {'format': 'lego_appearance_package_v1', 'status': 'PENDING_NEW_RENDER_AND_EVENT_AUDIT',
        **sampling,
        'model': str(model), 'baseline': str(baseline), 'candidate': str(candidate),
        'geometry_repair_report': str(model / 'rebuild-report.json'), 'appearance_refinement': appearance,
        'motion_check': checked, 'all_canonical_xyz_quaternions_exact': True,
        'retained_vertex_sha256': rows_sha256(readback[:retained]),
        'input_sha256': {'old_source': sha256(old_source), 'old_binding': sha256(old_binding),
                        'old_correction': sha256(args.candidate / 'correction.json')},
        'artifact_sha256': {'source_ply': source_hash, 'binding': sha256(new_binding),
                            'baseline_cache_metadata': sha256(baseline / 'cache.json'),
                            'baseline_poses': sha256(baseline / 'poses.npy'),
                            'dynamic_indices': sha256(baseline / 'dynamic_indices.npy'),
                            'correction': sha256(candidate / 'correction.json')},
        'event_approved': False, 'event_audit_inherited': False, 'old_inputs_not_modified': True,
        'notes': ['Only poses and indices are shared; old rendered_gray.npy and event-audit.json are deliberately absent.',
                  'Symlink storage is opened read-only by this tool, not claimed OS-enforced immutable.']}
    write_json(output / 'package-report.json', package_report)
    print(json.dumps({'output': str(output), 'model': str(model), 'baseline': str(baseline),
                      'candidate': str(candidate), 'motion_check': checked}, indent=2), flush=True)
    return package_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('base-rebuild', 'baseline', 'candidate', 'appearance', 'output-dir', 'parts', 'joint-report'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--pose-storage', choices=('symlink', 'copy'), default='symlink')
    package(parser.parse_args())


if __name__ == '__main__':
    main()
