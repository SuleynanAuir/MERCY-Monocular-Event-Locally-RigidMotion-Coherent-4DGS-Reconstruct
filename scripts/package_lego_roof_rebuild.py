"""Package an explicitly prior-assisted roof-only Gaussian replacement.

The source cloud and all trajectories remain read-only. The small extension
must be present at render time; a fresh all-frame event audit is mandatory.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.export_lego_event_preview import verify_candidate
from utils.rigid_checkpoint import sha256_file
from utils.static_surface_rebuild import FORMAT, verify_static_surface_rebuild


ARRAYS = ('retired_indices', 'new_xyz', 'new_quaternions', 'new_log_scale',
          'new_opacity_logit', 'new_dc_gray', 'part_ids', 'triangle_ids')


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.baseline, args.candidate, args.selection = (path.resolve(strict=True) for path in
                                                   (args.baseline, args.candidate, args.selection))
    args.output = args.output.resolve()
    cache, record, audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if cache.get('static_surface_rebuild') or cache.get('gaussian_cleanup'):
        raise ValueError('Use the original audited completion parent, not another experiment')
    completion = json.loads(Path(cache['multiview_completion']['report']).read_text())
    rebuild = json.loads(Path(completion['parent_report']).read_text())
    with np.load(args.selection, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ARRAYS}
        metadata = json.loads(str(archive['metadata_json']))
    if metadata['input_sha256'].get('source_ply') != cache['source_sha256']:
        raise ValueError('Roof selection identifies a different cloud')
    count, retired = len(arrays['new_xyz']), len(arrays['retired_indices'])
    if not 0 < count <= 100000 or not retired:
        raise ValueError('Empty or excessive roof replacement')
    model, baseline, candidate = [args.output/name for name in ('model', 'baseline', 'candidate')]
    for folder in (model, baseline, candidate):
        folder.mkdir(parents=True, exist_ok=False)
    extension = model/'roof-gaussians.npz'
    arrays['metadata_json'] = np.asarray(json.dumps({'format': FORMAT,
        'source_sha256': cache['source_sha256'], 'selection_sha256': sha256_file(args.selection),
        'unobserved_geometry_is_prior': True}))
    np.savez_compressed(extension, **arrays)
    report = {'format': FORMAT, 'source': cache['source'], 'source_sha256': cache['source_sha256'],
        'parent_baseline': str(args.baseline), 'parent_candidate': str(args.candidate),
        'parent_baseline_cache_sha256': sha256_file(args.baseline/'cache.json'),
        'parent_correction_sha256': sha256_file(args.candidate/'correction.json'),
        'parent_event_audit_sha256': sha256_file(args.candidate/'event-audit.json'),
        'parent_poses_sha256': sha256_file(args.baseline/'poses.npy'),
        'parent_dynamic_indices_sha256': sha256_file(args.baseline/'dynamic_indices.npy'),
        'mesh_prior': completion['mesh_prior'], 'mesh_prior_sha256': completion['mesh_prior_sha256'],
        'parts': completion['parts'], 'parts_sha256': completion['parts_sha256'],
        'selection': str(args.selection), 'selection_sha256': sha256_file(args.selection),
        'extension': str(extension), 'extension_sha256': sha256_file(extension),
        'source_gaussian_count': cache['gaussian_count'], 'retired_gaussian_count': retired,
        'added_gaussian_count': count, 'rendered_gaussian_count': cache['gaussian_count']+count,
        'retained_gaussian_count': rebuild['retained_gaussian_count'],
        'allowed_part_ids': [11, 12, 13, 14, 15], 'retirement_bounds': metadata['retirement_bounds'],
        'unobserved_geometry_is_prior': True, 'selection_metadata': metadata,
        'scope': 'Only five static cabin roof parts are rebuilt using actual supplied mesh triangles. '
                 'Original source rows remain recoverable; retired rows become transparent and static new rows append. '
                 'Video-supported visible colors and unobserved material priors are distinguished. '
                 'Other parts and every inherited motion parameter remain unchanged. '
                 'No complete-view, hidden-truth or event approval is inherited.'}
    report_path = model/'roof-report.json'
    write_json(report_path, report)
    contract = {'format': FORMAT, 'report': str(report_path), 'report_sha256': sha256_file(report_path),
                'extension': str(extension), 'extension_sha256': sha256_file(extension)}
    for name in ('poses.npy', 'dynamic_indices.npy'):
        (baseline/name).symlink_to((args.baseline/name).resolve(strict=True))
    child_cache = copy.deepcopy(cache)
    child_cache.update(static_surface_rebuild=contract,
                       rendered_gaussian_count=report['rendered_gaussian_count'],
                       status='ROOF_SURFACE_REBUILD_REQUIRES_NEW_EVENT_AND_VIEW_AUDIT')
    write_json(baseline/'cache.json', child_cache)
    child_record = copy.deepcopy(record)
    child_record['static_surface_rebuild'] = contract
    child_record['joint_structure']['input_sha256']['baseline_cache_metadata'] = sha256_file(baseline/'cache.json')
    write_json(candidate/'correction.json', child_record)
    verify_static_surface_rebuild(child_cache, child_record, baseline=baseline)
    print(json.dumps({'output': str(args.output), 'retired': retired, 'added': count,
                      'source_ply_unchanged': True, 'motion_unchanged': True, 'audit': 'pending'}, indent=2))


if __name__ == '__main__':
    main()
