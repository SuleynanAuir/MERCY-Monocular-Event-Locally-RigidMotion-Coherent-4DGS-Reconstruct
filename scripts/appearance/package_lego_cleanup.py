"""Package a new static-attribute overlay without copying or mutating the PLY.

Every geometry/motion row is preserved. The resulting Gaussian appearance must
be rendered and independently audited before deployment; no audit is copied.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData
from scripts.events.export_lego_event_preview import verify_candidate
from scripts.appearance.preview_lego_sparse_cleanup import protected_selection
from utils.gaussian_cleanup import FORMAT, cleanup_contract, verify_gaussian_cleanup
from utils.rigid_checkpoint import sha256_file


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--key', default='guarded_surface_indices')
    p.add_argument('--opacity-factor', type=float, default=0.)
    p.add_argument('--attribute', choices=('opacity', 'dc_gray'), default='opacity')
    args = p.parse_args()
    if args.output.exists() or not 0 <= args.opacity_factor < 1:
        raise ValueError('Use new output directory and opacity factor in [0,1)')
    args.baseline, args.candidate, args.selection = (path.resolve(strict=True) for path in
                                                   (args.baseline, args.candidate, args.selection))
    args.output = args.output.resolve()
    cache, record, audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if cleanup_contract(cache, record, audit) is not None:
        raise ValueError('Cannot stack opacity cleanups')
    completion = json.loads(Path(cache['multiview_completion']['report']).read_text())
    rebuild = json.loads(Path(completion['parent_report']).read_text())
    with np.load(args.selection, allow_pickle=False) as archive:
        indices = protected_selection(archive[args.key], cache['gaussian_count'],
            rebuild['retained_gaussian_count'], np.load(args.baseline/'dynamic_indices.npy', allow_pickle=False))
        selection_meta = json.loads(str(archive['metadata_json']))
        colors = archive['dc_gray'] if args.attribute == 'dc_gray' else None
    selection_hashes = selection_meta.get('input_sha256', selection_meta.get('artifact_sha256', {}))
    if not len(indices) or cache['source_sha256'] != selection_hashes.get('source_ply'):
        raise ValueError('Selection must identify this exact cloud and contain at least one static row')
    if args.key == 'guarded_surface_indices' and (selection_meta.get('frames_checked') != 400
            or selection_meta.get('original_camera') != cache['camera']
            or selection_meta['input_sha256'].get('correction') != sha256_file(args.candidate/'correction.json')
            or selection_meta.get('selected_count') != len(indices)):
        raise ValueError('Guarded selection differs from its complete camera/motion scan')
    source = PlyData.read(cache['source'])['vertex'].data
    old = np.asarray(source['opacity'][indices], np.float64)
    if np.isnan(old).any() or (args.attribute == 'opacity' and np.any(old < -30)):
        raise ValueError('Invalid selected source opacity')
    if args.attribute == 'dc_gray':
        if (colors.dtype != np.float32 or colors.shape != indices.shape or not np.isfinite(colors).all()
                or np.any(colors < 0) or np.any(colors > 1)):
            raise ValueError('Invalid target grayscale colors')
        values = colors
    elif args.opacity_factor == 0:
        values = np.full(len(indices), -30, np.float32)
    else:
        activated = np.exp(-np.logaddexp(0, -old))*args.opacity_factor
        values = (np.log(activated)-np.log1p(-activated)).astype(np.float32)
        if np.any(values < -30):
            raise ValueError('Attenuation would exceed supported lower bound')
    model, baseline, candidate = [args.output/name for name in ('model', 'baseline', 'candidate')]
    for folder in (model, baseline, candidate):
        folder.mkdir(parents=True, exist_ok=False)
    overlay = model/'attribute-cleanup.npz'
    attributes = {'dc_gray' if args.attribute == 'dc_gray' else 'opacity_logit': values}
    np.savez_compressed(overlay, indices=indices.astype(np.int32), **attributes,
                        metadata_json=np.asarray(json.dumps({'format': FORMAT,
                            'source_sha256': cache['source_sha256'], 'selection': str(args.selection),
                            'selection_sha256': sha256_file(args.selection), 'selection_key': args.key})))
    report = {'format': FORMAT, 'source': cache['source'], 'source_sha256': cache['source_sha256'],
        'parent_baseline': str(args.baseline), 'parent_candidate': str(args.candidate),
        'parent_baseline_cache_sha256': sha256_file(args.baseline/'cache.json'),
        'parent_correction_sha256': sha256_file(args.candidate/'correction.json'),
        'parent_event_audit_sha256': sha256_file(args.candidate/'event-audit.json'),
        'parent_poses_sha256': sha256_file(args.baseline/'poses.npy'),
        'parent_dynamic_indices_sha256': sha256_file(args.baseline/'dynamic_indices.npy'),
        'overlay': str(overlay), 'overlay_sha256': sha256_file(overlay),
        'gaussian_count': cache['gaussian_count'], 'retained_gaussian_count': rebuild['retained_gaussian_count'],
        'modified_gaussian_count': len(indices), 'attributes': [args.attribute],
        'opacity_factor': args.opacity_factor if args.attribute == 'opacity' else None,
        'selection': str(args.selection),
        'selection_sha256': sha256_file(args.selection), 'selection_key': args.key,
        'selection_metadata': selection_meta,
        'scope': 'Explicit static Gaussian attribute overlay. Source PLY rows, XYZ, quaternion, scale, '
                 'all dynamic and elastic rows, compact completion and event-corrected motion are unchanged. '
                 'No 2D filter or display-only masking. This is an unaudited candidate until freshly rendered; '
                 'back-side geometry/appearance are still priors, not additional observations.'}
    report_path = model/'cleanup-report.json'
    write_json(report_path, report)
    contract = {'format': FORMAT, 'report': str(report_path), 'report_sha256': sha256_file(report_path),
                'overlay': str(overlay), 'overlay_sha256': sha256_file(overlay)}
    for name in ('poses.npy', 'dynamic_indices.npy'):
        (baseline/name).symlink_to((args.baseline/name).resolve(strict=True))
    new_cache = copy.deepcopy(cache)
    new_cache.update(gaussian_cleanup=contract, status='STATIC_ATTRIBUTE_CLEANUP_REQUIRES_NEW_EVENT_AND_VIEW_AUDIT')
    write_json(baseline/'cache.json', new_cache)
    new_record = copy.deepcopy(record)
    new_record['gaussian_cleanup'] = contract
    new_record['joint_structure']['input_sha256']['baseline_cache_metadata'] = sha256_file(baseline/'cache.json')
    write_json(candidate/'correction.json', new_record)
    verify_gaussian_cleanup(new_cache, new_record, baseline=baseline)
    print(json.dumps({'output': str(args.output), 'modified_gaussians': len(indices),
                      'motion_unchanged': True, 'source_ply_unchanged': True, 'audit': 'pending'}, indent=2))


if __name__ == '__main__':
    main()
