"""Package a converged full-timeline guarded surface preview; never deploy it."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from utils.protected_surface_detail import FORMAT, additional_guard_views, validate_preview, verify_protected_surface_detail
from utils.rigid_checkpoint import sha256_file


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'preview', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--variant', choices=['guarded_remove_add'], default='guarded_remove_add')
    parser.add_argument('--require-additional-guard', action='store_true',
                        help='Require the extra 24 construction views as well as the canonical 36.')
    parser.add_argument('--require-global-support-preservation', action='store_true',
                        help='Require source-bound roof and zero-addition part support preservation.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output = args.output.resolve()
    for name in ('baseline', 'candidate', 'selection', 'preview'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    preview_dir = args.preview if args.preview.is_dir() else args.preview.parent
    preview_json_path, preview_npz_path = preview_dir/'preview.json', preview_dir/'preview.npz'
    from scripts.export_lego_event_preview import verify_candidate
    cache, record, audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if not cache.get('color_residual') or any(cache.get(k) for k in
        ('protected_surface_detail', 'gaussian_cleanup', 'static_surface_rebuild')):
        raise ValueError('Use the exact audited color-residual stage with no additional geometry overlays')
    with np.load(args.selection, allow_pickle=False) as data:
        selection = {k: data[k] for k in data.files}
    preview_json = json.loads(preview_json_path.read_text())
    if (args.require_global_support_preservation and not isinstance(
            preview_json.get('variants', {}).get(args.variant, {}).get('global_support_guard'), dict)):
        raise ValueError('This package requires global static-support preservation in the final preview')
    extra_views = additional_guard_views(preview_json, required=args.require_additional_guard)
    needed = ('proposed_retired_indices', 'retirement_maximum_probe_exposure', 'metadata_json',
              args.variant+'_active', args.variant+'_retired_indices', args.variant+'_maximum_probe_exposure')
    with np.load(preview_npz_path, allow_pickle=False) as data:
        preview = {k: data[k] for k in needed}
    if json.loads(str(preview['metadata_json'])) != preview_json:
        raise ValueError('Preview JSON and NPZ metadata do not agree')
    arrays, metadata = validate_preview(selection, preview, preview_json, cache['frame_count'], args.variant)
    if metadata.get('source_sha256') != cache['source_sha256']:
        raise ValueError('Selection identifies another source')
    completed = json.loads(Path(cache['multiview_completion']['report']).read_text())
    rebuild = json.loads(Path(completed['parent_report']).read_text())
    color = cache['color_residual']
    paths = {'source_ply': Path(cache['source']), 'baseline_cache_metadata': args.baseline/'cache.json',
        'correction': args.candidate/'correction.json', 'event_audit': args.candidate/'event-audit.json',
        'baseline_poses': args.baseline/'poses.npy', 'dynamic_indices': args.baseline/'dynamic_indices.npy',
        'color_residual': Path(color['residual']), 'color_residual_report': Path(color['report']),
        'selection': args.selection, 'preview_json': preview_json_path, 'preview_npz': preview_npz_path,
        'mesh': Path(completed['mesh_prior']), 'parts': Path(completed['parts']), 'binding': Path(rebuild['binding'])}
    if extra_views:
        paths['additional_guard_masks'] = Path(preview_json['additional_guard_masks'])
    hashes = {k: sha256_file(v) for k, v in paths.items()}
    for key in ('source_ply', 'correction', 'baseline_cache_metadata', 'selection', 'color_residual', 'color_residual_report', 'mesh'):
        if preview_json.get('artifact_sha256', {}).get(key) != hashes[key]:
            raise ValueError('Full preview was not produced from the exact parent: '+key)
    if extra_views and hashes['additional_guard_masks'] != preview_json['additional_guard_masks_sha256']:
        raise ValueError('Additional construction guard mask checksum differs')
    model, baseline, candidate = [args.output/name for name in ('model', 'baseline', 'candidate')]
    for folder in (model, baseline, candidate):
        folder.mkdir(parents=True, exist_ok=False)
    extension = model/'surface-detail.npz'
    arrays['metadata_json'] = np.asarray(json.dumps({'format': FORMAT, 'source_sha256': cache['source_sha256'],
        'selection_sha256': hashes['selection'], 'preview_npz_sha256': hashes['preview_npz']}))
    np.savez_compressed(extension, **arrays)
    count, retired = len(arrays['new_xyz']), len(arrays['retired_indices'])
    report = {'format': FORMAT, 'source': cache['source'], 'source_sha256': cache['source_sha256'],
        'parent_baseline': str(args.baseline), 'parent_candidate': str(args.candidate),
        'child_baseline': str(baseline), 'child_candidate': str(candidate),
        'input_paths': {k: str(v) for k, v in paths.items()}, 'input_sha256': hashes,
        'extension': str(extension), 'extension_sha256': sha256_file(extension),
        'source_gaussian_count': cache['gaussian_count'], 'retained_gaussian_count': rebuild['retained_gaussian_count'],
        'retired_gaussian_count': retired, 'added_gaussian_count': count,
        'rendered_gaussian_count': cache['gaussian_count']+count, 'frame_count': cache['frame_count'],
        'allowed_part_ids': metadata['target_part_ids'], 'retirement_bounds': metadata['retirement_bounds'],
        'variant': args.variant, 'unobserved_geometry_is_prior': True, 'unobserved_appearance_is_prior': True,
        'additional_guard_views_used_for_selection': bool(extra_views),
        'additional_guard_view_count': len(extra_views), 'construction_guard_view_count': 36+len(extra_views),
        'approved_for_export': False, 'fresh_event_audit': 'pending',
        'scope': 'Explicit static mesh-prior detail selected by converged full400 original-camera raster exposure guards '
                 'and zero NEW target-interior holes at finite preview views. Exact optimized parent color and motion remain '
                 'unchanged on retained rows. Unseen geometry/colors are priors, not observations. A fresh whole-timeline '
                 'event audit and broader actual-render checks are required; no parent acceptance is inherited. '
                 'Any additional 24 guard views participated in selection and are NOT independent validation.'}
    report_path = model/'surface-detail-report.json'
    write_json(report_path, report)
    contract = {'format': FORMAT, 'report': str(report_path), 'report_sha256': sha256_file(report_path),
                'extension': str(extension), 'extension_sha256': sha256_file(extension)}
    for name in ('poses.npy', 'dynamic_indices.npy'):
        (baseline/name).symlink_to((args.baseline/name).resolve(strict=True))
    child_cache = copy.deepcopy(cache)
    child_cache.update(protected_surface_detail=contract, rendered_gaussian_count=report['rendered_gaussian_count'],
                       status='PROTECTED_SURFACE_DETAIL_REQUIRES_FRESH_RGB_EVENT_AND_VIEW_AUDITS')
    write_json(baseline/'cache.json', child_cache)
    child_record = copy.deepcopy(record)
    child_record['protected_surface_detail'] = contract
    child_record['joint_structure']['input_sha256']['baseline_cache_metadata'] = sha256_file(baseline/'cache.json')
    write_json(candidate/'correction.json', child_record)
    verify_protected_surface_detail(child_cache, child_record, baseline=baseline)
    print(json.dumps({'output': str(args.output), 'added': count, 'retired': retired,
        'source_unchanged': True, 'motion_unchanged': True, 'fresh_event_audit': 'pending', 'deployed': False}), flush=True)


if __name__ == '__main__':
    main()
