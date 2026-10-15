"""Compare actual reloaded surface renders before an explicitly unapproved UI preview.

Mesh edges/coverage are prior-consistency checks, not unseen RGB ground truth.
Event acceptance thresholds are never changed by this release check.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import binary_erosion
from utils.rigid_checkpoint import sha256_file
from utils.surface_detail_metrics import surface_comparison, mesh_structure_edges, edge_agreement


def check_observed_quality(parent_event, event, parent_error, error):
    failures = sorted(set(event['failed_intervals']) - set(parent_event['failed_intervals']))
    if failures:
        raise ValueError('New failed event intervals: ' + str(failures))
    for split in ('all', 'validation_events'):
        old, new = parent_event['summary'][split], event['summary'][split]
        if new['intervals'] != old['intervals'] or new['passed'] < old['passed']:
            raise ValueError('Event count regression: ' + split)
        for key, tolerance, direction in (('cosine', 1e-4, -1), ('polarity_agreement', 1e-4, -1),
                                           ('relative_l1', 1e-3, 1), ('unsupported_change_fraction', 1e-5, 1)):
            if not np.isfinite([old[key], new[key]]).all() or direction * (new[key]-old[key]) > tolerance:
                raise ValueError('Event metric regression beyond stated numerical trust region: ' + split + '/' + key)
    parent_error, error = np.asarray(parent_error), np.asarray(error)
    if (parent_error.shape != (400,) or error.shape != (400,)
            or not np.isfinite(parent_error).all() or not np.isfinite(error).all()):
        raise ValueError('Require all 400 finite original-view errors')
    if error.mean() > parent_error.mean()*1.001 + 1e-7 or np.max(error-parent_error) > 2e-5:
        raise ValueError('Original-view RGB error regression')
    return {'parent_mean_mae': float(parent_error.mean()), 'candidate_mean_mae': float(error.mean()),
            'maximum_per_frame_mae_increase': float(np.max(error-parent_error)),
            'new_failed_intervals': failures,
            'remaining_failed_intervals': event['failed_intervals']}


def view_checks(old, new, masks, targets):
    old_meta, new_meta = (json.loads(str(item['metadata_json'])) for item in (old, new))
    mask_meta = json.loads(str(masks['metadata_json']))
    if old_meta['views'] != new_meta['views'] or new_meta['views'] != mask_meta['views']:
        raise ValueError('Multiview plans disagree')
    old_gray, new_gray, old_alpha, new_alpha = (old['gray'], new['gray'], old['alpha'], new['alpha'])
    part_maps, depths = masks['part_ids'], masks['depth']
    records = []
    for index, view in enumerate(new_meta['views']):
        parts, depth = part_maps[index], depths[index]
        before, after = old_gray[index], new_gray[index]
        alpha_old, alpha_new = old_alpha[index], new_alpha[index]
        row = {'view': view, **surface_comparison(before, after, alpha_old, alpha_new, parts, targets)}
        interior = np.zeros(parts.shape, bool)
        for part in targets:
            interior |= binary_erosion(parts == part, iterations=1)
        count = int(((alpha_old >= .5) & (alpha_new < .5) & interior).sum())
        row['new_target_interior_holes'] = count
        if count:
            raise ValueError(f'New target surface interior holes at view {index}: {count}')
        edges, region = mesh_structure_edges(parts, depth, targets)
        row['parent_prior_edge'] = edge_agreement(before, edges, region)
        row['candidate_prior_edge'] = edge_agreement(after, edges, region)
        records.append(row)
    return records


def require_fresh_views(views, used_views):
    used = {(v['frame'], v['yaw'], v['pitch']) for v in used_views}
    fresh = {(v['frame'], v['yaw'], v['pitch']) for v in views}
    if len(views) != 12 or len(fresh) != 12 or used & fresh:
        raise ValueError('Fresh detail check cameras overlap restoration/development cameras or are duplicated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'candidate', 'supervision', 'masks', 'heldout_masks', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--detail-masks', type=Path,
                        help='Fresh 12-view check required after using the former 24-view check for restoration')
    parser.add_argument('--detail-plan-name', choices=('detail-check', 'detail-check-v2'), default='detail-check')
    parser.add_argument('--previous-diagnostic-masks', type=Path, action='append', default=[],
                        help='Already inspected camera plans that informed a later algorithm change; never independent')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from scripts.export_lego_event_preview import verify_candidate
    cache, _, event = verify_candidate(args.candidate/'baseline', args.candidate/'candidate', allow_unapproved=True)
    _, _, parent_event = verify_candidate(args.parent/'baseline', args.parent/'candidate', allow_unapproved=True)
    contract = cache['protected_surface_detail']
    report = json.loads(Path(contract['report']).read_text())
    if Path(report['parent_candidate']) != (args.parent/'candidate').resolve():
        raise ValueError('Wrong parent for this surface layer')
    observed_hash = sha256_file(args.supervision)
    if any(record['artifact_sha256']['supervision'] != observed_hash for record in (parent_event, event)):
        raise ValueError('Wrong original observations')
    old = np.load(args.parent/'candidate/rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    new = np.load(args.candidate/'candidate/rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    with np.load(args.supervision, allow_pickle=False) as archive:
        reference = archive['images']
    if old.shape != (400, 640, 640) or new.shape != old.shape or reference.dtype != np.uint8:
        raise ValueError('Wrong original observation/render shape or encoding')
    errors = [[], []]
    for frame in range(400):
        target = reference[frame].astype(np.float32)/255
        for index, array in enumerate((old, new)):
            errors[index].append(float(np.abs(array[frame].astype(np.float32)-target).mean()))
    result = {'observed': check_observed_quality(parent_event, event, *errors), 'view_checks': {}}
    preview = json.loads(Path(report['input_paths']['preview_json']).read_text())
    parent_roof = preview['variants']['original']['front_roof_mae']
    candidate_roof = preview['variants']['guarded_remove_add']['front_roof_mae']
    if candidate_roof > parent_roof*1.001 + 1e-7:
        raise ValueError('Original-camera roof ROI error regression')
    result['roof_observed_probe'] = {'frames': preview['front_roof_metric_frames'],
                                   'parent_mae': parent_roof, 'candidate_mae': candidate_roof}
    with np.load(report['input_paths']['preview_npz'], allow_pickle=False) as archive:
        actual_preview = archive['guarded_remove_add_front_gray']
    if actual_preview.shape != new.shape:
        raise ValueError('Packaged geometry and full preview timeline disagree')
    parity_sum, parity_max = 0., 0.
    for frame in range(400):
        delta = np.abs(actual_preview[frame].astype(np.float16).astype(np.float32)-new[frame].astype(np.float32))
        parity_sum += float(delta.mean()); parity_max = max(parity_max, float(delta.max()))
    del actual_preview
    if parity_sum/400 > 1e-7 or parity_max > .001:
        raise ValueError('Reloaded compact overlay differs from the actual full preview')
    result['reloaded_preview_parity'] = {'mean_abs_difference': parity_sum/400, 'maximum_abs_difference': parity_max,
        'scope': 'Compare after identical float16 storage; at most sparse one-bin raster-rounding differences.'}
    extra_guards_used = bool(preview.get('additional_guard_views'))
    if extra_guards_used and args.detail_masks is None:
        raise ValueError('Restoration used the former held-out cameras; require the fresh 12-view check')
    if extra_guards_used and sha256_file(args.heldout_masks) != report['input_sha256']['additional_guard_masks']:
        raise ValueError('Former held-out masks differ from the actual restoration guard')
    diagnostic_views = []
    result['previous_diagnostic_plans'] = []
    for path in args.previous_diagnostic_masks:
        with np.load(path, allow_pickle=False) as archive:
            previous_meta = json.loads(str(archive['metadata_json']))
        diagnostic_views.extend(previous_meta['views'])
        result['previous_diagnostic_plans'].append({'path': str(path.resolve()),
            'sha256': sha256_file(path), 'views': previous_meta['views'],
            'scope': 'Previously inspected during algorithm development, not independent validation.'})
    plans = [('multiview', args.masks, 36), ('heldout', args.heldout_masks, 24)]
    if args.detail_masks is not None:
        plans.append((args.detail_plan_name, args.detail_masks, 12))
    for name, mask_path, count in plans:
        new_path, old_path = (folder/'qa'/name/'render.npz' for folder in (args.candidate, args.parent))
        with np.load(old_path, allow_pickle=False) as before, np.load(new_path, allow_pickle=False) as after, \
                np.load(mask_path, allow_pickle=False) as masks:
            meta = json.loads(str(after['metadata_json']))
            if (len(meta['views']) != count or meta['artifact_sha256']['masks'] != sha256_file(mask_path)
                    or meta['artifact_sha256']['correction'] != event['artifact_sha256']['correction']
                    or meta.get('protected_surface_detail') != contract):
                raise ValueError('Multiview render does not identify the candidate/mask plan')
            before_meta = json.loads(str(before['metadata_json']))
            if before_meta['artifact_sha256']['masks'] != sha256_file(mask_path):
                raise ValueError('Parent multiview masks differ')
            for render_meta, audited in ((before_meta, parent_event), (meta, event)):
                for key in ('source_ply', 'correction', 'baseline_cache_metadata', 'color_residual', 'color_residual_report'):
                    if render_meta['artifact_sha256'].get(key) != audited['artifact_sha256'].get(key):
                        raise ValueError('Multiview render/audit provenance mismatch: '+key)
            if name == args.detail_plan_name:
                require_fresh_views(meta['views'],
                    preview['views']+preview.get('additional_guard_views', [])+diagnostic_views)
            rows = view_checks(before, after, masks, report['allowed_part_ids'])
        result['view_checks'][name] = {'views': rows, 'render_sha256': sha256_file(new_path),
            'used_for_restoration': name == 'multiview' or (name == 'heldout' and extra_guards_used)}
    rear = [row for row in result['view_checks']['multiview']['views']
            if row['view']['frame'] == 100 and 90 <= row['view']['yaw'] <= 270 and row['view']['pitch'] >= 0]
    edge_gain = np.mean([row['candidate_prior_edge']['f1']-row['parent_prior_edge']['f1'] for row in rear])
    if not np.isfinite(edge_gain) or edge_gain <= 0:
        raise ValueError('No measured rear structural-edge improvement')
    result['rear_prior_edge_mean_f1_gain'] = float(edge_gain)
    result.update(format='lego_protected_surface_preview_release_v1',
        acceptable_as_explicitly_unapproved_preview=True,
        event_approved_for_export=event['approved_for_export'],
        event_audit_sha256=sha256_file(args.candidate/'candidate/event-audit.json'),
        protected_surface_detail=contract,
        former_heldout_24_now_used_for_restoration=extra_guards_used,
        scope='Actual Gaussian render comparisons, not image sharpening. Full original-view RGB and raw-event '
              'regression checks. Every view check states whether it was used for restoration; the extra 12-view '
              'plan, when present, is checked after the frozen fix. All novel views use mesh priors only. '
              'No unseen-view texture truth or complete event approval is inferred. Native images require visual review.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['observed'], indent=2), flush=True)


if __name__ == '__main__':
    main()
