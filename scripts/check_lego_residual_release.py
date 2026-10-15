"""Fail closed on image/detail/event regression before diagnostic UI rollout.

This may accept a better PREVIEW with unresolved event intervals. It never
converts that preview into an event-approved reconstruction.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.rigid_checkpoint import sha256_file


def validate_quality(parent_event, candidate_event, detail):
    regressions = sorted(set(candidate_event['failed_intervals'])-set(parent_event['failed_intervals']))
    if regressions:
        raise ValueError('New event failures: '+str(regressions))
    for split in ('all', 'validation_events'):
        old, new = parent_event['summary'][split], candidate_event['summary'][split]
        if new['passed'] < old['passed'] or new['intervals'] != old['intervals']:
            raise ValueError('Event pass count regressed: '+split)
        for key in ('cosine', 'polarity_agreement'):
            if new[key] < old[key]-1e-7:
                raise ValueError('Event metric regressed: '+split+'/'+key)
        for key in ('relative_l1', 'unsupported_change_fraction'):
            if new[key] > old[key]+1e-7:
                raise ValueError('Event error regressed: '+split+'/'+key)
    overall = detail['full_image']
    if overall['candidate'] > .95*overall['parent']:
        raise ValueError('Require at least 5% image-error improvement for this preview')
    for name, values in detail['rois'].items():
        before, after = values['parent'], values['candidate']
        if after['rgb_l1'] >= before['rgb_l1'] or after['edge_f1'] < before['edge_f1']:
            raise ValueError('Local image/detail regression: '+name)
    return {'new_event_failures': regressions,
            'fixed_event_intervals': sorted(set(parent_event['failed_intervals'])-set(candidate_event['failed_intervals'])),
            'global_mae_reduction_percent': 100*(1-overall['candidate']/overall['parent']),
            'local_mae_reduction_percent': {k: 100*(1-v['candidate']['rgb_l1']/v['parent']['rgb_l1']) for k, v in detail['rois'].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'candidate', 'fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent_event = json.loads((args.parent/'candidate/event-audit.json').read_text())
    event = json.loads((args.candidate/'candidate/event-audit.json').read_text())
    fit_event = json.loads((args.fit/'residual-event-audit.json').read_text())
    fit_finished = json.loads((args.fit/'completion.json').read_text())
    detail = json.loads((args.fit/'heldout-detail.json').read_text())
    if (event['summary'] != fit_event['summary'] or event['failed_intervals'] != fit_event['failed_intervals']
            or sha256_file(args.candidate/'candidate/rendered_gray.npy') != fit_finished['rendered_gray_sha256']):
        raise ValueError('Reloaded candidate differs from the actual optimized renders/audit')
    report = validate_quality(parent_event, event, detail)
    report['view_checks'] = {}
    for name in ('multiview', 'heldout'):
        path = args.candidate/'qa'/name/'render.npz'
        with np.load(path, allow_pickle=False) as new, np.load(args.parent/'qa'/name/'render.npz', allow_pickle=False) as old:
            meta, old_meta = json.loads(str(new['metadata_json'])), json.loads(str(old['metadata_json']))
            if meta['views'] != old_meta['views'] or meta['artifact_sha256']['correction'] != event['artifact_sha256']['correction']:
                raise ValueError('Multiview camera or actual candidate identity differs')
            alpha_error = float(np.abs(new['alpha']-old['alpha']).max())
            gray_change = np.abs(new['gray']-old['gray']).mean((1, 2))
            if alpha_error > 1e-6 or gray_change.max() > .01:
                raise ValueError('Novel-view coverage or appearance trust region regressed: '+name)
            report['view_checks'][name] = {'count': len(meta['views']), 'alpha_max_absolute_error': alpha_error,
                'maximum_view_gray_mae_change': float(gray_change.max()), 'render_sha256': sha256_file(path)}
    report.update(format='lego_residual_preview_release_check_v1',
        acceptable_as_explicitly_unapproved_preview=True,
        event_approved_for_export=event['approved_for_export'], remaining_failed_intervals=event['failed_intervals'],
        source_ply_sha256=event['artifact_sha256']['source_ply'],
        correction_sha256=event['artifact_sha256']['correction'],
        residual_sha256=event['artifact_sha256']['color_residual'],
        event_audit_sha256=sha256_file(args.candidate/'candidate/event-audit.json'),
        scope='Image metrics use same-camera timestamps withheld from the current residual stage only. '
              'Parent-stage RGB independence is not claimed. Novel-view references are previous Gaussian renders, '
              'not unseen geometry/texture truth. Unresolved event intervals remain unapproved and visible in UI.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
