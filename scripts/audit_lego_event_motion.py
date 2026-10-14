"""Check actual Gaussian renders against raw events; fail export on violations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.event_supervision import log_intensity, signed_event_change, consistency_metrics, acceptance_gate


def pool(image, factor):
    h, w = image.shape
    if h % factor or w % factor:
        raise ValueError('Invalid comparison scale')
    return image.reshape(h // factor, factor, w // factor, factor).mean((1, 3))


def audit(rendered, supervision, factor=8):
    with np.load(supervision, allow_pickle=False) as archive:
        images, counts = archive['images'], archive['counts']
        meta = json.loads(str(archive['metadata_json']))
    if (rendered.ndim != 3 or len(rendered) < 2 or rendered.shape != images.shape
            or counts.shape != (len(images) - 1, 2, *images.shape[1:])):
        raise ValueError('Render/video frame shapes disagree')
    if (not np.isfinite(meta['fps']) or meta['fps'] <= 0
            or meta['frame_count'] != len(images) or factor not in (1, 2, 4, 8)):
        raise ValueError('Invalid audit frame count, FPS or comparison factor')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Invalid event counts')
    floor, thresholds = meta['calibration']['log_floor'], meta['calibration']['thresholds']
    records = []
    for i, count in enumerate(counts):
        observed = pool(signed_event_change(count.astype(np.float32), thresholds), factor)
        activity = pool(count.astype(np.float32).sum(0), factor)
        predicted = pool(log_intensity(rendered[i + 1], floor) - log_intensity(rendered[i], floor), factor)
        reference = pool(log_intensity(images[i + 1] / 255, floor) - log_intensity(images[i] / 255, floor), factor)
        # At 80x80, one-cell tolerance is 8 source pixels; magnitudes/polarities
        # are still checked at their original cells, not shifted to find a match.
        metrics = consistency_metrics(predicted, observed, activity, tolerance=1, minimum_change=.01)
        reference_metrics = consistency_metrics(reference, observed, activity, tolerance=1, minimum_change=.01)
        gate = acceptance_gate(metrics, reference_metrics)
        reference_gate = acceptance_gate(reference_metrics, reference_metrics)
        records.append({'interval': i, 'start_seconds': i / meta['fps'], 'end_seconds': (i + 1) / meta['fps'],
                        'split': 'validation_events' if i % 5 == 4 else 'train_events',
                        'reference_self_check_passed': reference_gate['passed'],
                        'reference_self_check_reasons': reference_gate['reasons'],
                        'metrics': metrics, 'reference_metrics': reference_metrics, **gate})
    observable = [r for r in records if r['metrics']['observable']]
    failures = [r['interval'] for r in records if not r['passed']]
    stats = {}
    for split in ('all', 'train_events', 'validation_events'):
        selected = [r for r in observable if split == 'all' or r['split'] == split]
        stats[split] = {'intervals': len(selected), 'passed': sum(r['passed'] for r in selected),
                       **{key: float(np.mean([r['metrics'][key] for r in selected])) if selected else None
                          for key in ('cosine', 'polarity_agreement', 'relative_l1', 'unsupported_change_fraction')}}
    return {'format': 'lego_event_motion_audit_v1', 'approved_for_export': not failures,
            'frame_count': len(images), 'fps': meta['fps'],
            'render_resolution': list(rendered.shape[1:]),
            'comparison_resolution': [rendered.shape[1] // factor, rendered.shape[2] // factor],
            'scope': 'Coarse image-space event consistency, not complete/unique 3D reconstruction or pixel-perfect motion.',
            'validation_note': 'Validation event intervals are withheld from event fitting; adjacent RGB frames may be used in fitting.',
            'calibration': meta['calibration'], 'failed_intervals': failures,
            'failure_reason_counts': dict(Counter(reason for record in records for reason in record['reasons'])),
            'reference_self_check_failed_intervals': [r['interval'] for r in records if not r['reference_self_check_passed']],
            'reference_note': 'Reference self-check failures flag weak/incompatible evidence; they remain unapproved, not silently exempted.',
            'summary': stats, 'intervals': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--factor', type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(np.load(args.renders, mmap_mode='r', allow_pickle=False), args.supervision, args.factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'intervals'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
