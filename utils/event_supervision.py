"""Raw-event supervision and fail-closed image-space motion checks.

These checks constrain observable brightness changes, not unobservable 3D
depth. A positive polarity is increasing log brightness. Never use event-video
colors as event polarities, or interpret absence of events as proof of rest.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter


class EventStream:
    def __init__(self, path, width, height, time_scale=1.):
        self.path = Path(path)
        self.events = np.load(self.path, mmap_mode='r', allow_pickle=False)
        a = self.events
        if a.ndim != 2 or a.shape[1] != 4 or len(a) < 1 or a.dtype.kind not in 'fiu':
            raise ValueError('Events must be a nonempty numeric [N,4] array in x,y,t,p order')
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('Event time scale must be finite and positive')
        if not np.isfinite(a).all():
            raise ValueError('Non-finite event')
        if np.any(a[1:, 2] < a[:-1, 2]):
            raise ValueError('Event timestamps must be nondecreasing')
        if (np.any(a[:, :2] != np.floor(a[:, :2])) or np.any(a[:, :2] < 0) or
                np.any(a[:, 0] >= width) or np.any(a[:, 1] >= height)):
            raise ValueError('Invalid event pixel coordinates')
        if not np.all((a[:, 3] == 1) | (a[:, 3] == -1)):
            raise ValueError('Event polarity must be +1 or -1')
        self.width, self.height, self.time_scale = width, height, time_scale
        self.times = a[:, 2] * time_scale

    def counts(self, start, end):
        """Positive and negative counts in (start,end], preserving both signs."""
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('Invalid event interval')
        left, right = np.searchsorted(self.times, [start, end], side='right')
        a = self.events[left:right]
        index = a[:, 1].astype(np.int64) * self.width + a[:, 0].astype(np.int64)
        planes = [np.bincount(index[a[:, 3] == polarity], minlength=self.width * self.height)
                  .reshape(self.height, self.width) for polarity in (1, -1)]
        return np.asarray(planes, dtype=np.float32)


def log_intensity(gray, floor=1 / 255):
    gray = np.asarray(gray, dtype=np.float32)
    if not 0 < floor < 1 or not np.isfinite(gray).all() or np.any(gray < 0):
        raise ValueError('Expected finite nonnegative normalized intensity and a valid log floor')
    return np.log(np.maximum(gray, floor))


def signed_event_change(counts, thresholds):
    positive, negative = thresholds
    if not np.isfinite([positive, negative]).all() or min(positive, negative) <= 0:
        raise ValueError('Both contrast thresholds must be finite and positive')
    return counts[0] * positive - counts[1] * negative


def fit_thresholds(counts, changes):
    """Fit separate positive/negative thresholds; do not fit on validation data."""
    from scipy.optimize import nnls
    counts = np.asarray(counts)
    target = np.asarray(changes)
    active = counts.sum(axis=1) >= 1
    x = np.stack((counts[:, 0][active], -counts[:, 1][active]), -1)
    y = target[active]
    if len(y) < 10:
        raise ValueError('Insufficient calibration events')
    thresholds, _ = nnls(x.astype(np.float64), y.astype(np.float64))
    if np.any(thresholds < .001) or np.any(thresholds > 2):
        raise ValueError(f'Unusable event calibration: {thresholds.tolist()}')
    return thresholds.astype(np.float32)


def consistency_metrics(predicted, observed, activity=None, tolerance=2, minimum_change=.05):
    """Explicit missing-motion and unsupported-change penalties defeat freezing.

Thresholds are application-specific; these numbers are diagnostics, not a proof
of physical correctness. Cancellation pixels retain activity for support checks.
"""
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError('Expected two aligned 2D change images')
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError('Non-finite change image')
    obs_abs, pred_abs = np.abs(observed), np.abs(predicted)
    strong = obs_abs >= minimum_change
    support = strong if activity is None else np.asarray(activity) > 0
    near = maximum_filter(support.astype(np.uint8), size=2 * tolerance + 1) > 0
    energy = float(np.sum(obs_abs[strong]))
    if energy == 0:
        return {'observable': False, 'cosine': None, 'polarity_agreement': None,
                'relative_l1': None, 'unsupported_change_fraction': None,
                'predicted_change_mean': float(pred_abs.mean())}
    dot = float(np.sum(predicted * observed))
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(observed))
    same = (predicted * observed > 0) & (pred_abs >= minimum_change)
    return {'observable': True, 'cosine': dot / max(denominator, 1e-12),
            'polarity_agreement': float(obs_abs[strong & same].sum() / energy),
            'relative_l1': float(np.abs(predicted - observed).sum() / max(obs_abs.sum(), 1e-12)),
            'unsupported_change_fraction': float(pred_abs[~near].sum() / max(pred_abs.sum(), 1e-12)),
            'observed_change_mass': energy}


def acceptance_gate(metrics, reference, minimum_cosine=.35, minimum_polarity=.65,
                    max_unsupported=.25, max_relative_l1=1.1):
    """Fail closed: no automatic approval for unobservable or mismatched windows.

Both absolute floors and a same-window reference-video check are required. A
bad reference cannot lower the absolute acceptance floor. Caller must retain
the actual metrics, tolerances and rejected timestamps in the output report.
"""
    reasons = []
    if not metrics.get('observable') or not reference.get('observable'):
        return {'passed': False, 'reasons': ['insufficient_event_evidence']}
    thresholds = {'cosine': max(minimum_cosine, .7 * reference['cosine']),
                  'polarity_agreement': max(minimum_polarity, reference['polarity_agreement'] - .12),
                  'unsupported_change_fraction': min(max_unsupported, reference['unsupported_change_fraction'] + .12),
                  'relative_l1': min(max_relative_l1, reference['relative_l1'] + .35)}
    for key in ('cosine', 'polarity_agreement'):
        if not np.isfinite(metrics[key]) or metrics[key] < thresholds[key]:
            reasons.append(key)
    for key in ('unsupported_change_fraction', 'relative_l1'):
        if not np.isfinite(metrics[key]) or metrics[key] > thresholds[key]:
            reasons.append(key)
    return {'passed': not reasons, 'reasons': reasons, 'thresholds': thresholds}


def require_event_export_approval(report):
    """Reject incomplete, stale-policy or failed audit records before publication.

    Callers must also tie this report to the actual model/renders, not reuse a
    report from another asset. This is a quality check, not a digital signature.
    """
    if (report.get('format') != 'lego_event_motion_audit_v1'
            or report.get('approved_for_export') is not True
            or report.get('event_optimization_performed') is not True):
        raise ValueError('Event-supervised export rejected: missing correction or failed audit')
    count = report.get('frame_count', 0)
    records = report.get('intervals', [])
    if (not isinstance(count, int) or count < 2 or len(records) != count - 1
            or report.get('failed_intervals') != []):
        raise ValueError('Event-supervised export rejected: incomplete interval coverage')
    for index, record in enumerate(records):
        if (record.get('interval') != index or record.get('passed') is not True
                or not acceptance_gate(record['metrics'], record['reference_metrics'])['passed']):
            raise ValueError(f'Event-supervised export rejected at interval {index}')
