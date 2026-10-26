"""Frozen metric names, temporal splits and auditable experiment identities."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SECTIONS = {'comparison': '01_main_comparison', 'ablation': '02_ablation',
            'sensitivity': '03_sensitivity', 'efficiency': '04_efficiency',
            'significance': '05_significance'}
METRICS = [
    ('rgb_l1', 'RGB L1 ↓', -1), ('psnr', 'PSNR ↑', 1), ('ssim', 'SSIM ↑', 1),
    ('event_relative_l1', '留出事件相对 L1 ↓', -1), ('event_pass', '事件通过区间 ↑', 1),
    ('visible_missing', '可见缺失率 ↓', -1), ('back_missing', '背面缺失率 ↓', -1),
    ('joint_gap_max', '最大关节间隙 ↓', -1), ('angular_jerk', '角 jerk ↓', -1),
    ('position_jerk', '三维 jerk ↓', -1)]
METHODS = [('C1', 'Bind-GS'), ('C2', 'Align-GS'), ('C3 / A0', 'MERCY')]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def temporal_split(count, fps, guard=1):
    """Contiguous 60/20/20 split; guarded boundaries never share an endpoint."""
    if count < 15 or not np.isfinite(fps) or fps <= 0 or guard < 1:
        raise ValueError('Invalid timeline or boundary guard')
    a, b = int(count*.6), int(count*.8)
    frames = {'train': list(range(0, a-guard)),
              'validation': list(range(a+guard, b-guard)),
              'test': list(range(b+guard, count))}
    if any(len(v) < 2 for v in frames.values()):
        raise ValueError('Too few frames after temporal guards')
    intervals = {k: [i for i in v if i+1 in set(v)] for k, v in frames.items()}
    return dict(frame_count=count, fps=float(fps), frames=frames, intervals=intervals,
                guard_frames=guard, policy='contiguous_60_20_20',
                exposure='Parent G0/animation exposure must be declared separately')


def unavailable(reason):
    return {'value': None, 'status': 'unavailable', 'reason': reason}


def measured(value, **details):
    if not np.isfinite(value):
        raise ValueError('Metric must be finite; record exact-match PSNR separately')
    return {'value': float(value), 'status': 'measured', **details}
