"""Read-only residual diagnosis. Fitted gains below are NOT applied to renders."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--supervision', type=Path, required=True)
    args = p.parse_args()
    with np.load(args.supervision, allow_pickle=False) as data:
        counts = data['counts']
        metadata = json.loads(str(data['metadata_json']))
    positive, negative = metadata['calibration']['thresholds']
    floor = metadata['calibration']['log_floor']
    source = np.load(args.candidate / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    pool = lambda v: v.reshape(80, 8, 80, 8).mean((1, 3))
    logs = np.stack([pool(np.log(np.maximum(frame.astype(np.float32), floor))) for frame in source])
    rows = []
    for i, count in enumerate(counts):
        observed = pool(count[0].astype(np.float32)*positive-count[1].astype(np.float32)*negative)
        mass = np.abs(observed).sum()
        if i % 5 == 4 or mass < 3:
            continue
        predicted = logs[i+1] - logs[i]
        gain = float((predicted*observed).sum() / max(np.square(predicted).sum(), 1e-9))
        error = float(np.abs(predicted-observed).sum() / mass)
        fitted = float(np.abs(gain*predicted-observed).sum() / mass)
        options = []
        for lag in range(-3, 4):
            j = i+lag
            if 0 <= j < len(source)-1:
                options.append((float(np.abs((logs[j+1]-logs[j])-observed).sum() / mass), lag))
        best, lag = min(options)
        rows.append({'interval': i, 'gain_diagnostic_only': gain, 'relative_l1': error,
                     'gain_fitted_relative_l1_diagnostic_only': fitted, 'best_lag': lag, 'best_lag_l1': best})
    print(json.dumps({'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'interval'},
                      'lag_counts': {lag: sum(r['best_lag'] == lag for r in rows) for lag in range(-3,4)},
                      'gain_percentiles': np.percentile([r['gain_diagnostic_only'] for r in rows], [10,50,90]).tolist()}, indent=2))


if __name__ == '__main__':
    main()
