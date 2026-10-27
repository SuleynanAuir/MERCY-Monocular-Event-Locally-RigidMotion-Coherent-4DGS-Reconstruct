"""Fit and validate a shared empirical event response using original RGB only.

Calibration intervals are fixed before fitting, disjoint from reported validation
windows. No method render or model is read. This is not a sensor calibration.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.event_supervision import EventStream, log_intensity, fit_thresholds, signed_event_change, consistency_metrics
from utils.experiment_format import sha256, save_json


def calibrate(video_path, events_path, output):
    import cv2
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    images = []
    try:
        while True:
            ok, bgr = video.read()
            if not ok: break
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    finally:
        video.release()
    if len(images) < 12 or fps <= 0: raise ValueError('Insufficient RGB observations')
    images = np.asarray(images)
    stream = EventStream(events_path, images.shape[2], images.shape[1])
    if abs(stream.times[-1]-(len(images)-1)/fps) > 2/fps:
        raise ValueError('Event/RGB duration mismatch')
    # Whole two-frame windows; calibration cannot leak into step-2 validation.
    windows = [(i, i+2) for i in range(0, len(images)-2, 2)]
    selected = np.arange(len(windows)) % 5 == 1
    if selected.sum() < 2: raise ValueError('Too few independent calibration windows')
    counts = np.asarray([stream.counts(a/fps, b/fps) for a,b in windows])
    trials = []
    for floor in (1/255, .01, .03):
        changes = np.asarray([log_intensity(images[b]/255, floor)-log_intensity(images[a]/255, floor) for a,b in windows])
        try:
            thresholds = fit_thresholds(counts[selected], changes[selected])
            scores = [consistency_metrics(d, signed_event_change(c, thresholds), c.sum(0))
                      for d,c in zip(changes, counts)]
            def average(mask):
                values = [r['cosine'] for i,r in enumerate(scores) if mask[i] and r['observable']]
                return float(np.mean(values)) if values else -1.
            trials.append(dict(log_floor=floor, thresholds=thresholds.tolist(), offset_frames=0,
                               calibration_cosine=average(selected), validation_cosine=average(~selected)))
        except ValueError:
            continue
    if not trials: raise ValueError('No usable empirical response')
    best = max(trials, key=lambda row: row['calibration_cosine'])
    if min(best['calibration_cosine'], best['validation_cosine']) < .5:
        raise ValueError('RGB/event response failed fixed cosine >= 0.5 check: '+json.dumps(best))
    result = dict(format='scgs_empirical_event_response_v1', calibration=best, trials=trials,
                  calibration_ids=[i for use,(a,b) in zip(selected,windows) if use for i in range(a,b)],
                  validation_windows=[list(w) for use,w in zip(selected,windows) if not use],
                  video_sha256=sha256(video_path), events_sha256=sha256(events_path), fps=fps,
                  provenance='Original RGB/raw-event empirical fit; no method outputs. Fixed split, zero timestamp offset; not independent sensor calibration.')
    save_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(calibrate(a.video, a.events, a.output), indent=2))
