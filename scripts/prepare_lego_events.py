"""Calibrate raw Lego events against RGB timing; no Blender animation is read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from utils.event_supervision import EventStream, log_intensity, fit_thresholds, signed_event_change, consistency_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    video = cv2.VideoCapture(str(args.data / 'Lego.mkv'))
    fps = video.get(cv2.CAP_PROP_FPS)
    images = []
    while True:
        ok, image = video.read()
        if not ok:
            break
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    video.release()
    images = np.asarray(images)
    frames, height, width = images.shape
    events = EventStream(args.data / 'Lego_event', width, height)
    if abs(events.times[-1] - (frames - 1) / fps) > 2 / fps:
        raise ValueError('Event/video duration mismatch: verify timestamp units before optimization')
    # Fixed split before any calibration. Keep every fifth interval held out.
    calibration_ids = np.arange(5, frames - 1, 17)
    calibration_ids = calibration_ids[calibration_ids % 5 != 4]
    trials = []
    for offset_frames in (-1., -.5, 0., .5, 1.):
        counts = np.stack([events.counts(i / fps + offset_frames / fps,
                                        (i + 1) / fps + offset_frames / fps) for i in calibration_ids])
        for floor in (1 / 255, .01, .03):
            changes = np.stack([log_intensity(images[i + 1] / 255, floor) - log_intensity(images[i] / 255, floor)
                                for i in calibration_ids])
            try:
                thresholds = fit_thresholds(counts, changes)
            except ValueError as error:
                trials.append({'offset_frames': offset_frames, 'log_floor': floor,
                               'calibration_cosine': -1., 'error': str(error)})
                continue
            scores = [consistency_metrics(change, signed_event_change(count, thresholds), count.sum(0))
                      for change, count in zip(changes, counts)]
            score = float(np.mean([s['cosine'] for s in scores if s['observable']]))
            trial = {'offset_frames': offset_frames, 'log_floor': floor, 'thresholds': thresholds.tolist(), 'calibration_cosine': score}
            trials.append(trial)
            print(json.dumps(trial), flush=True)
    best = max(trials, key=lambda trial: trial['calibration_cosine'])
    if best['calibration_cosine'] < .5:
        raise ValueError('Event calibration is unreliable; do not optimize Gaussians against unaligned events')
    all_counts, report = [], []
    offset = best['offset_frames'] / fps
    for i in range(frames - 1):
        counts = events.counts(i / fps + offset, (i + 1) / fps + offset)
        if counts.max() > 65535:
            raise ValueError('Event count exceeds uint16 capacity')
        all_counts.append(counts.astype(np.uint16))
        delta = log_intensity(images[i + 1] / 255, best['log_floor']) - log_intensity(images[i] / 255, best['log_floor'])
        observed = signed_event_change(counts, best['thresholds'])
        metrics = consistency_metrics(delta, observed, counts.sum(0))
        report.append({'interval': i, 'start_seconds': i / fps, 'end_seconds': (i + 1) / fps,
                       'split': 'validation' if i % 5 == 4 else 'train', **metrics})
    counts = np.asarray(all_counts)
    metadata = {'format': 'lego_raw_event_supervision_v1', 'event_file': str(events.path),
                'event_count': len(events.events), 'event_order': 'x,y,t,p',
                'time_unit': 'seconds (duration and RGB change consistency checked)',
                'width': width, 'height': height, 'fps': fps, 'frame_count': frames,
                'event_start': float(events.times[0]), 'event_end': float(events.times[-1]),
                'calibration_ids': calibration_ids.tolist(), 'calibration': best, 'trials': trials,
                'intervals': report, 'event_generation_provenance': 'not supplied',
                'scope': 'Single-view image-space evidence only; not a proof of unique 3D motion.'}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'supervision.npz', images=images, counts=counts,
                        metadata_json=np.asarray(json.dumps(metadata)))
    (args.output / 'calibration.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Selected calibration', json.dumps(best), 'Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
