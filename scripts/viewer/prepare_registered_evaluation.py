"""Build source RGB/event evaluation inputs in a registered model coordinate frame.

Camera registration is an estimate; masks are RGB background proxies, not part GT.
No event contrast threshold is invented. UI preview images are not used as GT.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from scripts.viewer.prepare_reference_videos import probe, frames
from utils.experiment_format import sha256


def prepare(data, stem, parts_path, binding_path, output, allow_missing_events=False):
    video = data / (stem + '.mkv')
    event_path = data / (stem + '_event.npy')
    info = probe(video)
    intrinsics = json.loads((data / 'camera_intrinsics.json').read_text())
    binding = json.loads(binding_path.read_text())
    # Deforming exports record the same source identity inside provenance.
    registration = binding.get('provenance', binding)
    expected_sha = registration.get('parts_sha256', registration.get('source_sha256'))
    if sha256(parts_path) != expected_sha:
        raise ValueError('Motion source identity mismatch')
    with np.load(parts_path, allow_pickle=False) as source:
        meta = json.loads(str(source['metadata_json']))
        source_frames = source['source_frames'] if 'source_frames' in source else np.asarray(meta['source_frames'])
    if len(source_frames) != info['frame_count']:
        raise ValueError('RGB and motion frame counts differ')
    w, h, fps = info['width'], info['height'], info['fps']
    if (w, h) != (intrinsics['width'], intrinsics['height']):
        raise ValueError('RGB/intrinsics resolution mismatch')
    reg = np.asarray(registration['registration'])
    scale = np.cbrt(np.linalg.det(reg[:3, :3]))
    if scale <= 0 or not np.allclose((reg[:3, :3]/scale).T @ (reg[:3, :3]/scale), np.eye(3), atol=1e-6):
        raise ValueError('Expected a proper similarity registration')
    pose = reg @ np.asarray(meta['camera_pose'])
    pose[:3, :3] /= scale
    camera = dict(pose_opengl_c2w=pose.tolist(), width=w, height=h,
                  K=intrinsics['K'], fovy_degrees=float(np.rad2deg(2*np.arctan(h/(2*intrinsics['fy'])))))
    events = None
    if event_path.exists() or not allow_missing_events:
        events = np.load(event_path, mmap_mode='r', allow_pickle=False)
        if events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all():
            raise ValueError('Invalid raw events')
        xy = events[:, :2]
        if (np.any(np.diff(events[:, 2]) < 0) or not np.isin(events[:, 3], [-1, 1]).all()
                or not np.equal(xy, np.floor(xy)).all() or (xy < 0).any()
                or (xy[:, 0] >= w).any() or (xy[:, 1] >= h).any()
                or events[0, 2] < 0 or events[-1, 2] > (info['frame_count']-1)/fps+1e-6):
            raise ValueError('Raw event pixel/time convention mismatch')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    total = 0
    for i, rgb in enumerate(frames(video, w, h)):
        # Source videos have a black background; validate the border before using it.
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        if np.quantile(border.max(axis=1), .90) > 12:
            raise ValueError('Black-background proxy is not valid for this source')
        Image.fromarray(rgb).save(output / f'{i:04d}-rgb.png')
        Image.fromarray((rgb.max(axis=2) > 12).astype(np.uint8)*255).save(output / f'{i:04d}-mask.png')
        row = dict(frame=i, time_seconds=i/fps, rgb=f'{i:04d}-rgb.png', foreground_mask=f'{i:04d}-mask.png')
        if i and events is not None:
            lo, hi = np.searchsorted(events[:, 2], [(i-1)/fps, i/fps], side='right')
            event = events[lo:hi]
            counts = {}
            for sign, name in [(1, 'positive'), (-1, 'negative')]:
                selected = event[event[:, 3] == sign]
                counts[name] = np.bincount((selected[:, 1]*w+selected[:, 0]).astype(np.int64), minlength=h*w).reshape(h,w).astype(np.uint32)
            np.savez_compressed(output / f'{i:04d}-events.npz', **counts, t0=(i-1)/fps, t1=i/fps)
            row['events'] = f'{i:04d}-events.npz'
            total += len(event)
        rows.append(row)
    if len(rows) != info['frame_count']:
        raise ValueError('Decoded RGB frame count mismatch')
    record = dict(format='scgs_experiment_reference_v1', camera=camera, background='dark', background_rgb=[0.,0.,0.],
        mask_quality='rgb_background_proxy', mask_definition='Source RGB max channel > 12/255; dark material may be omitted',
        samples=rows, provenance=dict(rgb_sha256=sha256(video), events_sha256=sha256(event_path) if events is not None else None,
            parts_sha256=sha256(parts_path), binding_sha256=sha256(binding_path), intrinsics_sha256=sha256(data/'camera_intrinsics.json'),
            camera='Blender source camera mapped by existing similarity registration; estimated model alignment, not independent calibration',
            masks='Source RGB black-background threshold, shared across all versions', source_frames=source_frames.tolist(),
            time_mapping='RGB index / supplied video FPS; Blender source frame indices from motion archive', event_count=total),
        unavailable=dict(part_mask='No independent semantic part masks supplied',
                         events='Raw event file is not available on this host' if events is None else 'Raw event windows supplied',
                         event_response='Independent contrast thresholds/response calibration are unavailable'))
    (output/'evaluation.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(frames=len(rows), events=total, output=str(output/'evaluation.json'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parts', 'binding', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--allow-missing-events', action='store_true')
    a = p.parse_args()
    prepare(a.data, a.stem, a.parts, a.binding, a.output, a.allow_missing_events)
