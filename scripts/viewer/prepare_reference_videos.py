#!/usr/bin/env python3
"""Prepare loss-minimized RGB/event frame pairs and audit their image-space alignment.

Uses supplied event visualizations, never treats their colors as training labels.
This checks image-space registration, not physical stereo camera calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_propagation, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def probe(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
        '-show_frames', '-show_entries',
        'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time', '-of', 'json', str(path)]))
    stream = data['streams'][0]
    num, den = map(int, stream['avg_frame_rate'].split('/'))
    times = np.asarray([float(f['best_effort_timestamp_time']) for f in data['frames']])
    fps = num / den
    if not np.allclose(times, np.arange(len(times)) / fps, rtol=0, atol=.0011):
        raise ValueError('Reference must start at zero with a constant frame rate')
    return dict(width=stream['width'], height=stream['height'], fps=fps, frame_count=len(times))


def frames(path, width, height):
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-vf',
        f'scale={width}:{height}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    try:
        while True:
            data = process.stdout.read(width * height * 3)
            if not data:
                break
            if len(data) != width * height * 3:
                raise ValueError('Truncated video frame')
            yield np.frombuffer(data, np.uint8).reshape(height, width, 3)
        if process.wait() != 0:
            raise ValueError('Video decoding failed')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def cosine(a, b):
    norm = np.sqrt(np.sum(a*a, (1, 2)) * np.sum(b*b, (1, 2)))
    valid = norm > 1e-6
    if not np.any(valid):
        raise ValueError('Not enough motion to verify reference alignment')
    return float(np.mean(np.sum(a*b, (1, 2))[valid] / norm[valid]))


def audit(rgb_path, event_path, count):
    # Reduce only for registration checking; output retains native resolution.
    rgb = np.stack(list(frames(rgb_path, 320, 320)))
    event = np.stack(list(frames(event_path, 320, 320)))
    delta = np.abs(np.diff(np.log(.03 + rgb.mean(-1)/255), axis=0))
    activity = event.max(-1).astype(float) - event.min(-1)
    delta = gaussian_filter(delta, (0, 1, 1))
    activity = gaussian_filter(activity, (0, 1, 1))
    ids = np.arange(3, count-3)
    train, heldout = ids[ids % 5 != 0], ids[ids % 5 == 0]
    scores = {offset: cosine(activity[train+offset], delta[train-1]) for offset in range(-2, 3)}
    offset = max(scores, key=scores.get)
    # Only the known end-anchored interval convention is supported here.
    # Never silently crop, duplicate end frames, or apply an unverified warp.
    if offset != -1 or scores[offset] < .6:
        raise ValueError(f'Unexpected event timing; inspect before exporting: {scores}')
    ids = heldout
    expected = delta[ids-1, 8:-8, 8:-8]
    spatial = {(dx, dy): cosine(activity[ids+offset, 8+dy:312+dy, 8+dx:312+dx], expected)
               for dy in range(-3, 4) for dx in range(-3, 4)}
    if max(spatial, key=spatial.get) != (0, 0):
        raise ValueError('Event imagery is spatially shifted; requires explicit registration')
    return dict(space='rgb_image', label='事件相机 · 对齐 RGB',
        method='Identity pixel mapping verified against RGB log-intensity changes; no crop or warp',
        event_frame_offset=offset, interval='(t_previous, t_current]',
        initial_frame='Empty: no preceding interval at t=0',
        temporal_scores={str(k): v for k, v in scores.items()},
        heldout_cosine=cosine(activity[heldout+offset], delta[heldout-1]),
        spatial_best_shift_check_pixels=[0, 0], spatial_check_resolution=[320, 320],
        spatial_scores={f'{x},{y}': v for (x, y), v in spatial.items()},
        note='Image-space consistency check of supplied videos, not independent physical camera calibration')


def white_rgb_preview(rgb, coverage=None):
    """Replace border-connected black backdrop, preserving enclosed dark details.

    A source-geometry matte also removes enclosed background without keying out
    black parts. Without a verified matte, retain the conservative border rule.
    """
    if coverage is not None:
        coverage = np.asarray(coverage, dtype=np.float32)
        if (coverage.shape != rgb.shape[:2] or not np.isfinite(coverage).all()
                or coverage.min() < 0 or coverage.max() > 1):
            raise ValueError('Invalid RGB foreground coverage')
        # Reject shifted/incorrect camera mattes instead of whitening real parts.
        bright = rgb.max(axis=2) > 32
        supported = binary_dilation(coverage > 0, iterations=2)
        if np.count_nonzero(bright & ~supported) > max(8, np.count_nonzero(bright) * .005):
            raise ValueError('Geometry matte is not aligned with the RGB foreground')
        # Original RGB is already composited over black. Add only the missing
        # background contribution in linear light; do not multiply alpha twice.
        value = rgb.astype(np.float32) / 255
        linear = np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)
        linear = np.clip(linear + 1 - coverage[..., None], 0, 1)
        value = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
        result = np.rint(value * 255).clip(0, 255).astype(np.uint8)
        result[coverage == 1] = rgb[coverage == 1]
        return result
    dark = rgb.max(axis=2) <= 8
    seed = np.zeros_like(dark)
    seed[0] = dark[0]; seed[-1] = dark[-1]
    seed[:, 0] = dark[:, 0]; seed[:, -1] = dark[:, -1]
    background = binary_propagation(seed, mask=dark)
    result = rgb.copy()
    result[background] = 255
    return result


class GeometryMatteSequence:
    """Require exact source identity, dimensions and complete frame coverage."""
    def __init__(self, directory, rgb_path, video):
        self.root = Path(directory)
        self.metadata = json.loads((self.root / 'matte.json').read_text())
        meta = self.metadata
        if (meta.get('format') != 'rgb_geometry_matte_v1'
                or any(meta.get(k) != video[k] for k in ('width', 'height', 'frame_count', 'fps'))
                or meta.get('rendered_frames') != video['frame_count']
                or meta.get('rgb_sha256') != hashlib.sha256(Path(rgb_path).read_bytes()).hexdigest()
                or len(meta.get('source_frames', [])) != video['frame_count']):
            raise ValueError('Geometry matte source or timeline does not match RGB')
        if any(not (self.root / f'{i:04d}.png').is_file() for i in range(video['frame_count'])):
            raise ValueError('Incomplete geometry matte sequence')

    def frame(self, index):
        with Image.open(self.root / f'{index:04d}.png') as image:
            if image.mode != 'L' or image.size != (self.metadata['width'], self.metadata['height']):
                raise ValueError('Invalid geometry matte image format or dimensions')
            return np.asarray(image, dtype=np.float32) / 255


def white_event_preview(event):
    result = event.copy()
    hi, lo = event.max(axis=2), event.min(axis=2)
    result[(lo >= 235) & (hi.astype(int)-lo <= 18)] = 255
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--white-background', action='store_true', help='Display-only white matte; preserve original videos')
    parser.add_argument('--matte-directory', type=Path, help='Verified source-geometry coverage; removes enclosed black backdrop')
    parser.add_argument('--event-reference', type=Path, help='Existing aligned white reference.json; preserve its event pixels exactly in lossless PNG pairs')
    args = parser.parse_args()
    meta, other = probe(args.rgb), probe(args.events)
    if meta != other or meta['frame_count'] < 12:
        raise ValueError('RGB and event video dimensions, FPS and frame count must match')
    if args.matte_directory and not args.white_background:
        parser.error('--matte-directory requires --white-background')
    mattes = GeometryMatteSequence(args.matte_directory, args.rgb, meta) if args.matte_directory else None
    alignment = audit(args.rgb, args.events, meta['frame_count'])
    existing = None
    if args.event_reference:
        # The existing pair has already applied the -1 event interval offset.
        # Read the matching frame directly; never offset it a second time.
        from utils.viewer_reference import ReferenceSequence
        expected = dict(meta, times_seconds=(np.arange(meta['frame_count']) / meta['fps']).tolist())
        existing = ReferenceSequence(args.event_reference, expected)
        previous = json.loads(args.event_reference.read_text())
        if (not args.white_background or previous.get('display_background') != 'white'
                or previous.get('alignment', {}).get('event_frame_offset') != alignment['event_frame_offset']
                or previous['width'] != meta['width'] or previous['height'] != meta['height']
                or previous.get('sources', {}).get('event', {}).get('sha256') != hashlib.sha256(args.events.read_bytes()).hexdigest()):
            raise ValueError('Existing event reference does not match the source and display settings')
    args.output.mkdir(parents=True, exist_ok=False)
    event_frames = frames(args.events, meta['width'], meta['height'])
    try:
        for i, rgb in enumerate(frames(args.rgb, meta['width'], meta['height'])):
            event = np.full_like(rgb, 255) if i == 0 else next(event_frames)
            if args.white_background:
                rgb, event = white_rgb_preview(rgb, mattes.frame(i) if mattes else None), white_event_preview(event)
            if existing:
                with Image.open(existing.frame_path(i)) as pair:
                    if pair.size != (2 * meta['width'], meta['height']):
                        raise ValueError('Existing event reference has incorrect dimensions')
                    event = np.asarray(pair.convert('RGB'))[:, meta['width']:].copy()
                Image.fromarray(np.concatenate([rgb, event], axis=1)).save(args.output / f'{i:04d}.png')
            else:
                Image.fromarray(np.concatenate([rgb, event], axis=1)).save(
                    args.output / f'{i:04d}.jpg', quality=97, subsampling=0)
        # Drain the final forward interval, whose end lies outside this timeline.
        for _ in event_frames:
            pass
    finally:
        event_frames.close()
    manifest = dict(format='rgb_event_reference_v1', **meta,
        times_seconds=(np.arange(meta['frame_count'])/meta['fps']).tolist(), alignment=alignment,
        display_background='white' if args.white_background else 'original',
        display_only=True, pixel_coordinates_unchanged=True,
        background_method='source_geometry_coverage' if mattes else 'border_connected' if args.white_background else 'original',
        foreground_matte=mattes.metadata if mattes else None,
        frame_extension='.png' if existing else '.jpg',
        event_pixels_preserved_from=str(args.event_reference) if existing else None,
        sources={name: dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for name, p in [('rgb', args.rgb), ('event', args.events)]})
    (args.output / 'reference.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), frames=meta['frame_count'],
                         heldout_cosine=alignment['heldout_cosine'], event_offset=-1)))


if __name__ == '__main__':
    main()
