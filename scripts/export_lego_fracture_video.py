"""Diagnostic video of actual cached Gaussian renders beside the RGB reference.

This exports evidence only. It does not replace the interactive Gaussian UI,
change a model, interpolate frames, or grant event approval.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def validate_render_frame(frame):
    if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
        raise ValueError('Rendered grayscale must be finite and within [0, 1]')


def select_worst_frames(scores, count=8, minimum_gap=12):
    selected = []
    ranking = np.argsort(np.asarray(scores), kind='stable')[::-1]
    for index in ranking:
        if all(abs(int(index) - prior) >= minimum_gap for prior in selected):
            selected.append(int(index))
            if len(selected) == min(count, len(scores)):
                break
    return sorted(selected)


def labeled_panel(gray, label, index, fps, zoom):
    height, width = gray.shape
    gray = np.rint(np.clip(gray, 0, 1) * 255).astype(np.uint8)
    gray = cv2.resize(gray, (width * zoom, height * zoom), interpolation=cv2.INTER_NEAREST)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panel = cv2.copyMakeBorder(panel, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(panel, label[:60], (7, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f'f={index}  t={index / fps:.3f}s', (7, 37), cv2.FONT_HERSHEY_SIMPLEX,
                .42, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--renders', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', help='ASCII labels in the same order as --renders')
    parser.add_argument('--roi', type=int, nargs=4, default=[160, 140, 380, 380])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, help='Number of consecutive frames from zero; default is all source frames')
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--zoom', type=int, default=2)
    args = parser.parse_args()
    report_path = args.output.with_suffix('.json')
    contact_path = args.output.with_name(args.output.stem + '-worst.png')
    for path in (args.output, report_path, contact_path):
        if path.exists():
            raise FileExistsError(path)
    if args.output.suffix.lower() != '.mp4':
        raise ValueError('Output must have .mp4 extension')
    if not np.isfinite(args.fps) or args.fps <= 0 or args.zoom not in (1, 2, 3, 4):
        raise ValueError('fps must be positive; zoom must be 1, 2, 3 or 4')
    labels = args.labels or [path.name for path in args.renders]
    if (len(labels) != len(args.renders) or len(set(labels)) != len(labels)
            or any(not label or not label.isascii() for label in labels)):
        raise ValueError('Supply one unique, nonempty ASCII label per rendered sequence')
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
    if images.ndim != 3 or images.dtype != np.uint8:
        raise ValueError('Expected reference images [T,H,W] uint8')
    frame_count = len(images) if args.frames is None else args.frames
    if not 1 <= frame_count <= len(images):
        raise ValueError('frames must be positive and no larger than the source sequence')
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= images.shape[2] and 0 <= y0 < y1 <= images.shape[1]):
        raise ValueError('Invalid ROI')
    rendered = [np.load(path / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False) for path in args.renders]
    for path, frames in zip(args.renders, rendered):
        if frames.shape != images.shape or not np.issubdtype(frames.dtype, np.floating):
            raise ValueError(f'Render sequence shape/dtype differs from the reference: {path}')
    width = (x1 - x0) * args.zoom * (len(rendered) + 1)
    height = (y1 - y0) * args.zoom + 48
    if width % 2 or height % 2:
        raise ValueError('MP4 requires even output dimensions; choose zoom=2 or an even ROI size')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV MP4 encoder could not be opened')
    records = [[] for _ in rendered]
    try:
        for index in range(frame_count):
            reference = images[index, y0:y1, x0:x1].astype(np.float32) / 255
            panels = [labeled_panel(reference, 'RGB reference (not a Gaussian render)', index, args.fps, args.zoom)]
            for model_index, (label, frames) in enumerate(zip(labels, rendered)):
                # Validate the full frame, not only the selected diagnostic ROI.
                validate_render_frame(frames[index])
                actual = np.asarray(frames[index, y0:y1, x0:x1], dtype=np.float32)
                panels.append(labeled_panel(actual, label, index, args.fps, args.zoom))
                expected = reference > .12
                foreground = actual > .08
                records[model_index].append({
                    'frame': index, 'roi_rgb_l1': float(np.abs(actual - reference).mean()),
                    'missing_reference_bright_fraction': float((expected & ~foreground).sum() / max(expected.sum(), 1)),
                    'foreground_in_reference_dark_fraction': float((foreground & (reference < .03)).sum() / max(foreground.sum(), 1)),
                })
            writer.write(np.concatenate(panels, axis=1))
            if index % 100 == 0:
                print(f'Exported diagnostic frame {index}/{frame_count}', flush=True)
    finally:
        writer.release()
    # Check that this is a decodable one-pass output, not a silently truncated file.
    capture = cv2.VideoCapture(str(args.output))
    decoded = 0
    try:
        encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError('Decoded MP4 shape mismatch')
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count or abs(encoded_fps - args.fps) > .01:
        raise ValueError(f'MP4 verification failed: frames={decoded}/{frame_count}, fps={encoded_fps}')
    worst = select_worst_frames([row['roi_rgb_l1'] for row in records[-1]])
    sheets = []
    for index in worst:
        reference = images[index, y0:y1, x0:x1].astype(np.float32) / 255
        panels = [labeled_panel(reference, 'RGB reference', index, args.fps, args.zoom)]
        for label, frames in zip(labels, rendered):
            panels.append(labeled_panel(np.asarray(frames[index, y0:y1, x0:x1], dtype=np.float32),
                                        label, index, args.fps, args.zoom))
        sheets.append(np.concatenate(panels, axis=1))
    if not cv2.imwrite(str(contact_path), np.concatenate(sheets, axis=0)):
        raise RuntimeError('Could not write contact sheet')
    report = {
        'format': 'lego_fracture_diagnostic_video_v1',
        'scope': __doc__, 'reference': str(args.supervision),
        'frame_count': frame_count, 'source_frame_count': len(images), 'fps': args.fps,
        'duration_seconds': frame_count / args.fps, 'frame_indices': list(range(frame_count)),
        'playback_loops_encoded': 1, 'temporal_interpolation': False,
        'roi_xyxy': args.roi, 'nearest_neighbor_zoom': args.zoom,
        'dimensions': [width, height], 'codec': 'mp4v', 'decoded_frame_count': decoded,
        'contact_sheet': str(contact_path), 'contact_frames': worst,
        'contact_selection': 'Largest final-candidate ROI RGB L1, selected at least 12 source frames apart',
        'models': {label: {'render_path': str(path), 'summary': {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'frame'},
            'frames': rows} for label, path, rows in zip(labels, args.renders, records)},
        'metric_caveat': 'ROI brightness metrics are 2D diagnostics, not proof of 3D completeness, part attachment, or event acceptance.',
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'frames': decoded, 'contact_frames': worst,
                      'models': {label: model['summary'] for label, model in report['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
