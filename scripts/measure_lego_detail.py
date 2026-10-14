"""Reference-guided, fixed-ROI detail checks of actual Gaussian render caches.

This is an image diagnostic, not an event gate or proof of 3D completeness.
No sharpening, resampling, or learned enhancement is applied to evaluated images.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_ROIS = {'arm': [160, 140, 380, 380], 'gear': [360, 230, 500, 325]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gray_float(frame):
    frame = np.asarray(frame)
    if frame.ndim != 2 or min(frame.shape) < 7:
        raise ValueError('Expected a grayscale image at least 7 by 7 pixels')
    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255
    result = frame.astype(np.float32)
    if not np.isfinite(result).all() or result.min() < -1e-6 or result.max() > 1 + 1e-6:
        raise ValueError('Render must be finite and within [0, 1]')
    return np.clip(result, 0, 1)


def sobel(image):
    """Standard 3 by 3 Sobel derivatives divided by 8; reflect boundary."""
    a = np.pad(image, 1, mode='reflect')
    gx = (a[:-2, 2:] + 2 * a[1:-1, 2:] + a[2:, 2:]
          - a[:-2, :-2] - 2 * a[1:-1, :-2] - a[2:, :-2]) / 8
    gy = (a[2:, :-2] + 2 * a[2:, 1:-1] + a[2:, 2:]
          - a[:-2, :-2] - 2 * a[:-2, 1:-1] - a[:-2, 2:]) / 8
    return gx, gy, np.hypot(gx, gy)


def thin_edges(gx, gy, magnitude, threshold):
    """Four-direction nonmaximum suppression; omit the two-pixel ROI border."""
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    output = np.zeros(magnitude.shape, dtype=bool)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
    sectors = np.floor((angle + 22.5) / 45).astype(int) % 4
    for sector, (dy, dx) in enumerate(directions):
        before = np.roll(magnitude, (dy, dx), (0, 1))
        after = np.roll(magnitude, (-dy, -dx), (0, 1))
        output |= ((sectors == sector) & (magnitude >= threshold)
                   & (magnitude >= before) & (magnitude >= after))
    output[:2] = output[-2:] = False
    output[:, :2] = output[:, -2:] = False
    return output


def oriented_matches(edges, gx, gy, magnitude, target_edges, tx, ty, tm):
    """Match within one pixel with signed gradient direction within 30 degrees.

    Signed direction rejects contrast inversions. Precision is evaluated with
    the reverse call, so newly invented background edges remain penalized.
    """
    matched = np.zeros_like(edges)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted_edge = np.roll(target_edges, (dy, dx), (0, 1))
            sx = np.roll(tx, (dy, dx), (0, 1))
            sy = np.roll(ty, (dy, dx), (0, 1))
            sm = np.roll(tm, (dy, dx), (0, 1))
            aligned = gx * sx + gy * sy >= np.cos(np.pi / 6) * magnitude * sm
            matched |= edges & shifted_edge & aligned
    return int(matched.sum())


def measure_detail(reference, rendered):
    reference, rendered = gray_float(reference), gray_float(rendered)
    if reference.shape != rendered.shape:
        raise ValueError('Reference/render dimensions differ')
    rx, ry, rm = sobel(reference)
    ax, ay, am = sobel(rendered)
    positive = rm[2:-2, 2:-2]
    positive = positive[positive >= .01]
    threshold = max(.04, float(np.quantile(positive, .70))) if positive.size else .04
    ref_edges = thin_edges(rx, ry, rm, threshold)
    actual_edges = thin_edges(ax, ay, am, threshold)
    ref_count, actual_count = int(ref_edges.sum()), int(actual_edges.sum())
    ref_matched = oriented_matches(ref_edges, rx, ry, rm, actual_edges, ax, ay, am)
    actual_matched = oriented_matches(actual_edges, ax, ay, am, ref_edges, rx, ry, rm)
    recall = ref_matched / ref_count if ref_count else None
    precision = actual_matched / actual_count if actual_count else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.)
    if not ref_count and not actual_count:
        f1 = None
    bright, dark = reference > .12, reference < .03
    missing, extra = bright & (rendered < .08), dark & (rendered > .08)
    interior = (slice(1, -1), slice(1, -1))
    return {
        'rgb_l1': float(np.abs(reference - rendered).mean()),
        'sobel_magnitude_l1': float(np.abs(rm - am)[interior].mean()),
        'sobel_vector_l1': float((np.abs(rx - ax) + np.abs(ry - ay))[interior].mean()),
        'reference_edge_threshold': threshold,
        'reference_edge_pixels': ref_count, 'render_edge_pixels': actual_count,
        'matched_reference_edges': ref_matched, 'matched_render_edges': actual_matched,
        'edge_recall': recall, 'edge_precision': precision, 'edge_f1': f1,
        'bright_reference_pixels': int(bright.sum()), 'missing_bright_pixels': int(missing.sum()),
        'bright_missing_fraction': float(missing.sum() / bright.sum()) if bright.any() else None,
        'dark_reference_pixels': int(dark.sum()), 'extra_dark_pixels': int(extra.sum()),
        'dark_extra_fraction': float(extra.sum() / dark.sum()) if dark.any() else None,
    }


def summarize(rows):
    result = {'frames': len(rows)}
    for key in ('rgb_l1', 'sobel_magnitude_l1', 'sobel_vector_l1'):
        result[key] = float(np.mean([row[key] for row in rows]))
        result[key + '_worst'] = float(max(row[key] for row in rows))
    for key in ('reference_edge_pixels', 'render_edge_pixels', 'matched_reference_edges',
                'matched_render_edges', 'bright_reference_pixels', 'missing_bright_pixels',
                'dark_reference_pixels', 'extra_dark_pixels'):
        result[key] = sum(row[key] for row in rows)
    for label, numerator, denominator in (
        ('edge_recall', 'matched_reference_edges', 'reference_edge_pixels'),
        ('edge_precision', 'matched_render_edges', 'render_edge_pixels'),
        ('bright_missing_fraction', 'missing_bright_pixels', 'bright_reference_pixels'),
        ('dark_extra_fraction', 'extra_dark_pixels', 'dark_reference_pixels'),
    ):
        result[label] = result[numerator] / result[denominator] if result[denominator] else None
    p, r = result['edge_precision'], result['edge_recall']
    result['edge_f1'] = 2*p*r/(p+r) if p is not None and r is not None and p+r else 0.
    if not result['reference_edge_pixels'] and not result['render_edge_pixels']:
        result['edge_f1'] = None
    return result


def frame_mapping(length, reference_length, sparse_frames=None):
    if length == reference_length:
        return {i: i for i in range(length)}
    if sparse_frames is None or len(sparse_frames) != length:
        raise ValueError('Sparse renders require explicit --render-frames matching array length')
    if len(set(sparse_frames)) != length or any(i < 0 or i >= reference_length for i in sparse_frames):
        raise ValueError('Sparse render frame indices are repeated or out of range')
    return {index: row for row, index in enumerate(sparse_frames)}


def save_detail(path, crops, index, roi_name):
    width, height = crops[0][1].shape[::-1]
    canvas = Image.new('RGB', (width * len(crops), height + 50), '#171c23')
    draw = ImageDraw.Draw(canvas)
    for column, (name, crop) in enumerate(crops):
        image = Image.fromarray(np.rint(gray_float(crop) * 255).astype(np.uint8))
        canvas.paste(image.convert('RGB'), (column * width, 50))
        draw.text((column * width + 4, 3),
                  f'{name[:28]}\n{roi_name} f={index}\nNative pixels; no filter', fill='white')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', required=True, type=Path)
    parser.add_argument('--renders', required=True, type=Path, nargs='+')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--output', required=True, type=Path, help='New output directory')
    parser.add_argument('--samples', type=int, default=40, choices=[8, 40, 400])
    parser.add_argument('--frames', type=int, nargs='+', help='Explicit evaluated original frame indices')
    parser.add_argument('--render-frames', type=int, nargs='+', help='Mapping only for sparse render arrays')
    parser.add_argument('--detail-frames', type=int, nargs='+', help='Default: 8 evenly spaced evaluated frames')
    parser.add_argument('--arm-roi', type=int, nargs=4, default=DEFAULT_ROIS['arm'])
    parser.add_argument('--gear-roi', type=int, nargs=4, default=DEFAULT_ROIS['gear'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = args.labels or [path.stem if path.suffix == '.npy' else path.name for path in args.renders]
    if len(labels) != len(args.renders) or len(set(labels)) != len(labels):
        raise ValueError('Labels must be distinct, one per render input')
    with np.load(args.supervision, allow_pickle=False) as archive:
        reference = archive['images']
    if reference.ndim != 3:
        raise ValueError('Supervision images must have shape [frames, height, width]')
    count, height, width = reference.shape
    frames = args.frames if args.frames is not None else np.linspace(
        0, count, min(args.samples, count), endpoint=False, dtype=int).tolist()
    if not frames or len(set(frames)) != len(frames) or any(i < 0 or i >= count for i in frames):
        raise ValueError('Evaluated frame indices must be distinct and in range')
    details = args.detail_frames if args.detail_frames is not None else [frames[i] for i in
        np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int)]
    if not set(details).issubset(frames):
        raise ValueError('Detail frames must be a subset of evaluated frames')
    rois = {'arm': args.arm_roi, 'gear': args.gear_roi}
    for bounds in rois.values():
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and min(x1-x0, y1-y0) >= 7):
            raise ValueError('Invalid ROI bounds')
    sources = []
    for label, path in zip(labels, args.renders):
        path = path if path.suffix == '.npy' else path / 'rendered_gray.npy'
        rendered = np.load(path, mmap_mode='r', allow_pickle=False)
        if rendered.ndim != 3 or rendered.shape[1:] != reference.shape[1:]:
            raise ValueError(f'Render dimensions differ: {path}')
        mapping = frame_mapping(len(rendered), count, args.render_frames)
        if not set(frames).issubset(mapping):
            raise ValueError(f'Render input is missing selected frames: {path}')
        sources.append((label, path, rendered, mapping))
    report = {
        'format': 'lego_reference_detail_v1',
        'scope': 'Same reference-camera image detail only; assumes synchronized camera and timestamps. '
                 'Not an event gate, held-out test, or proof of complete geometry. Read metrics jointly; '
                 'noise can increase edge energy without recovering real structure.',
        'reference': str(args.supervision), 'reference_sha256': sha256(args.supervision),
        'frame_indices': frames, 'detail_frame_indices': details, 'roi_xyxy': rois,
        'parameters': {'sobel_divisor': 8, 'threshold_from_reference_only': True,
                       'edge_threshold': 'max(0.04, reference gradient >=0.01 70th percentile)',
                       'edge_nonmaximum_suppression': '4 directions', 'edge_roi_border_excluded': 2,
                       'edge_matching_tolerance_pixels': 1, 'signed_direction_tolerance_degrees': 30,
                       'bright_reference_threshold': .12, 'missing_render_threshold': .08,
                       'dark_reference_threshold': .03, 'extra_render_threshold': .08,
                       'visual_resampling': 'none; native pixel crops'},
        'models': {},
    }
    args.output.mkdir(parents=True)
    for label, path, rendered, mapping in sources:
        model = {'render_path': str(path), 'render_sha256': sha256(path), 'rois': {}}
        for name, (x0, y0, x1, y1) in rois.items():
            rows = [{'frame_index': index, **measure_detail(
                reference[index, y0:y1, x0:x1], rendered[mapping[index], y0:y1, x0:x1])}
                for index in frames]
            model['rois'][name] = {'summary': summarize(rows), 'frames': rows}
        report['models'][label] = model
        print(json.dumps({label: {name: data['summary'] for name, data in model['rois'].items()}}), flush=True)
    for index in details:
        for name, (x0, y0, x1, y1) in rois.items():
            crops = [('reference', reference[index, y0:y1, x0:x1])] + [
                (label, rendered[mapping[index], y0:y1, x0:x1]) for label, _, rendered, mapping in sources]
            save_detail(args.output / f'{name}-f{index:03d}.png', crops, index, name)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
