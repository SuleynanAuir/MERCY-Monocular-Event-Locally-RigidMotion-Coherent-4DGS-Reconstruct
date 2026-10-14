"""Independent sampled, visible per-part coverage against RGB; not 3D completeness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, label


def coverage(reference, rendered, region, min_pixels=16):
    expected = region & (reference > .12)
    missing = expected & (rendered < .08)
    count = int(expected.sum())
    components, component_count = label(missing)
    areas = np.bincount(components.ravel())[1:] if component_count else np.array([], dtype=int)
    return {'reference_bright_pixels': count, 'missing_pixels': int(missing.sum()),
            'missing_fraction': float(missing.sum() / count) if count else None,
            'coverage': float(1 - missing.sum() / count) if count else None,
            'largest_missing_component_pixels': int(areas.max()) if len(areas) else 0,
            'eligible': count >= min_pixels}


def summarize(rows):
    eligible = [row for row in rows if row['eligible']]
    if not eligible:
        return {'eligible_frames': 0}
    worst = max(eligible, key=lambda row: row['missing_fraction'])
    total_reference = sum(row['reference_bright_pixels'] for row in eligible)
    total_missing = sum(row['missing_pixels'] for row in eligible)
    return {
        'eligible_frames': len(eligible), 'reference_bright_pixels': total_reference,
        'missing_pixels': total_missing,
        'pixel_weighted_missing_fraction': total_missing / total_reference,
        'mean_missing_fraction': float(np.mean([row['missing_fraction'] for row in eligible])),
        'worst_frame_index': worst['frame_index'],
        'worst_frame_missing_fraction': worst['missing_fraction'],
        'worst_frame_reference_bright_pixels': worst['reference_bright_pixels'],
        'largest_missing_component_pixels': max(row['largest_missing_component_pixels'] for row in eligible),
    }


def save_roi_comparison(path, reference, rendered, roi, frame_index, name, part_mask):
    x0, y0, x1, y1 = roi
    reference = reference[y0:y1, x0:x1]
    rendered = rendered[y0:y1, x0:x1]
    moving_region = part_mask[y0:y1, x0:x1]
    ref_rgb = np.repeat(np.clip(reference[..., None] * 255, 0, 255).astype(np.uint8), 3, -1)
    out_rgb = np.repeat(np.clip(rendered[..., None] * 255, 0, 255).astype(np.uint8), 3, -1)
    overlay = out_rgb.copy()
    missing = (reference > .12) & (rendered < .08)
    overlay[missing] = [255, 32, 32]
    overlay[missing & moving_region] = [255, 200, 32]
    panels = [Image.fromarray(array).resize(((x1-x0)*2, (y1-y0)*2))
              for array in (ref_rgb, out_rgb, overlay)]
    canvas = Image.new('RGB', (sum(panel.width for panel in panels), panels[0].height + 32))
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * panel.width, 32))
        draw.text((i * panel.width + 5, 8),
                  [f'RGB frame {frame_index}', name[:50], 'missing: red; moving part: yellow'][i], fill='white')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--renders', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roi', type=int, nargs=4, default=[160, 140, 380, 380])
    parser.add_argument('--min-pixels', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.min_pixels < 1:
        raise ValueError('min-pixels must be positive')
    with np.load(args.masks, allow_pickle=False) as archive:
        masks = archive['part_ids']
        frame_indices = archive['frame_indices']
        metadata = json.loads(str(archive['metadata_json']))
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
    if masks.shape != (len(frame_indices), *images.shape[1:]):
        raise ValueError('Mask/image dimensions differ')
    if frame_indices.min() < 0 or frame_indices.max() >= len(images):
        raise ValueError('Mask frame index outside RGB sequence')
    if np.max(np.abs(np.asarray(metadata['fovy_degrees']) - 39.59775410217631)) > .1:
        raise ValueError('Part-mask field of view differs from the reconstruction camera')
    moving_parts = metadata['moving_rigid_parts']
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= images.shape[2] and 0 <= y0 < y1 <= images.shape[1]):
        raise ValueError('Invalid ROI bounds')
    roi = np.zeros(images.shape[1:], dtype=bool)
    roi[y0:y1, x0:x1] = True
    report = {
        'format': 'lego_visible_per_part_completeness_v1',
        'scope': 'Reference-camera 2D bright foreground coverage within Blender-prior visible '
                 'moving rigid parts only. Does not prove 3D completeness, physical attachment, '
                 'or event consistency. Masks follow allowed Blender initialization, not fitted motion.',
        'mask_path': str(args.masks), 'mask_sha256': hashlib.sha256(args.masks.read_bytes()).hexdigest(),
        'sampled_frame_indices': frame_indices.tolist(), 'frame_stride': 10,
        'reference_threshold': .12, 'render_missing_threshold': .08,
        'erosion_pixels': 1, 'erosion_neighborhood': '3x3 full (8-neighbor)',
        'min_reference_pixels_for_summary': args.min_pixels, 'roi_xyxy': args.roi,
        'moving_rigid_parts': moving_parts, 'excluded_deforming_parts': metadata['deforming_objects'],
        'models': {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interiors = []
    for mask in masks:
        interiors.append({part: binary_erosion(mask == part, structure=np.ones((3, 3), bool), border_value=0)
                          for part in moving_parts if np.any(mask == part)})
    # Silhouette alignment is reported independently so a bad camera/mask cannot silently pass.
    alignment = []
    for sample, index in enumerate(frame_indices):
        bright = images[index] / 255 > .12
        visible_mesh = masks[sample] >= 0
        alignment.append({'frame_index': int(index), 'bright_rgb_pixels': int(bright.sum()),
                          'bright_rgb_outside_mesh_fraction': float((bright & ~visible_mesh).sum() / max(bright.sum(), 1)),
                          'mesh_vs_bright_rgb_silhouette_iou': float((bright & visible_mesh).sum() / max((bright | visible_mesh).sum(), 1)),
                          'caveat': 'Blender mesh includes dark material and shading pixels excluded by the bright RGB threshold.'})
    report['mask_alignment'] = alignment
    palette = np.random.default_rng(17).integers(48, 256, (len(metadata['objects']), 3), dtype=np.uint8)
    for sample, index in enumerate(frame_indices):
        if index not in (0, 50, 100, 150):
            continue
        mask = masks[sample]
        visible = mask >= 0
        color = np.zeros((*mask.shape, 3), np.uint8)
        color[visible] = palette[mask[visible]]
        ref_rgb = np.repeat(images[index, ..., None], 3, -1)
        overlay = ref_rgb.copy()
        overlay[visible] = (.55 * ref_rgb[visible] + .45 * color[visible]).astype(np.uint8)
        outside = (images[index] / 255 > .12) & ~visible
        overlay[outside] = [255, 0, 0]
        canvas = Image.new('RGB', (640 * 3, 672))
        draw = ImageDraw.Draw(canvas)
        for i, (array, title) in enumerate(zip((ref_rgb, color, overlay),
                                              ('RGB', 'Blender visible part IDs', 'Overlay; outside mesh = red'))):
            canvas.paste(Image.fromarray(array), (640 * i, 32))
            draw.text((640 * i + 5, 8), f'{title} frame {index}', fill='white')
        canvas.save(args.output.parent / f'{args.output.stem}-mask-alignment-{index:03d}.png')
    for path in args.renders:
        rendered = np.load(path / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
        if rendered.shape != images.shape:
            raise ValueError(f'Rendered dimensions differ: {path}')
        part_rows = {part: [] for part in moving_parts}
        roi_rows, moving_roi_rows = [], []
        for sample, index in enumerate(frame_indices):
            reference = images[index].astype(np.float32) / 255
            actual = np.asarray(rendered[index], dtype=np.float32)
            moving_interior = np.zeros_like(roi)
            for part, interior in interiors[sample].items():
                moving_interior |= interior
                part_rows[part].append({'frame_index': int(index), **coverage(reference, actual, interior, args.min_pixels)})
            roi_rows.append({'frame_index': int(index), **coverage(reference, actual, roi, args.min_pixels)})
            moving_roi_rows.append({'frame_index': int(index), **coverage(reference, actual, roi & moving_interior, args.min_pixels)})
        parts = [{'part_id': part, 'object_name': metadata['objects'][part],
                  'summary': summarize(rows), 'frames': rows} for part, rows in part_rows.items()]
        parts.sort(key=lambda part: part['summary'].get('worst_frame_missing_fraction', -1), reverse=True)
        eligible_part_frames = [row for part in parts for row in part['frames'] if row['eligible']]
        global_summary = summarize(eligible_part_frames)
        global_summary['eligible_part_frames'] = global_summary.pop('eligible_frames')
        global_summary['parts_with_eligible_frames'] = sum(bool(part['summary']['eligible_frames']) for part in parts)
        global_summary['part_frames_missing_over_10_percent'] = sum(row['missing_fraction'] > .1 for row in eligible_part_frames)
        global_summary['part_frames_missing_over_25_percent'] = sum(row['missing_fraction'] > .25 for row in eligible_part_frames)
        report['models'][path.name] = {'render_path': str(path), 'summary': global_summary,
                                       'parts': parts, 'arm_roi': {'summary': summarize(roi_rows), 'frames': roi_rows},
                                       'moving_part_arm_roi': {'summary': summarize(moving_roi_rows), 'frames': moving_roi_rows}}
        worst_roi = sorted(roi_rows, key=lambda row: row['missing_fraction'] or 0, reverse=True)[:5]
        sample_map = {int(index): sample for sample, index in enumerate(frame_indices)}
        for row in worst_roi:
            index = row['frame_index']; sample = sample_map[index]
            moving_region = np.zeros_like(roi)
            for interior in interiors[sample].values():
                moving_region |= interior
            save_roi_comparison(args.output.parent / f'{args.output.stem}-{path.name}-roi-{index:03d}.png',
                                images[index] / 255, np.asarray(rendered[index], dtype=np.float32),
                                args.roi, index, path.name, moving_region)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {'parts': model['summary'], 'arm_roi': model['arm_roi']['summary'],
                             'moving_part_arm_roi': model['moving_part_arm_roi']['summary'],
                             'worst_parts': [{'part_id': part['part_id'], 'name': part['object_name'], **part['summary']}
                                             for part in model['parts'][:12]]}
                      for name, model in report['models'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
