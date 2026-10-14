"""Read-only, localized evidence for parts whose visible coverage worsened."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import binary_dilation, binary_erosion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    model_names = list(report['models'])
    if len(model_names) < 2:
        raise ValueError('At least two models are required')
    with np.load(args.masks, allow_pickle=False) as archive:
        masks, frames = archive['part_ids'], archive['frame_indices']
    with np.load(args.parts, allow_pickle=False) as archive:
        matrices = archive['object_matrices'].astype(np.float64)
        metadata = json.loads(str(archive['metadata_json']))
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
    delta = matrices @ np.linalg.inv(matrices[0])
    sampled = np.linspace(0, len(matrices) - 1, 17).astype(int)
    features = delta[sampled, :, :3].transpose(1, 0, 2, 3).reshape(len(matrices[0]), -1)
    clusters = fclusterdata(features, .002, criterion='distance', method='complete')
    by_model = {name: {part['part_id']: part for part in model['parts']}
                for name, model in report['models'].items()}
    comparisons = []
    for name in model_names[1:]:
        rows, raw_rows = [], []
        for part, old in by_model[model_names[0]].items():
            new = by_model[name][part]
            old_frames = {frame['frame_index']: frame for frame in old['frames']}
            observed = [frame for frame in new['frames'] if frame['reference_bright_pixels']]
            worst_raw = max(observed, key=lambda frame: frame['missing_pixels'] -
                            old_frames[frame['frame_index']]['missing_pixels']) if observed else None
            old_missing = sum(frame['missing_pixels'] for frame in old['frames'])
            new_missing = sum(frame['missing_pixels'] for frame in new['frames'])
            raw_rows.append({
                'part_id': part, 'object_name': metadata['objects'][part],
                'source_motion_cluster': int(clusters[part]),
                'has_eligible_frames': bool(new['summary']['eligible_frames']),
                'reference_bright_interior_pixels': sum(frame['reference_bright_pixels'] for frame in new['frames']),
                'old_missing_pixels': old_missing, 'new_missing_pixels': new_missing,
                'extra_missing_pixels': new_missing - old_missing,
                'new_largest_missing_component_pixels': max(
                    (frame['largest_missing_component_pixels'] for frame in new['frames']), default=0),
                'largest_regression_frame_index': worst_raw['frame_index'] if worst_raw else None,
            })
            if not old['summary']['eligible_frames'] or not new['summary']['eligible_frames']:
                continue
            worsened = new['summary']['missing_pixels'] - old['summary']['missing_pixels']
            old_bad = sum(frame['eligible'] and frame['missing_fraction'] > .25
                          for frame in old['frames'])
            new_bad = sum(frame['eligible'] and frame['missing_fraction'] > .25
                          for frame in new['frames'])
            rows.append({'part_id': part, 'object_name': metadata['objects'][part],
                         'source_motion_cluster': int(clusters[part]),
                         'same_cluster_parts': np.flatnonzero(clusters == clusters[part]).tolist(),
                         'extra_missing_pixels': worsened,
                         'old_frames_missing_over_25_percent': old_bad,
                         'new_frames_missing_over_25_percent': new_bad,
                         'additional_frames_missing_over_25_percent': new_bad - old_bad,
                         'old': old['summary'], 'new': new['summary']})
        rows.sort(key=lambda row: row['extra_missing_pixels'], reverse=True)
        comparisons.append({'baseline': model_names[0], 'candidate': name,
                            'eligible_parts_compared': len(rows),
                            'parts_with_more_missing_pixels': sum(row['extra_missing_pixels'] > 0 for row in rows),
                            'parts_with_fewer_missing_pixels': sum(row['extra_missing_pixels'] < 0 for row in rows),
                            'parts_with_additional_bad_frames': sum(row['additional_frames_missing_over_25_percent'] > 0
                                                                    for row in rows),
                            'all_part_comparisons': rows,
                            'all_source_parts_raw_interior_comparisons': raw_rows,
                            'raw_small_part_regressions': [row for row in raw_rows
                                                           if not row['has_eligible_frames'] and row['extra_missing_pixels'] > 0],
                            'worst_regressions': rows[:12]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for comparison in comparisons:
        for row in comparison['worst_regressions'][:6]:
            key = (row['part_id'], row['new']['worst_frame_index'])
            candidates[key] = row
        for row in comparison['raw_small_part_regressions']:
            key = (row['part_id'], row['largest_regression_frame_index'])
            candidates[key] = row
    rendered = {name: np.load(Path(model['render_path']) / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
                for name, model in report['models'].items()}
    frame_map = {int(frame): index for index, frame in enumerate(frames)}
    outputs = []
    for (part, frame), row in candidates.items():
        mask = masks[frame_map[frame]] == part
        yy, xx = np.nonzero(mask)
        cx, cy = int(np.mean(xx)), int(np.mean(yy))
        half = max(48, int(max(xx.max() - xx.min(), yy.max() - yy.min()) / 2) + 12)
        x0, y0, x1, y1 = max(0, cx-half), max(0, cy-half), min(640, cx+half), min(640, cy+half)
        edge = binary_dilation(mask) & ~mask
        interior = binary_erosion(mask, structure=np.ones((3, 3), bool))
        reference = images[frame].astype(np.float32) / 255
        panels = []
        for name, actual in [('RGB reference', reference), *[(name, np.asarray(values[frame], np.float32))
                                                            for name, values in rendered.items()]]:
            color = np.repeat(np.rint(np.clip(actual, 0, 1)[..., None] * 255).astype(np.uint8), 3, -1)
            color[edge] = [255, 220, 40]
            if name != 'RGB reference':
                missing = interior & (reference > .12) & (actual < .08)
                color[missing] = [255, 40, 40]
            panel = Image.fromarray(color[y0:y1, x0:x1]).resize(((x1-x0)*3, (y1-y0)*3), Image.Resampling.NEAREST)
            decorated = Image.new('RGB', (panel.width, panel.height + 48))
            decorated.paste(panel, (0, 48))
            draw = ImageDraw.Draw(decorated)
            draw.text((5, 5), name, fill='white')
            draw.text((5, 24), f'part {part}, frame {frame}; red = missing', fill='white')
            panels.append(decorated)
        sheet = Image.new('RGB', (sum(panel.width for panel in panels), panels[0].height))
        offset = 0
        for panel in panels:
            sheet.paste(panel, (offset, 0)); offset += panel.width
        filename = args.output.parent / f'{args.output.stem}-part-{part}-f{frame:03d}.png'
        sheet.save(filename)
        outputs.append({'part': part, 'frame': frame, 'crop_xyxy': [x0, y0, x1, y1], 'image': str(filename)})
    result = {'scope': 'Localized reference-view diagnosis. Red marks low-brightness holes inside the '
                      'Blender-prior visible part. A previous model may spuriously fill these pixels '
                      'with the wrong part; coverage alone cannot prove identity or 3D completeness.',
              'source_motion_cluster_method': 'Complete-link clustering of 17 sampled rigid deltas, threshold .002, one-based SciPy labels',
              'comparisons': comparisons, 'images': outputs}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'report': str(args.output), 'comparisons': [
        {'candidate': item['candidate'], 'parts': [{key: row[key] for key in ('part_id', 'source_motion_cluster', 'extra_missing_pixels')}
                                                   for row in item['worst_regressions'][:8]]} for item in comparisons]}, indent=2))


if __name__ == '__main__':
    main()
