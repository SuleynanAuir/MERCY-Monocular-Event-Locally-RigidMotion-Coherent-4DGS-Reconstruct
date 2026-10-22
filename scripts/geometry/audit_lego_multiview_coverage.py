"""Independent CPU Gaussian-alpha coverage of sampled rigid mesh PRIOR views.

This measures geometric image support, not observed back-side ground truth,
per-Gaussian ownership, depth correctness, appearance quality, or event approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, label


THRESHOLDS = (.2, .5, .8)
BACKSIDE_YAWS = (135, 180, 225)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def region_stats(alpha, expected, threshold):
    """Eight-connected missing support, on a caller-provided (possibly cropped) region."""
    if (alpha.shape != expected.shape or alpha.ndim != 2 or expected.dtype != np.bool_
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid alpha/region/threshold')
    count = int(expected.sum())
    missing = expected & (alpha < threshold)
    components, number = label(missing, structure=np.ones((3, 3), bool))
    sizes = np.bincount(components.ravel())[1:] if number else np.empty(0, np.int64)
    missing_count = int(missing.sum())
    return {'expected_pixels': count, 'missing_pixels': missing_count,
            'coverage': float(1 - missing_count / count) if count else None,
            'missing_fraction': float(missing_count / count) if count else None,
            'largest_missing_component_pixels': int(sizes.max()) if len(sizes) else 0,
            'missing_component_count': int(number)}


def aggregate(rows, threshold, region):
    key = str(threshold)
    valid = [(row, row['thresholds'][key][region]) for row in rows
             if row['thresholds'][key][region]['expected_pixels'] > 0]
    expected = sum(stats['expected_pixels'] for _, stats in valid)
    missing = sum(stats['missing_pixels'] for _, stats in valid)
    per_part = {}
    for row, stats in valid:
        per_part.setdefault(row['part_id'], []).append(stats['missing_fraction'])
    worst = max(valid, key=lambda pair: pair[1]['missing_fraction']) if valid else None
    return {
        'visible_part_view_count': len(rows), 'eligible_part_view_count': len(valid),
        'zero_interior_or_empty_part_view_count': len(rows) - len(valid),
        'eligible_unique_parts': len(per_part),
        'tiny_part_views_under_16_pixels': sum(stats['expected_pixels'] < 16 for _, stats in valid),
        'expected_pixels': expected, 'missing_pixels': missing,
        'pixel_weighted_missing_fraction': missing / expected if expected else None,
        'equal_weight_part_view_missing_fraction': float(np.mean([stats['missing_fraction'] for _, stats in valid])) if valid else None,
        'equal_weight_part_mean_missing_fraction': float(np.mean([np.mean(values) for values in per_part.values()])) if per_part else None,
        'largest_missing_component_pixels': max((stats['largest_missing_component_pixels'] for _, stats in valid), default=0),
        'worst_part_view': ({'part_id': worst[0]['part_id'], 'name': worst[0]['name'],
                            'view_index': worst[0]['view_index'], 'frame': worst[0]['frame'],
                            'yaw': worst[0]['yaw'], 'pitch': worst[0]['pitch'], **worst[1]} if worst else None),
    }


def validate_renders(alpha, gray, metadata, masks, mask_metadata):
    if (alpha.shape != masks.shape or gray.shape != masks.shape or alpha.dtype != np.float32
            or gray.dtype != np.float32 or not np.isfinite(alpha).all() or not np.isfinite(gray).all()
            or alpha.min() < 0 or alpha.max() > 1 or gray.min() < 0 or gray.max() > 1):
        raise ValueError('Expected finite float32 alpha/gray in [0,1] matching mask dimensions')
    if metadata.get('views') != mask_metadata['views']:
        raise ValueError('Rendered view identities/cameras differ from the exact mesh mask plan')


def measure(alpha, masks, metadata, group_ids, static_groups=(0,)):
    objects = metadata['objects']
    parts, view_rows = [], []
    for index, view in enumerate(metadata['views']):
        mask, rendered = masks[index], alpha[index]
        foreground = mask >= 0
        moving = np.zeros(mask.shape, bool)
        moving[foreground] = ~np.isin(group_ids[mask[foreground]], static_groups)
        foreground_summary = {}
        for threshold in THRESHOLDS:
            actual = rendered >= threshold
            intersection = actual & foreground
            outside = actual & ~foreground
            foreground_summary[str(threshold)] = {
                'all_rigid_foreground': region_stats(rendered, foreground, threshold),
                'moving_rigid_foreground': region_stats(rendered, moving, threshold),
                'rendered_foreground_pixels': int(actual.sum()),
                'outside_rigid_prior_pixels': int(outside.sum()),
                'outside_rigid_prior_fraction_of_rendered': float(outside.sum() / actual.sum()) if actual.any() else None,
                'silhouette_iou_diagnostic_only': float(intersection.sum() / (actual | foreground).sum()) if (actual | foreground).any() else None,
            }
        view_rows.append({'view_index': index, 'frame': view['frame'], 'yaw': view['yaw'],
                          'pitch': view['pitch'], 'thresholds': foreground_summary})
        for part in np.unique(mask[foreground]):
            yy, xx = np.nonzero(mask == part)
            box = (slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1))
            region = mask[box] == part
            interior = binary_erosion(region, structure=np.ones((3, 3), bool), border_value=0)
            row = {'view_index': index, 'part_id': int(part), 'name': objects[part],
                   'group_id': int(group_ids[part]), 'moving_rigid': int(group_ids[part]) not in static_groups,
                   'frame': view['frame'], 'yaw': view['yaw'], 'pitch': view['pitch'],
                   'bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                   'thresholds': {}}
            for threshold in THRESHOLDS:
                row['thresholds'][str(threshold)] = {
                    'foreground': region_stats(rendered[box], region, threshold),
                    'interior': region_stats(rendered[box], interior, threshold),
                }
            parts.append(row)
        print(json.dumps({'view_index': index, 'frame': view['frame'], 'yaw': view['yaw'],
                          'pitch': view['pitch'], 'visible_parts': int(len(np.unique(mask[foreground])))}), flush=True)
    scopes = {
        'moving_rigid_all_views': [row for row in parts if row['moving_rigid']],
        'moving_rigid_backside_all_pitches': [row for row in parts if row['moving_rigid'] and row['yaw'] % 360 in BACKSIDE_YAWS],
        'moving_rigid_backside_level': [row for row in parts if row['moving_rigid'] and row['yaw'] % 360 in BACKSIDE_YAWS and row['pitch'] == 0],
        'all_rigid_all_views': parts,
        'all_rigid_backside_all_pitches': [row for row in parts if row['yaw'] % 360 in BACKSIDE_YAWS],
        'static_rigid_all_views': [row for row in parts if not row['moving_rigid']],
    }
    for yaw in BACKSIDE_YAWS:
        scopes[f'moving_rigid_yaw_{yaw}'] = [row for row in parts if row['moving_rigid'] and row['yaw'] % 360 == yaw]
    summary = {scope: {str(threshold): {region: aggregate(selected, threshold, region)
                                      for region in ('foreground', 'interior')}
                       for threshold in THRESHOLDS}
               for scope, selected in scopes.items()}
    return {'summary': summary, 'views': view_rows, 'parts': parts}


def compare(before, after):
    previous = {(row['view_index'], row['part_id']): row for row in before['parts']}
    changes = []
    for row in after['parts']:
        key = row['view_index'], row['part_id']
        old = previous[key]
        if not row['moving_rigid']:
            continue
        first, second = old['thresholds']['0.5']['interior'], row['thresholds']['0.5']['interior']
        if not first['expected_pixels']:
            continue
        changes.append({'view_index': row['view_index'], 'part_id': row['part_id'], 'name': row['name'],
                        'frame': row['frame'], 'yaw': row['yaw'], 'pitch': row['pitch'],
                        'expected_pixels': first['expected_pixels'],
                        'missing_pixels_before': first['missing_pixels'], 'missing_pixels_after': second['missing_pixels'],
                        'missing_pixel_change': second['missing_pixels'] - first['missing_pixels'],
                        'missing_fraction_change': second['missing_fraction'] - first['missing_fraction'],
                        'largest_component_change': second['largest_missing_component_pixels'] - first['largest_missing_component_pixels']})
    regressions = sorted([row for row in changes if row['missing_pixel_change'] > 0],
                         key=lambda row: (-row['missing_fraction_change'], -row['missing_pixel_change']))
    improvements = sorted([row for row in changes if row['missing_pixel_change'] < 0],
                          key=lambda row: (row['missing_fraction_change'], row['missing_pixel_change']))
    return {'threshold': .5, 'scope': 'Moving rigid 1px-eroded part interiors; empty interiors excluded and reported separately.',
            'compared_part_views': len(changes), 'improved_part_views': len(improvements),
            'regressed_part_views': len(regressions), 'unchanged_part_views': len(changes) - len(improvements) - len(regressions),
            'regressions': regressions, 'largest_fraction_improvements': improvements[:50]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'masks', 'motion', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--static-groups', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.static_groups or min(args.static_groups) < 0:
        raise ValueError('Static groups must be nonnegative group IDs')
    with np.load(args.masks, allow_pickle=False) as archive:
        masks = archive['part_ids']
        mask_metadata = json.loads(str(archive['metadata_json']))
    with np.load(args.motion, allow_pickle=False) as archive:
        group_ids = archive['group_ids']
        motion_metadata = json.loads(str(archive['metadata_json']))
    if (mask_metadata.get('format') != 'lego_corrected_rigid_mesh_multiview_masks_v1'
            or mask_metadata.get('motion_sha256') != sha256(args.motion)
            or masks.ndim != 3 or masks.dtype.kind not in 'iu' or masks.shape[0] != len(mask_metadata['views'])
            or group_ids.shape != (len(mask_metadata['objects']),) or group_ids.dtype.kind not in 'iu'
            or motion_metadata.get('objects') != mask_metadata['objects'] or np.any(masks < -1)
            or np.any(masks >= len(group_ids)) or np.any(group_ids[masks[masks >= 0]] < 0)
            or np.isin(masks, mask_metadata['excluded_deforming_objects']).any()):
        raise ValueError('Invalid mesh mask/motion identity, ownership or excluded-band scope')
    report = {
        'format': 'lego_sampled_multiview_alpha_mesh_coverage_v1',
        'scope': 'Independent CPU alpha coverage of sampled opaque rigid mesh PRIOR silhouettes '
                 'and per-part interiors, using candidate-corrected poses. Not observed back-side '
                 'truth, full 3D completeness, correct part ownership/depth, appearance/sharpness '
                 'quality, or event acceptance. Alpha from any Gaussian can cover a prior part pixel.',
        'limitations': ['Elastic bands are excluded from prior geometry and occlusion but may be present in Gaussian alpha.',
                       'Outside-mask alpha and silhouette IoU therefore count bands, mesh/model mismatch and soft Gaussian edges; diagnostics only.',
                       'Material transparency is not modeled by opaque triangle masks.',
                       'Finite sampled cameras and times do not prove completeness at all continuous views or times.',
                       'Raw regions include thin parts. One-pixel erosion can erase them; empty interiors and tiny regions are reported, not called complete.',
                       'The large base can dominate pixel-weighted/global scores. Primary scopes are moving-rigid, backside and equal-weight per-part scores.'],
        'primary_scope': 'moving_rigid_backside_all_pitches', 'primary_threshold': .5,
        'primary_region': 'interior', 'primary_summary': 'equal_weight_part_mean_missing_fraction',
        'threshold_sensitivity': list(THRESHOLDS), 'backside_yaws': list(BACKSIDE_YAWS),
        'erosion_pixels': 1, 'erosion_neighborhood': '3x3 full, border zero',
        'missing_component_connectivity': 8, 'static_groups': args.static_groups,
        'mask_sha256': sha256(args.masks), 'motion_sha256': sha256(args.motion),
        'mask_metadata': mask_metadata, 'models': {},
    }
    for name, path in (('before', args.before), ('after', args.after)):
        with np.load(path, allow_pickle=False) as archive:
            alpha, gray = archive['alpha'], archive['gray']
            metadata = json.loads(str(archive['metadata_json']))
        validate_renders(alpha, gray, metadata, masks, mask_metadata)
        result = measure(alpha, masks, mask_metadata, group_ids, tuple(args.static_groups))
        result.update(path=str(path), sha256=sha256(path), metadata=metadata)
        report['models'][name] = result
        del alpha, gray
    report['change_primary_threshold'] = compare(report['models']['before'], report['models']['after'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'primary_summary': {
        name: value['summary'][report['primary_scope']]['0.5']['interior']
        for name, value in report['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
