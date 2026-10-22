"""Read-only comparison of initial Gaussian depth against the allowed Blender prior.

Depths are in Gaussian scene-coordinate units, not calibrated metres. The prior
is not hidden-depth ground truth and the diagnostic does not grant event approval.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw


def camera_vectors(height, width, fovy_degrees):
    focal = height / (2 * np.tan(np.deg2rad(fovy_degrees) / 2))
    yy, xx = np.mgrid[:height, :width]
    vectors = np.stack(((xx + .5 - width / 2) / focal,
                        -(yy + .5 - height / 2) / focal, -np.ones_like(xx)), -1)
    return vectors, focal


def ray_distance_to_z(distance, vectors, scale):
    return distance * scale / np.linalg.norm(vectors, axis=-1)


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def stats(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return {'pixels': 0}
    return {'pixels': len(values), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'p99': float(np.percentile(values, 99)),
            'mean': float(values.mean())}


def render_depth(source, camera):
    import torch
    from gaussian_renderer import render
    from render_ply import make_camera
    from scene import GaussianModel
    with torch.no_grad():
        model = GaussianModel(3, with_motion_mask=False)
        model.load_ply(str(source))
        if torch.count_nonzero(model._features_rest).item():
            raise ValueError('Expected DC-only source')
        model._features_rest = torch.empty((len(model._xyz), 0, 3), device='cuda')
        model.active_sh_degree = model.max_sh_degree = 0
        view = make_camera(np.asarray(camera['pose']), 640, 640, np.deg2rad(camera['fovy_degrees']))
        result = render(view, model, SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False),
                        torch.zeros(3, device='cuda'), 0., 0., 0.)
        alpha = result['alpha'].squeeze().cpu().numpy()
        depth = result['depth'].squeeze().cpu().numpy() / np.maximum(alpha, 1e-6)
        rgb = result['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return depth, alpha, rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--part-mask', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=.40884)
    parser.add_argument('--roi', type=int, nargs=4, default=[160, 140, 380, 380])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.scale) or args.scale <= 0:
        raise ValueError('scale must be finite and positive')
    metadata = json.loads((args.baseline / 'cache.json').read_text())
    camera = metadata['camera']
    with np.load(args.parts, allow_pickle=False) as data:
        parts_meta = json.loads(str(data['metadata_json']))
        matrices = data['object_matrices'].astype(np.float64)
    with np.load(args.part_mask, allow_pickle=False) as data:
        mask, distance = data['part_ids'], data['ray_distance']
        mask_meta = json.loads(str(data['metadata_json']))
    with np.load(args.binding, allow_pickle=False) as data:
        actual_registration = data['registration']
    if mask_meta['objects'] != parts_meta['objects']:
        raise ValueError('Part IDs differ')
    if abs(mask_meta['fovy_degrees'] - camera['fovy_degrees']) > .001:
        raise ValueError('Part-mask camera FOV differs')
    vectors, focal = camera_vectors(*mask.shape, camera['fovy_degrees'])
    prior_depth = ray_distance_to_z(distance, vectors, args.scale)
    depth, alpha, rgb = render_depth(metadata['source'], camera)
    method = 'Actual GS alpha-composited rasterizer depth divided by accumulated alpha; not a nearest-center approximation.'
    valid = (mask >= 0) & (distance > 0) & (alpha > .8) & np.isfinite(depth) & (depth > 0)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    rigid = np.ones(len(moving), bool)
    rigid[parts_meta['deforming_objects']] = False
    moving_ids = np.flatnonzero(moving & rigid)
    static_ids = np.flatnonzero(~moving & rigid)
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= mask.shape[1] and 0 <= y0 < y1 <= mask.shape[0]):
        raise ValueError('Invalid ROI bounds')
    roi = np.zeros_like(valid); roi[y0:y1, x0:x1] = True
    regions = {'all_visible': valid, 'moving_rigid': valid & np.isin(mask, moving_ids),
               'static': valid & np.isin(mask, static_ids), 'arm_roi': valid & roi,
               'moving_rigid_arm_roi': valid & roi & np.isin(mask, moving_ids)}
    for part in (183, 227, 213):
        regions[f'part_{part}'] = valid & (mask == part)
    difference = depth - prior_depth
    report = {'scope': __doc__, 'method': method, 'units': 'Gaussian scene coordinates; not calibrated metres',
              'source': metadata['source'], 'camera_registration_scale': args.scale,
              'ray_to_z_formula': 'z = normalized_ray_distance * .40884 / sqrt(1 + x_normalized^2 + y_normalized^2)',
              'alpha_threshold': .8, 'roi_xyxy': args.roi, 'focal_pixels': focal,
              'registration': {'used_for_primary_depth': 'Exact source-camera to GS-camera similarity using .40884 scale',
                               'binding_registration': actual_registration.tolist()},
              'regions': {}}
    for name, region in regions.items():
        report['regions'][name] = {'signed_depth_difference': stats(difference[region]),
                                  'absolute_depth_difference': stats(np.abs(difference[region])),
                                  'relative_depth_difference': stats(np.abs(difference[region]) / prior_depth[region]),
                                  'prior_depth': stats(prior_depth[region]),
                                  'gaussian_depth': stats(depth[region])}
    print(json.dumps({'initial_depth_results': report['regions']}, indent=2), flush=True)
    # At time 0 the two depths lie on exactly the same camera ray and therefore
    # have zero image displacement. Rotate BOTH hypothetical points using the
    # same known source part transform to isolate depth-driven future parallax.
    pose = np.asarray(camera['pose'])
    anchor_registration = pose @ np.diag([args.scale] * 3 + [1]) @ np.linalg.inv(parts_meta['camera_pose'])
    expected_points = transform(vectors * prior_depth[..., None], pose)
    rendered_points = transform(vectors * depth[..., None], pose)
    inverse_pose = np.linalg.inv(pose)
    motion_pixels = np.full(mask.shape, np.nan, np.float32)
    motion = anchor_registration @ delta[100] @ np.linalg.inv(anchor_registration)
    for part in np.unique(mask[valid]):
        region = valid & (mask == part)
        expected = transform(transform(expected_points[region], motion[part]), inverse_pose)
        actual = transform(transform(rendered_points[region], motion[part]), inverse_pose)
        expected_uv = focal * expected[:, :2] / -expected[:, 2:]
        actual_uv = focal * actual[:, :2] / -actual[:, 2:]
        motion_pixels[region] = np.linalg.norm(expected_uv - actual_uv, axis=-1)
    for name, region in regions.items():
        report['regions'][name]['hypothetical_f100_parallax_pixels'] = stats(motion_pixels[region])
    report['parallax_scope'] = ('At f0 same-ray depths have ZERO pixel displacement. Reported f100 displacement '
                               'applies the SAME prior rigid part transform to both same-ray depths; it isolates '
                               'depth sensitivity and is not measured flow error or evidence of correct ownership.')
    transformed_camera = actual_registration @ np.asarray(parts_meta['camera_pose'])
    report['registration']['actual_binding_camera_origin_offset'] = float(np.linalg.norm(transformed_camera[:3, 3] - pose[:3, 3]))
    report['registration']['actual_binding_similarity_scale'] = float(np.cbrt(np.linalg.det(actual_registration[:3, :3])))
    source_local = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True) * distance[..., None]
    source_world = transform(source_local, np.asarray(parts_meta['camera_pose']))
    registered_view = transform(transform(source_world, actual_registration), inverse_pose)
    actual_binding_z = -registered_view[..., 2]
    report['actual_binding_registered_hit_depth_caveat'] = ('Same source ray hits under the fitted ICP registration '
                                                         'may reproject to different pixels; these numbers are '
                                                         'a coordinate sensitivity comparison, not pixel-aligned depth truth.')
    report['actual_binding_registered_hit_depth'] = {
        name: stats(np.abs(depth[region] - actual_binding_z[region])) for name, region in regions.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), gs_depth=depth, gs_alpha=alpha,
                        prior_camera_z=prior_depth.astype(np.float32), signed_difference=difference.astype(np.float32),
                        same_part_f100_parallax=motion_pixels, valid=valid, part_ids=mask)
    limit = .08
    heat = np.zeros_like(rgb)
    ratio = np.clip(np.abs(difference) / limit, 0, 1)
    heat[..., 0] = np.where(difference >= 0, ratio, 0)
    heat[..., 2] = np.where(difference < 0, ratio, 0)
    heat[..., 1] = .2 * ratio
    heat[~valid] = 0
    overlay = .35 * rgb + .65 * heat
    canvas = Image.new('RGB', (1920, 672))
    draw = ImageDraw.Draw(canvas)
    for i, (array, title) in enumerate(zip((rgb, heat, overlay),
                                          ('Initial GS render', 'GS minus Blender-prior z: red behind / blue in front; saturates .08',
                                           'Overlay; units are scene coordinates, not metres'))):
        canvas.paste(Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)), (640 * i, 32))
        draw.text((640 * i + 5, 8), title, fill='white')
    canvas.save(args.output.with_suffix('.png'))
    print(json.dumps({'report': str(args.output), 'f100_parallax': {name: record['hypothetical_f100_parallax_pixels']
                                                                  for name, record in report['regions'].items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
