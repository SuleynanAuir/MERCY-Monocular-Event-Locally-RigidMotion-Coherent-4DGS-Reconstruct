"""Compare canonical Fan refinement under identical views and rigid motion.

Image smoothness/edge/coverage scores are proxies, not reconstruction ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def model_from_vertices(base):
    import torch
    from scene import GaussianModel
    model = GaussianModel(3, with_motion_mask=False)
    for name, fields in [('_xyz', ['x', 'y', 'z']), ('_rotation', [f'rot_{i}' for i in range(4)]),
                         ('_scaling', [f'scale_{i}' for i in range(3)]), ('_opacity', ['opacity'])]:
        setattr(model, name, torch.from_numpy(np.column_stack([base[k] for k in fields]).copy()).cuda())
    model._features_dc = torch.from_numpy(np.column_stack([base[f'f_dc_{i}'] for i in range(3)]).copy()).cuda()[:, None, :]
    rest = np.column_stack([base[f'f_rest_{i}'] for i in range(45)])
    if not np.any(rest):
        model._features_rest = torch.empty((len(base), 0, 3), device='cuda')
        model.active_sh_degree = model.max_sh_degree = 0
    else:
        model._features_rest = torch.from_numpy(rest.reshape(-1, 3, 15).transpose(0, 2, 1).copy()).cuda()
        model.active_sh_degree = 3
    return model


def main():
    import torch
    from PIL import Image
    from scipy import ndimage
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera
    from utils.viewer_state import orbit_pose, validate_request
    from utils.event_refinement import qmul, qrotate
    from utils.gs_clay import gray_gaussians
    from utils.gs_pseudocolor import color_gaussians, pseudocolor_materials
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', default='0,25,50,75,99')
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--display-smoothing', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.manifest.read_text()); root = args.manifest.parent
    with np.load(root/cfg['sources']['right']) as z:
        original = z['base_vertices']; meta = json.loads(str(z['metadata_json']))
    with np.load(root/cfg['hybrid_paths']['right']) as z: original_groups = z['groups']
    with np.load(root/cfg['surface_refinement']) as z:
        refined = z['base_vertices']; refined_groups = z['groups']; parents = z['parent_rows']
    with np.load(root/cfg['mesh_paths']['right']) as z:
        motion = [torch.from_numpy(z[k]).cuda() for k in ('motion_quaternions', 'motion_translations')]
    camera = meta['preview_camera']; fovy = camera['fovy_degrees']
    frames = [int(f) for f in args.frames.split(',')]
    views = [(0, 0), (35, 20), (80, 0), (180, 0), (0, 35), (0, -35)]
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
    bounds = np.stack([original[k] for k in ('x', 'y', 'z')], 1)
    # Both versions use exactly the same pseudo-color bounds/material chart.
    anchor = torch.from_numpy(bounds).cuda()
    source_material = pseudocolor_materials([anchor])[0]
    metrics = []
    with torch.no_grad():
        for name, base, groups in [('before', original, original_groups), ('after', refined, refined_groups)]:
            model = model_from_vertices(base); rest = model._xyz.clone(), model._rotation.clone()
            material = source_material if name == 'before' else source_material[torch.from_numpy(parents).cuda()]
            groups = torch.from_numpy(groups.astype(np.int64)).cuda()
            for frame in frames:
                q, t = [a[frame, groups] for a in motion]
                model._xyz = qrotate(q, rest[0])+t; model._rotation = qmul(q, rest[1])
                for view, (yaw, pitch) in enumerate(views):
                    request = validate_request(dict(yaw=yaw, pitch=pitch), meta['frame_count'])
                    cam = make_camera(orbit_pose(np.array(camera['pose']), np.array(camera['target']), request),
                                      args.resolution, args.resolution, np.deg2rad(fovy))
                    result = render(cam, model, pipe, black, 0., 0., 0.)
                    color = render(cam, model, pipe, black, 0., 0., 0., override_color=material)['render']
                    images = dict(gs=result['render']+(1-result['alpha']),
                        clay=gray_gaussians(result['render'], result['depth'], result['alpha'], fovy, white),
                        pseudo=color_gaussians(result['render'], result['depth'], result['alpha'], color, fovy, white))
                    if name == 'after' and args.display_smoothing:
                        from utils.hybrid_sampling import smooth_surface
                        images = {key: smooth_surface(rgb, result['depth'], result['alpha']) for key, rgb in images.items()}
                    alpha = result['alpha'].squeeze().cpu().numpy()
                    stem = f'f{frame:03d}-v{view}'
                    if name == 'before': np.save(args.output/f'{stem}-alpha.npy', alpha)
                    for mode, rgb in images.items():
                        rgb = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                        Image.fromarray(rgb).save(args.output/f'{name}-{mode}-{stem}.png')
                        if name == 'after':
                            before = np.array(Image.open(args.output/f'before-{mode}-{stem}.png'))
                            old_alpha = np.load(args.output/f'{stem}-alpha.npy')
                            union = (old_alpha > .5) | (alpha > .5)
                            common = ndimage.binary_erosion((old_alpha > .98) & (alpha > .98), iterations=3)
                            gray = [x.mean(2)/255 for x in (before, rgb)]
                            roughness = [float(np.abs(ndimage.laplace(x))[common].mean()) for x in gray]
                            edge = np.hypot(ndimage.sobel(gray[0], 0), ndimage.sobel(gray[0], 1))
                            strong = common & (edge > .6)
                            new_edge = np.hypot(ndimage.sobel(gray[1], 0), ndimage.sobel(gray[1], 1))
                            metrics.append(dict(frame=frame, view=view, yaw=yaw, pitch=pitch, mode=mode,
                                silhouette_iou=float(((old_alpha > .5) & (alpha > .5)).sum()/max(1, union.sum())),
                                roughness_before=roughness[0], roughness_after=roughness[1],
                                strong_edge_ratio=float(new_edge[strong].mean()/max(1e-6,edge[strong].mean())) if strong.any() else None,
                                mean_rgb_change=float(np.abs(rgb.astype(float)-before).mean()/255)))
                            Image.fromarray(np.concatenate([before, rgb], axis=1)).save(args.output/f'compare-{mode}-{stem}.jpg', quality=96)
                    print(name, stem, flush=True)
            del model, rest, material, result, color, images, q, t
            torch.cuda.empty_cache()
    for mode in ('gs', 'clay', 'pseudo'):
        rows = [x for x in metrics if x['mode'] == mode]
        print(mode, 'roughness ratio', sum(x['roughness_after'] for x in rows)/sum(x['roughness_before'] for x in rows),
              'min silhouette IoU', min(x['silhouette_iou'] for x in rows), flush=True)
    (args.output/'metrics.json').write_text(json.dumps(dict(note=__doc__, display_smoothing=args.display_smoothing, measurements=metrics), indent=2)+'\n')


if __name__ == '__main__': main()
