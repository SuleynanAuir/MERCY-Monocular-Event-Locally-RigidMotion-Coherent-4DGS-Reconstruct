"""Fit bounded, spatially regularized Mesh colors to its own multiview Gaussian renders.

Geometry, topology and all embedded motion remain byte-for-byte unchanged.
Validation views are evaluated separately and never enter the optimizer.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageFilter
from utils.viewer_state import orbit_pose, validate_request
from utils.viewer_mesh import load_mesh_data, projection_matrix


def validate_capture(capture, frame_count):
    if (capture.get('format') != 'gaussian_mesh_color_views_v1'
            or capture.get('scene', {}).get('frame_count') != frame_count
            or not capture.get('scene', {}).get('comparison')):
        raise ValueError('Invalid paired Gaussian capture or timeline')
    cameras = {'train': set(), 'validation': set()}
    for view in capture.get('views', []):
        if view.get('split') not in cameras:
            raise ValueError('Unknown view split')
        request = validate_request(view['request'], frame_count)
        if request['render_mode'] != 'gaussian' or request['background'] != 'black':
            raise ValueError('Expected actual Gaussian captures on black')
        key = (request['yaw'] % 360, *(request[k] for k in
               ('pitch', 'zoom', 'pan_x', 'pan_y', 'fovy')))
        cameras[view['split']].add(key)
    if not all(cameras.values()) or cameras['train'] & cameras['validation']:
        raise ValueError('Fitting and validation need separate camera views')


def detail_color_limits(colors, faces, maximum):
    """Protect already smooth areas; allow larger corrections at existing details."""
    contrast = np.zeros(len(colors), np.float32)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = faces[:, i], faces[:, j]
        difference = np.max(np.abs(colors[a] - colors[b]), axis=1)
        np.maximum.at(contrast, a, difference)
        np.maximum.at(contrast, b, difference)
    return maximum * (.05 + .95 * np.minimum(contrast / .08, 1) ** 2)


def main():
    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr
    from utils.event_refinement import qrotate
    from utils.rigid_checkpoint import sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--views', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--max-color-delta', type=float, default=.20)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.epochs <= 100 or not 0 < args.max_color_delta <= .3:
        parser.error('Use a new output directory and valid fitting limits')
    torch.set_num_threads(4); torch.manual_seed(42)
    started = time.perf_counter()
    with np.load(args.input, allow_pickle=False) as z:
        archive = {k: z[k] for k in z.files}
    meta = json.loads(str(archive['metadata_json']))
    data = load_mesh_data(args.input, meta['group_count'], meta['source_fingerprints'])
    if meta['format'] != 'gaussian_tsdf_mesh_v2':
        raise ValueError('Independent embedded source motion is required')
    side = 0 if meta['source_side'] == 'left' else 1
    settings = json.loads(args.comparison.read_text())
    with np.load(settings['mesh_paths'][meta['source_side']], allow_pickle=False) as z:
        active_meta = json.loads(str(z['metadata_json']))
    if active_meta['source_fingerprints'] != meta['source_fingerprints']:
        raise ValueError('Input source differs from active comparison')
    capture = json.loads((args.views / 'views.json').read_text())
    validate_capture(capture, meta['frame_count'])
    with np.load(settings['mesh_paths']['right'], allow_pickle=False) as z:
        right_meta = json.loads(str(z['metadata_json']))
    for key, value in right_meta['source_fingerprints'].items():
        if capture['scene'].get(key) != value:
            raise ValueError('Captured Gaussian source differs from comparison')
    camera = json.loads((Path(settings['baseline']) / 'cache.json').read_text())['camera']
    vertices = torch.from_numpy(data['vertices']).cuda()
    faces = torch.from_numpy(data['faces']).cuda()
    groups = torch.from_numpy(data['vertex_groups'].astype(np.int64)).cuda()
    qs = torch.from_numpy(data['motion_quaternions'].astype(np.float32)).cuda()
    ts = torch.from_numpy(data['motion_translations'].astype(np.float32)).cuda()
    base = torch.from_numpy(data['colors']).cuda()
    limit = torch.from_numpy(detail_color_limits(data['colors'], data['faces'], args.max_color_delta)).cuda()[:, None]
    colors = torch.nn.Parameter(base.clone())
    context = dr.RasterizeCudaContext()
    prepared = []
    with torch.no_grad():
        for index, view in enumerate(capture['views']):
            image_path = args.views / view['file']
            if sha256_file(image_path) != view['sha256']:
                raise ValueError('Captured image checksum mismatch')
            request = validate_request(view['request'], meta['frame_count'])
            n = request['width']
            if request['height'] != n or request['background'] != 'black':
                raise ValueError('Expected square Gaussian captures on black')
            paired = Image.open(image_path)
            if paired.size != (2*n, n):
                raise ValueError('Captured image does not contain two full-size panels')
            image = paired.crop((side*n, 0, (side+1)*n, n))
            raw = torch.from_numpy(np.asarray(image).copy()).cuda().float() / 255
            # Suppress subpixel Gaussian grain while retaining visible texture edges.
            target = torch.from_numpy(np.asarray(image.filter(ImageFilter.GaussianBlur(.45))).copy()).cuda().float() / 255
            frame = request['frame']
            xyz = qrotate(qs[frame, groups], vertices) + ts[frame, groups]
            pose = orbit_pose(camera['pose'], camera['target'], request)
            cam = torch.as_tensor(pose, device='cuda')
            xyz = (xyz - cam[:3, 3]) @ cam[:3, :3]
            projection = torch.as_tensor(projection_matrix(n, n, request['fovy']), device='cuda')
            clip = (torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1) @ projection.T)[None].contiguous()
            raster, _ = dr.rasterize(context, clip, faces, [n, n], grad_db=False)
            visible = raster[0, :, :, 3].flip(0) > 0
            foreground = raw.max(dim=2).values > .025
            mask = (visible & foreground).float()[None, None]
            mask = (1 - F.max_pool2d(1 - mask, 3, stride=1, padding=1))[0, 0] > .5
            prepared.append(dict(raster=raster.cpu(), target=target.cpu(), mask=mask.cpu(),
                                 split=view['split'], request=request))
            print(f'Prepared {index + 1}/{len(capture["views"])} {view["split"]}', flush=True)
    train = [v for v in prepared if v['split'] == 'train']
    validation = [v for v in prepared if v['split'] == 'validation']
    if not train or not validation:
        raise ValueError('Separate fitting and validation views are required')
    optimizer = torch.optim.Adam([colors], lr=.003)
    # Fixed local regularization prevents sparse per-vertex fitting from bringing
    # back the grain removed by surface denoising. Never connect motion groups.
    def render_view(view):
        rgb, _ = dr.interpolate(colors[None].contiguous(), view['raster'].cuda(), faces)
        return rgb[0].flip(0), view['target'].cuda(), view['mask'].cuda()

    @torch.no_grad()
    def evaluate(views):
        return float(np.mean([float((pred[mask]-target[mask]).abs().mean())
                              for pred, target, mask in (render_view(v) for v in views)]))

    initial = dict(train=evaluate(train), validation=evaluate(validation))
    history = []
    for epoch in range(args.epochs):
        optimizer.param_groups[0]['lr'] = .003 * (.1 + .9 * .5 * (1 + np.cos(np.pi * epoch / args.epochs)))
        for index in torch.randperm(len(train)).tolist():
            optimizer.zero_grad(set_to_none=True)
            pred, target, mask = render_view(train[index])
            residual = pred - target
            loss = F.smooth_l1_loss(pred[mask], target[mask], beta=.06)
            dy, dx = mask[1:] & mask[:-1], mask[:, 1:] & mask[:, :-1]
            loss += .15 * ((residual[1:] - residual[:-1])[dy].abs().mean()
                           + (residual[:, 1:] - residual[:, :-1])[dx].abs().mean())
            delta = colors - base
            # Resample the entire surface, rather than leaving most vertices
            # unregularized by using one fixed tiny edge subset.
            edge_faces = faces[torch.randint(len(faces), (200000,), device='cuda')].long()
            edge_a, edge_b = edge_faces[:, 0], edge_faces[:, 1]
            edge_weight = torch.exp(-((base[edge_a] - base[edge_b]) ** 2).mean(1) / (2 * .1**2))
            loss += .1 * delta.abs().mean()
            loss += .3 * ((delta[edge_a] - delta[edge_b]).abs().mean(1) * edge_weight).mean()
            loss.backward(); optimizer.step()
            with torch.no_grad():
                colors.copy_(torch.maximum(base - limit, torch.minimum(base + limit, colors)).clamp(0, 1))
        row = dict(epoch=epoch + 1, train=evaluate(train), validation=evaluate(validation))
        history.append(row); print(json.dumps(row), flush=True)
    # Export one predeclared training duration; validation never supplies gradients
    # or per-view colors. Inspect its results before deploying any candidate.
    report = dict(method='Bounded multiview Gaussian-to-Mesh color fitting',
                  source_mesh_sha256=sha256_file(args.input), capture_sha256=sha256_file(args.views / 'views.json'),
                  training_views=len(train), validation_views=len(validation), epochs=args.epochs,
                  max_color_delta=args.max_color_delta, initial=initial, history=history,
                  flat_surface_protection=dict(contrast_scale=.08, minimum_limit_ratio=.05,
                                               mean_color_limit=float(limit.mean())),
                  optimizer=dict(name='Adam', initial_learning_rate=.003, schedule='cosine to 10 percent',
                                 edge_samples_per_step=200000, delta_prior_weight=.1,
                                 delta_neighbor_weight=.3, image_gradient_weight=.15, seed=42),
                  geometry_and_motion_unchanged=True, target_blur_radius=.45,
                  elapsed_seconds=time.perf_counter()-started)
    meta['color_refinement'] = report
    archive['colors'] = colors.detach().cpu().numpy()
    archive['metadata_json'] = np.asarray(json.dumps(meta))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'mesh.npz', **archive)
    (args.output / 'report.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
