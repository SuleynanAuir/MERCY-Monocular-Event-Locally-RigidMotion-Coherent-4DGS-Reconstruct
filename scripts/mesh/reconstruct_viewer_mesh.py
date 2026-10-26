"""Fuse actual optimized Gaussian RGB/depth renders into independently moving meshes.

No source mesh is read. All surfaces come from the current Gaussian renderer.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from utils.viewer_state import orbit_pose, validate_request


def valid_depth(depth_sum, alpha, threshold):
    depth = np.asarray(depth_sum, np.float32) / np.maximum(alpha, 1e-6)
    return np.where((alpha >= threshold) & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)


def projected_samples(vertices, pose, focal, size):
    """Project canonical mesh vertices into exactly the Gaussian render camera."""
    camera = (vertices - pose[:3, 3]) @ pose[:3, :3]
    z = -camera[:, 2]
    denominator = np.maximum(z, 1e-8)
    x = focal * camera[:, 0] / denominator + (size - 1) / 2
    y = -focal * camera[:, 1] / denominator + (size - 1) / 2
    valid = (z > .01) & (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
    return x, y, z, valid


def bilinear(image, x, y):
    ix, iy = x.astype(np.int64), y.astype(np.int64)
    fx, fy = x - ix, y - iy
    if image.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((1 - fx) * (1 - fy) * image[iy, ix] + fx * (1 - fy) * image[iy, ix + 1]
            + (1 - fx) * fy * image[iy + 1, ix] + fx * fy * image[iy + 1, ix + 1])


def continuous_surface(mesh, voxel_size, depth):
    """Screened Poisson regularization, limited to the observed TSDF neighborhood."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud()
    cloud.points, cloud.normals, cloud.colors = mesh.vertices, mesh.vertex_normals, mesh.vertex_colors
    cloud = cloud.voxel_down_sample(voxel_size * 1.5)
    valid = np.linalg.norm(np.asarray(cloud.normals), axis=1) > .5
    cloud = cloud.select_by_index(np.flatnonzero(valid)); cloud.normalize_normals()
    if len(cloud.points) < 30:
        return mesh
    refined, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, scale=1.05, linear_fit=True, n_threads=4)
    distance, nearest = cKDTree(np.asarray(mesh.vertices)).query(np.asarray(refined.vertices), workers=4)
    refined.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[nearest])
    # Do not extrapolate sheets into unobserved space or close large mechanical openings.
    refined.remove_vertices_by_mask((distance > voxel_size * 5) | (np.asarray(density) < np.quantile(density, .005)))
    refined.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    return refined


def main():
    import open3d as o3d
    import torch
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera
    from scene import GaussianModel
    from utils.comparison_renderer import ComparisonRenderer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--side', choices=('left', 'right'), default='right')
    parser.add_argument('--left-binding', type=Path, help='Original GS attachment labels; no triangle mesh is read')
    parser.add_argument('--reuse-tsdf', type=Path, help='Reuse a matching earlier Gaussian TSDF mesh')
    parser.add_argument('--surface-method', choices=('tsdf', 'poisson'), default='poisson')
    parser.add_argument('--poisson-depth', type=int, default=10)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--voxel-size', type=float, default=.0015)
    parser.add_argument('--alpha-threshold', type=float, default=.65)
    parser.add_argument('--yaw-step', type=int, default=20)
    parser.add_argument('--truncation-voxels', type=float, default=4.)
    parser.add_argument('--smooth-iterations', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    if (not 128 <= args.resolution <= 1280 or not 0 < args.voxel_size < 1
            or not 0 < args.alpha_threshold < 1 or not 1 <= args.yaw_step <= 90
            or not 1 <= args.truncation_voxels <= 8 or not 0 <= args.smooth_iterations <= 10
            or not 7 <= args.poisson_depth <= 12):
        parser.error('Invalid reconstruction settings')
    if args.side == 'left' and args.left_binding is None:
        parser.error('--left-binding is required for the original source')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    with torch.no_grad():
        renderer = ComparisonRenderer(args.comparison, 'cpu')
        renderer.load(prewarm=False)
        side = 0 if args.side == 'left' else 1
        renderer.apply_side_pose(side, 0)
        model = renderer.model
        groups = torch.zeros(len(model._xyz), dtype=torch.long, device='cuda')
        # All immutable rows are assigned to a stationary root; elastic dynamic
        # rows remain -1 and are omitted, not silently frozen in the Mesh view.
        roots = [i for i, parent in enumerate(renderer.correction.parents) if parent == -1]
        if not roots:
            raise ValueError('No stationary root for the static Gaussian rows')
        groups.fill_(roots[0])
        settings = json.loads(args.comparison.read_text())
        from utils.rigid_checkpoint import sha256_file
        motion_check = None
        if side == 0:
            from utils.mesh_motion import baseline_groups, recover_rigid_motion
            baseline = Path(settings['baseline'])
            source_meta = json.loads((baseline / 'cache.json').read_text())
            record = json.loads((Path(settings['candidate']) / 'correction.json').read_text())
            with np.load(args.left_binding, allow_pickle=False) as binding:
                if not np.array_equal(binding['dynamic_indices'], renderer.indices[0].cpu().numpy()):
                    raise ValueError('Original binding does not match baseline motion rows')
                labels = baseline_groups(binding['labels'], source_meta['source_animation']['objects'],
                                         record['joint_structure']['source_groups'])
            if len(labels) != len(model._xyz):
                raise ValueError('Original binding does not match baseline Gaussians')
            groups = torch.from_numpy(labels).cuda()
            q_frames, t_frames, motion_check = recover_rigid_motion(
                renderer.poses[0], labels[renderer.indices[0].cpu().numpy()], renderer.correction.group_count)
            print('Validated original motion: ' + json.dumps(motion_check), flush=True)
            fingerprints = dict(source_sha256=sha256_file(source_meta['source']),
                                poses_sha256=sha256_file(baseline / 'poses.npy'),
                                dynamic_indices_sha256=sha256_file(baseline / 'dynamic_indices.npy'))
        else:
            groups[renderer.indices[1]] = renderer.correction.row_groups
            transforms = [renderer.correction.group_transforms(i) for i in range(renderer.info['frame_count'])]
            q_frames = np.stack([q.cpu().numpy() for q, _ in transforms])
            t_frames = np.stack([t.cpu().numpy() for _, t in transforms])
            fingerprints = {k: renderer.info[k] for k in ('candidate_correction_sha256', 'right_source_sha256')}
        if side == 1 and renderer.completion is not None:
            completion = renderer.completion
            groups[completion.prefix_count:completion.prefix_count + len(completion.groups)] = completion.groups
        excluded = int((groups < 0).sum())
        reused = None
        if args.reuse_tsdf:
            from utils.viewer_mesh import load_mesh_data
            reused = load_mesh_data(args.reuse_tsdf, renderer.correction.group_count, fingerprints)
        # The exported scene is canonical; group transforms are applied only by
        # the interactive renderer, never baked a second time into mesh vertices.
        q0, t0 = renderer.correction.group_transforms(0)
        if not torch.allclose(q0[:, 0].abs(), torch.ones_like(q0[:, 0]), atol=1e-5) or not torch.allclose(t0, torch.zeros_like(t0), atol=1e-5):
            raise ValueError('Expected canonical identity motion at frame zero')
        plan = [(yaw, pitch) for pitch in (-60, -30, 0, 30, 60) for yaw in range(0, 360, args.yaw_step)]
        n = args.resolution
        fovy = renderer.info['fovy']
        focal = n / (2 * np.tan(np.deg2rad(fovy) / 2))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(n, n, focal, focal, (n - 1) / 2, (n - 1) / 2)
        vertices, triangles, colors, vertex_groups, reports = [], [], [], [], []
        offset = 0
        background = torch.zeros(3, device='cuda')
        for group in sorted(set(groups.cpu().tolist()) - {-1}):
            rows = torch.nonzero((groups == group) & (model.get_opacity[:, 0] >= .05), as_tuple=True)[0]
            if len(rows) < 3:
                continue
            part = GaussianModel(0, with_motion_mask=False)
            for field in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
                setattr(part, field, getattr(model, field)[rows].contiguous())
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=args.voxel_size, sdf_trunc=args.voxel_size * args.truncation_voxels,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for index, (yaw, pitch) in enumerate(plan if reused is None else []):
                request = validate_request({'yaw': yaw, 'pitch': pitch, 'zoom': 1.05}, renderer.info['frame_count'])
                pose = orbit_pose(renderer.base_pose, renderer.target, request)
                camera = make_camera(pose, n, n, np.deg2rad(fovy))
                result = render(camera, part, renderer.pipe, background, 0., 0., 0.)
                alpha = result['alpha'].squeeze().cpu().numpy()
                depth = valid_depth(result['depth'].squeeze().cpu().numpy(), alpha, args.alpha_threshold)
                # Remove the black-background alpha contribution before fusing color.
                rgb = result['render'].permute(1, 2, 0).cpu().numpy() / np.maximum(alpha[..., None], 1e-6)
                rgb = np.ascontiguousarray((rgb.clip(0, 1) * 255).astype(np.uint8))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                    depth_scale=1., depth_trunc=100., convert_rgb_to_intensity=False)
                extrinsic = np.linalg.inv(pose.astype(np.float64)); extrinsic[1:3] *= -1
                volume.integrate(rgbd, intrinsic, extrinsic)
                if (index + 1) % 12 == 0:
                    print(f'Group {group}: fused {index + 1}/{len(plan)} Gaussian views', flush=True)
            if reused is None:
                mesh = volume.extract_triangle_mesh()
            else:
                selected = np.flatnonzero(reused['vertex_groups'] == group)
                remap = np.full(len(reused['vertices']), -1, np.int32); remap[selected] = np.arange(len(selected))
                faces = reused['faces'][reused['face_groups'] == group]
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(reused['vertices'][selected]),
                                                o3d.utility.Vector3iVector(remap[faces]))
                mesh.vertex_colors = o3d.utility.Vector3dVector(reused['colors'][selected])
            mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
            if len(mesh.triangles):
                labels, counts, _ = mesh.cluster_connected_triangles()
                mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 30)
                mesh.remove_unreferenced_vertices()
                if args.surface_method == 'poisson':
                    print(f'Group {group}: regularizing observed surface', flush=True)
                    mesh = continuous_surface(mesh, args.voxel_size, args.poisson_depth)
                if args.smooth_iterations:
                    mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations,
                                                     filter_scope=o3d.geometry.FilterScope.Vertex)
                mesh.compute_vertex_normals()
                v = np.asarray(mesh.vertices).astype(np.float32)
                normals = np.asarray(mesh.vertex_normals).astype(np.float32)
                baked = np.asarray(mesh.vertex_colors).astype(np.float32).copy()
                best = np.zeros(len(v), np.float32)
                supported = np.zeros(len(v), np.int32)
                empty = np.zeros(len(v), np.int32)
                # Pick a visible, well-aligned high-resolution observation instead
                # of blurring colors across every view. No reference mesh/RGB is used.
                for view_index, (yaw, pitch) in enumerate(plan):
                    request = validate_request({'yaw': yaw, 'pitch': pitch, 'zoom': 1.05}, renderer.info['frame_count'])
                    pose = orbit_pose(renderer.base_pose, renderer.target, request)
                    x, y, z, in_view = projected_samples(v, pose, focal, n)
                    ids = np.flatnonzero(in_view)
                    if not len(ids):
                        continue
                    result = render(make_camera(pose, n, n, np.deg2rad(fovy)), part, renderer.pipe, background, 0., 0., 0.)
                    alpha = result['alpha'].squeeze().cpu().numpy()
                    depth = valid_depth(result['depth'].squeeze().cpu().numpy(), alpha, args.alpha_threshold)
                    a = bilinear(alpha, x[ids], y[ids])
                    d = bilinear(depth, x[ids], y[ids])
                    difference = np.abs(z[ids] - d)
                    toward_camera = pose[:3, 3] - v[ids]
                    toward_camera /= np.maximum(np.linalg.norm(toward_camera, axis=1, keepdims=True), 1e-8)
                    facing = np.abs(np.sum(normals[ids] * toward_camera, axis=1))
                    supported[ids] += ((a >= args.alpha_threshold) & (difference <= args.voxel_size * 4))
                    empty[ids] += ((facing > .35) & ((a < .05) |
                                  ((d > 0) & (z[ids] < d - args.voxel_size * 6))))
                    score = a * facing**2 / (1 + difference / args.voxel_size)
                    if yaw == 0 and pitch == 0:
                        score *= 2  # Prefer the actual calibrated view when visible.
                    score[(a < args.alpha_threshold) | (difference > args.voxel_size * 3)] = 0
                    take = score > best[ids]
                    chosen = ids[take]
                    rgb = result['render'].permute(1, 2, 0).cpu().numpy() / np.maximum(alpha[..., None], 1e-6)
                    baked[chosen] = bilinear(rgb, x[chosen], y[chosen]).clip(0, 1)
                    best[chosen] = score[take]
                    if (view_index + 1) % 30 == 0:
                        print(f'Group {group}: baked sharp colors {view_index + 1}/{len(plan)}', flush=True)
                mesh.vertex_colors = o3d.utility.Vector3dVector(baked)
                if args.surface_method == 'poisson':
                    # Keep genuine openings visible in the source GS. Occluded
                    # samples never count as empty-space evidence.
                    carved = (empty >= 3) & (empty > supported * 2)
                    print(f'Group {group}: removed {int(carved.sum())} unsupported surface vertices', flush=True)
                    mesh.remove_vertices_by_mask(carved)
                    mesh.remove_unreferenced_vertices()
            count = len(mesh.triangles)
            reports.append(dict(group=group, gaussians=len(rows), vertices=len(mesh.vertices), triangles=count))
            print(json.dumps(reports[-1]), flush=True)
            if count:
                v = np.asarray(mesh.vertices).astype(np.float32)
                vertices.append(v); triangles.append(np.asarray(mesh.triangles, np.int32) + offset)
                colors.append(np.asarray(mesh.vertex_colors).astype(np.float32).clip(0, 1))
                vertex_groups.append(np.full(len(v), group, np.int32)); offset += len(v)
                o3d.io.write_triangle_mesh(str(args.output / f'group-{group:02d}.ply'), mesh)
            del part, volume
        if not triangles:
            raise RuntimeError('No surface could be reconstructed from Gaussian depth')
        meta = dict(format='gaussian_tsdf_mesh_v2', source_side=args.side,
                    source_fingerprints=fingerprints, frame_count=renderer.info['frame_count'],
                    motion_validation=motion_check, embedded_motion=True,
                    method='Per-rigid-group TSDF fusion of alpha-normalized Gaussian rendered depths',
                    source_mesh_used=False, frame=0, group_count=renderer.correction.group_count,
                    excluded_elastic_gaussians=excluded, voxel_size=args.voxel_size,
                    alpha_threshold=args.alpha_threshold, resolution=n, views_per_group=len(plan),
                    truncation_voxels=args.truncation_voxels, smooth_iterations=args.smooth_iterations,
                    color_method='Best visible high-resolution Gaussian view, with calibrated-view preference',
                    surface_method=args.surface_method, poisson_depth=args.poisson_depth,
                    empty_space_carving=args.surface_method == 'poisson',
                    reused_tsdf_sha256=sha256_file(args.reuse_tsdf) if args.reuse_tsdf else None,
                    groups=reports, elapsed_seconds=time.perf_counter() - started,
                    scope='Approximate surface reconstructed from optimized Gaussian renders; not ground truth. '
                          'Transparent regions, small components and elastic Gaussian rows are omitted.')
        np.savez_compressed(args.output / 'mesh.npz', vertices=np.concatenate(vertices), triangles=np.concatenate(triangles),
                            colors=np.concatenate(colors), vertex_groups=np.concatenate(vertex_groups),
                            motion_quaternions=q_frames, motion_translations=t_frames,
                            metadata_json=np.asarray(json.dumps(meta)))
        (args.output / 'report.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'vertices': offset,
                          'triangles': sum(len(t) for t in triangles), 'elapsed': meta['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
