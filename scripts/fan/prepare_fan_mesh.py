"""Reconstruct Fan Mesh and triangle-bound GS for its before/after motion versions.

Canonical geometry is shared only after validating identical Gaussian rows.
Three independently reconstructed groups prevent a triangle from connecting the
rotor to static support. Source motions are recovered and checked on every frame.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.fan_mesh import comparison_groups, validate_source_pair


def reconstruct(renderer, groups, output, size, voxel, yaw_step):
    import open3d as o3d
    import torch
    from gaussian_renderer import render
    from render_ply import make_camera
    from scene import GaussianModel
    from scripts.mesh.reconstruct_viewer_mesh import valid_depth, continuous_surface, projected_samples, bilinear
    from utils.viewer_state import orbit_pose, validate_request

    fovy = renderer.info['fovy']
    focal = size / (2 * np.tan(np.deg2rad(fovy) / 2))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(size, size, focal, focal, (size-1)/2, (size-1)/2)
    plan = [(yaw, pitch) for pitch in (-60, -30, 0, 30, 60) for yaw in range(0, 360, yaw_step)]
    collected = {key: [] for key in ('vertices', 'triangles', 'colors', 'vertex_groups')}
    reports, offset = [], 0
    for group in range(3):
        ids = np.flatnonzero(groups == group)
        part = GaussianModel(0, with_motion_mask=False)
        rows = torch.as_tensor(ids, device='cuda')
        for field in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
            setattr(part, field, getattr(renderer.model, field)[rows].contiguous())
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=voxel*3,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        def observation(yaw, pitch):
            req = validate_request({'yaw': yaw, 'pitch': pitch, 'zoom': 1.05}, renderer.info['frame_count'])
            pose = orbit_pose(renderer.base_pose, renderer.target, req)
            result = render(make_camera(pose, size, size, np.deg2rad(fovy)), part, renderer.pipe,
                            renderer.backgrounds['black'], 0., 0., 0.)
            alpha = result['alpha'].squeeze().cpu().numpy()
            depth = valid_depth(result['depth'].squeeze().cpu().numpy(), alpha, .65)
            rgb = result['render'].permute(1, 2, 0).cpu().numpy()/np.maximum(alpha[..., None], 1e-6)
            return pose, alpha, depth, rgb.clip(0, 1)

        for index, (yaw, pitch) in enumerate(plan):
            pose, alpha, depth, rgb = observation(yaw, pitch)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray((rgb*255).astype(np.uint8))),
                o3d.geometry.Image(depth), depth_scale=1., depth_trunc=100., convert_rgb_to_intensity=False)
            extrinsic = np.linalg.inv(pose.astype(np.float64)); extrinsic[1:3] *= -1
            volume.integrate(rgbd, intrinsic, extrinsic)
            if (index+1) % 18 == 0: print(f'Group {group}: fused {index+1}/{len(plan)}', flush=True)
        mesh = volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        if not len(mesh.triangles): raise ValueError(f'No observed surface for group {group}')
        labels, counts, _ = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 20)
        mesh.remove_unreferenced_vertices()
        # Observed-neighborhood Poisson regularization suppresses TSDF speckle;
        # shallow Taubin smoothing avoids rounding away the thin blade ribs.
        mesh = continuous_surface(mesh, voxel, 10)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=2,
                                        filter_scope=o3d.geometry.FilterScope.Vertex)
        mesh.compute_vertex_normals()
        v = np.asarray(mesh.vertices).astype(np.float32)
        normals = np.asarray(mesh.vertex_normals).astype(np.float32)
        colors = np.asarray(mesh.vertex_colors).astype(np.float32).copy()
        best = np.zeros(len(v), np.float32)
        supported, empty = np.zeros(len(v), np.int32), np.zeros(len(v), np.int32)
        for index, (yaw, pitch) in enumerate(plan):
            pose, alpha, depth, rgb = observation(yaw, pitch)
            x, y, z, visible = projected_samples(v, pose, focal, size)
            ids = np.flatnonzero(visible)
            if not len(ids): continue
            a = bilinear(alpha, x[ids], y[ids]); d = bilinear(depth, x[ids], y[ids])
            difference = np.abs(z[ids]-d)
            toward = pose[:3, 3]-v[ids]
            toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-8)
            facing = np.abs((normals[ids]*toward).sum(1))
            supported[ids] += (a >= .65) & (difference <= voxel*4)
            empty[ids] += (facing > .35) & ((a < .05) | ((d > 0) & (z[ids] < d-voxel*6)))
            score = a*facing**2/(1+difference/voxel)
            if yaw == 0 and pitch == 0: score *= 2
            score[(a < .65) | (difference > voxel*3)] = 0
            chosen = ids[score > best[ids]]
            colors[chosen] = bilinear(rgb, x[chosen], y[chosen]).clip(0, 1)
            best[chosen] = score[score > best[ids]]
            if (index+1) % 30 == 0: print(f'Group {group}: sharp color {index+1}/{len(plan)}', flush=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.remove_vertices_by_mask((empty >= 3) & (empty > supported*2))
        mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        if not len(mesh.triangles): raise ValueError(f'No supported surface for group {group}')
        v = np.asarray(mesh.vertices).astype(np.float32)
        collected['vertices'].append(v)
        collected['triangles'].append(np.asarray(mesh.triangles, np.int32)+offset)
        collected['colors'].append(np.asarray(mesh.vertex_colors, np.float32).clip(0, 1))
        collected['vertex_groups'].append(np.full(len(v), group, np.int32))
        offset += len(v)
        reports.append(dict(group=group, gaussians=int((groups == group).sum()),
                            vertices=len(v), triangles=len(mesh.triangles)))
        o3d.io.write_triangle_mesh(str(output/f'group-{group}.ply'), mesh)
        print(reports[-1], flush=True)
        del part, volume
    return {key: np.concatenate(value) for key, value in collected.items()}, reports


def attach(mesh, groups, xyz):
    import open3d as o3d
    from utils.mesh_gaussian_binding import attachment_centers
    assigned = np.full(len(xyz), -1, np.int32)
    weights, offsets = np.zeros((len(xyz), 3), np.float32), np.zeros((len(xyz), 3), np.float32)
    for group in range(3):
        rows = np.flatnonzero(groups == group)
        vertices = np.flatnonzero(mesh['vertex_groups'] == group)
        faces = np.flatnonzero(mesh['vertex_groups'][mesh['triangles'][:, 0]] == group)
        local_faces = np.searchsorted(vertices, mesh['triangles'][faces]).astype(np.uint32)
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(mesh['vertices'][vertices])),
                            o3d.core.Tensor(np.ascontiguousarray(local_faces)))
        for start in range(0, len(rows), 100000):
            selected = rows[start:start+100000]
            nearest = scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(xyz[selected])), nthreads=4)
            uv = nearest['primitive_uvs'].numpy()
            bary = np.column_stack([1-uv.sum(1), uv]).clip(0, 1); bary /= bary.sum(1, keepdims=True)
            face_ids = faces[nearest['primitive_ids'].numpy()]
            center = attachment_centers(mesh['vertices'], mesh['triangles'], face_ids, bary, np.zeros_like(bary))
            assigned[selected], weights[selected], offsets[selected] = face_ids, bary, xyz[selected]-center
        print(f'Bound group {group}: {len(rows)} Gaussians', flush=True)
    if np.any(assigned < 0): raise ValueError('Incomplete Fan Mesh binding')
    return dict(rows=np.arange(len(xyz), dtype=np.int32), face_indices=assigned,
                barycentric=weights, offsets=offsets, groups=groups)


def main():
    import torch
    from viewer import Renderer
    from utils.gaussian_sequence import GaussianSequence
    from utils.mesh_motion import recover_rigid_motion
    from utils.mesh_gaussian_binding import pose_fingerprint, validate_attachment
    from utils.rigid_checkpoint import sha256_file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--voxel-size', type=float, default=.001)
    parser.add_argument('--yaw-step', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    if not (128 <= args.resolution <= 1280 and 0 < args.voxel_size < .01 and 1 <= args.yaw_step <= 90):
        parser.error('Invalid reconstruction settings')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    start = time.perf_counter()
    paths = dict(left=args.before.resolve(), right=args.after.resolve())
    with GaussianSequence(paths['left']) as left, GaussianSequence(paths['right']) as right:
        validate_source_pair(left.metadata, right.metadata)
        groups = comparison_groups(len(left.base_vertices), left.dynamic_indices, right.dynamic_indices)
    motion, fingerprints, canonical = {}, {}, {}
    renderer = None
    for name, path in paths.items():
        r = Renderer(path, 'cpu'); r.load()
        canonical[name] = (r.model._xyz.detach().cpu().numpy().copy(),
                           r.model._rotation.detach().cpu().numpy().copy())
        fingerprints[name] = {'sequence_sha256': sha256_file(path)}
        ids = r.dynamic_indices.cpu().numpy()
        q, t, report = recover_rigid_motion(r.poses.numpy(), groups[ids], 3)
        motion[name] = (q, t, report)
        print(name, 'source motion verified:', report, flush=True)
        if name == 'left':
            renderer = r
        else:
            del r
    with torch.no_grad():
        # The load-time warmup is frame zero. Retain the exact canonical GS
        # rather than flattening centers onto the extracted surface.
        xyz = renderer.model._xyz.cpu().numpy().copy()
        rotation = renderer.model._rotation.cpu().numpy().copy()
        mesh, reports = reconstruct(renderer, groups, args.output, args.resolution, args.voxel_size, args.yaw_step)
    binding = attach(mesh, groups, xyz)
    for name in paths:
        from utils.mesh_gaussian_binding import attachment_centers
        # Rows restored to static use their original PLY quaternion; moving
        # rows use baked normalized quaternions. Preserve each source exactly.
        xyz, rotation = canonical[name]
        surface = attachment_centers(mesh['vertices'], mesh['triangles'], binding['face_indices'],
                                     binding['barycentric'], np.zeros_like(binding['offsets']))
        side_binding = dict(binding, offsets=(xyz-surface).astype(np.float32))
        q, t, check = motion[name]
        meta = dict(format='gaussian_tsdf_mesh_v2', source_side=name, source_fingerprints=fingerprints[name],
                    frame_count=len(q), group_count=3, embedded_motion=True, motion_validation=check,
                    source_mesh_used=False, method='Gaussian RGB/depth TSDF with observed-neighborhood Poisson',
                    canonical_geometry_shared=True, groups=reports, resolution=args.resolution,
                    voxel_size=args.voxel_size, surface_denoising=True, color_refinement=True,
                    scope='V1 before binding repair; V3 after binding repair. Video-fitted motion, not event validation.')
        mesh_path = args.output/f'{name}-mesh.npz'
        np.savez_compressed(mesh_path, **mesh, motion_quaternions=q, motion_translations=t,
                            metadata_json=np.asarray(json.dumps(meta)))
        record = dict(format='mesh_gaussian_binding_v1', source_side=name,
                      source_fingerprints=fingerprints[name], mesh_sha256=sha256_file(mesh_path),
                      gaussian_count=len(xyz), canonical_pose_sha256=pose_fingerprint(xyz, rotation),
                      appearance_preserved=True, offsets_preserved=True, unbound_gaussians=0)
        data = dict(side_binding, metadata_json=np.asarray(json.dumps(record)))
        validate_attachment(data, mesh['vertices'], mesh['triangles'], mesh['vertex_groups'],
                            xyz, rotation, meta, record['mesh_sha256'])
        np.savez_compressed(args.output/f'{name}-binding.npz', **data)
    manifest = dict(format='fan_mesh_comparison_v1', sources={k:str(v) for k,v in paths.items()},
                    source_fingerprints=fingerprints,
                    mesh_paths={k:f'{k}-mesh.npz' for k in paths},
                    hybrid_paths={k:f'{k}-binding.npz' for k in paths})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(dict(groups=reports, source_fingerprints=fingerprints,
        motion={k:v[2] for k,v in motion.items()}, elapsed_seconds=time.perf_counter()-start), indent=2)+'\n')
    print('Ready:', args.output/'manifest.json', flush=True)


if __name__ == '__main__': main()
