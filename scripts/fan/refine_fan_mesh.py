"""Fill only small Gaussian-supported Mesh holes and prefer calibrated-view color.

Original vertices and triangle indices remain intact, so existing barycentric
attachments retain their meaning. New triangles stay within a single part.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def main():
    import open3d as o3d
    import torch
    from viewer import Renderer
    from scene import GaussianModel
    from gaussian_renderer import render
    from render_ply import make_camera
    from utils.viewer_state import orbit_pose, validate_request
    from scripts.mesh.reconstruct_viewer_mesh import valid_depth, projected_samples, bilinear
    from utils.rigid_checkpoint import sha256_file
    from utils.mesh_gaussian_binding import validate_attachment
    from utils.fan_mesh import comparison_groups
    from utils.gaussian_sequence import GaussianSequence
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    config = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    with np.load(root/config['mesh_paths']['left']) as z: base = {k:z[k] for k in z.files}
    with GaussianSequence(config['sources']['left']) as left, GaussianSequence(config['sources']['right']) as right:
        groups = comparison_groups(len(left.base_vertices), left.dynamic_indices, right.dynamic_indices)
    torch.set_num_threads(4)
    r = Renderer(config['sources']['left'], 'cpu'); r.load()
    n, fovy = 1280, r.info['fovy']
    focal = n/(2*np.tan(np.deg2rad(fovy)/2))
    patches, reports = [], []
    colors = base['colors'].copy()
    with torch.no_grad():
        for group in range(3):
            rows = torch.as_tensor(np.flatnonzero(groups == group), device='cuda')
            part = GaussianModel(0, with_motion_mask=False)
            for field in ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest'):
                setattr(part, field, getattr(r.model, field)[rows].contiguous())
            ids = np.flatnonzero(base['vertex_groups'] == group)
            faces = base['triangles'][base['vertex_groups'][base['triangles'][:, 0]] == group]
            local = np.searchsorted(ids, faces).astype(np.int64)
            mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.ascontiguousarray(base['vertices'][ids])),
                                             o3d.core.Tensor(np.ascontiguousarray(local)))
            filled = mesh.fill_holes(hole_size=.003)
            all_faces = filled.triangle.indices.numpy()
            if (not np.array_equal(filled.vertex.positions.numpy(), base['vertices'][ids])
                    or not np.array_equal(all_faces[:len(local)], local)):
                raise ValueError('Hole filling changed original vertices or triangle order')
            candidates = ids[all_faces[len(local):]]
            centers = base['vertices'][candidates].mean(1)
            supported = np.zeros(len(candidates), bool)
            colored = 0
            for yaw in (0, 90, 180, 270):
                pose = orbit_pose(r.base_pose, r.target, validate_request({'yaw':yaw}, 100))
                result = render(make_camera(pose, n, n, np.deg2rad(fovy)), part, r.pipe, r.backgrounds['black'], 0.,0.,0.)
                alpha = result['alpha'].squeeze().cpu().numpy()
                depth = valid_depth(result['depth'].squeeze().cpu().numpy(), alpha, .65)
                x,y,z,visible = projected_samples(centers, pose, focal, n)
                selected = np.flatnonzero(visible)
                if len(selected):
                    a = bilinear(alpha,x[selected],y[selected]); d = bilinear(depth,x[selected],y[selected])
                    supported[selected] |= (a>.98) & (np.abs(d-z[selected])<.004)
                if yaw == 0:
                    x,y,z,visible = projected_samples(base['vertices'][ids],pose,focal,n)
                    selected = np.flatnonzero(visible)
                    a = bilinear(alpha,x[selected],y[selected]); d = bilinear(depth,x[selected],y[selected])
                    selected = selected[(a>.95) & (np.abs(d-z[selected])<.003)]
                    rgb = result['render'].permute(1,2,0).cpu().numpy()/np.maximum(alpha[...,None],1e-6)
                    colors[ids[selected]] = bilinear(rgb,x[selected],y[selected]).clip(0,1)
                    colored = len(selected)
            patches.append(candidates[supported])
            reports.append(dict(group=group, small_hole_candidates=len(candidates), supported_patch_triangles=int(supported.sum()), calibrated_color_vertices=colored))
            print(reports[-1], flush=True)
            del part, mesh, filled
    triangles = np.concatenate([base['triangles'], *patches]).astype(np.int32)
    for side in ('left', 'right'):
        with np.load(root/config['mesh_paths'][side]) as z: data = {k:z[k] for k in z.files}
        if not np.array_equal(data['vertices'],base['vertices']) or not np.array_equal(data['triangles'],base['triangles']):
            raise ValueError('Expected identical canonical geometry on both sides')
        meta=json.loads(str(data['metadata_json']))
        meta['local_hole_repair'] = dict(maximum_hole_radius=.003, gaussian_depth_tolerance=.004, groups=reports)
        data.update(triangles=triangles, colors=colors, metadata_json=np.asarray(json.dumps(meta)))
        mesh_path=args.output/f'{side}-mesh.npz'
        np.savez_compressed(mesh_path,**data)
        with np.load(root/config['hybrid_paths'][side]) as z: binding={k:z[k] for k in z.files}
        record=json.loads(str(binding['metadata_json']));record['mesh_sha256']=sha256_file(mesh_path)
        binding['metadata_json']=np.asarray(json.dumps(record))
        # Existing faces/vertices are exact prefixes, so attachments and all
        # Gaussian offsets remain byte-identical after the local repair.
        np.savez_compressed(args.output/f'{side}-binding.npz',**binding)
    (args.output/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'surface-repair.json').write_text(json.dumps(dict(groups=reports, triangles_before=len(base['triangles']),triangles_after=len(triangles)),indent=2)+'\n')
    print('Saved',args.output,flush=True)


if __name__ == '__main__': main()
