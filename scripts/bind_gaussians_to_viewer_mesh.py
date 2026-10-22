"""Bind Gaussian appearance to its own reconstructed Mesh, preserving offsets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def main():
    import torch
    import open3d as o3d
    from utils.comparison_renderer import ComparisonRenderer
    from utils.mesh_motion import baseline_groups
    from utils.mesh_gaussian_binding import attachment_centers, pose_fingerprint, validate_attachment
    from utils.rigid_checkpoint import sha256_file
    from utils.event_refinement import qmul, qrotate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--left-binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    torch.set_num_threads(4)
    settings=json.loads(args.comparison.read_text())
    renderer=ComparisonRenderer(args.comparison, 'cpu'); renderer.load(prewarm=False)
    source=json.loads((Path(settings['baseline'])/'cache.json').read_text())
    record=json.loads((Path(settings['candidate'])/'correction.json').read_text())
    args.output.mkdir(parents=True)
    for side, name in enumerate(('left','right')):
        with torch.no_grad(): renderer.apply_side_pose(side,0)
        xyz=renderer.model._xyz.detach().cpu().numpy().copy()
        rotation=renderer.model._rotation.detach().cpu().numpy().copy()
        mesh_path=Path(settings['mesh_paths'][name])
        with np.load(mesh_path,allow_pickle=False) as z: mesh={k:z[k] for k in z.files}
        metadata=json.loads(str(mesh['metadata_json']))
        if metadata['source_side']!=name: raise ValueError('Wrong Mesh side')
        if side==0:
            with np.load(args.left_binding,allow_pickle=False) as z:
                if not np.array_equal(z['dynamic_indices'],renderer.indices[0].cpu().numpy()):
                    raise ValueError('Original binding does not match source motion rows')
                groups=baseline_groups(z['labels'],source['source_animation']['objects'],record['joint_structure']['source_groups'])
            if len(groups)!=len(xyz): raise ValueError('Binding length mismatch')
        else:
            root=[i for i,p in enumerate(renderer.correction.parents) if p==-1][0]
            groups=np.full(len(xyz),root,np.int32)
            groups[renderer.indices[1].cpu().numpy()]=renderer.correction.row_groups.cpu().numpy()
            if renderer.completion is not None:
                c=renderer.completion
                groups[c.prefix_count:c.prefix_count+len(c.groups)]=c.groups.cpu().numpy()
        assigned=np.full(len(xyz),-1,np.int32)
        weights=np.zeros((len(xyz),3),np.float32)
        offsets=np.zeros((len(xyz),3),np.float32)
        fgroups=mesh['vertex_groups'][mesh['triangles'][:,0]]
        reports=[]
        for group in np.unique(groups[groups>=0]):
            rows=np.flatnonzero(groups==group); ids=np.flatnonzero(fgroups==group)
            if not len(ids): continue
            vertices=np.flatnonzero(mesh['vertex_groups']==group)
            faces=np.searchsorted(vertices,mesh['triangles'][ids]).astype(np.uint32)
            scene=o3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(mesh['vertices'][vertices])),
                                o3d.core.Tensor(np.ascontiguousarray(faces)))
            distances=[]
            for start in range(0,len(rows),100000):
                selected=rows[start:start+100000]
                nearest=scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(xyz[selected])),nthreads=4)
                uv=nearest['primitive_uvs'].numpy()
                bary=np.column_stack([1-uv.sum(1),uv]).clip(0,1);bary/=bary.sum(1,keepdims=True)
                face_ids=ids[nearest['primitive_ids'].numpy()]
                center=attachment_centers(mesh['vertices'],mesh['triangles'],face_ids,bary,np.zeros_like(bary))
                assigned[selected]=face_ids; weights[selected]=bary; offsets[selected]=xyz[selected]-center
                distances.append(np.linalg.norm(offsets[selected],axis=1))
            report=dict(group=int(group),gaussians=len(rows),offset_p95=float(np.percentile(np.concatenate(distances),95)))
            reports.append(report);print(name,json.dumps(report),flush=True)
            del scene
        rows=np.flatnonzero(assigned>=0)
        meta=dict(format='mesh_gaussian_binding_v1', source_side=name,mesh_sha256=sha256_file(mesh_path),
                  source_fingerprints=metadata['source_fingerprints'],gaussian_count=len(xyz),
                  canonical_pose_sha256=pose_fingerprint(xyz,rotation),groups=reports,
                  appearance_preserved=True,offsets_preserved=True,unbound_gaussians=len(xyz)-len(rows))
        data=dict(rows=rows.astype(np.int32),face_indices=assigned[rows],barycentric=weights[rows],offsets=offsets[rows],
                  groups=groups[rows],metadata_json=np.asarray(json.dumps(meta)))
        centers,_=validate_attachment(data,mesh['vertices'],mesh['triangles'],mesh['vertex_groups'],xyz,rotation,metadata,meta['mesh_sha256'])
        # Cover every time sample, including all moving rows. Static rows are
        # sampled because their transforms are identity throughout the clip.
        moving=renderer.indices[side].cpu().numpy()
        if side==1 and renderer.completion is not None:
            c=renderer.completion;moving=np.union1d(moving,np.arange(c.prefix_count,c.prefix_count+len(c.groups)))
        check=np.union1d(np.intersect1d(rows,moving),rows[::max(1,len(rows)//4096)])
        local=np.searchsorted(rows,check)
        rest_xyz=torch.from_numpy(centers[local]).cuda();rest_q=torch.from_numpy(rotation[check]).cuda()
        q=torch.from_numpy(mesh['motion_quaternions']).cuda();t=torch.from_numpy(mesh['motion_translations']).cuda()
        ids=torch.from_numpy(groups[check].astype(np.int64)).cuda()
        worst_xyz=worst_q=0.
        with torch.no_grad():
            for frame in range(metadata['frame_count']):
                renderer.apply_side_pose(side,frame)
                expected=renderer.model._xyz[check];actual=qrotate(q[frame,ids],rest_xyz)+t[frame,ids]
                error=float((expected-actual).norm(dim=-1).max());worst_xyz=max(worst_xyz,error)
                a=torch.nn.functional.normalize(qmul(q[frame,ids],rest_q),dim=-1)
                b=torch.nn.functional.normalize(renderer.model._rotation[check],dim=-1)
                error=float(torch.minimum((a-b).norm(dim=-1),(a+b).norm(dim=-1)).max());worst_q=max(worst_q,error)
        if max(worst_xyz,worst_q)>1e-4: raise ValueError(f'Hybrid motion differs from original: {worst_xyz}, {worst_q}')
        meta['motion_check']=dict(frames=metadata['frame_count'],checked_rows=len(check),max_position_error=worst_xyz,max_rotation_error=worst_q)
        data['metadata_json']=np.asarray(json.dumps(meta))
        np.savez_compressed(args.output/f'{name}.npz',**data)
        (args.output/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('Saved',name,json.dumps(meta['motion_check']),flush=True)
    print('Binding completed',flush=True)


if __name__=='__main__':main()
