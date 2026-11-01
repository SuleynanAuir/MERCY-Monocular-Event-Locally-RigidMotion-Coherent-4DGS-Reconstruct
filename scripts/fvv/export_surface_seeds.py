"""Blender-only bounded surface/motion anchors for optional missing-support seeds.

All anchors come from the supplied animated mesh, at ray-prior training times.
Finite differences follow the same triangle/barycentric point, not random
nearest neighbors at adjacent frames. Source files are never saved or modified.
"""
import argparse,json,sys,time
from pathlib import Path
import bpy
import numpy as np


def mesh_triangles(name,transform):
    obj=bpy.data.objects[name].evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
    try:
        mesh.calc_loop_triangles();xyz=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',xyz)
        matrix=transform@np.asarray(obj.matrix_world);xyz=xyz.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3]
        faces=np.empty(len(mesh.loop_triangles)*3,np.int32);mesh.loop_triangles.foreach_get('vertices',faces)
        faces=faces.reshape(-1,3)
        return xyz[faces],faces
    finally:obj.to_mesh_clear()


def detect_deforming_parts(names,transform,moving):
    """Include local vertex deformation that object matrix tracks cannot see."""
    result=set(moving);bpy.context.scene.frame_set(1)
    baseline={i:mesh_triangles(name,transform)[0] for i,name in enumerate(names) if i not in result}
    for index in [127,273,398]:
        bpy.context.scene.frame_set(1+index*10)
        for part,initial in baseline.items():
            current,_=mesh_triangles(names[part],transform)
            if current.shape!=initial.shape:raise ValueError('Changing evaluated topology')
            if np.max(np.linalg.norm(current-initial,axis=-1))>1e-5:result.add(part)
    return sorted(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binding','parts','dataset','rays','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--points-per-time',type=int,default=1024,choices=[512,1024])
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.binding,allow_pickle=False) as d:registration=np.array(json.loads(str(d['metadata_json']))['initial_registration'])
    with np.load(a.parts,allow_pickle=False) as d:names=json.loads(str(d['metadata_json']))['objects']
    manifest=json.loads((a.rays/'views.json').read_text());moving=manifest['moving_parts']
    times=sorted({v['time'] for v in manifest['views'] if v['split']=='train'})
    if not 1<=len(times)<=64:raise ValueError('Seed time budget exceeded')
    transform=np.array(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    moving=detect_deforming_parts(names,transform,moving)
    rng=np.random.default_rng(3126);out={key:[] for key in ['points','normals','velocities','times','parts']};start=time.monotonic()
    for timestamp in times:
        index=round(timestamp*60);bpy.context.scene.frame_set(1+index*10)
        triangles=[];probabilities=[];labels=[];face_ids=[];faces_by_part={}
        for part in moving:
            tri,faces=mesh_triangles(names[part],transform);faces_by_part[part]=faces
            area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            keep=area>1e-12
            if not keep.any():continue
            triangles.append(tri[keep]);probabilities.append(area[keep]/np.sqrt(area[keep].sum()))
            labels.extend([part]*int(keep.sum()));face_ids.extend(np.where(keep)[0].tolist())
        tri=np.concatenate(triangles);prob=np.concatenate(probabilities);prob/=prob.sum()
        selected=rng.choice(len(tri),a.points_per_time,p=prob);tri=tri[selected]
        labels=np.array(labels,np.int16)[selected];face_ids=np.array(face_ids)[selected]
        uv=rng.random((len(selected),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
        def barycentric(triangle,u):return triangle[:,0]+u[:,:1]*(triangle[:,1]-triangle[:,0])+u[:,1:]*(triangle[:,2]-triangle[:,0])
        points=barycentric(tri,uv);normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        endpoints=[];indices=[max(0,index-1),min(398,index+1)]
        for frame in indices:
            bpy.context.scene.frame_set(1+frame*10);position=np.empty_like(points)
            for part in np.unique(labels):
                rows=np.where(labels==part)[0];triangle,faces=mesh_triangles(names[part],transform)
                if not np.array_equal(faces,faces_by_part[part]):raise ValueError('Changing topology cannot be seeded as a moving surface')
                position[rows]=barycentric(triangle[face_ids[rows]],uv[rows])
            endpoints.append(position)
        velocity=(endpoints[1]-endpoints[0])/((indices[1]-indices[0])/60)
        for key,value in dict(points=points,normals=normals,velocities=velocity,times=np.full((len(points),1),timestamp),parts=labels).items():out[key].append(value)
        print(json.dumps(dict(done=len(out['times']),total=len(times),elapsed=time.monotonic()-start)),flush=True)
        if time.monotonic()-start>3600:raise TimeoutError('Seed export budget exceeded')
    arrays={key:np.concatenate(value).astype(np.int16 if key=='parts' else np.float32) for key,value in out.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite source anchors')
    arrays['metadata_json']=np.asarray(json.dumps(dict(source=bpy.data.filepath,scope='Explicit animated source mesh anchors; missing-support initialization only, requires fitting and independent validation',points_per_time=a.points_per_time,training_times=times,objects=names)))
    np.savez_compressed(a.output,**arrays)


if __name__=='__main__':main()
