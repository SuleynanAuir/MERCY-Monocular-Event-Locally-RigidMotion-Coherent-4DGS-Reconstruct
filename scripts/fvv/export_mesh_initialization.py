"""Blender-only bounded full-surface initialization from supplied Lego geometry.

This is explicit source-mesh initialization, not reconstruction from RGB alone.
No source file is modified. Appearance still requires real rendering/training.
"""
import argparse,json,sys,time
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_surface_seeds import mesh_triangles,detect_deforming_parts


def sample(names,labels,transform,count,rng):
    triangles=[];probabilities=[];parts=[];ids=[];topology={}
    for part in labels:
        tri,faces=mesh_triangles(names[part],transform);topology[part]=faces
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        keep=area>1e-12
        if not keep.any():continue
        triangles.append(tri[keep]);probabilities.append(area[keep]/np.sqrt(area[keep].sum()))
        parts.extend([part]*int(keep.sum()));ids.extend(np.flatnonzero(keep))
    triangles=np.concatenate(triangles);prob=np.concatenate(probabilities);prob/=prob.sum()
    chosen=rng.choice(len(triangles),count,p=prob);tri=triangles[chosen]
    uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    points=barycentric(tri,uv);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    return points,normal,np.asarray(parts,np.int16)[chosen],np.asarray(ids)[chosen],uv,topology


def barycentric(tri,uv):
    return tri[:,0]+uv[:,:1]*(tri[:,1]-tri[:,0])+uv[:,1:]*(tri[:,2]-tri[:,0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binding','parts','dataset','rays','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--points-per-time',type=int,default=4096,choices=[4096,16384])
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.binding,allow_pickle=False) as data:registration=np.asarray(json.loads(str(data['metadata_json']))['initial_registration'])
    with np.load(a.parts,allow_pickle=False) as data:names=json.loads(str(data['metadata_json']))['objects']
    moving=set(json.loads((a.rays/'views.json').read_text())['moving_parts'])
    transform=np.asarray(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    # Object matrices miss deformation in local mesh coordinates (the two
    # animated rubber bands in this source). Check evaluated triangles too.
    moving=detect_deforming_parts(names,transform,moving);static=sorted(set(range(len(names)))-set(moving))
    rng=np.random.default_rng(110126);start=time.monotonic();bpy.context.scene.frame_set(1)
    static_points,static_normals,static_parts,*_=sample(names,static,transform,98304,rng)
    arrays={key:[] for key in ['points','normals','velocities','times','parts']}
    indices=np.linspace(0,398,64).round().astype(int)
    for step,index in enumerate(indices):
        bpy.context.scene.frame_set(1+int(index)*10)
        points,normals,parts,face_ids,uv,topology=sample(names,moving,transform,a.points_per_time,rng)
        neighbors=[max(0,int(index)-1),min(398,int(index)+1)];positions=[]
        for frame in neighbors:
            bpy.context.scene.frame_set(1+frame*10);position=np.empty_like(points)
            for part in np.unique(parts):
                rows=np.flatnonzero(parts==part);tri,faces=mesh_triangles(names[part],transform)
                if not np.array_equal(faces,topology[part]):raise ValueError('Changing source topology')
                position[rows]=barycentric(tri[face_ids[rows]],uv[rows])
            positions.append(position)
        velocity=(positions[1]-positions[0])/((neighbors[1]-neighbors[0])/60)
        for key,value in dict(points=points,normals=normals,parts=parts,velocities=velocity,times=np.full((a.points_per_time,1),index/60)).items():arrays[key].append(value)
        print(json.dumps(dict(done=step+1,total=64,elapsed=time.monotonic()-start)),flush=True)
        if time.monotonic()-start>3600:raise TimeoutError('Mesh initialization export budget')
    arrays={key:np.concatenate(value).astype(np.int16 if key=='parts' else np.float32) for key,value in arrays.items()}
    arrays.update(static_points=static_points.astype(np.float32),static_normals=static_normals.astype(np.float32),static_parts=static_parts)
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite source geometry')
    arrays['metadata_json']=np.asarray(json.dumps(dict(source=bpy.data.filepath,scope='Explicit source-mesh production initialization; geometry timestamps are not held out',objects=names,frames=indices.tolist(),dynamic_points=64*a.points_per_time,static_points=98304,moving_parts=moving,motion_detection='Source matrix tracks plus evaluated world triangles at frames 0,127,273,398')))
    temporary=a.output.with_suffix('.partial.npz')
    try:
        np.savez_compressed(temporary,**arrays)
        if temporary.stat().st_size>64*1024**2:raise OSError('Source initialization exceeds 64 MiB')
        temporary.replace(a.output)
    finally:temporary.unlink(missing_ok=True)


if __name__=='__main__':main()
