"""Blender-only bounded surface samples from the supplied animated source.

These are explicit known-mesh priors, not newly reconstructed observations.
Only timestamps in the training split are exported; source files stay read-only.
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binding','parts','dataset','views','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.output.exists():raise FileExistsError(a.output)
    views=json.loads((a.views/'views.json').read_text())
    times=sorted(set(v['time'] for v in views['views'] if v['split']=='train'))
    if not 1<=len(times)<=64:raise ValueError('Surface prior time budget exceeded')
    with np.load(a.binding,allow_pickle=False) as data:
        registration=np.array(json.loads(str(data['metadata_json']))['initial_registration'])
    with np.load(a.parts,allow_pickle=False) as data:
        names=json.loads(str(data['metadata_json']))['objects']
    transform=np.array(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    rng=np.random.default_rng(6666);samples=[];normals=[]
    for timestamp in times:
        bpy.context.scene.frame_set(1+round(timestamp*60)*10)
        graph=bpy.context.evaluated_depsgraph_get();triangles=[];probabilities=[]
        for name in names:
            obj=bpy.data.objects[name].evaluated_get(graph);mesh=obj.to_mesh()
            try:
                mesh.calc_loop_triangles()
                xyz=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',xyz)
                matrix=transform@np.asarray(obj.matrix_world)
                xyz=xyz.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3]
                faces=np.empty(len(mesh.loop_triangles)*3,np.int32);mesh.loop_triangles.foreach_get('vertices',faces)
                tri=xyz[faces.reshape(-1,3)]
                area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
                keep=area>1e-12;tri=tri[keep];area=area[keep]
                if len(area):
                    triangles.append(tri)
                    # Balance surface area and small parts so the base plate
                    # does not consume nearly every geometric target.
                    probabilities.append(area/np.sqrt(area.sum()))
            finally:obj.to_mesh_clear()
        tri=np.concatenate(triangles);prob=np.concatenate(probabilities);prob/=prob.sum()
        selected=tri[rng.choice(len(tri),8192,p=prob)]
        uv=rng.random((8192,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
        points=selected[:,0]+uv[:,:1]*(selected[:,1]-selected[:,0])+uv[:,1:]*(selected[:,2]-selected[:,0])
        normal=np.cross(selected[:,1]-selected[:,0],selected[:,2]-selected[:,0])
        normal/=np.linalg.norm(normal,axis=1,keepdims=True).clip(1e-12)
        if not np.isfinite(points).all():raise ValueError('Non-finite surface prior')
        samples.append(points.astype(np.float32));normals.append(normal.astype(np.float32));print('SURFACE_DONE',len(samples),len(times),flush=True)
    metadata=dict(source=bpy.data.filepath,scope='explicit supplied animated mesh prior, not captured multiview geometry',
                  views=str(a.views),training_timestamps_only=True,points_per_time=8192,
                  coordinate_system='same CV coordinates and initial registration as source-scene image supervision')
    np.savez_compressed(a.output,times=np.array(times,np.float32),points=np.stack(samples),normals=np.stack(normals),metadata_json=np.asarray(json.dumps(metadata)))


if __name__=='__main__':main()
