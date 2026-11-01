"""Blender-only per-part inverse-depth rays from explicit supplied Lego geometry.

No RGB generation, source edits, optimizer files or full-resolution frame cache.
Default training timestamps exclude supplied validation splits and holdouts.
Production all-time fitting is explicit and recorded in the manifest.
"""
import argparse,json,sys,time,math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_surface_seeds import detect_deforming_parts


def tree_at(names, transform):
    graph=bpy.context.evaluated_depsgraph_get();vertices=[];faces=[];labels=[];offset=0
    for label,name in enumerate(names):
        obj=bpy.data.objects[name].evaluated_get(graph);mesh=obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            xyz=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',xyz)
            matrix=transform@np.asarray(obj.matrix_world)
            xyz=xyz.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3]
            triangles=np.empty(len(mesh.loop_triangles)*3,np.int32);mesh.loop_triangles.foreach_get('vertices',triangles)
            vertices.extend(xyz.tolist());faces.extend((triangles.reshape(-1,3)+offset).tolist())
            labels.extend([label]*len(mesh.loop_triangles));offset+=len(xyz)
        finally:obj.to_mesh_clear()
    return BVHTree.FromPolygons(vertices,faces,all_triangles=True),np.asarray(labels,np.int16)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binding','parts','dataset','views','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exclude-views',type=Path,action='append',default=[])
    p.add_argument('--times',type=int,default=64)
    p.add_argument('--production-times',action='store_true',help='Fit every source timestamp; explicitly gives up held-out geometry-time claims')
    p.add_argument('--angles-per-time',type=int,default=8,choices=[2,8])
    p.add_argument('--resolution',type=int,default=160,choices=[160,320])
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not 2<=a.times<=(399 if a.production_times else 64):raise ValueError('Time budget exceeded')
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.binding,allow_pickle=False) as d:registration=np.array(json.loads(str(d['metadata_json']))['initial_registration'])
    with np.load(a.parts,allow_pickle=False) as d:
        names=json.loads(str(d['metadata_json']))['objects'];matrices=d['object_matrices']
        moving=np.max(np.abs(matrices-matrices[:1]),axis=(0,2,3))>1e-5
    transform=np.array(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    moving[detect_deforming_parts(names,transform,np.flatnonzero(moving))]=True
    views=json.loads((a.views/'views.json').read_text())['views']
    excluded={127,273}
    for folder in [a.views,*a.exclude_views]:
        for v in json.loads((folder/'views.json').read_text())['views']:
            if v['split']=='validation':excluded.add(round(v['time']*60))
    allowed=list(range(399)) if a.production_times else sorted(set(range(399))-excluded)
    stamps=[allowed[i] for i in np.linspace(0,len(allowed)-1,a.times).round().astype(int)]
    templates=[min((v for v in views if v.get('angle',0)==angle),key=lambda v:abs(v.get('elevation',0))) for angle in sorted(set(v.get('angle',0) for v in views))]
    if len(templates)!=8:raise ValueError('Expected eight calibrated angular templates')
    records=[dict(v,time=i/60,split='train') for k,i in enumerate(stamps) for v in
             (templates if a.angles_per_time==8 else [templates[k%4],templates[k%4+4]])]
    records += [dict(v) for v in views if v['split']=='validation']
    if len(records)>1536:raise ValueError('Ray view budget exceeded')
    records.sort(key=lambda v:v['time']);last=None;start=time.monotonic();size=a.resolution
    yy,xx=np.mgrid[:size,:size]
    for index,v in enumerate(records):
        frame=round(v['time']*60)
        if frame!=last:
            bpy.context.scene.frame_set(1+frame*10);tree,labels=tree_at(names,transform);last=frame
        c2w=np.array(v['c2w']);focal=.5*size/math.tan(math.radians(v['fov'])/2)
        rays=np.stack(((xx+.5-size/2)/focal,(yy+.5-size/2)/focal,np.ones_like(xx)),axis=-1)@c2w[:3,:3].T
        rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
        origin=Vector(c2w[:3,3]);depth=np.zeros((size,size),np.float32);part=np.full((size,size),-1,np.int16)
        for y in range(size):
            for x in range(size):
                hit,_,face,_=tree.ray_cast(origin,Vector(rays[y,x]),100.)
                if face is not None:
                    z=np.dot(np.asarray(hit)-c2w[:3,3],c2w[:3,2])
                    if z>.01:depth[y,x]=1/z;part[y,x]=labels[face]
        name=f'{index:04d}.npz';np.savez_compressed(a.output/name,inverse_depth=depth,parts=part,moving=np.where(part>=0,moving[np.maximum(part,0)],False))
        v['file']=name
        if index%8==0:print(json.dumps(dict(done=index+1,total=len(records),elapsed=time.monotonic()-start)),flush=True)
        if time.monotonic()-start>7200 or sum(f.stat().st_size for f in a.output.iterdir())>256*1024**2:raise RuntimeError('Ray export budget exceeded')
    (a.output/'views.json').write_text(json.dumps(dict(schema=1,expected_views=len(records),views=records,objects=names,moving_parts=np.where(moving)[0].tolist(),excluded_frames=[] if a.production_times else sorted(excluded),production_geometry_fit=a.production_times,resolution=size,scope='Explicit supplied source mesh ray prior; not captured depth; production all-time fit' if a.production_times else 'Explicit supplied source mesh ray prior; not captured depth; withheld validation times',depth_convention='alpha-weighted inverse camera-z; background zero',elapsed=time.monotonic()-start),indent=2))


if __name__=='__main__':main()
