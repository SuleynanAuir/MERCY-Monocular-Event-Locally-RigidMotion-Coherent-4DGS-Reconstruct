"""Patch small enclosed image gaps with bounded part-local 3D surface patches."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from utils.viewer_state import orbit_pose,validate_request


def local_patch(ring_points, rays, max_error):
    """Intersect rays with a nearby fitted plane; reject folds or steep views."""
    # Robust consensus rejects isolated back-surface hits at a thin rim, but
    # requires a clear majority so we do not bridge two separate surfaces.
    rng=np.random.default_rng(0)
    best=np.ones(len(ring_points),bool);best_count=0
    for _ in range(48):
        a,b,c=ring_points[rng.choice(len(ring_points),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-12:continue
        inliers=np.abs((ring_points-a)@(normal/length))<max_error
        if inliers.sum()>best_count:best,best_count=inliers,int(inliers.sum())
    if best_count < max(6,int(np.ceil(.7*len(ring_points)))):return None
    ring_points=ring_points[best]
    center=ring_points.mean(0)
    _,_,vt=np.linalg.svd(ring_points-center,full_matrices=False)
    normal=vt[-1]
    error=np.abs((ring_points-center)@normal)
    denom=rays@normal
    if np.quantile(error,.9)>max_error or np.min(np.abs(denom))<.25:return None
    depth=(center@normal)/denom
    if np.any(depth<=0):return None
    return rays*depth[:,None]


def main():
    import open3d as o3d
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','source','camera','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-area',type=int,default=96);p.add_argument('--max-distance',type=float,default=.008)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    if not 1<=args.max_area<=256 or not 0<args.max_distance<.1:p.error('Invalid patch bounds')
    with np.load(args.input) as z:data={k:z[k] for k in z.files}
    if 'clay_normals' in data:p.error('Patch holes before feature reconstruction; otherwise canonical normals become stale')
    with np.load(args.source) as z:source={k:z[k] for k in ('vertices','vertex_groups')}
    from utils.rigid_checkpoint import sha256_file
    meta=json.loads(str(data['metadata_json']))
    if meta['clay_regularization']['source_mesh_sha256']!=sha256_file(args.source):raise ValueError('Source mismatch')
    camera=json.loads(args.camera.read_text());n=1024;focal=n/(2*np.tan(np.deg2rad(camera['fovy_degrees'])/2))
    yy,xx=np.mgrid[:n,:n];camera_rays=np.stack([(xx+.5-n/2)/focal,-(yy+.5-n/2)/focal,-np.ones_like(xx)],axis=-1)
    trees={int(g):cKDTree(source['vertices'][source['vertex_groups']==g]) for g in np.unique(source['vertex_groups'])}
    reports=[]
    # Patches exist in object space and move with their own group at every frame.
    for yaw,pitch in [(0,0),(35,15),(-35,15),(90,0),(180,0),(270,0)]:
        request=validate_request(dict(yaw=yaw,pitch=pitch),meta['frame_count'])
        pose=orbit_pose(np.asarray(camera['pose']),np.asarray(camera['target']),request)
        directions=camera_rays@pose[:3,:3].T
        rays=np.concatenate([np.broadcast_to(pose[:3,3],directions.shape),directions],axis=-1).astype(np.float32)
        scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(data['vertices'],np.float32)),o3d.core.Tensor(np.ascontiguousarray(data['triangles'],np.uint32)))
        cast=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)
        depth=cast['t_hit'].numpy();primitive=cast['primitive_ids'].numpy();visible=np.isfinite(depth)
        labels,count=ndimage.label(~visible);sizes=np.bincount(labels.ravel())
        sizes[np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))]=0;sizes[0]=0
        candidate=np.flatnonzero((sizes>0)&(sizes<=args.max_area));slices=ndimage.find_objects(labels)
        face_groups=data['vertex_groups'][data['triangles'][:,0]]
        new_v,new_f,new_g=[],[],[];offset=len(data['vertices']);accepted=0;pixels=0
        for ident in candidate:
            box=slices[ident-1]
            y0,y1=max(0,box[0].start-3),min(n,box[0].stop+3);x0,x1=max(0,box[1].start-3),min(n,box[1].stop+3)
            hole=labels[y0:y1,x0:x1]==ident
            ring=ndimage.binary_dilation(hole,iterations=2)&visible[y0:y1,x0:x1]
            ry,rx=np.where(ring);ry+=y0;rx+=x0
            if len(ry)<8:continue
            gs=face_groups[primitive[ry,rx]]
            if not np.all(gs==gs[0]):continue
            group=int(gs[0])
            points=camera_rays[ry,rx]*depth[ry,rx,None]
            # Include a one-pixel overlap to hide cracks at the patch boundary.
            region=ndimage.binary_dilation(hole,iterations=1)
            hy,hx=np.where(region);hy+=y0;hx+=x0
            corners=np.concatenate([np.stack([hy,hx],1),np.stack([hy,hx+1],1),np.stack([hy+1,hx+1],1),np.stack([hy+1,hx],1)])
            corners,inverse=np.unique(corners,axis=0,return_inverse=True)
            cr=np.column_stack([(corners[:,1]-n/2)/focal,-(corners[:,0]-n/2)/focal,-np.ones(len(corners))])
            plane=local_patch(points,cr,args.max_distance*.35)
            if plane is None:continue
            world=plane@pose[:3,:3].T+pose[:3,3]
            distance,_=trees[group].query(world,workers=1)
            if distance.max()>args.max_distance:continue
            quad=inverse.reshape(4,-1).T
            tri=np.concatenate([quad[:,[0,1,2]],quad[:,[0,2,3]]])
            new_v.append(world.astype(np.float32));new_f.append(tri.astype(np.int32)+offset);new_g.append(np.full(len(world),group,np.int32));offset+=len(world)
            accepted+=1;pixels+=int(hole.sum())
        if new_v:
            data['vertices']=np.concatenate([data['vertices'],*new_v]);data['triangles']=np.concatenate([data['triangles'],*new_f]);data['vertex_groups']=np.concatenate([data['vertex_groups'],*new_g])
        row=dict(yaw=yaw,pitch=pitch,candidates=len(candidate),patched_regions=accepted,patched_pixels=pixels,added_triangles=sum(len(f) for f in new_f));reports.append(row);print(json.dumps(row),flush=True)
    meta['clay_regularization']['local_patches']=dict(resolution=n,maximum_area_pixels=args.max_area,maximum_source_distance=args.max_distance,views=reports)
    meta.update(vertices=len(data['vertices']),triangles=len(data['triangles']))
    data['colors']=np.full_like(data['vertices'],.65);data['metadata_json']=np.asarray(json.dumps(meta))
    args.output.mkdir(parents=True);np.savez_compressed(args.output/'mesh.npz',**data)
    (args.output/'report.json').write_text(json.dumps(meta['clay_regularization'],indent=2)+'\n')

if __name__=='__main__':main()
