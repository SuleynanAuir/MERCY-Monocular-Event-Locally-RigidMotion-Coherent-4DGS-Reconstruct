"""Three actual GS versions attached to evaluated triangle motion, with provenance.

Preserves Gaussian scale/opacity; transports local offsets and orientations. This
is source-assisted deformation attachment, not event reconstruction.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.lamp.prepare_lamp_viewer import camera_pose,register
from scripts.motion.bind_lego_parts import transform
from utils.object_surface import resample_surface
from utils.gaussian_sequence import FORMAT,SPARSE_VERSION,POSE_FIELDS,GaussianSequence
from utils.gaussian_pair_renderer import file_sha

def basis(tri):
 e=tri[:,1]-tri[:,0];e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
 n=np.cross(e,tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
 return np.stack([e,np.cross(n,e),n],axis=2)

def bind(xyz,tri):
 # Choose the closest sampled candidate face; barycentric coordinates remain
 # within the triangle. Preserve the original residual offset exactly at t0.
 tree=cKDTree(tri.mean(1));_,ids=tree.query(xyz,k=min(12,len(tri)),workers=4)
 if ids.ndim==1:ids=ids[:,None]
 best=np.full(len(xyz),np.inf);chosen=np.zeros(len(xyz),np.int32);weights=np.zeros((len(xyz),3))
 for j in range(ids.shape[1]):
  ts=tri[ids[:,j]];v=ts[:,1]-ts[:,0];w=ts[:,2]-ts[:,0];r=xyz-ts[:,0]
  aa=(v*v).sum(1);bb=(v*w).sum(1);cc=(w*w).sum(1);dd=(v*r).sum(1);ee=(w*r).sum(1);det=aa*cc-bb*bb
  u=(cc*dd-bb*ee)/np.maximum(det,1e-20);t=(aa*ee-bb*dd)/np.maximum(det,1e-20);b=np.maximum(np.column_stack([1-u-t,u,t]),0);b/=b.sum(1,keepdims=True)
  q=np.einsum('ni,nij->nj',b,ts);dist=((q-xyz)**2).sum(1);take=(dist<best)&(det>1e-20)
  best[take]=dist[take];chosen[take]=ids[take,j];weights[take]=b[take]
 if not np.isfinite(best).all():raise ValueError('Degenerate binding')
 return chosen,weights,np.sqrt(best)

def bake(vertices,tri,preview,meta,out,name):
 xyz=np.column_stack([vertices[k] for k in ['x','y','z']]);q=np.column_stack([vertices[f'rot_{i}'] for i in range(4)])
 ids,w,d=bind(xyz,tri[0]);extent=np.linalg.norm(np.ptp(xyz,axis=0))
 if np.quantile(d,.95)>extent*.04 or d.max()>extent*.12:raise ValueError('Deforming surface registration too far')
 offset=xyz-np.einsum('ni,nij->nj',w,tri[0,ids]);base=basis(tri[0]);indices=np.arange(len(vertices));poses=np.empty((len(vertices),len(tri),7),np.float32);rot=Rotation.from_quat(q[:,[1,2,3,0]])
 entries=[]
 for frame,ts in enumerate(tri):
  r=basis(ts)@base.transpose(0,2,1);poses[:,frame,:3]=np.einsum('ni,nij->nj',w,ts[ids])+np.einsum('nij,nj->ni',r[ids],offset)
  poses[:,frame,3:]=(Rotation.from_matrix(r[ids])*rot).as_quat()[:,[3,0,1,2]]
  if frame==0:poses[:,frame,:3]=xyz;poses[:,frame,3:]=q
  entries.append(dict(file=f'frames/frame_{frame:04d}.ply',time_seconds=frame/meta['fps'],pose_sha256=hashlib.sha256(poses[:,frame].tobytes()).hexdigest()))
 rest=sum(k.startswith('f_rest_') for k in vertices.dtype.names);degree=int(round(np.sqrt(rest/3+1)-1))
 m=dict(format=FORMAT,version=SPARSE_VERSION,name=name,description='原始 Blender 非刚体三角面运动绑定；保持尺度，未做事件训练。',motion_label='源动画形变绑定',frame_count=len(tri),fps=meta['fps'],duration_seconds=len(tri)/meta['fps'],gaussians_per_frame=len(vertices),sh_degree=degree,pose_fields=POSE_FIELDS,frames=entries,base_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),preview_camera=preview,default_resolution=768,default_background='dark',default_stream_rate=30,default_stream_quality=90,default_still_quality=95,motion_is_baked=True,event_validation=False,provenance=meta)
 np.savez_compressed(out,base_vertices=vertices,poses=poses,dynamic_indices=indices,times_seconds=np.arange(len(tri))/meta['fps'],metadata_json=np.asarray(json.dumps(m)))
 with GaussianSequence(out) as seq:
  for i in range(len(tri)):seq.validate_pose_frame(i)
 return dict(path=str(out.resolve()),sha256=file_sha(out),gaussians=len(vertices),binding_distance_p95=float(np.quantile(d,.95)),binding_max=float(d.max()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();b=a.root;out=b/'animated/Robot';out.mkdir(parents=True,exist_ok=False)
 with np.load(b/'aligned-motion/Robot-deforming.npz') as z:source={k:z[k] for k in z.files}
 sm=json.loads(str(source['metadata_json']));preview=camera_pose(json.loads((b/'data/Robot/fixed_camera.json').read_text()))
 with np.load(b/'packed/robot-left.npz') as z:left=z['base_vertices']
 with np.load(b/'packed/robot-right.npz') as z:middle=z['base_vertices']
 xyz=np.column_stack([left[k] for k in ['x','y','z']]);reg,_=register(source['triangles'][0].reshape(-1,3),xyz,sm['camera_pose'],preview['pose'])
 tri=transform(source['triangles'].reshape(-1,3),reg).reshape(source['triangles'].shape)
 normals=source['normals']@np.linalg.inv(reg[:3,:3]);normals/=np.linalg.norm(normals,axis=2,keepdims=True)
 optimized,_,_,report=resample_surface(tri[0],normals,source['groups'],source['materials'],middle,spacing=.004)
 if len(optimized)>2000000:raise ValueError('Surface point budget exceeded')
 meta=dict(sm,registration=reg.tolist(),source_sha256=file_sha(b/'aligned-motion/Robot-deforming.npz'),limitation='Nearest source triangle attachment, fixed Gaussian scale, no event training')
 sides={}
 for key,vertices in [('left',left),('middle',middle),('right',optimized)]:
  print('bake',key,len(vertices),flush=True);sides[key]=bake(vertices,tri,preview,meta,out/(key+'.npz'),'Robot')
 cfg=dict(format='gaussian_pair_v1',name='Robot',description='原始 / 已有修正版 / 源表面优化，真实源动画形变绑定，未新增事件训练。',left=dict(sides['left'],label='V1 · 原始高斯'),right=dict(sides['right'],label='V3 · 表面约束优化'),material_reference=sides['middle'])
 (out/'manifest.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));(out/'middle.json').write_text(json.dumps(dict(format='gaussian_stage_v1',path=sides['middle']['path'],sha256=sides['middle']['sha256'],label='V2 · 已有修正版',branch='supplied-corrected',description='已有修正版的真实形变绑定'),ensure_ascii=False,indent=2));(out/'report.json').write_text(json.dumps(dict(surface=report,binding=sides,provenance=meta),indent=2));(out/'ready.json').write_text('{}')
if __name__=='__main__':main()
