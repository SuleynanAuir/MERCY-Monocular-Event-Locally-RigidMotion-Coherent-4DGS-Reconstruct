"""Transfer local canonical appearance onto V3 geometry without altering V1/V2.

Creates a small verified appearance stage; all parent poses/geometry are reused.
"""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--surface',type=Path);p.add_argument('--spacing',type=float,default=.003);p.add_argument('--material-gain',action='append',default=[],metavar='NAME=GAIN');a=p.parse_args();gains={}
 for spec in a.material_gain:
  name,value=spec.rsplit('=',1);value=float(value)
  if not name or not np.isfinite(value) or not .25<=value<=2:raise ValueError('Material gain must be finite and within [0.25, 2]')
  gains[name]=value
 if gains and not a.surface:raise ValueError('Material gains require a verified source surface')
 a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads(a.manifest.read_text())
 for k in ['left','right','material_reference','detail_finish']:
  if k in cfg:cfg[k]['path']=str((a.manifest.parent/cfg[k]['path']).resolve())
 parent=Path(cfg['right']['path']);assert file_sha(parent)==cfg['right']['sha256']
 with np.load(parent) as z:v=z['base_vertices'];parent_meta=json.loads(str(z['metadata_json']))
 with np.load(a.source) as z:s=z['base_vertices']
 xyz=lambda x:np.column_stack([x[k] for k in ['x','y','z']])
 def normal(x):
  q=np.column_stack([x[f'rot_{i}'] for i in [1,2,3,0]]);r=Rotation.from_quat(q).as_matrix();axis=np.argmin(np.column_stack([x[f'scale_{i}'] for i in range(3)]),axis=1)
  return r[np.arange(len(x)),:,axis]
 sx=xyz(s);vx=xyz(v);sn=normal(s);vn=normal(v);sc=np.column_stack([s[f'f_dc_{i}'] for i in range(3)])
 keep=s['opacity']>0;sx=sx[keep];sn=sn[keep];sc=sc[keep];tree=cKDTree(sx);color=np.empty((len(v),1,3),np.float32);distances=[]
 for start in range(0,len(v),16384):
  stop=min(start+16384,len(v));d,idx=tree.query(vx[start:stop],k=12,workers=4);dot=np.abs(np.einsum('nki,ni->nk',sn[idx],vn[start:stop]));sigma=np.maximum(d[:,3:4],.0005)
  weight=np.exp(-.5*(d/sigma)**2)*(.05+.95*dot**4);weight/=weight.sum(1,keepdims=True)
  color[start:stop,0]=np.sum(sc[idx]*weight[:,:,None],axis=1);distances.extend(d[:,0].tolist())
 material_report=[]
 if a.surface:
  with np.load(a.surface) as z:
   tri=z['triangles'];normals=z['normals'];materials=z['materials'];sm=json.loads(str(z['metadata_json']))
  if tri.ndim==4:
   reg=np.asarray(parent_meta['provenance']['registration']);tri=tri[0]@reg[:3,:3].T+reg[:3,3];normals=normals@np.linalg.inv(reg[:3,:3]);normals/=np.linalg.norm(normals,axis=2,keepdims=True)
  if set(gains)-set(sm['materials']):raise ValueError('Unknown source material name')
  area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
  positions=[];directions=[];labels=[]
  for face in np.flatnonzero(area>1e-15):
   count=max(4,int(np.ceil(area[face]/a.spacing**2)));k=np.arange(count)+.5;root=np.sqrt(k/count);uv=np.mod(k*.6180339887498949,1);w=np.column_stack((1-root,root*(1-uv),root*uv))
   positions.append(w@tri[face]);n=w@normals[face];n/=np.linalg.norm(n,axis=1,keepdims=True);directions.append(n);labels.append(np.full(count,materials[face]))
  points=np.concatenate(positions);n=np.concatenate(directions);labels=np.concatenate(labels)
  if points.shape!=vx.shape or not np.allclose(points,vx,atol=2e-6,rtol=0):raise ValueError('Surface sampling does not match parent geometry')
  rgb=color[:,0]*.28209479177387814+.5
  key=np.array([-.4,.8,.7]);key/=np.linalg.norm(key);fill=np.array([.7,.3,-.5]);fill/=np.linalg.norm(fill)
  lighting=.42+1.05*np.clip(n@key,0,1)+.18*np.clip(n@fill,0,1)
  for label in np.unique(labels):
   mask=labels==label;base=np.median(rgb[mask],axis=0)
   material_report.append(dict(name=sm['materials'][int(label)],rgb=base.tolist(),count=int(mask.sum())))
   rgb[mask]=np.clip(base[None,:]*gains.get(sm['materials'][int(label)],1.)*lighting[mask,None],0,1)
  color[:,0]=(rgb-.5)/.28209479177387814
 if not np.isfinite(color).all():raise ValueError('Invalid transferred appearance')
 meta=dict(format='scgs_appearance_stage_v1',parent_sha256=file_sha(parent),source_sha256=file_sha(a.source),method='Distance and normal-aware local DC transfer from original V2 to distinct V3 surface',geometry_unchanged=True,motion_unchanged=True,opacity_unchanged=True,scale_unchanged=True,limitation='Source appearance transfer, not newly observed detail or event training',distance_p95=float(np.quantile(distances,.95)))
 meta['material_regions']=material_report;meta['material_gains']=gains;meta['canonical_lighting']=dict(ambient=.42,key=1.05,fill=.18) if a.surface else None;meta['surface_sha256']=file_sha(a.surface) if a.surface else None;meta['limitation']+='; material-region median with declared canonical illumination removes view-baked texture and reflections when surface is supplied';stage=a.output/'appearance.npz';np.savez_compressed(stage,features_dc=color,opacity=np.asarray(v['opacity'],np.float32)[:,None],scaling=np.column_stack([v[f'scale_{i}'] for i in range(3)]).astype(np.float32),metadata_json=np.asarray(json.dumps(meta)))
 cfg.pop('detail_finish',None);cfg.pop('rgb_filter',None);cfg['appearance_refinement']=dict(path=stage.name,sha256=file_sha(stage));cfg['right']['label']='V3 · 表面与材质细化';cfg['description']='V3 保留独立表面高斯与源动画，使用原始版本的局部材质重估外观，避免继承修正版斑点；未新增事件训练。';cfg['right']['render_supersample']=2
 (a.output/'manifest.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));(a.output/'report.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
if __name__=='__main__':main()
