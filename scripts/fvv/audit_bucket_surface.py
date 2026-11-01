"""Bounded canonical all-part/all-normal support audit, without GPU rendering."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from rigid_bucket import read_arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','model','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.source.stat().st_size>64*1024**2:raise ValueError('Source exceeds budget')
    with np.load(a.source,allow_pickle=False) as d:points=d['points'];normals=d['normals'];parts=d['parts'];meta=json.loads(str(d['metadata_json']))
    if len(points)>1048576:raise ValueError('Source point budget exceeded')
    schema=json.loads((a.model/'manifest.json').read_text())['rigid_bucket']['schema']
    v=read_arrays(a.model/'rigid-bucket.npz',schema);selected=v.get('point_groups',np.zeros(len(v['points'])))==0
    gaussians=v['points'][selected];q=v['rotations'][selected][:,[1,2,3,0]]
    gaussian_normals=Rotation.from_quat(q).as_matrix()[:,:,2];_,indices=cKDTree(points).query(gaussians,workers=4);assigned=parts[indices]
    rows=[]
    for part in meta['bucket_parts']:
        mask=parts==part['id'];source=points[mask];normal=normals[mask];chosen=assigned==part['id']
        if not len(source) or not chosen.any():raise ValueError('Bucket component lacks surface or Gaussian support: '+part['name'])
        distance,near=cKDTree(gaussians[chosen]).query(source,k=min(8,int(chosen.sum())),workers=4)
        if distance.ndim==1:distance=distance[:,None];near=near[:,None]
        aligned=np.abs((gaussian_normals[chosen][near]*normal[:,None]).sum(-1))>.8
        distances=np.where(aligned,distance,np.inf).min(1)
        axis=np.argmax(np.abs(normal),axis=1);bins=axis*2+(normal[np.arange(len(normal)),axis]>=0)
        groups={}
        for i,name in enumerate(['-x','+x','-y','+y','-z','+z']):
            selected=distances[bins==i]
            if len(selected):groups[name]=dict(samples=len(selected),within_2sigma_fraction=float((selected<=.0056).mean()))
        rows.append(dict(**part,source_samples=len(source),gaussians=int(chosen.sum()),within_2sigma_fraction=float((distances<=.0056).mean()),normal_directions=groups))
    result=dict(scope='Sampled supplied source surface support, same inferred part and normal alignment; not continuous surface completeness or image quality proof',parts=rows,points=len(gaussians),worst_support=min(r['within_2sigma_fraction'] for r in rows))
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
