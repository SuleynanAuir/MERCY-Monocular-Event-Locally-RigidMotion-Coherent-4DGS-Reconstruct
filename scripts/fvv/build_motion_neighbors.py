"""Build bounded same-part, coplanar temporal neighbors for complete articulation."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from rigid_bucket import read_arrays
from motion_coherence import surface_pairs
from common import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['model','arm-source','bucket-source','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    meta=json.loads((a.model/'manifest.json').read_text());v=read_arrays(a.model/'rigid-bucket.npz',2)
    result={};report={}
    for name,path,mask in [('bucket',a.bucket_source,v['point_groups']==0),('arm',a.arm_source,v['point_groups']>0)]:
        with np.load(path,allow_pickle=False) as d:source=d['points'];parts=d['parts']
        points=v['points'][mask];distance,near=cKDTree(source).query(points,workers=4)
        if distance.max()>1e-5:raise ValueError('Source surface no longer matches '+name)
        normals=Rotation.from_quat(v['rotations'][mask][:,[1,2,3,0]]).as_matrix()[:,:,2]
        labels=parts[near];pairs=surface_pairs(points,normals,labels)
        if not 1000<len(pairs)<300000:raise ValueError('Unexpected neighbor count')
        result[name]=pairs
        report[name]=dict(points=len(points),pairs=len(pairs),parts=len(np.unique(labels)),covered_points=len(np.unique(pairs)),source_sha256=digest(path))
    report['parent_sidecar_sha256']=meta['rigid_bucket']['sha256'];report['scope']='Same physical source part and aligned tangent plane; never cross parts or average static texture.'
    result['metadata_json']=np.asarray(json.dumps(report));np.savez_compressed(a.output,**result)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
