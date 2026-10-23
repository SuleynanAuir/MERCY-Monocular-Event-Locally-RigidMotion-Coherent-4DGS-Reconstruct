"""Render identical views before/after clay repair and measure image proxies."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np


def main():
    from PIL import Image
    from scipy import ndimage
    import torch
    from utils.viewer_mesh import MeshRenderer
    from utils.viewer_state import validate_request
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('before','after','camera','output'): parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    camera=json.loads(args.camera.read_text())
    plan=[(0,0,0),(0,35,15),(30,0,0),(0,-20,25)]
    images={}
    for name,path in [('before',args.before),('after',args.after)]:
        with np.load(path) as z: meta=json.loads(str(z['metadata_json']))
        renderer=MeshRenderer(path,meta['source_fingerprints'],SimpleNamespace(group_count=meta['group_count'],frame_count=meta['frame_count']),np.asarray(camera['pose']),np.asarray(camera['target']))
        renderer.edge_antialias=False
        for frame,yaw,pitch in plan:
            request=validate_request(dict(render_mode='clay',frame=frame,yaw=yaw,pitch=pitch,width=768,height=768,background='black',fovy=camera['fovy_degrees']),meta['frame_count'])
            rgb=renderer.render_pixels(request)
            images[(name,frame,yaw)]=rgb
            Image.fromarray(rgb).save(args.output/f'{name}-{frame}-{yaw}.png')
        del renderer
        torch.cuda.empty_cache()
    rows=[]
    for frame,yaw,pitch in plan:
        a,b=[images[(name,frame,yaw)] for name in ('before','after')]
        masks=[rgb.mean(2)>25 for rgb in (a,b)]
        common=ndimage.binary_erosion(masks[0]&masks[1],iterations=3)
        roughness=[float(np.abs(ndimage.laplace(rgb.mean(2)/255))[common].mean()) for rgb in (a,b)]
        holes=[]
        for mask in masks:
            labels,n=ndimage.label(~mask)
            counts=np.bincount(labels.ravel());border=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]));counts[border]=0;counts[0]=0
            holes.append(dict(small_count=int(((counts>=2)&(counts<=64)).sum()),small_pixels=int(counts[(counts>=2)&(counts<=64)].sum())))
        rows.append(dict(frame=frame,yaw=yaw,pitch=pitch,silhouette_iou=float(np.sum(masks[0]&masks[1])/np.sum(masks[0]|masks[1])),interior_laplacian_before=roughness[0],interior_laplacian_after=roughness[1],holes_before=holes[0],holes_after=holes[1]))
        Image.fromarray(np.concatenate([a,b],axis=1)).save(args.output/f'comparison-{frame}-{yaw}.png')
    (args.output/'metrics.json').write_text(json.dumps(dict(note='Image roughness and small enclosed background regions are display proxies, not ground-truth reconstruction accuracy. Same camera, material and lighting; probe uses embedded rigid motion.',views=rows),indent=2)+'\n')
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
