"""Blender-only nearest visible source-part masks for whole-arm evaluation."""
import argparse,json,sys,math,time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_ray_prior import tree_at


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['views','binding','parts','dataset','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.views/'views.json').read_text())
    if len(meta['views'])!=meta['expected_views'] or len(meta['views'])>128:raise ValueError('Incomplete or oversized references')
    with np.load(a.binding,allow_pickle=False) as d:registration=np.asarray(json.loads(str(d['metadata_json']))['initial_registration'])
    with np.load(a.parts,allow_pickle=False) as d:names=json.loads(str(d['metadata_json']))['objects']
    transform=np.asarray(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    size=160;yy,xx=np.mgrid[:size,:size];last=None;rows=[];started=time.monotonic()
    for k,v in enumerate(meta['views']):
        frame=round(v['time']*60)
        if frame!=last:
            bpy.context.scene.frame_set(1+frame*10);selected=[p['id'] for p in meta['source_arm_parts']];tree,local_labels=tree_at([names[i] for i in selected],transform);labels=np.asarray(selected)[local_labels];full_tree,full_labels=tree_at(names,transform);last=frame
        pose=np.asarray(v['c2w']);focal=.5*size/math.tan(math.radians(v['fov'])/2)
        directions=np.stack(((xx+.5-size/2)/focal,(yy+.5-size/2)/focal,np.ones_like(xx)),axis=-1)@pose[:3,:3].T
        directions/=np.linalg.norm(directions,axis=-1,keepdims=True);origin=Vector(pose[:3,3]);parts=np.full((size,size),-1,np.int16);visible_parts=np.full((size,size),-1,np.int16)
        for y in range(size):
            for x in range(size):
                hit,_,face,_=tree.ray_cast(origin,Vector(directions[y,x]),100.)
                if face is not None:parts[y,x]=labels[face]
                hit,_,face,_=full_tree.ray_cast(origin,Vector(directions[y,x]),100.)
                if face is not None:visible_parts[y,x]=full_labels[face]
        name=f'{k:03d}.npz';np.savez_compressed(a.output/name,parts=parts,visible_parts=visible_parts);rows.append(dict(image=v['file'],mask=name))
        print('MASK_DONE',k+1,flush=True)
        if time.monotonic()-started>1800:raise TimeoutError('Mask budget exceeded')
    (a.output/'masks.json').write_text(json.dumps(dict(scope='Nearest visible supplied source triangle at 160-square pixel centers; not captured segmentation',objects=names,arm_parts=meta['source_arm_parts'],views=rows),indent=2))


if __name__=='__main__':main()
