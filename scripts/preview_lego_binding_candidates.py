"""Render sparse motion frames to reject fractured binding initializations early."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import torch
from gaussian_renderer import render
from render_ply import make_camera
from scene import GaussianModel
from utils.articulated_motion import ArticulatedBaker


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parts', 'source-ply', 'camera', 'supervision', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--bindings', type=Path, nargs='+', required=True)
    p.add_argument('--current', type=Path)
    p.add_argument('--frames', type=int, nargs='+', default=[0,25,50,75,100,125,150,175])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    vertices = PlyData.read(args.source_ply)['vertex'].data
    model = GaussianModel(3, with_motion_mask=False); model.load_ply(str(args.source_ply))
    if torch.count_nonzero(model._features_rest).item():
        raise ValueError('Expected DC-only source')
    model._features_rest = torch.empty((len(vertices),0,3),device='cuda')
    model.active_sh_degree = model.max_sh_degree = 0
    rest_xyz, rest_q = model._xyz.clone(), model._rotation.clone()
    camera = json.loads(args.camera.read_text())['camera']
    view = make_camera(np.asarray(camera['pose']),640,640,np.deg2rad(camera['fovy_degrees']))
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    background = torch.zeros(3,device='cuda')
    luma = torch.tensor([.299,.587,.114],device='cuda')[:,None,None]
    with np.load(args.supervision,allow_pickle=False) as data:
        reference = data['images'][args.frames].astype(np.float32)/255
    columns = [('reference', reference)]
    if args.current:
        columns.append((args.current.name,np.load(args.current/'rendered_gray.npy',mmap_mode='r')[args.frames]))
    report = {'frames':args.frames,'event_approved':False,'scope':'Sparse binding diagnostic only; no event approval.','models':{}}
    for path in args.bindings:
        baker = ArticulatedBaker(vertices,args.parts,path)
        indices = torch.as_tensor(baker.indices.astype(np.int64),device='cuda')
        frames = []
        for index in args.frames:
            pose = torch.from_numpy(baker.frame(index)).cuda()
            model._xyz.copy_(rest_xyz); model._rotation.copy_(rest_q)
            model._xyz[indices],model._rotation[indices] = pose[:,:3],pose[:,3:]
            rgb = render(view,model,pipe,background,0.,0.,0.)['render'].clamp(0,1)
            frame = (rgb*luma).sum(0).cpu().numpy(); frames.append(frame)
            Image.fromarray((frame*255).astype(np.uint8)).save(args.output/f'{path.parent.name}_{index:04d}.png')
        frames = np.asarray(frames)
        columns.append((path.parent.name,frames))
        wanted, actual = reference[:,140:380,160:380]>.08, frames[:,140:380,160:380]>.08
        report['models'][path.parent.name] = {'roi_rgb_l1':float(np.abs(frames[:,140:380,160:380]-reference[:,140:380,160:380]).mean()),
            'roi_missing_foreground':float((wanted&~actual).sum()/wanted.sum()),
            'roi_extra_foreground':float((actual&~wanted).sum()/max(actual.sum(),1)),
            'dynamic_count':len(baker.indices)}
        print(json.dumps(report['models'][path.parent.name]),flush=True)
    canvas = Image.new('RGB',(220*len(columns),275*len(args.frames)),'#171c23'); draw=ImageDraw.Draw(canvas)
    for col,(name,frames) in enumerate(columns):
        for row,index in enumerate(args.frames):
            image = Image.fromarray((np.clip(frames[row,140:380,160:380],0,1)*255).astype(np.uint8))
            canvas.paste(image,(col*220,row*275+35))
            draw.text((col*220+4,row*275+4),f'{name}\nf={index} t={index/60:.3f}s',fill='white')
    canvas.save(args.output/'joint-cycle.png')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
