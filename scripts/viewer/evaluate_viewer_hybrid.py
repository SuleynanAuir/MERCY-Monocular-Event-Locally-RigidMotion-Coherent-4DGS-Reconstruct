"""Evaluate source GS, pure Mesh and hybrid playback against 4x GS sampling.

The 4x reference measures sampling/appearance fidelity, not physical ground truth.
Temporal residuals compare successive image differences; they are not optical flow.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, binary_erosion


def main():
    import torch
    import torch.nn.functional as F
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera
    from utils.comparison_renderer import ComparisonRenderer
    from utils.mesh_gaussian_binding import MeshGaussianBinding
    from utils.viewer_mesh import MeshRenderer
    from utils.viewer_state import validate_request, orbit_pose
    from utils.hybrid_sampling import sampling_plan, resolve_hybrid_rgb
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--detail',action='store_true',help='Compare previous hybrid with new detail reconstruction')
    parser.add_argument('--probe',action='store_true',help='Render frame 100 only before the full dynamic evaluation')
    args=parser.parse_args()
    if args.output.exists() or not 256<=args.size<=768:parser.error('Use a new output and size 256..768')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    renderer=ComparisonRenderer(args.comparison,'cpu');renderer.load(prewarm=False)
    settings=json.loads(args.comparison.read_text());rows=[]
    modes=('hybrid','detail','cubic_experiment') if args.detail else ('gaussian','mesh','hybrid')
    frames=[100] if args.probe else list(range(36,228,4))
    panels=(('hybrid','Previous hybrid / 2x area'),('detail','Detail / 3x area'),('reference','4x sampling reference')) if args.detail else (('gaussian','Original Gaussian'),('mesh','Pure Mesh'),('hybrid','Mesh + Gaussian / 2x AA'))
    for side,name in enumerate(('left','right')):
        mesh_path=Path(settings['mesh_paths'][name])
        with np.load(mesh_path) as z:meta=json.loads(str(z['metadata_json']))
        mesh=MeshRenderer(mesh_path,meta['source_fingerprints'],renderer.correction,renderer.base_pose,renderer.target, name)
        with torch.no_grad():
            renderer.apply_side_pose(side,0)
            binding=MeshGaussianBinding(args.bindings/f'{name}.npz',mesh_path,mesh,renderer.model._xyz,renderer.model._rotation)
        for view,(yaw,pitch) in enumerate(((15,15),(165,45),(285,-10))):
            previous={};clip=[]
            for frame in frames:
                request=validate_request(dict(frame=frame,yaw=yaw,pitch=pitch,width=args.size,height=args.size,
                    render_mode='hybrid',background='black',fovy=renderer.info['fovy']),renderer.info['frame_count'])
                pose=orbit_pose(renderer.base_pose,renderer.target,request)
                images={};times={}
                with torch.no_grad():
                    variants=(('hybrid',2),('detail',3),('reference',4)) if args.detail else (('gaussian',1),('hybrid',2),('reference',4))
                    for mode,factor in variants:
                        renderer.apply_side_pose(side,frame)
                        if mode in ('hybrid','detail'):binding.apply(renderer.model,frame)
                        width,height=sampling_plan(args.size,args.size,'detail') if mode=='detail' else (args.size*factor,args.size*factor)
                        camera=make_camera(pose,width,height,np.deg2rad(request['fovy']))
                        torch.cuda.synchronize();start=time.perf_counter()
                        rgb=render(camera,renderer.model,renderer.pipe,renderer.backgrounds['black'],0.,0.,0.)['render'].clamp(0,1)
                        if mode=='detail':
                            area=resolve_hybrid_rgb(rgb,args.size,args.size,'detail')
                            images['detail']=(area.permute(1,2,0)*255).byte().cpu().numpy()
                            times['detail']=(time.perf_counter()-start)*1000
                            cubic=F.interpolate(rgb.unsqueeze(0),size=(args.size,args.size),mode='bicubic',align_corners=False,antialias=True)
                            upper=F.max_pool2d(area.unsqueeze(0),3,1,1);lower=-F.max_pool2d(-area.unsqueeze(0),3,1,1)
                            cubic=cubic.maximum(lower).minimum(upper)[0]
                            images['cubic_experiment']=(cubic.permute(1,2,0)*255).byte().cpu().numpy()
                            times['cubic_experiment']=(time.perf_counter()-start)*1000
                            rgb=area
                        elif factor>1:rgb=F.avg_pool2d(rgb,factor,stride=factor)
                        images[mode]=(rgb.permute(1,2,0)*255).byte().cpu().numpy()
                        if mode not in times:times[mode]=(time.perf_counter()-start)*1000
                    if not args.detail:
                        start=time.perf_counter();images['mesh']=mesh.render_pixels(request);times['mesh']=(time.perf_counter()-start)*1000
                ref=images['reference'].astype(float)/255
                mask=ref.max(2)>.05;refgray=ref.mean(2)
                edges=np.hypot(sobel(refgray,axis=0),sobel(refgray,axis=1))
                for mode in modes:
                    actual=images[mode].astype(float)/255
                    own=actual.max(2)>.05;area=mask|own;inside=binary_erosion(mask&own,iterations=2)
                    gray=actual.mean(2);gradient=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
                    residual=actual-ref
                    row=dict(side=name,view=view,frame=frame,mode=mode,foreground_mae=float(np.abs(residual[area]).mean()),
                        silhouette_iou=float((mask&own).sum()/area.sum()),edge_correlation=float(np.corrcoef(edges[inside],gradient[inside])[0,1]),
                        render_ms=times[mode])
                    if mode in previous:
                        old_residual,old_mask=previous[mode]
                        row['temporal_residual_mae']=float(np.abs(residual-old_residual)[area|old_mask].mean())
                    previous[mode]=(residual,area);rows.append(row)
                panel=Image.new('RGB',(args.size*3,args.size+32))
                draw=ImageDraw.Draw(panel)
                for col,(mode,label) in enumerate(panels):
                    panel.paste(Image.fromarray(images[mode]),(col*args.size,32));draw.text((col*args.size+8,8),label,fill='white')
                clip.append(panel)
                if frame==100:
                    panel.save(args.output/f'{name}-view-{view}-comparison.jpg',quality=95,subsampling=0)
                    for mode in images:Image.fromarray(images[mode]).save(args.output/f'{name}-view-{view}-{mode}.png')
            # GIF delays have 10 ms precision; alternate to preserve 15 FPS
            # rather than truncating every 66.7 ms source interval to 60 ms.
            clip[0].save(args.output/f'{name}-view-{view}.gif',save_all=True,append_images=clip[1:],
                         duration=([70,60,70]*16)[:len(frames)],loop=0,optimize=False)
            print('Completed',name,view,flush=True)
        del mesh,binding;gc.collect();torch.cuda.empty_cache()
    summary=[]
    for name in ('left','right'):
        for mode in modes:
            selected=[r for r in rows if r['side']==name and r['mode']==mode]
            summary.append(dict(side=name,mode=mode,**{key:float(np.mean([r[key] for r in selected if key in r])) for key in
                ('foreground_mae','silhouette_iou','edge_correlation','temporal_residual_mae','render_ms') if any(key in r for r in selected)}))
    report=dict(reference='4x sampling of the same original Gaussians; sampling reference, not independent ground truth',
        cameras=3,frames_per_camera=len(frames),detail_comparison=args.detail,output_size=args.size,summary=summary,results=rows)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
