"""Matched-camera multiview/motion render checks; image proxies are not accuracy metrics."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def main():
    import torch
    from PIL import Image
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from scipy import ndimage
    from scripts.fan.evaluate_fan_surface import model_from_vertices
    from scripts.lamp.prepare_lamp_viewer import camera_pose, rigid_deltas
    from scripts.motion.bind_lego_parts import transform
    from utils.event_refinement import qmul, qrotate
    from utils.viewer_state import validate_request, orbit_pose
    from utils.gs_clay import gray_gaussians
    from utils.gs_pseudocolor import palette
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parts', 'binding-report', 'candidate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--frames', default='0,25,49')
    p.add_argument('--resolution', type=int, default=1024)
    p.add_argument('--full-orbit', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binding = json.loads(args.binding_report.read_text())
    with np.load(args.parts, allow_pickle=False) as z:
        registration = np.array(binding['registration'])
        surface = transform(z['rest_points'], registration)
        groups = z['point_parts']; delta = rigid_deltas(z['object_matrices'], registration)
    camera = camera_pose(json.loads((args.data/'fixed_camera.json').read_text()))
    v1, before, after = [PlyData.read(f)['vertex'].data.copy() for f in
        (args.data/'lamp.ply', args.data/'lamb_corrected_g0.ply', args.candidate/'surface.ply')]
    with np.load(args.candidate/'binding.npz', allow_pickle=False) as z:
        after_groups = z['groups']
    tree = cKDTree(surface)
    points = [np.column_stack([v[k] for k in ('x', 'y', 'z')]) for v in (v1, before, after)]
    part_labels = [groups[tree.query(x, workers=4)[1]] for x in points[:2]]+[after_groups]
    chart = [torch.from_numpy(x).cuda() @ torch.tensor([.65,.20,.15],device='cuda') for x in points[:2]]
    low, high = min(x.min() for x in chart), max(x.max() for x in chart)
    frames = list(map(int, args.frames.split(',')))
    views = [(0,0),(35,20),(80,0),(180,0),(0,35),(0,-35)]
    if args.full_orbit:
        views += [(yaw,pitch) for yaw in (45,135,225,270,315) for pitch in (-25,25)]
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black, white = torch.zeros(3,device='cuda'), torch.ones(3,device='cuda')
    measurements = []
    with torch.no_grad():
        for name, base, labels, factor in [('v1',v1,part_labels[0],1), ('before',before,part_labels[1],1), ('after',after,part_labels[2],2)]:
            model = model_from_vertices(base); rest = model._xyz.clone(), model._rotation.clone()
            materials = palette((rest[0]@rest[0].new_tensor([.65,.20,.15])-low)/(high-low).clamp_min(1e-6))
            for frame in frames:
                q = Rotation.from_matrix(delta[frame,:,:3,:3]).as_quat()[:,[3,0,1,2]]
                q = torch.from_numpy(q[labels].astype(np.float32)).cuda()
                t = torch.from_numpy(delta[frame,labels,:3,3].astype(np.float32)).cuda()
                model._xyz=qrotate(q,rest[0])+t;model._rotation=qmul(q,rest[1])
                for view,(yaw,pitch) in enumerate(views):
                    req=validate_request(dict(yaw=yaw,pitch=pitch),len(delta))
                    cam=make_camera(orbit_pose(camera['pose'],camera['target'],req),args.resolution*factor,args.resolution*factor,np.deg2rad(camera['fovy_degrees']))
                    out=render(cam,model,pipe,black,0.,0.,0.)
                    material=render(cam,model,pipe,black,0.,0.,0.,override_color=materials)['render']
                    if factor>1:
                        out={k:torch.nn.functional.avg_pool2d(out[k],factor) for k in ('render','depth','alpha')}
                        material=torch.nn.functional.avg_pool2d(material,factor)
                    from utils.gs_pseudocolor import color_gaussians
                    images=dict(gs=out['render']+1-out['alpha'],
                        clay=gray_gaussians(out['render'],out['depth'],out['alpha'],camera['fovy_degrees'],white),
                        pseudo=color_gaussians(out['render'],out['depth'],out['alpha'],material,camera['fovy_degrees'],white))
                    alpha=out['alpha'].squeeze().cpu().numpy();stem=f'f{frame:03d}-v{view:02d}'
                    np.save(args.output/f'{name}-{stem}-alpha.npy',alpha)
                    for mode,rgb in images.items():
                        pixels=(rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                        Image.fromarray(pixels).save(args.output/f'{name}-{mode}-{stem}.png')
                        if name=='after':
                            old=np.asarray(Image.open(args.output/f'before-{mode}-{stem}.png'))
                            original=np.asarray(Image.open(args.output/f'v1-{mode}-{stem}.png'))
                            old_alpha=np.load(args.output/f'before-{stem}-alpha.npy')
                            original_alpha=np.load(args.output/f'v1-{stem}-alpha.npy')
                            common=ndimage.binary_erosion((alpha>.98)&(old_alpha>.98),iterations=3)
                            gray=[im.mean(2)/255 for im in (old,pixels)]
                            roughness=[float(np.abs(ndimage.laplace(im))[common].mean()) for im in gray]
                            masks=[old_alpha>.5,alpha>.5]
                            fragments=[]
                            for mask in masks:
                                connected,n=ndimage.label(mask);counts=np.bincount(connected.ravel());counts[0]=0
                                fragments.append(int(counts[(counts>0)&(counts<25)].sum()))
                            union=masks[0]|masks[1]
                            measurements.append(dict(frame=frame,view=view,yaw=yaw,pitch=pitch,mode=mode,
                                roughness_before=roughness[0],roughness_after=roughness[1],
                                detached_pixels_before=fragments[0],detached_pixels_after=fragments[1],
                                silhouette_iou=float((masks[0]&masks[1]).sum()/max(1,union.sum())),
                                source_silhouette_iou=float(((original_alpha>.5)&masks[1]).sum()/max(1,((original_alpha>.5)|masks[1]).sum()))))
                            triplet=np.concatenate([original,old,pixels],axis=1)
                            Image.fromarray(triplet).save(args.output/f'compare-{mode}-{stem}.jpg',quality=95)
                    print(name,stem,flush=True)
            del model,rest,materials,out,images,material,q,t
            torch.cuda.empty_cache()
    summary={}
    for mode in ('gs','clay','pseudo'):
        rows=[r for r in measurements if r['mode']==mode]
        summary[mode]=dict(roughness_ratio=sum(r['roughness_after'] for r in rows)/sum(r['roughness_before'] for r in rows),
            minimum_source_silhouette_iou=min(r['source_silhouette_iou'] for r in rows),
            detached_pixels_before=sum(r['detached_pixels_before'] for r in rows),
            detached_pixels_after=sum(r['detached_pixels_after'] for r in rows))
    (args.output/'metrics.json').write_text(json.dumps(dict(note=__doc__,resolution=args.resolution,summary=summary,measurements=measurements),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
