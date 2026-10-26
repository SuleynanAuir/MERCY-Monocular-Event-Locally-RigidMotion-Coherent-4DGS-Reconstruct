"""Compare surface denoising directly with native GS, without a high-sampling target.

Noise proxies measure image variation, not recovered geometry or ground truth.
Export the actual native/detail/surface images so visual changes remain reviewable.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter, minimum_filter, sobel, laplace, binary_erosion


def main():
    import torch
    from gaussian_renderer import render
    from scripts.rendering.render_ply import make_camera
    from utils.comparison_renderer import ComparisonRenderer
    from utils.mesh_gaussian_binding import MeshGaussianBinding
    from utils.viewer_mesh import MeshRenderer
    from utils.viewer_state import validate_request, orbit_pose
    from utils.hybrid_sampling import resolve_hybrid_rgb, smooth_surface
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--size', type=int, default=640)
    args = parser.parse_args()
    if args.output.exists() or not 256 <= args.size <= 768:
        parser.error('Use a new output directory and size 256..768')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    renderer = ComparisonRenderer(args.comparison, 'cpu'); renderer.load(prewarm=False)
    settings = json.loads(args.comparison.read_text())
    frames = [100] if args.probe else list(range(36, 228, 4))
    rows = []
    for side, name in enumerate(('left', 'right')):
        path = Path(settings['mesh_paths'][name])
        with np.load(path) as z: meta = json.loads(str(z['metadata_json']))
        mesh = MeshRenderer(path, meta['source_fingerprints'], renderer.correction, renderer.base_pose, renderer.target, name)
        with torch.no_grad():
            renderer.apply_side_pose(side, 0)
            binding = MeshGaussianBinding(args.bindings/f'{name}.npz', path, mesh, renderer.model._xyz, renderer.model._rotation)
        for view, (yaw, pitch) in enumerate(((0, 0), (35, 15), (165, 45))):
            clip, previous = [], {}
            for frame in frames:
                request = validate_request(dict(frame=frame, yaw=yaw, pitch=pitch, width=args.size, height=args.size,
                    background='black', fovy=renderer.info['fovy']), renderer.info['frame_count'])
                pose = orbit_pose(renderer.base_pose, renderer.target, request)
                images = {}; times = {}
                with torch.no_grad():
                    for mode, factor in (('gaussian', 1), ('native', 1), ('detail', 3)):
                        renderer.apply_side_pose(side, frame)
                        if mode != 'gaussian': binding.apply(renderer.model, frame)
                        camera = make_camera(pose, args.size*factor, args.size*factor, np.deg2rad(request['fovy']))
                        torch.cuda.synchronize(); start = time.perf_counter()
                        rendered = render(camera, renderer.model, renderer.pipe, renderer.backgrounds['black'], 0., 0., 0.)
                        rgb = resolve_hybrid_rgb(rendered['render'], args.size, args.size, mode if mode != 'gaussian' else 'native')
                        images[mode] = (rgb.permute(1,2,0)*255).byte().cpu().numpy()
                        times[mode] = (time.perf_counter()-start)*1000
                        if mode == 'native':
                            alpha = rendered['alpha']; depth = rendered['depth']
                            smooth = smooth_surface(rgb, depth, alpha)
                            images['surface'] = (smooth.permute(1,2,0)*255).byte().cpu().numpy()
                            times['surface'] = (time.perf_counter()-start)*1000
                            a = alpha[0].cpu().numpy(); z = (depth/alpha.clamp_min(1e-6))[0].cpu().numpy()
                            if frame == 100:
                                np.savez_compressed(args.output/f'{name}-view-{view}-guides.npz',
                                    rgb=rgb.cpu().numpy(), depth=depth.cpu().numpy(), alpha=alpha.cpu().numpy())
                original = images['gaussian'].astype(float)/255
                gray = original.mean(2)
                color_range = (maximum_filter(original, size=(3,3,1))-minimum_filter(original,size=(3,3,1))).max(2)
                zrange = (maximum_filter(z,3)-minimum_filter(z,3))/np.maximum(z,1e-6)
                inside = binary_erosion(a > .995, iterations=2)
                flat = inside & (color_range < .15) & (zrange < .005)
                protected = (minimum_filter(a,3) <= .98) | (color_range >= .25) | (zrange >= .02)
                foreground = gray > .05
                edge = np.hypot(sobel(gray,0),sobel(gray,1))
                strong = inside & (edge > .5)
                for mode in ('native', 'detail', 'surface'):
                    actual = images[mode].astype(float)/255
                    g = actual.mean(2); residual = actual-original
                    row = dict(side=name, view=view, frame=frame, mode=mode,
                        native_foreground_mae=float(np.abs(residual[foreground]).mean()),
                        protected_mae=float(np.abs(residual[protected & foreground]).mean()),
                        flat_high_frequency=float(np.abs(laplace(g))[flat].mean()),
                        strong_edge_ratio=float(np.hypot(sobel(g,0),sobel(g,1))[strong].sum()/edge[strong].sum()),
                        changed_fraction=float((np.abs(residual).max(2)>1/255)[foreground].mean()),
                        native_flat_high_frequency=float(np.abs(laplace(gray))[flat].mean()),
                        render_and_filter_ms=times[mode])
                    if mode in previous:
                        row['temporal_change_residual']=float(np.abs(residual-previous[mode])[foreground].mean())
                    previous[mode] = residual; rows.append(row)
                panel = Image.new('RGB', (args.size*3,args.size+32)); draw = ImageDraw.Draw(panel)
                for col,(mode,label) in enumerate((('gaussian','Standalone GS'),('detail','Previous Mesh + GS / 3x'),('surface','New Mesh + GS / surface'))):
                    panel.paste(Image.fromarray(images[mode]),(col*args.size,32)); draw.text((col*args.size+8,8),label,fill='white')
                clip.append(panel)
                if frame == 100:
                    panel.save(args.output/f'{name}-view-{view}-comparison.jpg',quality=95,subsampling=0)
                    for mode,pixels in images.items(): Image.fromarray(pixels).save(args.output/f'{name}-view-{view}-{mode}.png')
            clip[0].save(args.output/f'{name}-view-{view}.gif',save_all=True,append_images=clip[1:],
                duration=([70,60,70]*16)[:len(frames)],loop=0,optimize=False)
            print('Completed',name,view,flush=True)
        del mesh,binding; gc.collect(); torch.cuda.empty_cache()
    keys = ('native_foreground_mae','protected_mae','flat_high_frequency','strong_edge_ratio','changed_fraction',
            'native_flat_high_frequency','render_and_filter_ms','temporal_change_residual')
    summary=[]
    for side in ('left','right'):
        for mode in ('native','detail','surface'):
            selected=[r for r in rows if r['side']==side and r['mode']==mode]
            summary.append(dict(side=side,mode=mode,**{k:float(np.mean([r[k] for r in selected if k in r])) for k in keys if any(k in r for r in selected)}))
    report=dict(reference='Actual native GS at the same output dimensions and camera; variation proxies are not geometry ground truth',
        cameras=3,frames_per_camera=len(frames),size=args.size,summary=summary,results=rows)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
