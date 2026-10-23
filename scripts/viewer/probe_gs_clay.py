"""Capture native GS fields and compare soft-gray display without changing assets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from gaussian_renderer import render
from render_ply import make_camera
from utils.viewer_state import validate_request, orbit_pose
from utils.gs_clay import gray_gaussians, canonical_coordinates


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['lego', 'fan'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.scene == 'lego':
        from utils.comparison_renderer import ComparisonRenderer
        renderer = ComparisonRenderer(args.input, 'cpu');renderer.load(prewarm=False)
    else:
        from viewer import Renderer
        renderer = Renderer(args.input, 'cpu');renderer.load()
    material, _ = canonical_coordinates(renderer.side_geometry[1][0] if args.scene == 'lego' else renderer.model._xyz)
    for frame, yaw, pitch in [(0, 0, 0), (30, 0, 0), (0, 35, 15), (0, -20, 25)]:
        req = validate_request(dict(frame=frame, yaw=yaw, pitch=pitch, width=768, height=768,
                                    background='black', fovy=renderer.info['fovy']), renderer.info['frame_count'])
        pose = orbit_pose(renderer.base_pose, renderer.target, req)
        if args.scene == 'lego':
            renderer.apply_side_pose(1, frame)
        else:
            renderer.render(req)
        sampling = renderer.side_sampling[1]['render_supersample'] if args.scene == 'lego' else 1
        camera = make_camera(pose, 768*sampling, 768*sampling, np.deg2rad(req['fovy']))
        output = render(camera, renderer.model, renderer.pipe, renderer.backgrounds['black'], 0., 0., 0.)
        fields = {k:torch.nn.functional.avg_pool2d(output[k], sampling) for k in ('render', 'depth', 'alpha')}
        coordinate_output = render(camera, renderer.model, renderer.pipe, renderer.backgrounds['black'], 0., 0., 0., override_color=material)
        fields['coordinates'] = torch.nn.functional.avg_pool2d(coordinate_output['render'], sampling)
        np.savez_compressed(args.output/f'fields-{frame}-{yaw}.npz', **{k:v.cpu().numpy() for k,v in fields.items()}, fovy=req['fovy'])
        gs = fields['render'].clamp(0, 1)
        clay = gray_gaussians(gs, fields['depth'], fields['alpha'], req['fovy'], [0, 0, 0], coordinates=fields['coordinates'])
        panels = [(rgb.permute(1, 2, 0)*255).byte().cpu().numpy() for rgb in (gs, clay)]
        Image.fromarray(np.concatenate(panels, axis=1)).save(args.output/f'comparison-{frame}-{yaw}.png')
        Image.fromarray(panels[1]).save(args.output/f'clay-{frame}-{yaw}.png')
        print(args.scene, frame, yaw, flush=True)


if __name__ == '__main__': main()
