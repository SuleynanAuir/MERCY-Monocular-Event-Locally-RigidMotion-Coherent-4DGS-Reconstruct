"""Render transferred Lego motion against supplied video before full baking."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from gaussian_renderer import render
from render_ply import make_camera
from scene import GaussianModel
from utils.articulated_motion import ArticulatedBaker


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--inspection', required=True, type=Path)
    parser.add_argument('--parts', default='blender-parts.npz')
    parser.add_argument('--binding', default='part-binding.npz')
    args = parser.parse_args()
    path = args.data / 'fan_contrastive_g0/contrastive_g0.ply'
    vertices = PlyData.read(path)['vertex'].data
    baker = ArticulatedBaker(vertices, args.inspection / args.parts, args.inspection / args.binding)
    camera_meta = json.loads((args.inspection / 'inspection.json').read_text())['camera']
    camera = make_camera(np.asarray(camera_meta['pose']), 640, 640, np.deg2rad(camera_meta['fovy_degrees']))
    model = GaussianModel(3, with_motion_mask=False)
    model.load_ply(str(path))
    if torch.count_nonzero(model._features_rest).item() == 0:
        model._features_rest = torch.empty((len(vertices), 0, 3), device='cuda')
        model.active_sh_degree = model.max_sh_degree = 0
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    indices = torch.as_tensor(baker.indices.astype(np.int64), device='cuda')
    report = []
    for frame in (0, 50, 100, 200, 300, 399):
        pose = torch.from_numpy(baker.frame(frame)).to('cuda')
        model._xyz[indices] = pose[:, :3]
        model._rotation[indices] = pose[:, 3:]
        rgb = (render(camera, model, pipe, background, 0., 0., 0.)['render'].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
        Image.fromarray(rgb).save(args.inspection / f'dynamic_{frame:04d}.png')
        ref_path = args.inspection / f'video_{frame:04d}.png'
        if ref_path.exists():
            reference = np.asarray(Image.open(ref_path).convert('RGB'))
            a, b = rgb.max(-1) > 12, reference.max(-1) > 12
            report.append({'frame': frame, 'mae': float(np.abs(rgb.astype(float) - reference).mean() / 255),
                           'foreground_iou': float((a & b).sum() / (a | b).sum())})
        print('Rendered', frame, flush=True)
    result = {'comparison_to_video': report, 'rigid_rotation_max_error': baker.max_rigidity_error,
              'binding': baker.binding, 'source_animation': baker.source}
    (args.inspection / 'motion-preview.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
