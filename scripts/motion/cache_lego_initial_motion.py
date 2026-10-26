"""Cache a Blender-initialized motion and render every frame BEFORE event fitting."""
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
from scripts.rendering.render_ply import make_camera
from scene import GaussianModel
from utils.articulated_motion import ArticulatedBaker


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source-ply', type=Path, help='Explicit separate repaired Gaussian source; original input stays unchanged')
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--parts', type=Path, help='Optional existing surface/motion archive, without copying it')
    parser.add_argument('--camera', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source_ply or args.data / 'fan_contrastive_g0/contrastive_g0.ply'
    vertices = PlyData.read(source)['vertex'].data
    baker = ArticulatedBaker(vertices, args.parts or args.initialization / 'blender-surface-parts.npz',
                             args.initialization / 'initial-binding.npz')
    camera_meta = json.loads(args.camera.read_text())['camera']
    model = GaussianModel(3, with_motion_mask=False)
    model.load_ply(str(source))
    if torch.count_nonzero(model._features_rest).item() != 0:
        raise ValueError('Nonzero higher-order SH is unsupported by this motion transfer')
    model._features_rest = torch.empty((len(vertices), 0, 3), device='cuda')
    model.active_sh_degree = model.max_sh_degree = 0
    camera = make_camera(np.asarray(camera_meta['pose']), 640, 640, np.deg2rad(camera_meta['fovy_degrees']))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    indices = torch.as_tensor(baker.indices.astype(np.int64), device='cuda')
    args.output.mkdir(parents=True)
    frames = len(baker.delta)
    poses = np.lib.format.open_memmap(args.output / 'poses.npy', mode='w+', dtype=np.float32,
                                     shape=(frames, len(baker.indices), 7))
    rendered = np.lib.format.open_memmap(args.output / 'rendered_gray.npy', mode='w+', dtype=np.float16,
                                        shape=(frames, 640, 640))
    np.save(args.output / 'dynamic_indices.npy', baker.indices, allow_pickle=False)
    luma = torch.tensor([.299, .587, .114], device='cuda')[:, None, None]
    for i in range(frames):
        pose = baker.frame(i)
        poses[i] = pose
        gpu_pose = torch.from_numpy(pose).to('cuda')
        model._xyz[indices], model._rotation[indices] = gpu_pose[:, :3], gpu_pose[:, 3:]
        rgb = render(camera, model, pipe, background, 0., 0., 0.)['render'].clamp(0, 1)
        rendered[i] = (rgb * luma).sum(0).cpu().numpy()
        if i in (0, 50, 100, 200, 300, 399):
            Image.fromarray((rgb.permute(1, 2, 0) * 255).byte().cpu().numpy()).save(args.output / f'frame_{i:04d}.png')
        if i % 50 == 0:
            print('Cached baseline', i, '/', frames, flush=True)
    poses.flush(); rendered.flush()
    metadata = {'source': str(source), 'frame_count': frames, 'fps': 60,
                'gaussian_count': len(vertices), 'dynamic_count': len(baker.indices),
                'camera': camera_meta, 'binding': baker.binding, 'source_animation': baker.source,
                'event_supervised': False, 'status': 'INITIALIZATION_ONLY_NOT_APPROVED'}
    (args.output / 'cache.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Baseline cache complete', args.output, flush=True)


if __name__ == '__main__':
    main()
