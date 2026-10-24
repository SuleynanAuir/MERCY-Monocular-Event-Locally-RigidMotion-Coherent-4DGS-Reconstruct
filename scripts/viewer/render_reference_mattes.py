"""Render display-only foreground coverage from existing Blender mesh/animation exports.

Uses source part motion, including elastic vertices, never V3 corrections or GS.
The RGB camera and sampling must match the original reference video.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.viewer_mesh import projection_matrix


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('mesh', 'parts', 'rgb', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--fovy', type=float, required=True)
    p.add_argument('--supersample', type=int, default=4, choices=(2, 4))
    p.add_argument('--limit', type=int, help='Diagnostic prefix only; full export omits this option')
    args = p.parse_args()
    import torch
    import nvdiffrast.torch as dr
    from PIL import Image
    from scripts.viewer.prepare_reference_videos import probe

    video = probe(args.rgb)
    if not 0 < args.fovy < 180:
        raise ValueError('Invalid source camera field of view')
    with np.load(args.mesh, allow_pickle=False) as mesh, np.load(args.parts, allow_pickle=False) as parts:
        meta = json.loads(str(mesh['metadata_json']))
        motion_meta = json.loads(str(parts['metadata_json']))
        if (meta['format'] != 'lego_complete_mesh_prior_v1'
                or meta['parts_sha256'] != sha256(args.parts)
                or meta['objects'] != motion_meta['objects']
                or video['frame_count'] != motion_meta['frame_count']
                or video['fps'] != motion_meta['playback_fps']):
            raise ValueError('Source mesh, motion and RGB timeline do not match')
        matrices = parts['object_matrices']
        inverse = np.linalg.inv(np.asarray(meta['registration']))
        rest = mesh['vertices'] @ inverse[:3, :3].T + inverse[:3, 3]
        local, labels, slices, offset = [], [], {}, 0
        for i, row in enumerate(meta['parts_detail']):
            if row['part_id'] != i or row['name'] != meta['objects'][i]:
                raise ValueError('Unexpected source vertex ordering')
            section = slice(offset, offset + row['vertices']); slices[i] = section
            transform = np.linalg.inv(matrices[0, i])
            local.append(rest[section] @ transform[:3, :3].T + transform[:3, 3])
            labels.append(np.full(row['vertices'], i, np.int64)); offset += row['vertices']
        if offset != len(rest):
            raise ValueError('Incomplete source vertex ownership')
        vertices = torch.tensor(np.concatenate(local), dtype=torch.float32, device='cuda')
        labels = torch.tensor(np.concatenate(labels), device='cuda')
        faces = torch.tensor(np.ascontiguousarray(mesh['triangles']), dtype=torch.int32, device='cuda')
        elastic = {}
        for i in meta['deforming_objects']:
            values = parts[f'deformed_{i}']
            if values.shape != (len(matrices), slices[i].stop - slices[i].start, 3):
                raise ValueError('Elastic topology does not match the source mesh')
            elastic[i] = values
        camera = torch.tensor(motion_meta['camera_pose'], dtype=torch.float32, device='cuda')
        projection = torch.tensor(projection_matrix(video['width'], video['height'], args.fovy), device='cuda')
        context = dr.RasterizeCudaContext()
        count = min(args.limit or video['frame_count'], video['frame_count'])
        args.output.mkdir(parents=True, exist_ok=False)
        with torch.no_grad():
            for frame in range(count):
                transforms = torch.tensor(matrices[frame], device='cuda')
                xyz = torch.bmm(transforms[labels, :3, :3], vertices[..., None])[..., 0] + transforms[labels, :3, 3]
                for i, values in elastic.items():
                    xyz[slices[i]] = torch.tensor(values[frame], device='cuda')
                view = (xyz - camera[:3, 3]) @ camera[:3, :3]
                clip = torch.cat((view, torch.ones_like(view[:, :1])), -1) @ projection.T
                h, w, scale = video['height'], video['width'], args.supersample
                raster, _ = dr.rasterize(context, clip[None].contiguous(), faces, [h * scale, w * scale], grad_db=False)
                coverage = (raster[0, ..., 3] > 0).float().reshape(h, scale, w, scale).mean((1, 3)).flip(0)
                Image.fromarray((coverage.cpu().numpy() * 255).round().astype(np.uint8)).save(args.output / f'{frame:04d}.png')
                if frame % 25 == 0:
                    print(f'Foreground matte {frame + 1}/{count}', flush=True)
        manifest = dict(format='rgb_geometry_matte_v1', **video, rendered_frames=count,
                        rgb_sha256=sha256(args.rgb), mesh_sha256=sha256(args.mesh),
                        parts_sha256=meta['parts_sha256'], source_frames=parts['source_frames'][:count].tolist(),
                        fovy=args.fovy, supersample=args.supersample, camera_pose=motion_meta['camera_pose'],
                        display_only=True, motion='Uncorrected source Blender animation including elastic parts')
        (args.output / 'matte.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
