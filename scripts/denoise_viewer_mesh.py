"""Reduce Gaussian Mesh surface noise without filling holes or mixing motion groups."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.mesh_surface import denoise_surface
from utils.rigid_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=('balanced', 'smooth'), default='balanced',
                        help='smooth removes more small surface ripples while retaining the same displacement limits')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--color-iterations', type=int)
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = 64 if args.preset == 'smooth' else 16
    if args.color_iterations is None:
        args.color_iterations = 10 if args.preset == 'smooth' else 3
    if args.output.exists() or args.iterations < 0 or args.color_iterations < 0:
        parser.error('Use a new output directory and nonnegative iteration counts')
    with np.load(args.input, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads(str(data['metadata_json']))
    from utils.viewer_mesh import load_mesh_data
    load_mesh_data(args.input, meta['group_count'], meta['source_fingerprints'])
    voxel = meta.get('voxel_size', meta.get('voxel'))
    if voxel is None:
        raise ValueError('Source mesh must record its reconstruction voxel size')
    rows = []
    face_groups = data['vertex_groups'][data['triangles'][:, 0]]
    for group in np.unique(data['vertex_groups']):
        indices = np.flatnonzero(data['vertex_groups'] == group)
        faces = data['triangles'][face_groups == group]
        if not len(faces):
            continue
        local_faces = np.searchsorted(indices, faces)
        xyz, rgb, report = denoise_surface(data['vertices'][indices], local_faces,
                                           data['colors'][indices], voxel,
                                           args.iterations, args.color_iterations)
        data['vertices'][indices], data['colors'][indices] = xyz, rgb
        report['group'] = int(group)
        rows.append(report)
        print(json.dumps(report), flush=True)
    if meta.get('surface_denoising'):
        meta.setdefault('surface_denoising_history', []).append(meta['surface_denoising'])
    meta['surface_denoising'] = dict(source_mesh_sha256=sha256_file(args.input), groups=rows, preset=args.preset,
                                     topology_preserved=True, motion_preserved=True)
    # The source mesh may have advertised exact preservation before this pass.
    meta.pop('original_vertices_preserved', None)
    meta.pop('original_colors_preserved', None)
    data['metadata_json'] = np.asarray(json.dumps(meta))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'mesh.npz', **data)
    (args.output / 'report.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
