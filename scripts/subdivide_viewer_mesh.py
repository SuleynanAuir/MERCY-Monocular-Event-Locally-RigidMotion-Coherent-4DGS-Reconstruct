"""Refine large triangles of each independent Gaussian Mesh motion group."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.mesh_subdivision import refine_surface
from utils.rigid_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--edge-voxels', type=float, default=4 / 3)
    parser.add_argument('--offset-voxels', type=float, default=.15)
    parser.add_argument('--max-growth', type=float, default=2.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    with np.load(args.input, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data['metadata_json']))
    from utils.viewer_mesh import load_mesh_data
    load_mesh_data(args.input, meta['group_count'], meta['source_fingerprints'])
    voxel = meta.get('voxel_size', meta.get('voxel'))
    if voxel is None:
        raise ValueError('Source mesh must record voxel size')
    xyz, rgb, groups = [data[k] for k in ('vertices', 'colors', 'vertex_groups')]
    face_groups = groups[data['triangles'][:, 0]]
    vertices, colors, labels, faces, rows = [xyz], [rgb], [groups], [], []
    count = len(xyz)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        source_faces = data['triangles'][face_groups == group]
        if not len(source_faces):
            continue
        local_faces = np.searchsorted(indices, source_faces)
        v, f, c, _, _, report = refine_surface(xyz[indices], local_faces, rgb[indices],
                                              voxel * args.edge_voxels, voxel * args.offset_voxels,
                                              args.max_growth)
        added = len(v) - len(indices)
        mapping = np.concatenate([indices, np.arange(count, count + added)])
        vertices.append(v[len(indices):]); colors.append(c[len(indices):])
        labels.append(np.full(added, group, dtype=groups.dtype)); faces.append(mapping[f].astype(np.int32))
        count += added
        report['group'] = int(group); rows.append(report)
        print(json.dumps(report), flush=True)
    data.update(vertices=np.concatenate(vertices), colors=np.concatenate(colors),
                vertex_groups=np.concatenate(labels), triangles=np.concatenate(faces))
    if meta.get('surface_subdivision'):
        meta.setdefault('surface_subdivision_history', []).append(meta['surface_subdivision'])
    meta['surface_subdivision'] = dict(source_mesh_sha256=sha256_file(args.input), groups=rows,
                                       motion_preserved=True, original_vertices_preserved=True,
                                       original_colors_preserved=True)
    meta.update(vertices=len(data['vertices']), triangles=len(data['triangles']))
    meta['groups'] = [dict(group=r['group'], vertices=r['vertices'], triangles=r['triangles']) for r in rows]
    data['metadata_json'] = np.asarray(json.dumps(meta))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'mesh.npz', **data)
    (args.output / 'report.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(dict(vertices=meta['vertices'], triangles=meta['triangles'])), flush=True)


if __name__ == '__main__':
    main()
