"""Preserve detailed Gaussian TSDF triangles and add only nearby supported gap patches."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def main():
    import open3d as o3d
    from scipy.spatial import cKDTree
    from utils.rigid_checkpoint import sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gap-distance', type=float, default=.00075)
    parser.add_argument('--boundary-distance', type=float, default=.0045)
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.gap_distance < args.boundary_distance < .1:
        parser.error('Use a new directory and valid local patch distances')
    with np.load(args.base, allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    with np.load(args.surface, allow_pickle=False) as z:
        surface = {k: z[k] for k in z.files}
    old_meta, meta = [json.loads(str(z['metadata_json'])) for z in (base, surface)]
    if (old_meta['source_fingerprints'] != meta['source_fingerprints']
            or old_meta.get('source_side', 'right') != meta['source_side']
            or not meta.get('empty_space_carving') or meta.get('format') != 'gaussian_tsdf_mesh_v2'):
        raise ValueError('Patch surface must share the source and have passed empty-space carving')
    additions = []
    for group in np.unique(base['vertex_groups']):
        old_faces = base['triangles'][base['vertex_groups'][base['triangles'][:, 0]] == group]
        new_faces = surface['triangles'][surface['vertex_groups'][surface['triangles'][:, 0]] == group]
        if not len(old_faces) or not len(new_faces):
            continue
        edges = np.sort(np.concatenate([old_faces[:, [0, 1]], old_faces[:, [1, 2]], old_faces[:, [2, 0]]]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = np.unique(edges[counts == 1])
        if not len(boundary):
            continue
        centers = surface['vertices'][new_faces].mean(axis=1)
        near, _ = cKDTree(base['vertices'][boundary]).query(centers, workers=4)
        candidates = new_faces[near <= args.boundary_distance]
        centers = centers[near <= args.boundary_distance]
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(base['vertices'].astype(np.float32)),
                            o3d.core.Tensor(old_faces.astype(np.uint32)))
        distance = scene.compute_distance(o3d.core.Tensor(centers.astype(np.float32)), nthreads=4).numpy()
        patches = candidates[distance >= args.gap_distance]
        additions.append(patches)
        print(json.dumps(dict(group=int(group), original_faces=len(old_faces), patch_faces=len(patches))), flush=True)
    if not additions or not sum(len(p) for p in additions):
        raise RuntimeError('No supported local gap patches were found')
    faces = np.concatenate(additions)
    used = np.unique(faces)
    remap = np.full(len(surface['vertices']), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32) + len(base['vertices'])
    result = {k: np.concatenate([base[k], surface[k][used]]) for k in ('vertices', 'colors', 'vertex_groups')}
    result['triangles'] = np.concatenate([base['triangles'], remap[faces]])
    meta.update(surface_method='Detailed TSDF with supported local Poisson gap patches',
                base_mesh_sha256=sha256_file(args.base), patch_mesh_sha256=sha256_file(args.surface),
                original_vertices_preserved=True, original_colors_preserved=True,
                gap_distance=args.gap_distance, boundary_distance=args.boundary_distance,
                added_patch_triangles=len(faces), vertices=len(result['vertices']), triangles=len(result['triangles']))
    face_groups = result['vertex_groups'][result['triangles'][:, 0]]
    meta['groups'] = [dict(group=int(group), vertices=int((result['vertex_groups'] == group).sum()),
                          triangles=int((face_groups == group).sum()))
                      for group in np.unique(result['vertex_groups'])]
    result.update(metadata_json=np.asarray(json.dumps(meta)),
                  motion_quaternions=surface['motion_quaternions'], motion_translations=surface['motion_translations'])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'mesh.npz', **result)
    (args.output / 'report.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(dict(vertices=meta['vertices'], triangles=meta['triangles'])), flush=True)


if __name__ == '__main__':
    main()
