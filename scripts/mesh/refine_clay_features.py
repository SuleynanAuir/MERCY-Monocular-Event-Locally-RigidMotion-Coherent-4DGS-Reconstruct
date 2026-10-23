"""Refine an existing Clay mesh with robust, feature-aware surface patches."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.clay_features import reconstruct_features
from utils.rigid_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=.008)
    parser.add_argument('--max-move', type=float, default=.003)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    with np.load(args.input, allow_pickle=False) as z: data = {k:z[k] for k in z.files}
    meta = json.loads(str(data['metadata_json']))
    if not meta.get('clay_regularization'): parser.error('Expected a source-bound Clay mesh')
    from utils.viewer_mesh import load_mesh_data
    load_mesh_data(args.input, meta['group_count'], meta['source_fingerprints'])
    groups, faces = data['vertex_groups'], data['triangles']
    vertices = data['vertices'].copy()
    normals = np.empty_like(vertices)
    reports = []
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        local_faces = np.searchsorted(ids, faces[groups[faces[:, 0]] == group])
        radius = min(args.radius, np.ptp(vertices[ids], axis=0).max()/12)
        v, n, report = reconstruct_features(vertices[ids], local_faces, radius, min(args.max_move, radius*.45))
        vertices[ids], normals[ids] = v, n
        report['group'] = int(group)
        reports.append(report)
        print(json.dumps(report), flush=True)
    data.update(vertices=vertices, clay_normals=normals)
    meta['clay_regularization']['feature_reconstruction'] = dict(
        method='Robust patch guidance, static/dynamic normal filtering and bounded point-to-plane reconstruction',
        input_sha256=sha256_file(args.input), groups=reports,
        topology_preserved=True, canonical_normals=True, image_filtering=False)
    data['metadata_json'] = np.asarray(json.dumps(meta))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'mesh.npz', **data)
    (args.output/'report.json').write_text(json.dumps(meta['clay_regularization'], indent=2)+'\n')


if __name__ == '__main__': main()
