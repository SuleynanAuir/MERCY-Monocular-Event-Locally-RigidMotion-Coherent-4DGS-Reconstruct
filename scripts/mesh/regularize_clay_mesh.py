"""Build a separate regularized display mesh while preserving all group motion."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.clay_surface import regularize_part
from utils.rigid_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacing', type=float, default=.0025)
    parser.add_argument('--max-gap', type=float, default=.006)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    with np.load(args.input, allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    meta = json.loads(str(data['metadata_json']))
    from utils.viewer_mesh import load_mesh_data
    load_mesh_data(args.input, meta['group_count'], meta['source_fingerprints'])
    face_groups = data['vertex_groups'][data['triangles'][:,0]]
    vertices, faces, groups, reports = [], [], [], []
    offset = 0
    for group in np.unique(data['vertex_groups']):
        ids = np.flatnonzero(data['vertex_groups']==group)
        source_faces = data['triangles'][face_groups==group]
        if not len(source_faces): continue
        # Preserve fine isolated mechanisms with a smaller local sampling scale.
        extent = np.ptp(data['vertices'][ids], axis=0).max()
        spacing = min(args.spacing, extent/32)
        gap = min(args.max_gap, spacing*2.4)
        v,f,report = regularize_part(data['vertices'][ids], np.searchsorted(ids,source_faces), spacing,gap)
        if not len(f): raise ValueError('Regularization removed a whole motion group')
        vertices.append(v);faces.append(f+offset);groups.append(np.full(len(v),group,np.int32));offset+=len(v)
        report['group']=int(group);reports.append(report);print(json.dumps(report),flush=True)
    source_hash = sha256_file(args.input)
    meta['clay_regularization'] = dict(source_mesh_sha256=source_hash, groups=reports,
        method='Part-local oriented-sample regularization and bounded Poisson gap repair',
        motion_preserved=True, display_only=True)
    data.update(vertices=np.concatenate(vertices),triangles=np.concatenate(faces),vertex_groups=np.concatenate(groups))
    data.pop('clay_normals', None)
    data['colors']=np.full_like(data['vertices'],.65)
    meta.update(vertices=len(data['vertices']),triangles=len(data['triangles']))
    data['metadata_json']=np.asarray(json.dumps(meta))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'mesh.npz',**data)
    (args.output/'report.json').write_text(json.dumps(meta['clay_regularization'],indent=2)+'\n')
    print(str(args.output/'mesh.npz'),flush=True)


if __name__=='__main__': main()
