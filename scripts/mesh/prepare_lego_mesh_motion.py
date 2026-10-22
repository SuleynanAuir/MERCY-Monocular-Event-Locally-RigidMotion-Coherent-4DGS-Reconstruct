"""Export compact event-corrected group transforms for independent prior checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from utils.event_refinement import correction_from_record
from utils.rigid_checkpoint import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'parts', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cache = json.loads((args.baseline/'cache.json').read_text())
    record = json.loads((args.candidate/'correction.json').read_text())
    with np.load(args.parts, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
    groups = np.full(len(meta['objects']), -1, np.int32)
    for group in record['joint_structure']['source_groups']:
        groups[group['objects']] = group['group']
    if set(np.flatnonzero(groups < 0).tolist()) != set(meta['deforming_objects']):
        raise ValueError('All and only elastic source objects must lack rigid groups')
    torch.set_num_threads(2)
    model = correction_from_record(record, 'cpu')
    q, t = [], []
    with torch.no_grad():
        for index in range(cache['frame_count']):
            quaternion, translation = model.group_transforms(index)
            q.append(quaternion.numpy()); t.append(translation.numpy())
    info = {'format': 'lego_corrected_mesh_motion_v1', 'objects': meta['objects'],
            'excluded_deforming_objects': meta['deforming_objects'],
            'source_parts_sha256': sha256_file(args.parts),
            'source_correction_sha256': sha256_file(args.candidate/'correction.json'),
            'baseline_cache_sha256': sha256_file(args.baseline/'cache.json'),
            'scope': 'Exact group SE3 from existing event-corrected motion; no new fit. Elastic bands excluded.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, group_ids=groups, quaternions=np.asarray(q), translations=np.asarray(t),
                        camera_json=np.asarray(json.dumps(cache['camera'])), metadata_json=np.asarray(json.dumps(info)))
    print(json.dumps({'output': str(args.output), 'groups': model.group_count, 'frames': len(q)}), flush=True)


if __name__ == '__main__':
    main()
