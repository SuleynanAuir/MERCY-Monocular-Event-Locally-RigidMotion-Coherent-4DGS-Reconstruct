"""Repair a baked Fan sequence without refitting angles or deleting Gaussians."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from utils.gaussian_sequence import GaussianSequence
from utils.fan_motion_binding import refine_static_membership


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--motion-report', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    camera = json.loads((args.data/'fixed_camera.json').read_text())
    motion = json.loads(args.motion_report.read_text())
    cap = cv2.VideoCapture(str(args.data/'Fan.mkv'))
    frames = []
    while True:
        ok, image = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cap.release()
    with GaussianSequence(args.sequence) as sequence:
        if sequence.dynamic_indices is None: raise ValueError('Expected sparse baked Fan sequence')
        if len(frames) != len(sequence): raise ValueError('Video and sequence lengths differ')
        expected = sequence.metadata['input_sha256']
        if expected != motion['input_sha256']:
            raise ValueError('Motion report and sequence have different source inputs')
        for name in ('Fan.mkv', 'fixed_camera.json'):
            if sha(args.data/name) != expected[name]:
                raise ValueError(f'Changed input: {name}')
        raw = sequence.base_vertices
        xyz = np.column_stack([raw[k] for k in ('x','y','z')])
        moving = np.zeros(len(raw), bool); moving[sequence.dynamic_indices] = True
        corrected, changed, report = refine_static_membership(
            xyz, moving, motion['center'], motion['angles_rad'], frames, camera)
        if not len(changed): raise ValueError('No supported static reassignment found')
        keep = corrected[sequence.dynamic_indices]
        poses = np.ascontiguousarray(sequence.poses[keep])
        indices = sequence.dynamic_indices[keep]
        meta = copy.deepcopy(sequence.metadata)
        for index, frame in enumerate(meta['frames']):
            sequence.validate_pose_frame(index)
            frame['pose_sha256'] = hashlib.sha256(poses[:, index].tobytes()).hexdigest()
        meta['motion_binding_repair'] = report
        motion.update(moving_gaussians=len(indices), static_gaussians=len(raw)-len(indices),
                      motion_binding_repair=report)
        args.output.mkdir(parents=True)
        np.save(args.output/'reassigned-static-indices.npy', changed)
        (args.output/'motion-report.json').write_text(json.dumps(motion, indent=2))
        partial = args.output/'fan-motion.npz.partial'
        with partial.open('wb') as stream:
            np.savez(stream, base_vertices=raw, poses=poses, dynamic_indices=indices,
                     metadata_json=np.asarray(json.dumps(meta)),
                     times_seconds=np.asarray([f['time_seconds'] for f in meta['frames']]))
        partial.rename(args.output/'fan-motion.npz')
    with GaussianSequence(args.output/'fan-motion.npz') as checked:
        for i in range(len(checked)): checked.validate_pose_frame(i)
        assert checked.metadata['base_vertices_sha256'] == meta['base_vertices_sha256']
    print(json.dumps(report, indent=2), flush=True)
    print('All remaining poses and canonical Gaussians preserved; 100-frame checks passed', flush=True)

if __name__ == '__main__': main()
