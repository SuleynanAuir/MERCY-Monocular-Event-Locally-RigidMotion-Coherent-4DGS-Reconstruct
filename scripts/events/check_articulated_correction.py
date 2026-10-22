"""Check every v3 pose for finite values and preservation of within-group shape."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.events.export_lego_event_preview import verify_candidate
from utils.event_refinement import correction_from_record, TemporalRigidMotionCorrection


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, record, _ = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if record['format'] != 'event_supervised_articulated_alignment_v3':
        raise ValueError('Expected articulated v3 model')
    model = correction_from_record(record, 'cuda')
    poses = np.load(args.baseline / 'poses.npy', mmap_mode='r')
    rest = torch.from_numpy(poses[0].copy()).cuda()
    group = model.groups.cpu().numpy()
    pairs = []
    for g in np.unique(group):
        rows = np.flatnonzero(group == g)
        chosen = rows[np.linspace(0, len(rows)-1, min(32, len(rows))).astype(int)]
        pairs.extend(zip(chosen[:len(chosen)//2], chosen[len(chosen)//2:]))
    pairs = torch.as_tensor(pairs, dtype=torch.long, device='cuda')
    pair_error, contact_max, contact_rms, anchor_max = 0., 0., [], 0.
    for i in range(len(poses)):
        baked = torch.from_numpy(poses[i].copy()).cuda()
        before = baked if i == 0 else TemporalRigidMotionCorrection.forward(model, rest, baked, i)
        after = baked if i == 0 else model(rest, baked, i)
        if not torch.isfinite(after).all() or torch.any(after[:, 3:].norm(dim=-1) < 1e-8):
            raise ValueError(f'Invalid Gaussian pose at {i}')
        length = lambda pose: (pose[pairs[:, 0], :3]-pose[pairs[:, 1], :3]).norm(dim=-1)
        pair_error = max(pair_error, float((length(before)-length(after)).abs().max()))
        delta = after[:, :3]-before[:, :3]
        if len(model.contacts):
            shift = (delta[model.contacts[:, 0]]-delta[model.contacts[:, 1]]).norm(dim=-1)
            contact_max = max(contact_max, float(shift.max()))
            contact_rms.append(float(shift.square().mean().sqrt()))
        if len(model.anchors):
            anchor_max = max(anchor_max, float(delta[model.anchors].norm(dim=-1).max()))
        if i == 0 and not torch.equal(after, rest):
            raise ValueError('First frame changed')
    if pair_error > 1e-5:
        raise ValueError(f'Unexpected nonrigid shape change: {pair_error}')
    report = {'frames_checked': len(poses), 'dynamic_gaussians': len(rest), 'motion_groups': len(np.unique(group)),
              'rigid_pairs_sampled_per_frame': len(pairs), 'all_poses_finite': True, 'first_frame_exact': True,
              'max_added_pair_distance_error': pair_error, 'contact_pairs': len(model.contacts),
              'max_contact_relative_displacement_scene_units': contact_max,
              'mean_frame_rms_contact_displacement_scene_units': float(np.mean(contact_rms)),
              'max_anchor_displacement_scene_units': anchor_max,
              'scope': 'Checks added correction rigidity, not ground-truth kinematics, hidden geometry or force/dynamics.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
