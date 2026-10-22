"""Compare actual V3/V4 moving Gaussian trajectories without changing the viewer."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from utils.event_refinement import correction_from_record, qrotate
from utils.rigid_checkpoint import sha256_file
from utils.scgs_v4_field import load_checkpoint
from utils.trajectory_stability import trajectory_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probes', type=int, default=1024)
    args = parser.parse_args()
    if args.probes < 13:
        parser.error('Use at least 13 probes to represent the motion groups')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    field, provenance = load_checkpoint(args.checkpoint, 'cpu')
    path = Path(provenance['candidate'])/'correction.json'
    if sha256_file(path) != provenance['baseline_artifact_sha256']['correction']:
        raise ValueError('V3 correction no longer matches the checkpoint')
    cache_path = Path(provenance['baseline'])/'cache.json'
    if sha256_file(cache_path) != provenance['baseline_artifact_sha256']['baseline_cache_metadata']:
        raise ValueError('V3 timeline metadata changed')
    cache = json.loads(cache_path.read_text())
    correction = correction_from_record(json.loads(path.read_text()), 'cpu')
    # Deterministic sampling in every moving group, including small parts.
    probes = []
    groups = field.groups.unique()
    for group in groups:
        rows = torch.where(field.groups == group)[0]
        count = min(len(rows), max(1, args.probes//len(groups)))
        probes.append(rows[torch.linspace(0, len(rows)-1, count).round().long()])
    selection = torch.cat(probes)
    original_rows = field.rows[selection].tolist()
    sample = copy.deepcopy(field)
    for key in ('groups', 'canonical', 'knn', 'offsets', 'weights', 'pins'):
        if hasattr(sample, key):
            setattr(sample, key, getattr(sample, key)[selection])
    sample.rows = torch.arange(len(selection))
    before, after = [], []
    rotations = torch.tensor([1., 0., 0., 0.]).expand(len(selection), -1)
    for frame in range(cache['frame_count']):
        q, t = correction.group_transforms(frame)
        base = qrotate(q[sample.groups], sample.canonical)+t[sample.groups]
        moved, _ = sample.deform(base, rotations, q, frame)
        before.append(base.numpy()); after.append(moved.numpy())
    before, after = np.asarray(before), np.asarray(after)
    report = dict(checkpoint_sha256=sha256_file(args.checkpoint), correction_sha256=sha256_file(path),
                  scope='Deterministic moving rigid Gaussian probes; excludes static, elastic-band, surface appearance and playback timing.',
                  interpretation='Acceleration, jerk and high-frequency energy can also be genuine motion; this diagnostic alone cannot identify perceptual jitter.',
                  source_gaussian_rows=original_rows,
                  v3=trajectory_metrics(before, cache['fps'], float(field.scene_scale)),
                  v4=trajectory_metrics(after, cache['fps'], float(field.scene_scale)),
                  max_position_difference=float(np.linalg.norm(after-before, axis=-1).max()),
                  exact_same_positions=bool(np.array_equal(before, after)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_gaussian_rows'}, indent=2))


if __name__ == '__main__':
    main()
