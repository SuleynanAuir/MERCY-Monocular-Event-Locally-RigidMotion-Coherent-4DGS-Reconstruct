"""Build a separate, auditable V3 surface asset without changing source motion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.fan_surface import regularize


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strength', type=float, default=.65)
    parser.add_argument('--split-budget', type=int, default=300000)
    parser.add_argument('--maximum-shift', type=float, default=.00065)
    parser.add_argument('--neighbors', type=int, default=32)
    parser.add_argument('--color-strength', type=float, default=.2)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    config = json.loads(args.manifest.read_text())
    if config.get('format') != 'fan_mesh_comparison_v1': parser.error('Expected Fan comparison')
    root = args.manifest.parent
    source, binding, mesh = [(root/config[key]['right']).resolve()
                             for key in ('sources', 'hybrid_paths', 'mesh_paths')]
    with np.load(source, allow_pickle=False) as z:
        base = z['base_vertices']; meta = json.loads(str(z['metadata_json']))
    with np.load(binding, allow_pickle=False) as z:
        rows, groups = z['rows'], z['groups']
    if not np.array_equal(rows, np.arange(len(base))):
        raise ValueError('Every source Gaussian must have a stable part binding')
    args.output.mkdir(parents=True)
    vertices, parents, groups, report = regularize(base, groups, strength=args.strength,
        split_budget=args.split_budget, maximum_shift=args.maximum_shift,
        neighbors=args.neighbors, color_strength=args.color_strength)
    record = dict(format='fan_surface_refinement_v1', source_sha256=sha(source),
        binding_sha256=sha(binding), mesh_sha256=sha(mesh),
        source_base_sha256=meta['base_vertices_sha256'], report=report,
        supervision='existing canonical Gaussians only; no RGB/event training',
        motion='unchanged embedded rigid group transforms, inherited by every child')
    np.savez_compressed(args.output/'surface.npz', base_vertices=vertices,
        parent_rows=parents.astype(np.int64), groups=groups.astype(np.int32),
        metadata_json=np.asarray(json.dumps(record)))
    # Keep V1 and the complete source/binding chain intact.
    for key in ('sources', 'mesh_paths', 'hybrid_paths'):
        config[key] = {side: str((root/path).resolve()) for side, path in config[key].items()}
    config['surface_refinement'] = 'surface.npz'
    (args.output/'manifest.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
