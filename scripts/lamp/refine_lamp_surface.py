"""Create a separate, auditable V3 surface candidate; retain both input PLYs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scripts.motion.bind_lego_parts import transform
from utils.gaussian_pair_renderer import file_sha
from utils.lamp_surface import reconstruct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parts', 'binding-report', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--split', type=int, choices=(1, 2, 4), default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.binding_report.read_text())
    if report['parts_sha256'] != file_sha(args.parts):
        raise ValueError('Motion binding does not match source parts')
    with np.load(args.parts, allow_pickle=False) as parts:
        surface = transform(parts['rest_points'], np.array(report['registration']))
        part_ids = parts['point_parts']
    sources = [args.data/name for name in ('lamp.ply', 'lamb_corrected_g0.ply')]
    vertices = [PlyData.read(path)['vertex'].data.copy() for path in sources]
    tree = cKDTree(surface)
    groups = [part_ids[tree.query(np.column_stack([v[k] for k in ('x','y','z')]), workers=4)[1]] for v in vertices]
    result, parents, labels, detail = reconstruct(*vertices, *groups, split=args.split)
    args.output.mkdir(parents=True)
    target = args.output/'surface.ply'
    PlyData([PlyElement.describe(result, 'vertex')], text=False).write(target)
    np.savez_compressed(args.output/'binding.npz', groups=labels, reference_parents=parents)
    record = dict(format='lamp_surface_v1', report=detail,
        reference_sha256=file_sha(sources[0]), corrected_sha256=file_sha(sources[1]),
        surface_sha256=file_sha(target), binding_sha256=file_sha(args.output/'binding.npz'),
        parts_sha256=file_sha(args.parts), original_binding_sha256=file_sha(args.binding_report))
    (args.output/'report.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
