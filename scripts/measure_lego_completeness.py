"""Measure visible foreground holes/floaters in the event-active region, not hidden 3D completeness."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--renders', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
        counts = archive['counts']
        active = binary_dilation(np.any(counts > 0, axis=(0, 1)), iterations=3)
    del counts
    report = {'scope': 'Reference-view visible silhouette only, in the union of observed event pixels dilated by 3 pixels.',
              'foreground_threshold': .08, 'active_pixels': int(active.sum()), 'models': {}}
    for path in args.renders:
        rendered = np.load(path / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
        if rendered.shape != images.shape:
            raise ValueError('Frame shapes differ')
        records = []
        for index in range(len(images)):
            expected = (images[index] / 255 > .08) & active
            actual = (rendered[index] > .08) & active
            records.append({'frame': index, 'missing_foreground_fraction': float(np.sum(expected & ~actual) / max(expected.sum(), 1)),
                            'unsupported_foreground_fraction': float(np.sum(actual & ~expected) / max(actual.sum(), 1)),
                            'silhouette_iou': float(np.sum(expected & actual) / max(np.sum(expected | actual), 1)),
                            'rgb_l1': float(np.abs(rendered[index].astype(np.float32) - images[index] / 255)[active].mean())})
        report['models'][path.name] = {'summary': {key: float(np.mean([row[key] for row in records])) for key in records[0] if key != 'frame'},
                                      'frames': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: model['summary'] for name, model in report['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
