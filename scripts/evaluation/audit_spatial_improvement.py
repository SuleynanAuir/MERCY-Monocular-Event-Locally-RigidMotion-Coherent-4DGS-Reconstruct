"""Audit all exported frames against shared RGB without changing report selection.

Reports signed V3-minus-V2 error and both improved and degraded foreground pixels.
Version IDs follow the export's displayed order; labels and references stay intact.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image


def audit(archive, output):
    with zipfile.ZipFile(archive) as bundle:
        manifests = [n for n in bundle.namelist() if n.endswith('/manifest.json') or n == 'manifest.json']
        if len(manifests) != 1:
            raise ValueError('Expected one experiment manifest')
        prefix = manifests[0][:-len('manifest.json')]
        manifest = json.loads(bundle.read(manifests[0]))
        versions = manifest['versions']
        if len(versions) != 3:
            raise ValueError('Expected three displayed versions')
        def image(path):
            return np.asarray(Image.open(io.BytesIO(bundle.read(prefix + path))).convert('RGB'), dtype=float) / 255
        if len(manifest['reference']['samples']) != len(manifest['samples']):
            raise ValueError('Incomplete references')
        rows = []
        for i, (sample, ref) in enumerate(zip(manifest['samples'], manifest['reference']['samples'])):
            if not ref.get('aligned') or not ref.get('rgb') or not ref.get('foreground_mask'):
                raise ValueError('Aligned shared RGB and foreground masks required')
            truth = image(ref['rgb'])
            mask = image(ref['foreground_mask']).max(-1) > 0
            if not mask.any():
                raise ValueError('Empty foreground')
            errors = [np.abs(image(f'versions/{v["id"]}/frames/{i:06d}.png') - truth).mean(-1) for v in versions]
            stats = [dict(mae=float(e.mean()), foreground_mae=float(e[mask].mean()),
                          foreground_p95=float(np.percentile(e[mask], 95))) for e in errors]
            delta = errors[2] - errors[1]
            rows.append(dict(frame=sample['frame'], versions=stats,
                             v3_better_than_both=all(stats[2][k] < min(s[k] for s in stats[:2]) for k in ('mae', 'foreground_mae')),
                             foreground_improved_fraction=float((delta[mask] < 0).mean()),
                             foreground_degraded_fraction=float((delta[mask] > 0).mean())))
        if len(rows) != len(manifest['samples']):
            raise ValueError('Incomplete references')
    result = dict(archive=str(archive), versions=versions, samples=rows,
                  all_frames_improved=all(r['v3_better_than_both'] for r in rows),
                  scope='Same-view source RGB; local regressions retained; improvement in MAE is not proof of every metric or unseen-view quality')
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.archive, a.output)
    print(json.dumps(dict(frames=len(result['samples']), all_frames_improved=result['all_frames_improved'])))
