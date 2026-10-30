"""Verify a Lamp candidate against a completed, same-camera V1/V2/V3 report.

Read-only deployment preflight. A report about another checkpoint cannot approve
this candidate; missing metrics and regressions fail closed. This checks the
development sequence, not unseen-view generalization.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


DIRECTIONS = dict(foreground_psnr=1, psnr=1, ssim=1, foreground_ssim=1,
                  lpips=-1, iou=1, missing=-1, part_missing=-1,
                  event_l1=-1, event_cosine=1, polarity=1, silent=-1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quality_gate(summary):
    rows = {row['version']: row for row in summary}
    deltas = {}
    for key, direction in DIRECTIONS.items():
        values = [rows[version].get(key) for version in ('V2', 'V3')]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError('Missing or nonfinite quality metric: ' + key)
        delta = values[1] - values[0]
        if direction * delta < 0:
            raise ValueError('V3 regresses against V2: ' + key)
        deltas[key] = delta
    if not any(deltas.values()):
        raise ValueError('Candidate has no measured improvement')
    return deltas


def verify(pair, report):
    pair, report = Path(pair).resolve(), Path(report).resolve()
    index = json.loads((report/'artifact_manifest.json').read_text())
    for name in ('data/analysis.json', 'data/input_manifest.json.gz'):
        if sha(report/name) != index['files'][name]['sha256']:
            raise ValueError('Report artifact changed: ' + name)
    manifest = json.loads(gzip.decompress((report/'data/input_manifest.json.gz').read_bytes()))
    if manifest['scene'] != 'lamp' or any(s['render_mode'] != 'gaussian' for s in manifest['samples']):
        raise ValueError('Expected native Gaussian Lamp evaluation')
    recorded = {str(Path(s['source']).resolve()): s['sha256'] for s in manifest['sources']}
    config = json.loads(pair.read_text())
    paths = [pair]
    for key in ('left', 'right', 'material_reference', 'appearance_refinement', 'detail_finish'):
        if key not in config:
            continue
        entry = config[key]
        path = (pair.parent/entry['path']).resolve()
        if sha(path) != entry['sha256']:
            raise ValueError('Candidate dependency changed: ' + str(path))
        paths.append(path)
    for path in paths:
        if recorded.get(str(path)) != sha(path):
            raise ValueError('Report evaluated a different asset: ' + str(path))
    summary = json.loads((report/'data/analysis.json').read_text())['summary']
    return dict(pair=str(pair), pair_sha256=sha(pair), report=str(report),
                sample_count=len(manifest['samples']), summary=summary,
                v3_minus_v2=quality_gate(summary), passed=True,
                scope='Same-view development sequence; not independent generalization')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.pair, args.report), indent=2))
