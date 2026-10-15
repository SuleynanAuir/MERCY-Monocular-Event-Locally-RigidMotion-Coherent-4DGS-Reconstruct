"""Package measured color residuals; never inherit old event/image acceptance."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_lego_event_preview import verify_candidate
from utils.color_residual import FORMAT, verify_color_residual
from utils.rigid_checkpoint import sha256_file


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output = args.output.resolve()
    for name in ('baseline', 'candidate', 'fit'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    cache, record, _ = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if any(cache.get(key) for key in ('color_residual', 'gaussian_cleanup', 'static_surface_rebuild')):
        raise ValueError('Cannot stack an appearance experiment on another experiment')
    fit_report = json.loads((args.fit/'optimization.json').read_text())
    finished = json.loads((args.fit/'completion.json').read_text())
    if (fit_report.get('parent_baseline') != str(args.baseline)
            or fit_report.get('parent_candidate') != str(args.candidate)
            or fit_report.get('selected_iteration', 0) <= 0
            or finished.get('source_inputs_rechecked') is not True or finished.get('alpha_unchanged') is not True
            or finished.get('deployed') is not False):
        raise ValueError('Require a completed, provenance-matched diagnostic fitting run')
    model, baseline, candidate = [args.output/key for key in ('model', 'baseline', 'candidate')]
    for folder in (model, baseline, candidate):
        folder.mkdir(parents=True, exist_ok=False)
    for name in ('optimization.json', 'residual.npz'):
        shutil.copyfile(args.fit/name, model/name)
    contract = {'format': FORMAT, 'report': str(model/'optimization.json'),
        'report_sha256': sha256_file(model/'optimization.json'), 'residual': str(model/'residual.npz'),
        'residual_sha256': sha256_file(model/'residual.npz')}
    child_cache = copy.deepcopy(cache)
    child_cache.update(color_residual=contract, status='COLOR_RESIDUAL_REQUIRES_FRESH_EVENT_AUDIT')
    write_new(baseline/'cache.json', child_cache)
    for name in ('poses.npy', 'dynamic_indices.npy'):
        (baseline/name).symlink_to((args.baseline/name).resolve(strict=True))
    child_record = copy.deepcopy(record)
    child_record['color_residual'] = contract
    child_record['joint_structure']['input_sha256']['baseline_cache_metadata'] = sha256_file(baseline/'cache.json')
    write_new(candidate/'correction.json', child_record)
    verify_color_residual(child_cache, child_record, baseline=baseline)
    print(json.dumps({'output': str(args.output), 'optimized_rows': fit_report['optimized_count'],
        'source_and_motion_unchanged': True, 'event_audit': 'pending'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
