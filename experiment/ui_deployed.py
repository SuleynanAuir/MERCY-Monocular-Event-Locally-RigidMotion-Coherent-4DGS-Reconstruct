"""Audit and capture an existing deployment without rebuilding its models.

Currently accepts the original Gaussian-pair catalogs. Lego's comparison and
Fan's composite renderer must retain their own artifact contracts; they are
never converted to Gaussian-pair inputs by this entry point.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from .protocol import save_json, sha256


def freeze(catalog, project):
    config = json.loads(catalog.read_text())
    files = {catalog.resolve()}
    scenes = []
    for entry in config['scenes']:
        if entry['type'] != 'gaussian_pair':
            raise ValueError('Retain the original renderer and its artifact contract: ' + entry['id'])
        manifest = (catalog.parent / entry['path']).resolve()
        files.add(manifest)
        settings = json.loads(manifest.read_text())
        if settings['format'] != 'gaussian_pair_v1':
            raise ValueError('Unexpected deployed manifest format')
        assets = {}
        for role in ('left', 'right', 'material_reference', 'detail_finish'):
            if role not in settings:
                continue
            item = settings[role]
            path = (manifest.parent / item['path']).resolve()
            digest = sha256(path)
            if digest != item['sha256']:
                raise ValueError('Deployed asset hash mismatch: ' + str(path))
            files.add(path)
            assets[role] = dict(path=str(path), sha256=digest, label=item.get('label'))
        reference = (catalog.parent / entry['reference_path']).resolve()
        files.update(p.resolve() for p in reference.parent.rglob('*') if p.is_file())
        scenes.append(dict(id=entry['id'], renderer='GaussianPairRenderer',
                           manifest=str(manifest), manifest_sha256=sha256(manifest), assets=assets,
                           reference=str(reference), versions=['V1', 'V3'],
                           v2_status='No separately deployed V2; do not invent or substitute one'))
    code = list(project.glob('*.py'))
    for folder in ('utils', 'viewer', 'scene', 'gaussian_renderer', 'scripts'):
        code.extend(p for p in (project/folder).rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.js', '.html', '.css'))
    return dict(format='original_ui_deployment_lock_v1', catalog=str(catalog.resolve()),
                project=str(project.resolve()), scenes=scenes,
                files_sha256={str(p): sha256(p) for p in sorted(files)},
                code_sha256={str(p.relative_to(project)): sha256(p) for p in sorted(set(code))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--process', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', required=True)
    args = p.parse_args()
    proc = json.loads(args.process.read_text())
    pid = proc['pid']
    if Path(f'/proc/{pid}/cwd').resolve() != args.project.resolve():
        raise ValueError('UI is running another code checkout')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if str(args.catalog.resolve()).encode() not in command:
        raise ValueError('UI is running another scene catalog')
    args.output.mkdir(parents=True, exist_ok=False)
    lock = freeze(args.catalog, args.project)
    save_json(args.output/'deployment-lock.json', lock)
    for scene in lock['scenes']:
        deadline = time.monotonic() + 900
        while True:
            with urllib.request.urlopen(args.url+'/api/status?scene='+scene['id'], timeout=30) as response:
                status = json.load(response)
            if status.get('error'):
                raise RuntimeError(status['error'])
            if status.get('ready'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Original UI did not initialize')
            time.sleep(2)
        for mode in ('gaussian', 'clay'):
            dest = args.output/'captures'/scene['id']/mode
            subprocess.run([sys.executable, '-u', '-m', 'experiment.ui_capture',
                            '--url', args.url, '--scene', scene['id'], '--mode', mode,
                            '--output', str(dest)], check=True)
    if freeze(args.catalog, args.project) != lock:
        raise ValueError('Original deployment changed during capture; reject these measurements')
    save_json(args.output/'verification.json', dict(
        deployment_unchanged=True, original_code_checkout=str(args.project),
        model_training_performed=False, asset_rebuild_performed=False,
        completed_scenes=[s['id'] for s in lock['scenes']], expected_scenes=5,
        absent_scenes=['lego', 'fan', 'lamp'],
        scope='Original deployment pixels, V1/V3; no substituted V2 or native model audit'))


if __name__ == '__main__':
    main()
