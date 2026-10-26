"""Run frozen UI observation stages; source preparation must finish first."""
import argparse
from pathlib import Path
import sys
import time

from .ui_rebuild import run_jobs


def observation_jobs(data, root, scenes):
    py = sys.executable
    inventory = root/'input-inventory.json'
    jobs = [('inventory', [py, '-m', 'experiment.run', 'audit', '--data', str(data),
                           '--output', str(inventory)], [inventory])]
    for scene, folder, stem in [('lamp','lamp_data','lamp'),
                                ('metronome','metronome_data','metronome'),
                                ('microphone','micphone_data','micphone')]:
        if scene not in scenes:
            continue
        dest = root/scene
        prepared = dest/'v1-prior'
        original = dest/'supplied-pair/left.npz'
        jobs.append((scene+'-v1-prior', [py, '-m', 'experiment.ui_stages', 'prepare-rigid',
            '--inventory', str(inventory), '--scene', scene, '--root', str(dest),
            '--source', str(data/folder/(stem+'.ply')), '--sequence', str(original),
            '--output', str(prepared)], [prepared/'prior.json', prepared/'supervision/metadata.json']))
        training = dest/'v2-training'
        jobs.append((scene+'-v2-train', [py, '-m', 'experiment.ui_stages', 'train',
            '--prepared', str(prepared), '--output', str(training), '--method', 'C2',
            '--steps', '200', '--seed', '17'], [training/'checkpoint.pt', training/'evaluation.json']))
        jobs.append((scene+'-v2-bake', [py, '-m', 'experiment.ui_stages', 'export',
            '--prepared', str(prepared), '--checkpoint', str(training/'checkpoint.pt'),
            '--method', 'C2', '--template', str(original), '--output', str(dest/'v2.npz')], [dest/'v2.npz']))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=['lamp','metronome','microphone'])
    p.add_argument('--stage', choices=('observation','auxiliary'), default='observation')
    p.add_argument('--only',nargs='+')
    p.add_argument('--wait-for',type=Path,help='Wait for a required completed upstream audit before using the GPU')
    args = p.parse_args()
    if args.wait_for:
        deadline=time.monotonic()+3600
        print('Waiting for upstream completed audit:',args.wait_for,flush=True)
        while not args.wait_for.is_file():
            if time.monotonic()>deadline:raise TimeoutError(args.wait_for)
            time.sleep(5)
    factory = observation_jobs if args.stage == 'observation' else auxiliary_jobs
    tasks=factory(args.data.resolve(), args.root.resolve(), args.scenes)
    if args.only:
        if set(args.only)-{job[0] for job in tasks}:p.error('Unknown job')
        tasks=[job for job in tasks if job[0] in args.only]
    run_jobs(tasks,
             args.root.resolve()/(args.stage+'-journal'))


def auxiliary_jobs(data, root, scenes):
    py = sys.executable
    jobs = []
    for scene in scenes:
        if scene not in ('lamp','metronome','microphone'):
            raise ValueError('Articulated/Fan auxiliary preparation requires its specific source baker')
        dest = root/scene
        sequence = dest/('surface-pair/right.npz' if scene == 'lamp' else 'surface/right.npz')
        manifest = dest/('detail/manifest.json' if scene == 'lamp' else 'surface/manifest-detail.json')
        prior = dest/'v3-prior'
        jobs.append((scene+'-v3-prior', [py, '-m', 'experiment.ui_stages', 'prepare-rigid',
            '--inventory', str(root/'input-inventory.json'), '--scene', scene, '--root', str(dest),
            '--source', str(dest/'surface/surface.ply'), '--sequence', str(sequence),
            '--labels', str(dest/'surface/binding.npz'), '--reference-prior', str(dest/'v1-prior/prior.json'),
            '--output', str(prior)], [prior/'prior.json', prior/'supervision/metadata.json']))
        training = dest/'v3-training'
        jobs.append((scene+'-v3-train', [py, '-m', 'experiment.ui_stages', 'train',
            '--prepared', str(prior), '--output', str(training), '--method', 'C3',
            '--initial-alignment', str(dest/'v2-training/checkpoint.pt'), '--freeze-alignment',
            '--event-only', '--steps', '200', '--seed', '17'], [training/'checkpoint.pt', training/'evaluation.json']))
        jobs.append((scene+'-v3-bake', [py, '-m', 'experiment.ui_stages', 'export',
            '--prepared', str(prior), '--checkpoint', str(training/'checkpoint.pt'), '--method', 'C3',
            '--template', str(sequence), '--output', str(dest/'v3.npz')], [dest/'v3.npz']))
        jobs.append((scene+'-package', [py, '-m', 'experiment.ui_stages', 'package',
            '--root', str(dest), '--name', scene.title(), '--surface-manifest', str(manifest)],
            [dest/'pair-v1-v2.json', dest/'pair-v1-v3.json']))
    return jobs


if __name__ == '__main__':
    main()
