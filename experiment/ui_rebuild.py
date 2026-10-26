"""Rebuild UI input assets from data in a new, journaled experiment directory.

This entry point deliberately does not relabel the earlier motion-adapter runs
as UI versions. Later stages must consume these assets and the UI renderer.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from .protocol import save_json, sha256


def commands(data, output, blender):
    py = sys.executable
    jobs = []
    for scene, folder, stem in [('lamp', 'lamp_data', 'lamp'),
                                ('metronome', 'metronome_data', 'metronome'),
                                ('microphone', 'micphone_data', 'micphone')]:
        dest = output / scene
        parts = dest / 'parts.npz'
        jobs.append((scene + '-parts', [blender, '-b', '--disable-autoexec',
            str(data/folder/(stem+'.blend')), '--python', 'scripts/lamp/export_lamp_motion.py',
            '--', '--output', str(parts)], [parts]))
        script = 'prepare_lamp_viewer.py' if scene == 'lamp' else 'prepare_object_viewer.py'
        cmd = [py, 'scripts/lamp/'+script, '--data', str(data/folder),
               '--parts', str(parts), '--output', str(dest/'supplied-pair')]
        if scene != 'lamp':
            cmd += ['--stem', stem, '--name', scene.title()]
        jobs.append((scene+'-supplied-pair', cmd, [dest/'supplied-pair/manifest.json',
                    dest/'supplied-pair/left.npz', dest/'supplied-pair/right.npz']))
    fan = output/'fan/source-motion'
    jobs.append(('fan-source-motion', [py, 'scripts/fan/prepare_fan_viewer.py',
        '--data', str(data/'Fan_data'), '--output', str(fan)],
        [fan/'fan-motion.npz', fan/'motion-report.json']))
    parts = output/'lego/parts.npz'
    jobs.append(('lego-parts', [blender, '-b', '--disable-autoexec',
        str(data/'Lego_data/Lego.blend'), '--python', 'scripts/motion/export_blender_parts.py',
        '--', '--output', str(parts)], [parts]))
    lego = output/'lego'
    jobs.append(('lego-inspection', [py, 'scripts/diagnostics/inspect_lego_data.py',
        '--data', str(data/'Lego_data'), '--output', str(lego/'inspection')],
        [lego/'inspection/inspection.json']))
    for version, source in [('original', 'Lego.ply'), ('corrected', 'fan_contrastive_g0/contrastive_g0.ply')]:
        cmd = [py, 'scripts/motion/bind_lego_parts.py', '--data', str(data/'Lego_data'),
            '--parts', str(parts), '--inspection', str(lego/'inspection/inspection.json'),
            '--source-ply', str(data/'Lego_data'/source), '--output', str(lego/f'{version}-binding.npz')]
        if version == 'corrected':
            cmd += ['--registration', str(lego/'original-binding.json')]
        jobs.append(('lego-'+version+'-binding', cmd, [lego/f'{version}-binding.npz']))
    return jobs


def run_jobs(jobs, output):
    journal_path = output/'build-journal.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
        'format': 'ui_rebuild_journal_v1', 'jobs': {},
        'scope': 'New data-to-UI source assets; not completed V1/V2/V3 evaluation'}
    for name, cmd, products in jobs:
        previous = journal['jobs'].get(name)
        if previous and previous['status'] == 'complete':
            if previous['command'] != cmd or any(
                not Path(p).is_file() or sha256(p) != h
                for p, h in previous['products'].items()):
                raise ValueError('Completed job identity changed: '+name)
            print('Verified completed:', name, flush=True)
            continue
        for product in products:
            product.parent.parent.mkdir(parents=True, exist_ok=True)
        (output/'logs').mkdir(parents=True, exist_ok=True)
        record = dict(command=cmd, status='running',
                      started=datetime.datetime.now(datetime.timezone.utc).isoformat())
        journal['jobs'][name] = record
        save_json(journal_path, journal)
        print('Running:', name, flush=True)
        with (output/'logs'/f'{name}.log').open('a') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        record['returncode'] = result.returncode
        record['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['status'] = 'complete' if result.returncode == 0 else 'failed'
        if result.returncode == 0:
            record['products'] = {str(p): sha256(p) for p in products}
        save_json(journal_path, journal)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {output}/logs/{name}.log')


def quality_commands(data, output, blender):
    py = sys.executable
    jobs = []
    lamp = output/'lamp'
    base = lamp/'supplied-pair'
    jobs.append(('lamp-surface', [py, 'scripts/lamp/refine_lamp_surface.py',
        '--data', str(data/'lamp_data'), '--parts', str(lamp/'parts.npz'),
        '--binding-report', str(base/'binding-report.json'),
        '--output', str(lamp/'surface'), '--split', '4'], [lamp/'surface/report.json']))
    jobs.append(('lamp-package', [py, 'scripts/lamp/package_lamp_surface.py',
        '--baseline', str(base/'manifest.json'), '--candidate', str(lamp/'surface'),
        '--parts', str(lamp/'parts.npz'), '--binding-report', str(base/'binding-report.json'),
        '--output', str(lamp/'surface-pair')], [lamp/'surface-pair/manifest.json']))
    jobs.append(('lamp-detail', [py, 'scripts/lamp/prepare_lamp_detail.py',
        '--manifest', str(lamp/'surface-pair/manifest.json'), '--parts', str(lamp/'parts.npz'),
        '--binding-report', str(base/'binding-report.json'), '--blend', str(data/'lamp_data/lamp.blend'),
        '--blender', blender, '--output', str(lamp/'detail')], [lamp/'detail/manifest.json']))
    for scene, folder, stem in [('metronome','metronome_data','metronome'),
                                ('microphone','micphone_data','micphone')]:
        dest = output/scene
        base = dest/'supplied-pair'
        jobs.append((scene+'-surface-export', [blender, '-b', '--disable-autoexec',
            str(data/folder/(stem+'.blend')), '--python', 'scripts/lamp/export_object_surface.py',
            '--', '--parts', str(dest/'parts.npz'), '--binding-report', str(base/'binding-report.json'),
            '--output', str(dest/'source-surface.npz')], [dest/'source-surface.npz']))
        jobs.append((scene+'-surface-refine', [py, 'scripts/lamp/refine_object_surface.py',
            '--surface', str(dest/'source-surface.npz'), '--baseline', str(base/'manifest.json'),
            '--parts', str(dest/'parts.npz'), '--binding-report', str(base/'binding-report.json'),
            '--output', str(dest/'surface'), '--spacing', '.0015'], [dest/'surface/manifest-detail.json']))
    return jobs


def reference_commands(data, output, blender):
    jobs = []
    for scene, folder, stem, events in [
        ('lego','Lego_data','Lego','Lego_event_video.mp4'),
        ('fan','Fan_data','Fan','Fan_events_video.mp4'),
        ('lamp','lamp_data','lamp','lamp_events.mp4'),
        ('metronome','metronome_data','metronome','metronome_events.mp4'),
        ('microphone','micphone_data','micphone','micphone_event.mp4')]:
        dest = output/scene/'references'
        command = [sys.executable,'scripts/viewer/prepare_reference_videos.py',
            '--rgb',str(data/folder/(stem+'.mkv')),'--events',str(data/folder/events),
            '--output',str(dest),'--white-background']
        if scene == 'metronome': command += ['--rgb-background','69','69','69']
        jobs.append((scene+'-reference',command,[dest/'reference.json']))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--blender', required=True)
    p.add_argument('--only', nargs='+')
    p.add_argument('--stage', choices=('sources', 'quality', 'references'), default='sources')
    args = p.parse_args()
    factory = {'sources':commands, 'quality':quality_commands, 'references':reference_commands}[args.stage]
    jobs = factory(args.data.resolve(), args.output.resolve(), args.blender)
    if args.only:
        unknown = set(args.only)-{job[0] for job in jobs}
        if unknown: p.error('Unknown jobs: '+', '.join(sorted(unknown)))
        jobs = [job for job in jobs if job[0] in args.only]
    run_jobs(jobs, args.output.resolve()/('reference-journal') if args.stage == 'references' else args.output.resolve())


if __name__ == '__main__':
    main()
