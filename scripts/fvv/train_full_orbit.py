"""Offline full-orbit refinement of the supplied Lego scene, with rollback inputs.

Run in the Hybrid Python environment. Produces a candidate and evaluation;
never changes the active model. Source-scene images are not captured cameras.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

try:
    from .training_resources import pause, resume
except ImportError:
    from training_resources import pause, resume


def merge_showcase_views(wide, close, output):
    """Weight dense-time arc supervision twice without duplicating image data."""
    rows=[];links=[]
    for prefix,source,repeats in [('wide',wide,1),('arc',close,2)]:
        data=json.loads((source/'views.json').read_text())
        if len(data['views'])!=data['expected_views']:raise ValueError('Incomplete source views')
        for view in data['views']:
            file=(source/view['file']).resolve()
            if not file.is_relative_to(source.resolve()) or not file.is_file():raise ValueError('Invalid view file')
            name=prefix+'-'+file.name;links.append((name,file))
            rows.extend([dict(view,file=name)]*(repeats if view['split']=='train' else 1))
    if len(rows)>512:raise ValueError('Mixed supervision exceeds 512 view entries')
    output.mkdir(exist_ok=False)
    for name,file in links:(output/name).symlink_to(file)
    (output/'views.json').write_text(json.dumps(dict(expected_views=len(rows),views=rows,
        scope='explicit source-scene prior; dense-time arc training repeated twice; validation never repeated'),indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'repo', 'model', 'blender', 'blend', 'binding', 'video'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--registry', type=Path, action='append', default=[])
    p.add_argument('--steps', type=int, default=18000, choices=[6000, 12000, 18000, 24000])
    p.add_argument('--focus-showcase',action='store_true',help='100 timestamps at the calibrated 8-degree Lego arc')
    p.add_argument('--wide-views',type=Path,help='Existing full-orbit source views retained as a constraint')
    p.add_argument('--geometry',type=Path,help='Optional explicit animated surface prior')
    p.add_argument('--surface-normal-weight',type=float,default=0,choices=[0,1.,2.,5.],
                   help='Optional dynamic surface thickness constraint; requires a prior containing normals')
    a = p.parse_args()
    if a.focus_showcase and not a.wide_views:p.error('--focus-showcase requires --wide-views')
    if a.surface_normal_weight and not a.geometry:p.error('--surface-normal-weight requires --geometry')
    a.root.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    views, candidate = a.root / 'views', a.root / 'candidate'
    state = dict(started_at=time.time(), parent=str(a.model), steps=a.steps,
                 scope='supplied source-scene supervision; no captured novel-view claim')

    def stage(name, command):
        state.update(stage=name, updated_at=time.time())
        (a.root / 'status.json').write_text(json.dumps(state, indent=2))
        with (a.root / (name + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=125*60)

    records = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Training pipeline interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    # Same lock as UI training/export, held during reference generation too.
    lock = a.model.parent / '.fvv-writer.lock'
    with lock.open('x') as stream:
        stream.write(f'full orbit pipeline pid={os.getpid()}\n')
    try:
        pause(a.registry, records)
        reference_pose=(['--times','100','--distances','.15','--angles','8','--elevations','1.5'] if a.focus_showcase else
                        ['--times','24','--distances','.3','--angles','-180','-135','-90','-45','0','45','90','135','--elevations','-15','0','15','--cycle-elevations'])
        stage('references', [a.blender, '--background', '--disable-autoexec', a.blend,
              '--python', scripts / 'render_lego_multiview.py', '--',
              '--binding', a.binding, '--dataset', a.model.parent.parent / 'dataset-v1',
              '--output', views, '--samples', '32',*reference_pose])
        training_views=views
        if a.focus_showcase:
            training_views=a.root/'mixed-views'
            merge_showcase_views(a.wide_views,views,training_views)
        extra=(['--appearance-weight','4','--subframe-samples','2'] if a.focus_showcase else [])
        if a.geometry:extra+=['--geometry',a.geometry,'--geometry-weight','.5']
        if a.surface_normal_weight:extra+=['--surface-normal-weight',str(a.surface_normal_weight)]
        stage('training', [sys.executable, scripts / 'refine_lego.py', '--repo', a.repo,
              '--model', a.model, '--video', a.video,
              '--dataset', a.model.parent.parent / 'dataset-v1', '--output', candidate,
              '--steps', str(a.steps), '--max-minutes', '120', '--all-frames',
              '--multiview', training_views, '--multiview-weight', '3', '--multiview-batch', '2',
              '--train-static', '--position-lr', '1e-5' if a.focus_showcase else '3e-5', '--opacity-lr', '.003',
              '--edge-weight', '.5', '--temporal-weight', '4' if a.focus_showcase else '2',
              '--midpoint-weight', '2' if a.focus_showcase else '1',*extra])
        stage('orbit-evaluation', [sys.executable, scripts / 'check_multiview.py',
              '--repo', a.repo, '--previous', a.model, '--model', candidate,
              '--views', views, '--output', a.root / 'orbit-evaluation'])
        if a.focus_showcase:
            stage('wide-evaluation',[sys.executable,scripts/'check_multiview.py','--repo',a.repo,
                  '--previous',a.model,'--model',candidate,'--views',a.wide_views,'--output',a.root/'wide-evaluation'])
        stage('motion-evaluation', [sys.executable, scripts / 'check_refinement.py',
              '--repo', a.repo, '--previous', a.model, '--model', candidate,
              '--video', a.video, '--output', a.root / 'motion-evaluation'])
        state.update(stage='evaluated', candidate=str(candidate))
    except BaseException as error:
        state.update(stage='failed', error=str(error))
        raise
    finally:
        try:
            resume(records)
        except Exception as error:
            state.update(stage='restore_failed', error=str(error))
            raise
        finally:
            lock.unlink(missing_ok=True)
            state.update(finished_at=time.time())
            (a.root / 'status.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
