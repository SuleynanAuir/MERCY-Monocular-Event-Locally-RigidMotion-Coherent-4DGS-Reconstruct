"""Rebuild the articulated Lego and video-fitted Fan branches, never old assets."""
import argparse
from pathlib import Path
import sys

from .ui_rebuild import run_jobs


def jobs(data, root, blender, scene):
    py = sys.executable
    d = root/scene
    inventory = root/'input-inventory.json'
    result = []
    def add(name, command, products):
        result.append((scene+'-'+name, command, products))
    def stage(*args):
        return [py, '-m', 'experiment.ui_stages', *map(str,args)]
    if scene == 'lego':
        add('mesh-export',[blender,'-b','--disable-autoexec',str(data/'Lego_data/Lego.blend'),
            '--python','scripts/geometry/export_lego_full_mesh.py','--','--parts',str(d/'parts.npz'),
            '--binding',str(d/'corrected-binding.npz'),'--output',str(d/'source-mesh.npz')], [d/'source-mesh.npz'])
        add('v1-prior',[py,'-m','experiment.ui_lego','prepare','--root',str(d),
            '--source',str(data/'Lego_data/Lego.ply'),'--binding',str(d/'original-binding.npz'),
            '--inventory',str(inventory),'--output',str(d/'v1-prior')], [d/'v1-prior/prior.json'])
    else:
        add('v1-prior',[py,'-m','experiment.ui_fan','--root',str(d),'--data',str(data/'Fan_data'),
            '--inventory',str(inventory),'--version','v1','--output',str(d/'v1-prior')], [d/'v1-prior/prior.json'])
    add('v1-bake',stage('export','--prepared',d/'v1-prior','--template',d/'v1-prior/canonical-template.npz',
        '--method','C1','--output',d/'supplied-pair/left.npz'),[d/'supplied-pair/left.npz'])
    add('v2-train',stage('train','--prepared',d/'v1-prior','--method','C2','--steps','200',
        '--output',d/'v2-training'),[d/'v2-training/checkpoint.pt',d/'v2-training/evaluation.json'])
    add('v2-bake',stage('export','--prepared',d/'v1-prior','--template',d/'v1-prior/canonical-template.npz',
        '--checkpoint',d/'v2-training/checkpoint.pt','--method','C2','--output',d/'v2.npz'),[d/'v2.npz'])
    if scene == 'lego':
        add('surface',[py,'-m','experiment.ui_lego','surface','--root',str(d),
            '--source',str(data/'Lego_data/fan_contrastive_g0/contrastive_g0.ply'),
            '--output',str(d/'surface')],[d/'surface/surface.ply',d/'surface/binding.npz'])
        add('v3-prior',[py,'-m','experiment.ui_lego','prepare','--root',str(d),
            '--source',str(d/'surface/surface.ply'),'--binding',str(d/'surface/binding.npz'),
            '--inventory',str(inventory),'--reference-prior',str(d/'v1-prior/prior.json'),
            '--output',str(d/'v3-prior')],[d/'v3-prior/prior.json'])
        add('v3-train',stage('train','--prepared',d/'v3-prior','--method','C3',
            '--initial-alignment',d/'v2-training/checkpoint.pt','--freeze-alignment','--event-only',
            '--steps','200','--output',d/'v3-training'),[d/'v3-training/checkpoint.pt',d/'v3-training/evaluation.json'])
        display = d/'surface/display.json'
    else:
        add('v3-prior',[py,'-m','experiment.ui_fan','--root',str(d),'--data',str(data/'Fan_data'),
            '--inventory',str(inventory),'--version','v3','--reference-prior',str(d/'v1-prior/prior.json'),
            '--output',str(d/'v3-prior')],[d/'v3-prior/prior.json',d/'v3-prior/surface.ply'])
        add('inherit-v2',stage('inherit-without-events','--prepared',d/'v3-prior',
            '--parent',d/'v2-training/checkpoint.pt','--output',d/'v3-training/checkpoint.pt'),[d/'v3-training/checkpoint.pt'])
        display = d/'v3-prior/display.json'
    add('v3-bake',stage('export','--prepared',d/'v3-prior','--template',d/'v3-prior/canonical-template.npz',
        '--checkpoint',d/'v3-training/checkpoint.pt','--method','C3','--output',d/'v3.npz'),[d/'v3.npz'])
    add('package',stage('package','--root',d,'--name',scene.title(),'--surface-manifest',display),
        [d/'pair-v1-v2.json',d/'pair-v1-v3.json'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data','root'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--blender',required=True)
    p.add_argument('--scene',required=True,choices=('lego','fan'))
    p.add_argument('--only',nargs='+')
    args = p.parse_args()
    tasks = jobs(args.data.resolve(),args.root.resolve(),args.blender,args.scene)
    if args.only:
        names = {name for name, _, _ in tasks}
        if set(args.only)-names: p.error('Unknown job name')
        tasks = [job for job in tasks if job[0] in args.only]
    run_jobs(tasks,args.root.resolve()/(args.scene+'-journal'))


if __name__ == '__main__':
    main()
