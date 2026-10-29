"""Blender-only: bounded RGB views from the supplied animated Lego scene.

Run with --disable-autoexec. Never saves or changes the source .blend file.
Generated source-scene supervision is explicitly separate from observed video.
"""
import argparse,json,sys,time,shutil
from pathlib import Path
import bpy
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['binding','dataset','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--times',type=int,default=40);p.add_argument('--samples',type=int,default=32)
    p.add_argument('--reuse',type=Path)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--distances',type=float,nargs='+',default=[.5,1.5])
    p.add_argument('--frames',type=int,nargs='+')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not 2<=a.times<=100 or not 16<=a.samples<=128:raise ValueError('Export exceeds budget')
    if not all(np.isfinite(d) and 0<=d<=3 for d in a.distances):raise ValueError('Distances must be finite and within 0..3')
    reuse_views={}
    if a.reuse:
        reuse_manifest=json.loads((a.reuse/'views.json').read_text())
        if reuse_manifest['source']!=bpy.data.filepath or reuse_manifest['samples']!=a.samples:
            raise ValueError('Reuse source or sample count differs')
        reuse_views={v['file']:v for v in reuse_manifest['views']}
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.binding,allow_pickle=False) as f:registration=np.array(json.loads(str(f['metadata_json']))['initial_registration'])
    provenance=json.loads((a.dataset/'provenance.json').read_text())
    transform=np.array(provenance['world_to_camera_cv'])@registration
    unit_scale=np.cbrt(np.linalg.det(transform[:3,:3]))
    scene=bpy.context.scene;scene.frame_set(1);original=scene.camera.matrix_world.copy()
    # CUDA rendering is isolated from the existing PyTorch environments.
    if not a.cpu:
        preferences=bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type='CUDA';preferences.get_devices()
        for device in preferences.devices:device.use=device.type=='CUDA'
    scene.cycles.device='CPU' if a.cpu else 'GPU'
    scene.render.threads_mode='FIXED';scene.render.threads=8
    scene.cycles.samples=a.samples;scene.cycles.use_denoising=True
    scene.render.use_persistent_data=True
    scene.render.image_settings.media_type='IMAGE';scene.render.image_settings.file_format='PNG'
    scene.render.resolution_percentage=100;scene.render.resolution_x=640;scene.render.resolution_y=640
    records=[];start=time.monotonic()
    indices=a.frames if a.frames else np.linspace(0,398,a.times).round().astype(int)
    if not all(0<=i<=398 for i in indices) or len(indices)*len(a.distances)>200:raise ValueError('View budget exceeded')
    for k,index in enumerate(indices):
        scene.frame_set(1+int(index)*10)
        for j,distance in enumerate(a.distances):
            scene.camera.matrix_world=original.copy()
            scene.camera.location+=original.col[2].xyz*(distance/unit_scale)
            name=f'{index:03d}-{j}.png';scene.render.filepath=str(a.output/name)
            reused=a.reuse/name if a.reuse else None
            previous=reuse_views.get(name,{})
            if reused and reused.is_file() and previous.get('distance')==distance and previous.get('time')==float(index/60):
                shutil.copyfile(reused,a.output/name)
            else:
                bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            c2w=transform@np.asarray(scene.camera.matrix_world)@np.diag([1,-1,-1,1])
            c2w[:3,:3]/=unit_scale
            records.append(dict(file=name,time=float(index/60),position=c2w[:3,3].tolist(),
                c2w=c2w.tolist(),fov=provenance['fov_degrees'],split='validation' if a.validation_only or k%8==4 else 'train',distance=distance))
            (a.output/'views.json').write_text(json.dumps(dict(source=bpy.data.filepath,
                scope='rendered supplied source scene, NOT newly captured real camera images',samples=a.samples,expected_views=len(indices)*len(a.distances),
                elapsed_seconds=time.monotonic()-start,views=records),indent=2))
            print('VIEW_DONE',len(records),flush=True)
            if sum(f.stat().st_size for f in a.output.iterdir())>256*1024**2:raise OSError('View export exceeds 256 MiB')
            if time.monotonic()-start>30*60:raise TimeoutError('View export exceeded 30 minutes')

if __name__=='__main__':main()
