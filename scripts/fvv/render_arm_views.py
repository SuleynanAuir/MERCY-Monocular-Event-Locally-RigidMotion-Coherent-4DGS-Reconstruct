"""Blender-only full arm angular references, with explicit source provenance.

Read-only scene; source lighting retained; non-arm camera occlusion disabled. Cameras target the
entire articulated arm bounding sphere, not a favorable projected face.
"""
import argparse,json,sys,time,math,shutil
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','binding','dataset','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',default=[0,99,199,299,398]);p.add_argument('--reuse',type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
    if len(a.frames)*24>128 or len(set(a.frames))!=len(a.frames) or any(i<0 or i>398 for i in a.frames):raise ValueError('Reference frame budget exceeded')
    reused={}
    if a.reuse:
        previous=json.loads((a.reuse/'views.json').read_text())
        if previous.get('isolated_arm') is not True or previous['samples']!=64 or previous['source']!=bpy.data.filepath:raise ValueError('Reuse provenance mismatch')
        reused={v['file']:v for v in previous['views']}
    with np.load(a.source,allow_pickle=False) as d:points=d['points'];tracks=d['tracks'];source_meta=json.loads(str(d['metadata_json']));groups=d['groups']
    center=(points.min(0)+points.max(0))/2;radius=float(np.linalg.norm(points-center,axis=1).max())
    with np.load(a.binding,allow_pickle=False) as d:registration=np.asarray(json.loads(str(d['metadata_json']))['initial_registration'])
    provenance=json.loads((a.dataset/'provenance.json').read_text());transform=np.asarray(provenance['world_to_camera_cv'])@registration
    scale=np.cbrt(np.linalg.det(transform[:3,:3]));scene=bpy.context.scene
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='CUDA';prefs.get_devices()
    for device in prefs.devices:device.use=device.type=='CUDA'
    arm_names={p['name'] for p in [part for group in source_meta['groups'] for part in group['parts']]}
    for obj in scene.objects:
        if obj.type=='MESH' and obj.name not in arm_names:obj.visible_camera=False
    scene.cycles.device='GPU';scene.cycles.samples=64;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=8;scene.render.use_persistent_data=True
    scene.render.image_settings.media_type='IMAGE';scene.render.image_settings.file_format='PNG'
    scene.render.resolution_percentage=100;scene.render.resolution_x=640;scene.render.resolution_y=640
    fov=float(provenance['fov_degrees'])
    # A source camera can retain animation/constraints/parent transforms.
    # Use a fresh offline-only camera; never save the scene.
    camera_data=bpy.data.cameras.new('ArmAuditCamera');camera_data.type='PERSP'
    camera_data.sensor_fit='HORIZONTAL';camera_data.angle=math.radians(fov)
    camera_data.shift_x=0;camera_data.shift_y=0;camera_data.clip_start=.001
    camera_object=bpy.data.objects.new('ArmAuditCamera',camera_data);scene.collection.objects.link(camera_object);scene.camera=camera_object
    distance=radius/math.sin(math.radians(fov)/2)*1.25
    rows=[];started=time.monotonic()
    # Middle time and intermediate azimuths are RGB validation only.
    for frame in a.frames:
        scene.frame_set(1+frame*10)
        posed=np.einsum('nij,nj->ni',tracks[frame,groups,:3,:3],points)+tracks[frame,groups,:3,3]
        target=(posed.min(0)+posed.max(0))/2;radius=float(np.linalg.norm(posed-target,axis=1).max())
        distance=radius/math.sin(math.radians(fov)/2)*1.25;track=np.eye(4)
        for elevation in [-65.,0.,65.]:
            for azimuth in range(-180,180,45):
                angle=azimuth+(22.5 if frame==199 else 0.)
                theta,phi=map(math.radians,[angle,elevation])
                offset=np.array([math.sin(theta)*math.cos(phi),-math.sin(phi),-math.cos(theta)*math.cos(phi)])
                # World-oriented orbit about the complete articulated arm extent;
                # recenter every posture, never frame just one projected segment.
                position=target+track[:3,:3]@offset*distance;forward=(target-position)/distance
                down=track[:3,:3]@np.array([0.,1.,0.]);right=np.cross(down,forward);right/=np.linalg.norm(right)
                down=np.cross(forward,right);pose=np.eye(4);pose[:3,:3]=np.stack([right,down,forward],axis=1);pose[:3,3]=position
                world=pose.copy();world[:3,:3]*=scale
                scene.camera.matrix_world=Matrix((np.linalg.inv(transform)@world@np.diag([1,-1,-1,1])).tolist())
                bpy.context.view_layer.update()
                actual=transform@np.asarray(scene.camera.matrix_world)@np.diag([1,-1,-1,1]);actual[:3,:3]/=scale
                if not np.allclose(actual,pose,atol=1e-5):raise ValueError('Reference camera transform changed')
                name=f'{frame:03d}-{elevation:+.0f}-{angle:+.1f}.png';scene.render.filepath=str(a.output/name)
                prior=reused.get(name)
                if prior and prior['time']==frame/60 and prior['fov']==fov and np.allclose(prior['c2w'],pose,atol=1e-6):
                    shutil.copyfile(a.reuse/name,a.output/name)
                else:bpy.ops.render.render(write_still=True)
                rows.append(dict(file=name,time=frame/60,position=position.tolist(),c2w=pose.tolist(),fov=fov,distance=0.,angle=angle,elevation=elevation,split='validation' if frame==199 else 'train'))
                meta=dict(schema=1,source=bpy.data.filepath,source_arm_parts=[part for group in source_meta['groups'] for part in group['parts']],isolated_arm=True,scope='Supplied source arm only to camera; other meshes retain shadow/light interaction; whole arm centered; not captured multi-camera truth',samples=64,expected_views=len(a.frames)*24,center=center.tolist(),radius=radius,orbit_distance=distance,views=rows,elapsed_seconds=time.monotonic()-started)
                (a.output/'views.json').write_text(json.dumps(meta,indent=2));print('ARM_VIEW_DONE',len(rows),flush=True)
                if time.monotonic()-started>7200 or sum(f.stat().st_size for f in a.output.iterdir())>128*1024**2:raise RuntimeError('Arm reference budget exceeded')


if __name__=='__main__':main()
