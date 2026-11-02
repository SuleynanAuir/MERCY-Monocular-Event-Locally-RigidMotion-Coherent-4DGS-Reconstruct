"""Blender source-asset multiview supervision, never presented as captured RGB."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--size',type=int,default=384)
p.add_argument('--samples',type=int,default=64)
p.add_argument('--validation-only',action='store_true')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.output.exists():raise FileExistsError(a.output)
if bpy.context.preferences.filepaths.use_scripts_auto_execute:raise ValueError('Disable autoexec')
a.output.mkdir(parents=True)
s=bpy.context.scene;s.frame_set(1)
with np.load(a.parent,allow_pickle=False) as z:meta=json.loads(str(z['metadata_json']))
reg=np.asarray(meta['provenance']['registration']);scale=np.cbrt(np.linalg.det(reg[:3,:3]))
cam=s.camera;original=cam.matrix_world.copy();up=original.to_3x3()@Vector((0,1,0))
g=bpy.context.evaluated_depsgraph_get();points=[]
for obj in s.objects:
 if obj.type=='MESH' and not obj.hide_render:
  ev=obj.evaluated_get(g);points.extend([ev.matrix_world@Vector(c) for c in ev.bound_box])
points=np.asarray(points);target=Vector((points.min(0)+points.max(0))/2)
delta=original.translation-target;radius=delta.length
forward=delta.normalized();right=up.cross(forward).normalized();up=forward.cross(right).normalized()
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=a.samples;s.cycles.use_denoising=True
s.render.threads_mode='FIXED';s.render.threads=8
s.render.resolution_x=s.render.resolution_y=a.size;s.render.resolution_percentage=100
s.render.film_transparent=True
if hasattr(s.render.image_settings,'media_type'):s.render.image_settings.media_type='IMAGE'
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA'
# Preserve source color management, lighting and materials; record all settings.
rows=[]
views=[(yaw,pitch,'train') for yaw in range(0,360,60) for pitch in [-45,0,45]]
views += [(yaw,0,'validation') for yaw in range(30,360,60)]
if a.validation_only:views=[(yaw,pitch,'validation') for yaw in (45,225) for pitch in (-70,70)]
total=sum(3 if split=='train' else 2 for _,_,split in views)
for vi,(yaw,pitch,split) in enumerate(views):
 theta,phi=math.radians(yaw),math.radians(pitch)
 direction=(forward*math.cos(theta)+right*math.sin(theta))*math.cos(phi)+up*math.sin(phi)
 position=target+radius*direction
 pose=(target-position).to_track_quat('-Z','Y').to_matrix().to_4x4();pose.translation=position
 # Disable camera animation/constraints only; source object motion stays intact.
 cam.animation_data_clear()
 for c in list(cam.constraints):cam.constraints.remove(c)
 for frame in ([0,12,23] if split=='train' else [6,18]):
  s.frame_set(1+frame*s.frame_step);cam.matrix_world=pose
  name=f'v{vi:02d}-f{frame:02d}.png';s.render.filepath=str(a.output/name);bpy.ops.render.render(write_still=True)
  c=reg@np.asarray(cam.matrix_world);c[:3,:3]/=scale
  deps=bpy.context.evaluated_depsgraph_get();proj=np.asarray(cam.calc_matrix_camera(deps,x=a.size,y=a.size,scale_x=1,scale_y=1))
  f=float(proj[1,1]*a.size/2)
  rows.append(dict(frame=frame,time_seconds=frame/30,rgb=name,view=f'{yaw}_{pitch}',split=split,camera=dict(pose_opengl_c2w=c.tolist(),width=a.size,height=a.size,K=[[f,0,a.size/2],[0,f,a.size/2],[0,0,1]])))
  record=dict(format='scgs_synthetic_multiview_v1',samples=rows,background_rgb=[0,0,0],provenance=dict(kind='synthetic source Blender asset, not captured RGB',blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),samples=a.samples,denoising=True,view_transform=s.view_settings.view_transform,look=s.view_settings.look,engine=s.render.engine))
  (a.output/'source.json').write_text(json.dumps(record,indent=2));print('MULTIVIEW',len(rows),total,flush=True)
