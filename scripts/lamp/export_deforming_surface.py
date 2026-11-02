"""Export fixed-topology evaluated source motion without freezing deformation."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);s=bpy.context.scene
 if a.output.exists():raise FileExistsError(a.output)
 if bpy.context.preferences.filepaths.use_scripts_auto_execute:raise ValueError('Autoexec must be disabled')
 objects=[o for o in s.objects if o.type=='MESH' and not o.hide_render];tracks=[];topology=[];normals=[];groups=[];materials=[];names=[]
 for o in objects:
  for m in o.modifiers:
   if m.type=='SUBSURF':m.levels=m.render_levels
 for i in range(a.frames):
  s.frame_set(1+i*s.frame_step);g=bpy.context.evaluated_depsgraph_get();tris=[]
  cam=np.array(s.camera.evaluated_get(g).matrix_world)
  if i==0:camera=cam
  elif not np.allclose(cam,camera,atol=1e-6):raise ValueError('Camera moves')
  for j,o in enumerate(objects):
   ev=o.evaluated_get(g);mesh=ev.to_mesh()
   try:
    mesh.calc_loop_triangles();xyz=np.array([v.co[:] for v in mesh.vertices]);faces=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int32).reshape(-1,3);mat=np.array(ev.matrix_world)
    if i==0:topology.append(faces)
    elif not np.array_equal(faces,topology[j]):raise ValueError('Changing source topology')
    if not len(faces):continue
    world=xyz@mat[:3,:3].T+mat[:3,3];tris.append(world[faces])
    if i==0:
     loops=np.array([t.loops[:] for t in mesh.loop_triangles]);n=np.array([x.vector[:] for x in mesh.corner_normals])@np.linalg.inv(mat[:3,:3]);n/=np.linalg.norm(n,axis=1,keepdims=True);normals.append(n[loops]);groups.append(np.full(len(faces),j,np.int32));ids=[]
     for t in mesh.loop_triangles:
      m=mesh.materials[t.material_index] if t.material_index<len(mesh.materials) else None;label=m.name if m else 'unassigned'
      if label not in names:names.append(label)
      ids.append(names.index(label))
     materials.append(np.array(ids,np.int32))
   finally:ev.to_mesh_clear()
  tracks.append(np.concatenate(tris).astype(np.float32));print(i+1,a.frames,flush=True)
 projection=np.array(s.camera.calc_matrix_camera(g,x=s.render.resolution_x,y=s.render.resolution_y,scale_x=s.render.pixel_aspect_x,scale_y=s.render.pixel_aspect_y))
 meta=dict(blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),camera_pose=camera.tolist(),fps=s.render.fps/s.render.fps_base,frame_count=a.frames,source_frames=[1+i*s.frame_step for i in range(a.frames)],objects=[o.name for o in objects],materials=names,autoexec_disabled=True,method='Evaluated fixed-topology triangles including nonrigid deformation',camera_fovy_degrees=float(np.rad2deg(2*np.arctan(1/projection[1,1]))))
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,triangles=np.stack(tracks),normals=np.concatenate(normals),groups=np.concatenate(groups),materials=np.concatenate(materials),metadata_json=np.asarray(json.dumps(meta)))
if __name__=='__main__':main()
