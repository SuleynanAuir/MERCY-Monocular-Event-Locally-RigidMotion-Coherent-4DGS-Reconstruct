"""Export evaluated source triangles, smooth normals and material identities.

Run inside Blender with --disable-autoexec. Source geometry is an explicit prior,
not a surface inferred from event data. Source files are never saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parts','binding-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists(): raise FileExistsError(args.output)
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise ValueError('Embedded scripts must remain disabled')
    with np.load(args.parts,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json'])); first=int(z['source_frames'][0])
    binding=json.loads(args.binding_report.read_text())
    if sha(bpy.data.filepath)!=meta['blend_sha256'] or sha(args.parts)!=binding['parts_sha256']:
        raise ValueError('Source geometry/motion identity mismatch')
    bpy.context.scene.frame_set(first)
    reg=np.array(binding['registration'])
    triangles=[]; normals=[]; groups=[]; materials=[]; material_names=[]
    for group,name in enumerate(meta['objects']):
        obj=bpy.data.objects[name]
        for modifier in obj.modifiers:
            if modifier.type=='SUBSURF': modifier.levels=modifier.render_levels
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            vertices=np.array([v.co[:] for v in mesh.vertices]);faces=np.array([t.vertices[:] for t in mesh.loop_triangles])
            loops=np.array([t.loops[:] for t in mesh.loop_triangles]);n=np.array([x.vector[:] for x in mesh.corner_normals])
            matrix=reg@np.array(evaluated.matrix_world)
            vertices=vertices@matrix[:3,:3].T+matrix[:3,3]
            n=n@np.linalg.inv(matrix[:3,:3]);n/=np.linalg.norm(n,axis=1,keepdims=True)
            triangles.append(vertices[faces]);normals.append(n[loops]);groups.append(np.full(len(faces),group,np.int16))
            ids=[]
            for t in mesh.loop_triangles:
                material=mesh.materials[t.material_index] if t.material_index<len(mesh.materials) else None
                label=material.name if material else 'unassigned'
                if label not in material_names:material_names.append(label)
                ids.append(material_names.index(label))
            materials.append(np.array(ids,np.int16));print(name,len(faces),flush=True)
        finally:evaluated.to_mesh_clear()
    record=dict(format='object_surface_prior_v1',parts_sha256=sha(args.parts),blend_sha256=meta['blend_sha256'],
        binding_sha256=sha(args.binding_report),objects=meta['objects'],materials=material_names,
        registration=reg.tolist(),source_frame=first,autoexec_disabled=True,
        limitation='Existing Blender geometry prior; not reconstructed from events')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,triangles=np.concatenate(triangles),normals=np.concatenate(normals),
        groups=np.concatenate(groups),materials=np.concatenate(materials),metadata_json=np.asarray(json.dumps(record)))


if __name__=='__main__':main()
