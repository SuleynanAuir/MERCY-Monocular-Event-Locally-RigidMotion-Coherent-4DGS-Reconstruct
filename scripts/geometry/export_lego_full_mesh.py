"""Blender-only: export the complete evaluated prior, including unseen faces.

Read the supplied .blend with --disable-autoexec. Never save or edit it.
Unlike the observed-surface exporter, this does not call unseen surfaces data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def material_gray(material):
    color = np.array(material.diffuse_color[:3] if material else [.7]*3)
    if material and material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                color = np.array(node.inputs['Base Color'].default_value[:3])
                break
    # A bounded material prior, not a measured back-side texture or RGB target.
    return float(np.clip(color @ [.299, .587, .114], .05, .9))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parts', 'binding', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.parts, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        first = int(archive['source_frames'][0])
    with np.load(args.binding, allow_pickle=False) as archive:
        registration = archive['registration'].astype(np.float64)
    bpy.context.scene.frame_set(first)
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles, labels, colors = [], [], [], []
    rows, offset = [], 0
    for part, name in enumerate(meta['objects']):
        obj = bpy.data.objects[name].evaluated_get(graph)
        mesh = obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            xyz = np.empty(len(mesh.vertices)*3, np.float32)
            mesh.vertices.foreach_get('co', xyz)
            transform = registration @ np.asarray(obj.matrix_world)
            xyz = xyz.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
            faces = np.empty(len(mesh.loop_triangles)*3, np.int32)
            mesh.loop_triangles.foreach_get('vertices', faces)
            faces = faces.reshape(-1, 3)
            materials = np.empty(len(faces), np.int32)
            mesh.loop_triangles.foreach_get('material_index', materials)
            palette = np.array([material_gray(m) for m in mesh.materials] or [.7])
            gray = palette[np.clip(materials, 0, len(palette)-1)]
            vertices.append(xyz.astype(np.float32)); triangles.append(faces+offset)
            labels.append(np.full(len(faces), part, np.int32)); colors.append(gray.astype(np.float32))
            rows.append({'part_id': part, 'name': name, 'vertices': len(xyz), 'triangles': len(faces),
                         'deforming': part in meta['deforming_objects']})
            offset += len(xyz)
        finally:
            obj.to_mesh_clear()
        if part % 50 == 0:
            print('Exporting complete mesh', part, '/', len(meta['objects']), flush=True)
    info = {'format': 'lego_complete_mesh_prior_v1', 'source': bpy.data.filepath,
            'source_sha256': sha256(bpy.data.filepath), 'source_frame': first,
            'parts': str(args.parts), 'parts_sha256': sha256(args.parts),
            'binding': str(args.binding), 'binding_sha256': sha256(args.binding),
            'registration': registration.tolist(), 'objects': meta['objects'],
            'deforming_objects': meta['deforming_objects'], 'parts_detail': rows,
            'coordinate_system': 'Canonical Gaussian world at source frame one',
            'autoexec_disabled': not bpy.context.preferences.filepaths.use_scripts_auto_execute,
            'scope': 'Full allowed Blender mesh PRIOR, including unobserved surfaces. '
                     'Geometry/materials are not recovered back-side observations; no event acceptance is implied.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vertices=np.concatenate(vertices), triangles=np.concatenate(triangles),
                        part_ids=np.concatenate(labels), material_gray=np.concatenate(colors),
                        metadata_json=np.asarray(json.dumps(info)))
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'vertices': offset,
                      'triangles': sum(len(x) for x in triangles)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
