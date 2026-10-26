"""Blender CLI: export original evaluated geometry for shared reference labels.

blender -b source.blend --disable-autoexec --python this.py -- --parts parts.npz --output mesh.npz
Only supports verified rigid source objects; elastic sources use their existing
full mesh/animation export. Never reads or modifies a method's reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists(): raise FileExistsError(args.output)
    with np.load(args.parts, allow_pickle=False) as parts:
        meta = json.loads(str(parts['metadata_json']))
        if meta.get('maximum_local_geometry_error', 1) > 1e-5:
            raise ValueError('Need verified rigid source topology')
        if hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest() != meta['blend_sha256']:
            raise ValueError('Original Blender source checksum mismatch')
        bpy.context.scene.frame_set(int(parts['source_frames'][0]))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        vertices, faces, details, offset = [], [], [], 0
        for i, name in enumerate(meta['objects']):
            obj = bpy.data.objects[name].evaluated_get(depsgraph)
            mesh = obj.to_mesh(); mesh.calc_loop_triangles()
            xyz = np.asarray([obj.matrix_world @ v.co for v in mesh.vertices], np.float32)
            triangles = np.asarray([tuple(t.vertices) for t in mesh.loop_triangles], np.int32)
            if not len(xyz) or not len(triangles): raise ValueError('Empty source object')
            vertices.append(xyz); faces.append(triangles+offset)
            details.append(dict(part_id=i, name=name, vertices=len(xyz))); offset += len(xyz)
            obj.to_mesh_clear()
        result = dict(format='source_reference_mesh_v1', source_sha256=meta['blend_sha256'],
                      parts_sha256=hashlib.sha256(args.parts.read_bytes()).hexdigest(), objects=meta['objects'],
                      registration=np.eye(4).tolist(), parts_detail=details, deforming_objects=[])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, vertices=np.concatenate(vertices), triangles=np.concatenate(faces),
                            metadata_json=np.asarray(json.dumps(result)))


if __name__ == '__main__': main()
