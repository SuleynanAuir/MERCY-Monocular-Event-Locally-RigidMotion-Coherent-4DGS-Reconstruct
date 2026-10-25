"""Transfer source normals within each verified Lamp part (run in Blender)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'parts', 'binding-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise ValueError('Run with --disable-autoexec')
    with np.load(args.parts, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        first_frame = int(archive['source_frames'][0])
    binding = json.loads(args.binding_report.read_text())
    if (metadata['blend_sha256'] != sha(bpy.data.filepath)
            or binding['parts_sha256'] != sha(args.parts)):
        raise ValueError('Source geometry and motion identities differ')
    with np.load(args.input, allow_pickle=False) as archive:
        xyz, groups = archive['xyz'], archive['groups']
    if (groups.shape != (len(xyz),) or groups.dtype.kind not in 'iu'
            or groups.min() < 0 or groups.max() >= len(metadata['objects'])
            or not np.isfinite(xyz).all()):
        raise ValueError('Invalid part-labelled Gaussian positions')
    bpy.context.scene.frame_set(first_frame)
    registration = np.array(binding['registration'])
    normals = np.empty_like(xyz)
    distances = np.empty(len(xyz), np.float32)
    for group, name in enumerate(metadata['objects']):
        obj = bpy.data.objects[name]
        for modifier in obj.modifiers:
            if modifier.type == 'SUBSURF':
                modifier.levels = modifier.render_levels
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            vertices = np.array([v.co[:] for v in mesh.vertices])
            faces = np.array([t.vertices[:] for t in mesh.loop_triangles])
            loops = np.array([t.loops[:] for t in mesh.loop_triangles])
            corner_normals = np.array([n.vector[:] for n in mesh.corner_normals])
            matrix = registration @ np.array(evaluated.matrix_world)
            vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            corner_normals = corner_normals @ np.linalg.inv(matrix[:3, :3])
            corner_normals /= np.linalg.norm(corner_normals, axis=1, keepdims=True)
            tree = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)
            ids = np.flatnonzero(groups == group)
            hits = np.empty((len(ids), 3)); face_ids = np.empty(len(ids), np.int32)
            for index, row in enumerate(ids):
                point, _, face, distance = tree.find_nearest(xyz[row])
                hits[index], face_ids[index], distances[row] = point, face, distance
            triangles = vertices[faces[face_ids]]
            u, v = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
            d = hits-triangles[:, 0]
            uu, uv, vv = (u*u).sum(1), (u*v).sum(1), (v*v).sum(1)
            du, dv = (d*u).sum(1), (d*v).sum(1)
            denominator = (uu*vv-uv*uv).clip(1e-20)
            b, c = (vv*du-uv*dv)/denominator, (uu*dv-uv*du)/denominator
            weights = np.stack([1-b-c, b, c], 1).clip(0, 1)
            weights /= weights.sum(1)[:, None]
            normals[ids] = (corner_normals[loops[face_ids]]*weights[:, :, None]).sum(1)
            normals[ids] /= np.linalg.norm(normals[ids], axis=1, keepdims=True)
            print(name, len(ids), flush=True)
        finally:
            evaluated.to_mesh_clear()
    record = dict(format='lamp_source_normal_projection_v1', input_sha256=sha(args.input),
        parts_sha256=sha(args.parts), blend_sha256=sha(bpy.data.filepath),
        binding_sha256=sha(args.binding_report), source_frame=first_frame,
        note='Only nearest-surface normal directions are transferred; Gaussian positions are unchanged.')
    np.savez_compressed(args.output, normals=normals, distance=distances,
                        metadata_json=np.asarray(json.dumps(record)))


if __name__ == '__main__':
    main()
