"""Blender-only: ray-cast sampled visible part IDs without modifying the scene file."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def scene_tree(graph, names):
    vertices, faces, ids, offset = [], [], [], 0
    for label, name in enumerate(names):
        evaluated = bpy.data.objects[name].evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            xyz = np.empty(len(mesh.vertices) * 3, np.float32)
            mesh.vertices.foreach_get('co', xyz)
            matrix = np.asarray(evaluated.matrix_world)
            xyz = xyz.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
            triangles = np.empty(len(mesh.loop_triangles) * 3, np.int32)
            mesh.loop_triangles.foreach_get('vertices', triangles)
            vertices.extend(xyz.tolist())
            faces.extend((triangles.reshape(-1, 3) + offset).tolist())
            ids.extend([label] * len(mesh.loop_triangles))
            offset += len(xyz)
        finally:
            evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True), np.asarray(ids, np.int16)


def camera_rays(scene, graph, width, height):
    camera = scene.camera.evaluated_get(graph)
    projection = np.asarray(camera.calc_matrix_camera(
        graph, x=width, y=height, scale_x=1., scale_y=1.))
    inverse = np.linalg.inv(projection)
    matrix = np.asarray(camera.matrix_world)
    yy, xx = np.mgrid[:height, :width]
    ndc = np.stack((2 * (xx + .5) / width - 1, 1 - 2 * (yy + .5) / height,
                    -np.ones_like(xx), np.ones_like(xx)), -1)
    local = ndc @ inverse.T
    local = local[..., :3] / local[..., 3:]
    directions = local @ matrix[:3, :3].T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return Vector(matrix[:3, 3]), directions, projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stride < 1:
        raise ValueError('stride must be positive')
    with np.load(args.parts, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        source_frames = archive['source_frames'].copy()
        matrices = archive['object_matrices']
        moving = np.max(np.abs(matrices - matrices[:1]), axis=(0, 2, 3)) > 1e-5
    deforming = set(meta.get('deforming_objects', []))
    moving_rigid = [int(i) for i in np.flatnonzero(moving) if i not in deforming]
    indices = np.arange(0, len(source_frames), args.stride, dtype=np.int32)
    masks = np.full((len(indices), 640, 640), -1, np.int16)
    camera_poses, fovys = [], []
    scene = bpy.context.scene
    for position, index in enumerate(indices):
        scene.frame_set(int(source_frames[index]))
        graph = bpy.context.evaluated_depsgraph_get()
        tree, ids = scene_tree(graph, meta['objects'])
        origin, directions, projection = camera_rays(scene, graph, 640, 640)
        for y in range(640):
            for x in range(640):
                _, _, triangle, _ = tree.ray_cast(origin, Vector(directions[y, x]))
                if triangle is not None:
                    masks[position, y, x] = ids[triangle]
        camera_poses.append(np.asarray(scene.camera.evaluated_get(graph).matrix_world).tolist())
        fovys.append(float(np.rad2deg(2 * np.arctan(1 / projection[1, 1]))))
        print(json.dumps({'sample': position + 1, 'samples': len(indices),
                          'frame_index': int(index), 'source_frame': int(source_frames[index]),
                          'visible_pixels': int((masks[position] >= 0).sum())}), flush=True)
        del tree, ids
    metadata = {
        'format': 'lego_visible_part_sequence_v1',
        'source': bpy.data.filepath, 'source_frames': source_frames[indices].tolist(),
        'objects': meta['objects'], 'moving_rigid_parts': moving_rigid,
        'deforming_objects': sorted(deforming), 'resolution': [640, 640],
        'fovy_degrees': fovys, 'camera_poses': camera_poses,
        'parts_sha256': hashlib.sha256(args.parts.read_bytes()).hexdigest(),
        'scope': 'Blender-prior visible part ownership for independent 2D diagnostics only; '
                 'not event approval or evidence of hidden 3D completeness.',
        'autoexec_disabled': not bpy.context.preferences.filepaths.use_scripts_auto_execute,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, frame_indices=indices, part_ids=masks,
                        metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps({'output': str(args.output), 'sample_count': len(indices),
                      'moving_rigid_parts': len(moving_rigid)}), flush=True)


if __name__ == '__main__':
    main()
