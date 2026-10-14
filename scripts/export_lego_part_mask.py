"""Blender: ray-cast ONLY the initial frame into existing part IDs. No scene save."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.parts, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
    scene = bpy.context.scene
    scene.frame_set(meta['source_frame_start'])
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, faces, ids, offset = [], [], [], 0
    for label, name in enumerate(meta['objects']):
        evaluated = bpy.data.objects[name].evaluated_get(graph)
        mesh = evaluated.to_mesh(); mesh.calc_loop_triangles()
        xyz = np.empty(len(mesh.vertices)*3, np.float32)
        mesh.vertices.foreach_get('co', xyz)
        matrix = np.asarray(evaluated.matrix_world)
        xyz = xyz.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = np.empty(len(mesh.loop_triangles)*3, np.int32)
        mesh.loop_triangles.foreach_get('vertices', triangles)
        vertices.extend(xyz.tolist()); faces.extend((triangles.reshape(-1,3)+offset).tolist())
        ids.extend([label]*len(mesh.loop_triangles)); offset += len(xyz)
        evaluated.to_mesh_clear()
    tree = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
    del vertices, faces
    camera = scene.camera.evaluated_get(graph)
    projection = np.asarray(camera.calc_matrix_camera(graph, x=640, y=640, scale_x=1., scale_y=1.))
    inverse = np.linalg.inv(projection)
    matrix = np.asarray(camera.matrix_world)
    yy, xx = np.mgrid[:640, :640]
    ndc = np.stack((2*(xx+.5)/640-1, 1-2*(yy+.5)/640, -np.ones_like(xx), np.ones_like(xx)), -1)
    local = ndc @ inverse.T
    local = local[..., :3] / local[..., 3:]
    directions = local @ matrix[:3, :3].T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origin = Vector(matrix[:3, 3])
    mask, depth = np.full((640,640), -1, np.int16), np.zeros((640,640), np.float32)
    for y in range(640):
        for x in range(640):
            location, normal, triangle, distance = tree.ray_cast(origin, Vector(directions[y,x]))
            if triangle is not None:
                mask[y,x], depth[y,x] = ids[triangle], distance
        if y % 160 == 0:
            print('Initial part-mask row', y, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'source': bpy.data.filepath, 'source_frame': meta['source_frame_start'],
                'objects': meta['objects'], 'resolution': [640,640],
                'fovy_degrees': float(np.rad2deg(2*np.arctan(1/projection[1,1]))),
                'scope': 'Blender initial-frame part initialization only; no event approval.'}
    np.savez_compressed(args.output, part_ids=mask, ray_distance=depth, metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps(metadata | {'objects': len(meta['objects']), 'foreground_pixels': int((mask >= 0).sum())}), flush=True)


main()
