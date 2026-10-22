"""Export evaluated rest surfaces and sampled part motion; run inside Blender.

The .blend is read with automatic embedded Python execution disabled. No scene
files are saved. This exports source animation, not newly invented keyframes.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def points(obj, graph, densify=False):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    xyz = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", xyz)
    xyz = xyz.reshape(-1, 3)
    transform = np.asarray(evaluated.matrix_world, dtype=np.float64)
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    if densify:
        # Vertex-only nearest neighbors miss the interiors of large flat bucket
        # faces. Add deterministic uniform surface samples on coarse triangles.
        mesh.calc_loop_triangles()
        triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', triangles)
        triangle = xyz[triangles.reshape(-1, 3)]
        area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
        count = np.ceil(area / .000025).astype(np.int32)
        count[area < .00005] = 0
        if count.sum() > 2000000:
            raise ValueError(f'Unexpectedly large visible surface: {obj.name}')
        repeated = np.repeat(np.arange(len(triangle)), count)
        rng = np.random.default_rng(0)
        uv = rng.random((len(repeated), 2))
        flip = uv.sum(1) > 1
        uv[flip] = 1 - uv[flip]
        selected = triangle[repeated]
        samples = selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
        xyz = np.concatenate((xyz, samples))
    evaluated.to_mesh_clear()
    return xyz.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = bpy.context.scene
    scene.frame_set(args.start)
    graph = bpy.context.evaluated_depsgraph_get()
    camera = np.asarray(scene.camera.evaluated_get(graph).matrix_world, dtype=np.float64)
    camera_inverse = np.linalg.inv(camera)
    # Object.hide_render alone misses objects in hidden asset-library
    # collections. Those prototypes must never participate in attachment.
    render_objects = set()
    def collect(layer):
        if layer.exclude or layer.collection.hide_render:
            return
        render_objects.update(obj.name for obj in layer.collection.objects)
        for child in layer.children:
            collect(child)
    collect(bpy.context.view_layer.layer_collection)
    objects, geometry, labels, deforming = [], [], [], []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render or obj.name not in render_objects:
            continue
        evaluated = obj.evaluated_get(graph)
        corners = np.asarray(evaluated.bound_box) @ np.asarray(evaluated.matrix_world)[:3, :3].T + np.asarray(evaluated.matrix_world)[:3, 3]
        view = corners @ camera_inverse[:3, :3].T + camera_inverse[:3, 3]
        # Exclude off-camera library prototypes, lights and background planes.
        # The complete excavator fits within this generous first-frame frustum.
        if np.min(-view[:, 2]) > 8 or np.max(-view[:, 2]) < .01:
            continue
        projected = view[:, :2] / np.maximum(-view[:, 2:3], .01)
        if np.any(projected.min(0) > .6) or np.any(projected.max(0) < -.6):
            continue
        elastic = any(m.type == "ARMATURE" for m in obj.modifiers)
        xyz = points(obj, graph, densify=not elastic)
        if not len(xyz):
            continue
        label = len(objects)
        objects.append(obj); geometry.append(xyz); labels.append(np.full(len(xyz), label, np.int16))
        if elastic:
            deforming.append(label)
    source_frames = args.start + np.arange(args.frames) * args.step
    if source_frames[-1] > scene.frame_end:
        raise ValueError("Sampling exceeds source animation")
    matrices = np.empty((args.frames, len(objects), 4, 4), dtype=np.float32)
    deformed = {label: np.empty((args.frames, len(geometry[label]), 3), dtype=np.float32) for label in deforming}
    for i, frame in enumerate(source_frames):
        scene.frame_set(int(frame)); graph = bpy.context.evaluated_depsgraph_get()
        for label, obj in enumerate(objects):
            matrices[i, label] = np.asarray(obj.evaluated_get(graph).matrix_world)
        for label in deforming:
            deformed[label][i] = points(objects[label], graph)
        if i % 50 == 0:
            print(f"Exported source animation {i}/{args.frames}, Blender frame {frame}", flush=True)
    metadata = {"source": bpy.data.filepath, "blender": bpy.app.version_string,
                "source_scene_fps": scene.render.fps / scene.render.fps_base,
                "source_frame_start": args.start, "source_frame_step": args.step,
                "frame_count": args.frames, "playback_fps": 60,
                "camera_pose": camera.tolist(), "objects": [obj.name for obj in objects],
                "deforming_objects": deforming, "autoexec_disabled": True,
                "note": "Sampling is matched to the supplied 400-frame 60 FPS video, not the scene's nominal 30 FPS."}
    arrays = {"rest_points": np.concatenate(geometry), "point_parts": np.concatenate(labels),
              "object_matrices": matrices, "source_frames": source_frames,
              "metadata_json": np.asarray(json.dumps(metadata)),
              "part_offsets": np.cumsum([0] + [len(x) for x in geometry])}
    arrays.update({f"deformed_{label}": value for label, value in deformed.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print("Saved", args.output, "objects", len(objects), "vertices", len(arrays["rest_points"]),
          "deforming", [(objects[k].name, v.shape) for k, v in deformed.items()], flush=True)


main()
