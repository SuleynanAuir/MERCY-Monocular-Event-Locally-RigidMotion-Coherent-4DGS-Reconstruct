"""Run with Blender --background --factory-startup --disable-autoexec FILE --python SCRIPT.

Only reads the supplied scene; writes an inspection JSON outside the .blend.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def matrix(value):
    return [list(row) for row in value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scene = bpy.context.scene
    objects = []
    for obj in scene.objects:
        entry = {"name": obj.name, "type": obj.type, "parent": obj.parent.name if obj.parent else None,
                 "parent_type": obj.parent_type, "parent_bone": obj.parent_bone,
                 "hide_render": obj.hide_render, "modifiers": [{"name": m.name, "type": m.type}
                                                               for m in obj.modifiers],
                 "matrix_world": matrix(obj.matrix_world)}
        if obj.type == "MESH":
            entry.update(vertices=len(obj.data.vertices), polygons=len(obj.data.polygons),
                         vertex_groups=[g.name for g in obj.vertex_groups])
        if obj.type == "ARMATURE":
            entry["bones"] = [{"name": b.name, "parent": b.parent.name if b.parent else None,
                               "head": list(b.head), "tail": list(b.tail),
                               "constraints": [{"name": c.name, "type": c.type} for c in b.constraints]}
                              for b in obj.pose.bones]
        if obj.animation_data:
            entry["action"] = obj.animation_data.action.name if obj.animation_data.action else None
            entry["drivers"] = [{"path": d.data_path, "expression": d.driver.expression}
                                for d in obj.animation_data.drivers]
        objects.append(entry)
    report = {"blender": bpy.app.version_string, "file": bpy.data.filepath, "scene": scene.name,
              "frame_start": scene.frame_start, "frame_end": scene.frame_end,
              "fps": scene.render.fps / scene.render.fps_base,
              "camera": scene.camera.name if scene.camera else None, "objects": objects,
              "actions": [{"name": action.name, "range": list(action.frame_range)} for action in bpy.data.actions],
              "samples": []}
    frames = sorted(set([scene.frame_start, (scene.frame_start + scene.frame_end) // 4,
                         (scene.frame_start + scene.frame_end) // 2, scene.frame_end]))
    for frame in frames:
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        sample = {"frame": frame, "objects": [], "bones": []}
        for obj in scene.objects:
            if obj.type not in {"MESH", "ARMATURE", "CAMERA"}:
                continue
            evaluated = obj.evaluated_get(graph)
            record = {"name": obj.name, "matrix_world": matrix(evaluated.matrix_world)}
            if obj.type == "MESH" and not obj.hide_render:
                mesh = evaluated.to_mesh()
                points = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", points)
                points = points.reshape(-1, 3)
                transform = np.asarray(evaluated.matrix_world)
                world = points @ transform[:3, :3].T + transform[:3, 3]
                record["bounds"] = [world.min(0).tolist(), world.max(0).tolist()] if len(world) else []
                record["evaluated_vertices"] = len(world)
                record["probe_vertices"] = world[np.linspace(0, len(world) - 1, min(16, len(world)), dtype=int)].tolist()
                evaluated.to_mesh_clear()
            sample["objects"].append(record)
            if obj.type == "ARMATURE":
                sample["bones"].extend({"name": b.name, "matrix_world": matrix(evaluated.matrix_world @ b.matrix)}
                                       for b in evaluated.pose.bones)
        report["samples"].append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"objects", "samples"}}, indent=2))
    print("Inspected", len(objects), "objects; wrote", args.output)


main()
