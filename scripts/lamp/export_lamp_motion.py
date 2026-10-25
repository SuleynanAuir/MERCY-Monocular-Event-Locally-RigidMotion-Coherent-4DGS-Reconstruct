"""Read Lamp's evaluated Blender parts and source animation; never save the blend.

Run with Blender --background --disable-autoexec lamp.blend --python ... --
--output parts.npz. Reject nonrigid geometry rather than silently freezing it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def evaluated_mesh(obj, graph):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        vertices = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get('co', vertices)
        mesh.calc_loop_triangles()
        faces = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get('vertices', faces)
        return vertices.reshape(-1, 3), faces.reshape(-1, 3), np.asarray(evaluated.matrix_world)
    finally:
        evaluated.to_mesh_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--name', default='Lamp', help='Object name for progress messages')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = bpy.context.scene
    frames = np.arange(scene.frame_start, scene.frame_end + 1, scene.frame_step)
    scene.frame_set(int(frames[0]))
    graph = bpy.context.evaluated_depsgraph_get()
    camera = np.asarray(scene.camera.evaluated_get(graph).matrix_world)
    projection = np.asarray(scene.camera.calc_matrix_camera(graph,
        x=scene.render.resolution_x, y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y))
    visible = set()
    def collect(layer):
        if layer.exclude or layer.collection.hide_render:
            return
        visible.update(o.name for o in layer.collection.objects)
        for child in layer.children:
            collect(child)
    collect(bpy.context.view_layer.layer_collection)
    objects = [o for o in scene.objects if o.type == 'MESH' and not o.hide_render and o.name in visible]
    if not objects:
        raise ValueError('No visible source meshes')
    rest, surfaces, groups, renderable, empty = [], [], [], [], []
    for obj in objects:
        # Match final-render subdivision, not interactive viewport quality.
        for modifier in obj.modifiers:
            if modifier.type == 'SUBSURF':
                modifier.levels = modifier.render_levels
        graph = bpy.context.evaluated_depsgraph_get()
        xyz, faces, matrix = evaluated_mesh(obj, graph)
        if len(faces) == 0:
            empty.append(obj)
            continue
        i = len(renderable)
        renderable.append(obj)
        rest.append((xyz, faces))
        world = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = world[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
        count = min(200000, max(1000, int(area.sum()/1e-5)))
        rng = np.random.default_rng(i)
        chosen = triangles[rng.choice(len(faces), count, p=area/area.sum())]
        uv = rng.random((count, 2)); uv[uv.sum(1) > 1] = 1-uv[uv.sum(1) > 1]
        samples = chosen[:, 0] + uv[:, :1]*(chosen[:, 1]-chosen[:, 0]) + uv[:, 1:]*(chosen[:, 2]-chosen[:, 0])
        surface = np.concatenate((world, samples))
        surfaces.append(surface.astype(np.float32)); groups.append(np.full(len(surface), i, np.int16))
    objects = renderable
    if not objects:
        raise ValueError('No renderable source triangles')
    matrices = np.empty((len(frames), len(objects), 4, 4), np.float64)
    maximum_local_error = 0.
    for i, frame in enumerate(frames):
        scene.frame_set(int(frame)); graph = bpy.context.evaluated_depsgraph_get()
        current_camera = np.asarray(scene.camera.evaluated_get(graph).matrix_world)
        if not np.allclose(camera, current_camera, atol=1e-6, rtol=0):
            raise ValueError('Expected the supplied fixed RGB camera')
        for obj in empty:
            if len(evaluated_mesh(obj, graph)[1]):
                raise ValueError(f'Initially empty mesh becomes visible: {obj.name}')
        for j, obj in enumerate(objects):
            xyz, faces, matrix = evaluated_mesh(obj, graph)
            original, topology = rest[j]
            if xyz.shape != original.shape or not np.array_equal(faces, topology):
                raise ValueError(f'Changing topology in {obj.name}')
            error = float(np.max(np.abs(xyz-original)))
            maximum_local_error = max(maximum_local_error, error)
            if error > 2e-5:
                raise ValueError(f'Nonrigid local geometry in {obj.name}: {error}')
            matrices[i, j] = matrix
        print(f'{args.name} source animation {i+1}/{len(frames)}', flush=True)
    metadata = dict(format='lamp_blender_parts_v1', source=bpy.data.filepath,
        blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        blender=bpy.app.version_string, objects=[o.name for o in objects],
        excluded_empty_meshes=[o.name for o in empty],
        camera_pose=camera.tolist(), frame_count=len(frames),
        camera_projection=projection.tolist(),
        camera_fovy_degrees=float(np.rad2deg(2*np.arctan(1/projection[1, 1]))),
        fps=scene.render.fps/scene.render.fps_base, source_frame_step=scene.frame_step,
        autoexec_disabled=not bpy.context.preferences.filepaths.use_scripts_auto_execute,
        maximum_local_geometry_error=maximum_local_error,
        timeline='All source render frames; playback uses scene output FPS, including frame_step.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rest_points=np.concatenate(surfaces), point_parts=np.concatenate(groups),
        object_matrices=matrices, source_frames=frames, metadata_json=np.asarray(json.dumps(metadata)))


if __name__ == '__main__':
    main()
