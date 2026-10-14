"""Blender CPU BVH masks for the complete mesh PRIOR at corrected rigid poses.

Run with --background --factory-startup --disable-autoexec --threads 3.
No scene render, GPU operation, source .blend write, or image ground truth is used.
Elastic parts are explicitly excluded from this rigid-mesh diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.viewer_state import orbit_pose


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def views(front_stride=None, frame_count=400, heldout=False):
    if type(heldout) is not bool or (heldout and front_stride is not None):
        raise ValueError('Held-out plan and front-only stride are mutually exclusive')
    if heldout:
        if type(frame_count) is not int or frame_count <= 150:
            raise ValueError('Held-out plan requires frames 50 and 150')
        plan = [(frame, yaw, 0) for frame in (50, 150) for yaw in (0, 90, 180, 270)]
        plan += [(150, yaw, pitch) for pitch in (-60, 60) for yaw in range(0, 360, 45)]
    elif front_stride is not None:
        if (type(front_stride) is not int or front_stride < 1 or type(frame_count) is not int
                or frame_count < 2):
            raise ValueError('Front-only stride and timeline must be positive integers')
        frames = sorted(set(range(0, frame_count, front_stride)) | {frame_count - 1})
        plan = [(frame, 0, 0) for frame in frames]
    else:
        plan = [(frame, yaw, 0) for frame in (0, 100, 200, 300) for yaw in (0, 90, 180, 270)]
        plan += [(100, yaw, 0) for yaw in (45, 135, 225, 315)]
        plan += [(100, yaw, pitch) for pitch in (-35, 35) for yaw in range(0, 360, 45)]
    return [{'frame': frame, 'yaw': yaw, 'pitch': pitch, 'zoom': 1., 'pan_x': 0., 'pan_y': 0.}
            for frame, yaw, pitch in plan]


def validate_identity(mesh_metadata, motion_metadata):
    if (mesh_metadata.get('format') != 'lego_complete_mesh_prior_v1'
            or motion_metadata.get('format') != 'lego_corrected_mesh_motion_v1'
            or mesh_metadata.get('objects') != motion_metadata.get('objects')
            or mesh_metadata.get('parts_sha256') != motion_metadata.get('source_parts_sha256')
            or sorted(mesh_metadata.get('deforming_objects', []))
            != sorted(motion_metadata.get('excluded_deforming_objects', []))):
        raise ValueError('Mesh/motion source identity or object order differs')


def vertex_part_ids(vertex_count, triangles, parts):
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or parts.shape != (len(triangles),)
            or triangles.dtype.kind not in 'iu' or parts.dtype.kind not in 'iu'
            or np.any(triangles < 0) or np.any(triangles >= vertex_count) or np.any(parts < 0)):
        raise ValueError('Invalid triangle ownership or indices')
    # min/max detect cross-part shared vertices, rather than last-assignment wins.
    minimum = np.full(vertex_count, np.iinfo(np.int32).max, np.int32)
    maximum = np.full(vertex_count, -1, np.int32)
    for axis in range(3):
        np.minimum.at(minimum, triangles[:, axis], parts)
        np.maximum.at(maximum, triangles[:, axis], parts)
    used = maximum >= 0
    if np.any(minimum[used] != maximum[used]):
        raise ValueError('A mesh vertex is shared by different source parts')
    return maximum


def quaternion_matrices(quaternions):
    q = np.asarray(quaternions, np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError('Invalid wxyz quaternion matrix')
    if np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) > 1e-4:
        raise ValueError('Motion quaternions must be unit length')
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(-1, 3, 3)


def posed_vertices(vertices, vertex_parts, group_ids, quaternions, translations, excluded):
    rotations = quaternion_matrices(quaternions)
    result = np.array(vertices, dtype=np.float64, copy=True)
    rigid = (vertex_parts >= 0) & ~np.isin(vertex_parts, sorted(excluded))
    vertex_groups = np.full(len(vertices), -1, np.int32)
    vertex_groups[rigid] = group_ids[vertex_parts[rigid]]
    if np.any(vertex_groups[rigid] < 0) or np.any(vertex_groups[rigid] >= len(rotations)):
        raise ValueError('A rigid part has no corrected motion group')
    for group in np.unique(vertex_groups[rigid]):
        chosen = rigid & (vertex_groups == group)
        result[chosen] = vertices[chosen] @ rotations[group].T + translations[group]
    return result


def camera_rays(pose, width, height, fovy):
    pose = np.asarray(pose, np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not 1 <= width <= 1280 or not 1 <= height <= 1280 or not 0 < fovy < 180):
        raise ValueError('Invalid ray camera')
    yy, xx = np.mgrid[:height, :width]
    tangent = np.tan(np.deg2rad(fovy) / 2)
    local = np.stack(((2*(xx+.5)/width-1)*tangent*width/height,
                      (1-2*(yy+.5)/height)*tangent, -np.ones_like(xx)), axis=-1)
    rays = local @ pose[:3, :3].T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return pose[:3, 3].copy(), rays


def write_png(path, rgb):
    """Small RGB PNG writer; Blender's bundled Python need not include Pillow."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('PNG expects uint8 RGB')
    height, width = rgb.shape[:2]
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    encoded = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    Path(path).write_bytes(encoded)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mesh', 'motion', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=640)
    plan_group = parser.add_mutually_exclusive_group()
    plan_group.add_argument('--front-stride', type=int,
                            help='Instead of orbit views, sample only the original camera at this '
                                 'time stride, always including the final frame')
    plan_group.add_argument('--heldout', action='store_true',
                            help='24 additional mesh-prior view/time checks not in the original '
                                 '36-view surface-selection plan; not held-out event data')
    return parser


def main():
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    parser = argument_parser()
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    preview_dir = args.output.with_suffix('')
    if args.output.exists() or args.output.with_suffix('.json').exists() or preview_dir.exists():
        raise FileExistsError('Mask output, metadata, or preview directory already exists')
    with np.load(args.mesh, allow_pickle=False) as archive:
        vertices = archive['vertices'].astype(np.float64)
        triangles = archive['triangles']
        part_ids = archive['part_ids']
        mesh_meta = json.loads(str(archive['metadata_json']))
    with np.load(args.motion, allow_pickle=False) as archive:
        group_ids = archive['group_ids']
        quaternions = archive['quaternions']
        translations = archive['translations']
        camera = json.loads(str(archive['camera_json']))
        motion_meta = json.loads(str(archive['metadata_json'])) if 'metadata_json' in archive else {}
    validate_identity(mesh_meta, motion_meta)
    excluded = set(mesh_meta['deforming_objects'])
    objects = mesh_meta['objects']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or len(objects) >= 32767 or group_ids.shape != (len(objects),)
            or group_ids.dtype.kind not in 'iu' or not len(triangles)
            or np.any(part_ids >= len(objects)) or quaternions.ndim != 3
            or quaternions.shape[0] < 301 or quaternions.shape[-1] != 4
            or translations.shape != (*quaternions.shape[:2], 3)
            or not np.isfinite(translations).all()):
        raise ValueError('Invalid mesh/motion schema or timeline')
    ownership = vertex_part_ids(len(vertices), triangles, part_ids)
    keep = ~np.isin(part_ids, sorted(excluded))
    faces, labels = triangles[keep], part_ids[keep]
    if not len(faces):
        raise ValueError('No rigid mesh faces remain')
    plan = views(args.front_stride, len(quaternions), args.heldout)
    masks = np.full((len(plan), args.height, args.width), -1, np.int16)
    depths = np.full(masks.shape, np.nan, np.float32)
    palette = np.random.default_rng(17).integers(48, 256, (len(objects), 3), dtype=np.uint8)
    info = {
        'format': 'lego_corrected_rigid_mesh_multiview_masks_v1',
        'scope': 'Blender full-mesh PRIOR in canonical Gaussian coordinates, transformed by the '
                 'actual candidate corrected rigid group SE3 and sampled at UI orbit cameras. '
                 'Not observed back-side truth, Gaussian appearance quality, complete geometry '
                 'proof, or event acceptance. Elastic source objects are excluded from both '
                 'geometry and occlusion; inferred rigid surfaces behind bands can therefore '
                 'appear in these masks. Unobserved-view quality is prior consistency only.',
        'mesh': str(args.mesh), 'mesh_sha256': sha256(args.mesh),
        'motion': str(args.motion), 'motion_sha256': sha256(args.motion),
        'mesh_metadata': mesh_meta, 'motion_metadata': motion_meta,
        'objects': objects, 'excluded_deforming_objects': sorted(excluded),
        'excluded_deforming_names': [objects[part] for part in sorted(excluded)],
        'resolution': [args.width, args.height], 'camera': camera,
        'depth_definition': 'Positive camera-space -Z in Gaussian scene units; NaN for no rigid hit.',
        'pixel_rays': 'OpenGL camera, pixel-center perspective, same orbit_pose as paired UI.',
        'rigid_triangle_count': int(keep.sum()), 'excluded_triangle_count': int((~keep).sum()),
        'autoexec_disabled': not bpy.context.preferences.filepaths.use_scripts_auto_execute,
        'cpu_only': True, 'views': [],
        'front_only_stride': args.front_stride,
        'heldout_prior_view_plan': args.heldout,
        'plan': ('heldout_prior_geometry_24' if args.heldout else
                 'original_camera_temporal_guard' if args.front_stride is not None else 'surface_selection_orbit_36'),
        'heldout_scope': ('These 24 camera/time tuples do not occur in the original 36-view '
                          'surface-selection plan. The two yaw=0,pitch=0 views at frames 50 and '
                          '150 are original-camera controls already used by front visibility '
                          'guards and the 400-frame exposure filter; only 22 tuples are novel views. '
                          'They use the same known mesh prior and inherited '
                          'motion, not independent observed hidden-surface truth, held-out events, '
                          'or proof of all-view completeness. Later tuning against them would make '
                          'them development diagnostics rather than untouched held-out views.'
                          if args.heldout else None),
        'heldout_original_camera_control_indices': [0, 4] if args.heldout else [],
        'heldout_novel_camera_time_count': 22 if args.heldout else None,
        'triangle_visibility': 'Opaque triangles; material transparency is not modeled.',
        'sampling_limitation': 'Finite sampled cameras/times; visibility between samples is not proven.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir()
    # Reuse one BVH for every camera at each sampled motion time.
    records = {}
    for frame in sorted(set(view['frame'] for view in plan)):
        start = time.monotonic()
        posed = posed_vertices(vertices, ownership, group_ids, quaternions[frame], translations[frame], excluded)
        tree = BVHTree.FromPolygons(posed.tolist(), faces.tolist(), all_triangles=True)
        print(json.dumps({'stage': 'bvh_ready', 'frame': frame, 'seconds': time.monotonic()-start}), flush=True)
        for index, view in enumerate(plan):
            if view['frame'] != frame:
                continue
            pose = orbit_pose(camera['pose'], camera['target'], view)
            origin, directions = camera_rays(pose, args.width, args.height, camera['fovy_degrees'])
            origin_vector = Vector(origin)
            negative_back = -pose[:3, 2].astype(np.float64)
            for y in range(args.height):
                for x in range(args.width):
                    location, _, triangle, _ = tree.ray_cast(origin_vector, Vector(directions[y, x]))
                    if triangle is not None:
                        masks[index, y, x] = labels[triangle]
                        depths[index, y, x] = float(np.dot(np.asarray(location) - origin, negative_back))
            selected = masks[index] >= 0
            if np.any(depths[index][selected] <= 0):
                raise ValueError('A mask hit lies behind the camera')
            visible, counts = np.unique(masks[index][selected], return_counts=True)
            preview = np.zeros((*masks.shape[1:], 3), np.uint8)
            preview[selected] = palette[masks[index][selected]]
            image_name = f'view-{index:02d}_frame-{frame:04d}_yaw-{view["yaw"]:03d}_pitch-{view["pitch"]:+03d}.png'
            write_png(preview_dir / image_name, preview)
            row = {'view_index': index, **view, 'camera_pose': pose.tolist(), 'preview': image_name,
                   'visible_rigid_pixels': int(selected.sum()),
                   'visible_part_pixels': {str(int(part)): int(count) for part, count in zip(visible, counts)}}
            records[index] = row
            print(json.dumps({'stage': 'view_ready', 'view_index': index, 'frame': frame,
                              'yaw': view['yaw'], 'pitch': view['pitch'], 'pixels': int(selected.sum())}), flush=True)
        del tree, posed
    info['views'] = [records[index] for index in range(len(plan))]
    info['view_count'] = len(plan)
    np.savez_compressed(args.output, part_ids=masks, depth=depths,
                        frame_indices=np.array([view['frame'] for view in plan], np.int32),
                        metadata_json=np.asarray(json.dumps(info)))
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'views': len(plan), 'excluded_parts': sorted(excluded)}), flush=True)


if __name__ == '__main__':
    main()
