"""Blender: export observed missing rigid-surface ray hits, without saving a scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bpy
import numpy as np
from mathutils import Vector
from scripts.export_lego_part_sequence import scene_tree, camera_rays


def eroded_parts(mask):
    interior = mask >= 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.full_like(mask, -2)
            ys, xs = slice(max(0, dy), min(mask.shape[0], mask.shape[0]+dy)), slice(max(0, dx), min(mask.shape[1], mask.shape[1]+dx))
            shifted[ys, xs] = mask[slice(max(0, -dy), min(mask.shape[0], mask.shape[0]-dy)),
                                     slice(max(0, -dx), min(mask.shape[1], mask.shape[1]-dx))]
            interior &= shifted == mask
    return interior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-facing-cosine', type=float, default=.2)
    parser.add_argument('--all-visible-rigid', action='store_true',
                        help='Export every observed visible moving rigid surface, not only current holes')
    parser.add_argument('--erosion-pixels', type=int, choices=(0, 1), default=1)
    parser.add_argument('--reference-threshold', type=float)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    threshold = args.reference_threshold if args.reference_threshold is not None else (.03 if args.all_visible_rigid else .12)
    if not 0 <= threshold < 1:
        raise ValueError('Invalid reference threshold')
    with np.load(args.parts, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        source_frames = archive['source_frames'].copy()
        matrices = archive['object_matrices'].astype(np.float64)
    with np.load(args.masks, allow_pickle=False) as archive:
        masks = archive['part_ids']; frames = archive['frame_indices']
        mask_meta = json.loads(str(archive['metadata_json']))
    with np.load(args.supervision, allow_pickle=False) as archive:
        images = archive['images']
    rendered = np.load(args.candidate / 'rendered_gray.npy', mmap_mode='r', allow_pickle=False)
    if images.shape != rendered.shape or mask_meta['objects'] != metadata['objects']:
        raise ValueError('Reference, candidate, or part identity mismatch')
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[metadata['deforming_objects']] = False
    scene = bpy.context.scene
    columns = {key: [] for key in ('frame_index', 'part_id', 'pixel_xy', 'source_world', 'canonical_source_xyz',
                                    'canonical_source_normal', 'source_camera_z', 'gray', 'facing_cosine')}
    rows = []
    for sample, index in enumerate(frames):
        index = int(index); mask = masks[sample]
        reference = images[index].astype(np.float32) / 255
        actual = np.asarray(rendered[index], np.float32)
        if not np.isfinite(actual).all() or actual.min() < 0 or actual.max() > 1:
            raise ValueError('Candidate grayscale is nonfinite or outside [0,1]')
        interior = eroded_parts(mask) if args.erosion_pixels else mask >= 0
        wanted = interior & np.isin(mask, np.flatnonzero(moving)) & (reference > threshold)
        if not args.all_visible_rigid:
            wanted &= actual < .08
        yy, xx = np.nonzero(wanted)
        if not len(yy):
            rows.append({'frame_index': index, 'missing_interior_pixels': 0, 'accepted_hits': 0})
            continue
        scene.frame_set(int(source_frames[index])); graph = bpy.context.evaluated_depsgraph_get()
        tree, labels = scene_tree(graph, metadata['objects'])
        origin, directions, projection = camera_rays(scene, graph, 640, 640)
        view = np.linalg.inv(np.asarray(scene.camera.evaluated_get(graph).matrix_world))
        inverse_delta = np.linalg.inv(delta[index])
        accepted, mismatched, grazing = 0, 0, 0
        for y, x in zip(yy, xx):
            location, normal, triangle, _ = tree.ray_cast(origin, Vector(directions[y, x]))
            if triangle is None or labels[triangle] != mask[y, x]:
                mismatched += 1; continue
            label = int(labels[triangle]); world = np.asarray(location, dtype=np.float64)
            normal = np.asarray(normal, dtype=np.float64)
            facing = abs(float(np.dot(normal, directions[y, x])))
            if facing < args.min_facing_cosine:
                grazing += 1; continue
            rest = inverse_delta[label, :3, :3] @ world + inverse_delta[label, :3, 3]
            rest_normal = inverse_delta[label, :3, :3] @ normal
            rest_normal /= np.linalg.norm(rest_normal)
            z = -float((view[:3, :3] @ world + view[:3, 3])[2])
            for key, value in {'frame_index': index, 'part_id': label, 'pixel_xy': [int(x), int(y)],
                               'source_world': world, 'canonical_source_xyz': rest,
                               'canonical_source_normal': rest_normal, 'source_camera_z': z,
                               'gray': float(reference[y, x]), 'facing_cosine': facing}.items():
                columns[key].append(value)
            accepted += 1
        rows.append({'frame_index': index, 'missing_interior_pixels': len(yy), 'accepted_hits': accepted,
                     'part_mismatch_rejected': mismatched, 'grazing_rejected': grazing})
        print(json.dumps(rows[-1] | {'sample': sample+1, 'samples': len(frames)}), flush=True)
        del tree, labels
    arrays = {key: np.asarray(value, dtype=np.int32 if key in ('frame_index', 'part_id', 'pixel_xy') else np.float32)
              for key, value in columns.items()}
    info = {'format': 'lego_observed_completion_surfaces_v1', 'objects': metadata['objects'],
            'parts': str(args.parts), 'parts_sha256': hashlib.sha256(args.parts.read_bytes()).hexdigest(),
            'masks': str(args.masks), 'candidate': str(args.candidate), 'supervision': str(args.supervision),
            'source': bpy.data.filepath, 'reference_threshold': threshold,
            'missing_threshold': None if args.all_visible_rigid else .08,
            'all_visible_rigid': args.all_visible_rigid,
            'erosion_pixels': args.erosion_pixels, 'min_facing_cosine': args.min_facing_cosine, 'sampled_frames': frames.tolist(),
            'frames': rows, 'hits': len(columns['frame_index']),
            'excluded_deforming_parts': metadata['deforming_objects'],
            'scope': 'Allowed Blender evaluated visible triangles and actual RGB observations only; '
                     'background/source holes are excluded. In all-visible-rigid mode candidate holes do not gate selection. '
                     'No source scene or Gaussian model is changed by this export.'}
    arrays['metadata_json'] = np.asarray(json.dumps(info))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'hits': info['hits']}), flush=True)


if __name__ == '__main__':
    main()
