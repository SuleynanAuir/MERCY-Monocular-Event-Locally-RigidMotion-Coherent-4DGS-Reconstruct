"""Export actual Lego Gaussian trajectories, sharing the unchanged static rows.

This is a baked sequence, NOT SC-GS deformation-network weights. Playback needs
only the resulting NPZ, not Blender, input data or a learned deformation model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement

from utils.articulated_motion import ArticulatedBaker
from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS
from utils.rigid_checkpoint import sha256_file


def write_npz(path, arrays):
    # Low compression level keeps this one-off CPU export responsive. ZIP64
    # supports archives whose uncompressed trajectory array exceeds 4 GB.
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for key, value in arrays.items():
            print('Compressing', key, value.nbytes / 1e9, 'GB', flush=True)
            with archive.open(key + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--parts', default='blender-surface-parts.npz')
    parser.add_argument('--binding', default='refined-binding.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.data / 'fan_contrastive_g0/contrastive_g0.ply'
    vertices = PlyData.read(source)['vertex'].data.copy()
    if any(np.any(vertices[k]) for k in ('nx', 'ny', 'nz')):
        raise ValueError('This Lego exporter expects zero normals')
    if any(np.any(vertices[k]) for k in vertices.dtype.names if k.startswith('f_rest_')):
        raise ValueError('This Lego exporter expects DC-only color; nonzero SH needs rotation')
    baker = ArticulatedBaker(vertices, args.inspection / args.parts, args.inspection / args.binding)
    frames, fps = len(baker.delta), baker.source['playback_fps']
    needed = len(baker.indices) * frames * 7 * 4 + vertices.nbytes * 3 + 1024**3
    if shutil.disk_usage(args.output.parent).free < needed:
        raise OSError('Insufficient data-disk space for conservative export bound')
    args.output.mkdir()
    poses = np.empty((len(baker.indices), frames, 7), dtype=np.float32)
    metadata = {'format': FORMAT, 'version': SPARSE_VERSION,
                'storage': 'sparse_lossless_baked_gaussian_trajectories',
                'frame_count': frames, 'fps': fps, 'duration_seconds': frames / fps,
                'gaussians_per_frame': len(vertices), 'dynamic_gaussians': len(baker.indices),
                'sh_degree': 3, 'time_unit': 'seconds', 'quaternion_order': 'wxyz',
                'stable_gaussian_row_order': True, 'motion_is_baked': True,
                'requires_deform_checkpoint_for_playback': False, 'training_steps': 0,
                'pose_fields': POSE_FIELDS, 'poses_layout': 'N_dynamic,T_frames,channels',
                'source_sha256': sha256_file(source),
                'base_vertices_sha256': hashlib.sha256(vertices.tobytes()).hexdigest(),
                'preview_camera': json.loads((args.inspection / 'inspection.json').read_text())['camera'],
                'motion_label': 'Lego · 原工程铲斗与连杆动画',
                'default_stream_rate': 30, 'default_resolution': 768,
                'motion_provenance_only': {'source_animation': baker.source, 'binding': baker.binding,
                                            'rigid_rotation_max_error': baker.max_rigidity_error},
                'scope': 'Original articulated keyframe animation transferred by approximate geometric binding; not a physics simulation or learned SC-GS network. Elastic bands use local rigid surface fits, no covariance stretching.',
                'frames': []}
    for i in range(frames):
        pose = baker.frame(i)
        poses[:, i] = pose
        metadata['frames'].append({'file': f'frames/frame_{i:04d}.ply', 'time_seconds': i / fps,
                                   'pose_sha256': hashlib.sha256(pose.tobytes()).hexdigest(),
                                   'source_blender_frame': int(baker.parts['source_frames'][i])})
        if i == 100:
            snapshot = vertices.copy()
            for column, key in enumerate(POSE_FIELDS):
                snapshot[key][baker.indices] = pose[:, column]
            PlyData([PlyElement.describe(snapshot, 'vertex')], text=False, byte_order='<').write(args.output / 'frame_0100.ply')
            del snapshot
        if i % 50 == 0:
            print('Baked', i, '/', frames, flush=True)
    partial = args.output / 'lego_motion.npz.partial'
    write_npz(partial, {'base_vertices': vertices, 'dynamic_indices': baker.indices,
                        'poses': poses, 'times_seconds': np.arange(frames, dtype=np.float64) / fps,
                        'metadata_json': np.asarray(json.dumps(metadata))})
    partial.rename(args.output / 'lego_motion.npz')
    shutil.copyfile(source, args.output / 'point_cloud.ply')
    shutil.copyfile(Path(__file__).parent / 'utils/gaussian_sequence.py', args.output / 'load_sequence.py')
    (args.output / 'sequence.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'README.txt').write_text(
        'LEGO BAKED MOVING GAUSSIANS\n\n'
        f'{len(vertices)} Gaussians; {len(baker.indices)} moving, {frames} frames at {fps} FPS.\n'
        'lego_motion.npz contains actual per-frame XYZ and wxyz quaternion states, sharing static Gaussian rows.\n'
        'No Blender, original PLY, deform.pth or training is needed for playback.\n'
        'point_cloud.ply is the supplied rest state. frame_0100.ply is a complete moving-state snapshot.\n'
        'A single PLY is static; use the NPZ-aware viewer for animation.\n\n'
        'python viewer.py --sequence_path PATH/lego_motion.npz --port 8891\n'
        'from load_sequence import GaussianSequence\n'
        'with GaussianSequence("lego_motion.npz") as sequence:\n'
        '    sequence.frame(100).write("snapshot.ply")\n\n'
        'Source Blender frames 1,11,...3991 are retimed to match the supplied 400-frame 60 FPS Lego.mkv.\n'
        'Motion is supplied keyframe kinematics, not motor/contact/force simulation.\n'
        'Part attachment is inferred from geometry, so small seams and residual artifacts are possible.\n'
        'Rigid parts preserve distances and Gaussian shapes; elastic bands use approximate local surface fits.\n')
    print('Finished', args.output, flush=True)


if __name__ == '__main__':
    main()
