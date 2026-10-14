"""Bake an audited event-corrected candidate for inspection, never silently approve it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from plyfile import PlyData
import torch

from export_lego_sequence import write_npz
from utils.event_refinement import correction_from_record
from utils.event_supervision import require_event_export_approval
from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS, GaussianSequence
from utils.rigid_checkpoint import sha256_file
from utils.render_sampling import require_same_sampling, require_native_sampling
from utils.multiview_completion import verify_multiview_completion, require_no_completion


def verify_candidate(baseline, candidate, allow_unapproved=False):
    report = json.loads((candidate / 'event-audit.json').read_text())
    correction = json.loads((candidate / 'correction.json').read_text())
    cache = json.loads((baseline / 'cache.json').read_text())
    require_same_sampling(cache, correction, report)
    if (report.get('format') != 'lego_event_motion_audit_v1'
            or report.get('event_optimization_performed') is not True
            or correction.get('format') not in ('event_supervised_rigid_motion_alignment_v1',
                                                 'event_supervised_temporal_rigid_alignment_v2',
                                                 'event_supervised_articulated_alignment_v3',
                                                 'event_supervised_hard_joint_alignment_v4')):
        raise ValueError('A recorded event-optimized and audited candidate is required')
    correction_from_record(correction)
    if correction.get('frame_count', cache['frame_count']) != cache['frame_count']:
        raise ValueError('Correction and baseline timeline disagree')
    if (correction['format'] == 'event_supervised_articulated_alignment_v3'
            and len(correction['group_structure']['groups']) != cache['dynamic_count']):
        raise ValueError('Articulated correction and baseline rows disagree')
    if (correction['format'] == 'event_supervised_hard_joint_alignment_v4'
            and len(correction['joint_structure']['row_groups']) != cache['dynamic_count']):
        raise ValueError('Hard-joint correction and baseline rows disagree')
    if report.get('approved_for_export') is True:
        require_event_export_approval(report)
    elif not allow_unapproved:
        raise ValueError('Candidate failed event audit; explicit --allow-unapproved-preview is required')
    elif report.get('approved_for_export') is not False:
        raise ValueError('Missing audit decision')
    for key, path in {'correction': candidate / 'correction.json',
                      'rendered_gray': candidate / 'rendered_gray.npy',
                      'baseline_cache_metadata': baseline / 'cache.json',
                      'baseline_poses': baseline / 'poses.npy',
                      'dynamic_indices': baseline / 'dynamic_indices.npy',
                      'source_ply': Path(cache['source'])}.items():
        if sha256_file(path) != report.get('artifact_sha256', {}).get(key):
            raise ValueError(f'Candidate/audit checksum mismatch: {key}')
    if (report['frame_count'] != cache['frame_count'] or report['fps'] != cache['fps']
            or len(report['intervals']) != cache['frame_count'] - 1
            or [r['interval'] for r in report['intervals']] != list(range(cache['frame_count'] - 1))):
        raise ValueError('Incomplete or incompatible audit timeline')
    verify_multiview_completion(cache, correction, report, baseline=baseline)
    from utils.gaussian_cleanup import verify_gaussian_cleanup
    verify_gaussian_cleanup(cache, correction, report, baseline=baseline)
    from utils.static_surface_rebuild import verify_static_surface_rebuild
    verify_static_surface_rebuild(cache, correction, report, baseline=baseline)
    return cache, correction, report


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-unapproved-preview', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cache, parameters, report = verify_candidate(args.baseline, args.candidate, args.allow_unapproved_preview)
    require_no_completion(cache, 'Baked sequence export')
    from utils.gaussian_cleanup import require_no_cleanup
    require_no_cleanup(cache, 'Baked sequence export')
    from utils.static_surface_rebuild import require_no_static_surface_rebuild
    require_no_static_surface_rebuild(cache, 'Legacy baked sequence export')
    require_native_sampling(cache, 'Baked sequence export')
    source = Path(cache['source'])
    vertices = PlyData.read(source)['vertex'].data.copy()
    for key in vertices.dtype.names:
        if (key.startswith('f_rest_') or key in ('nx', 'ny', 'nz')) and np.any(vertices[key]):
            raise ValueError('Preview requires zero normals and DC-only source appearance')
    baseline = np.load(args.baseline / 'poses.npy', mmap_mode='r', allow_pickle=False)
    indices = np.load(args.baseline / 'dynamic_indices.npy', allow_pickle=False)
    frames, count, channels = baseline.shape
    if (channels != 7 or frames != report['frame_count'] or len(indices) != count
            or np.any(indices < 0) or np.any(indices >= len(vertices)) or np.any(indices[1:] <= indices[:-1])):
        raise ValueError('Invalid cached Gaussian poses/indices')
    if shutil.disk_usage(args.output.parent).free < baseline.nbytes * 2 + vertices.nbytes + 1024**3:
        raise OSError('Insufficient data disk space')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correction = correction_from_record(parameters, device)
    rest = torch.from_numpy(baseline[0].copy()).to(device)
    poses = np.empty((count, frames, 7), dtype=np.float32)
    records = []
    for i in range(frames):
        baked = torch.from_numpy(baseline[i].copy()).to(device)
        pose = (baked if i == 0 else correction(rest, baked, i)).cpu().numpy()
        if not np.isfinite(pose).all() or np.any(np.linalg.norm(pose[:, 3:], axis=-1) < 1e-8):
            raise ValueError(f'Invalid corrected pose: {i}')
        poses[:, i] = pose
        records.append({'file': f'frames/frame_{i:04d}.ply', 'time_seconds': i / report['fps'],
                        'pose_sha256': hashlib.sha256(pose.tobytes()).hexdigest()})
        if i % 50 == 0:
            print('Baking event-corrected Gaussian preview', i, '/', frames, flush=True)
    summary = report['summary']['all']
    metadata = {'format': FORMAT, 'version': SPARSE_VERSION,
                'storage': 'sparse_lossless_baked_gaussian_trajectories',
                'frame_count': frames, 'fps': report['fps'], 'duration_seconds': frames / report['fps'],
                'gaussians_per_frame': len(vertices), 'dynamic_gaussians': count, 'sh_degree': 3,
                'time_unit': 'seconds', 'quaternion_order': 'wxyz', 'stable_gaussian_row_order': True,
                'motion_is_baked': True, 'requires_deform_checkpoint_for_playback': False,
                'pose_fields': POSE_FIELDS, 'poses_layout': 'N_dynamic,T_frames,channels',
                'source_sha256': report['artifact_sha256']['source_ply'],
                'base_vertices_sha256': hashlib.sha256(vertices.tobytes()).hexdigest(),
                'preview_camera': cache['camera'], 'motion_label': 'Lego · 原始事件校正后 · 候选预览',
                'default_stream_rate': 30, 'default_resolution': 640, 'default_background': 'black',
                'event_validation': {'optimized': True, 'approved': report['approved_for_export'],
                                     'preview_only': True, 'summary': summary,
                                     'failed_intervals': report['failed_intervals'],
                                     'comparison_resolution': report['comparison_resolution'],
                                     'scope': 'Single-view binned event consistency; novel views are not validated.'},
                'event_provenance': {'correction': parameters, 'artifact_sha256': report['artifact_sha256'],
                                     'audit_sha256': sha256_file(args.candidate / 'event-audit.json')},
                'scope': 'Inspection-only event-corrected candidate, not an accepted reconstruction or native SC-GS checkpoint.',
                'frames': records}
    args.output.mkdir(parents=True)
    partial = args.output / 'lego_event_candidate.npz.partial'
    write_npz(partial, {'base_vertices': vertices, 'dynamic_indices': indices, 'poses': poses,
                        'times_seconds': np.arange(frames, dtype=np.float64) / report['fps'],
                        'metadata_json': np.asarray(json.dumps(metadata))})
    del poses
    target = args.output / 'lego_event_candidate.npz'
    partial.rename(target)
    with GaussianSequence(target) as sequence:
        for i in range(len(sequence)):
            sequence.validate_pose_frame(i)
    shutil.copyfile(args.candidate / 'event-audit.json', args.output / 'event-audit.json')
    shutil.copyfile(args.candidate / 'correction.json', args.output / 'correction.json')
    (args.output / 'sequence.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Preview archive verified:', target, 'APPROVED:', report['approved_for_export'], flush=True)


if __name__ == '__main__':
    main()
