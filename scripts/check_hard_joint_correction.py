"""Audit every hard-joint pose, without claiming visual or event acceptance.

Requires the real candidate's correction.json and event-audit.json. Existing
artifact verification is reused; an event-unapproved candidate remains
unapproved even when all geometric invariants checked here pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from utils.event_refinement import correction_from_record, qmul, qrotate
from utils.joint_refinement import HardJointEventCorrection
from utils.multiview_completion import CompactCompletion, verify_multiview_completion


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def inspect_model(model, poses, progress=None):
    """Bounded-memory full-row/full-timeline checks; accepts arrays or tensors."""
    if not isinstance(model, HardJointEventCorrection) or len(poses) != model.frame_count:
        raise ValueError('Expected hard-joint model and its complete pose timeline')
    device, dtype = model.source_quaternions.device, model.source_quaternions.dtype

    def tensor(frame):
        value = poses[frame]
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.from_numpy(np.asarray(value).copy()).to(device=device, dtype=dtype)

    rest = tensor(0)
    if rest.shape != (len(model.row_groups), 7):
        raise ValueError('Pose rows disagree with the hard-joint structure')
    rigid = model.rigid_rows
    rigid_groups = model.row_groups[rigid]
    bands = torch.nonzero(model.row_groups < 0, as_tuple=True)[0]
    roots = torch.zeros(model.group_count, dtype=torch.bool, device=device)
    roots[model.roots] = True
    root_rows = rigid[roots[rigid_groups]]
    pairs = []
    for group in range(model.group_count):
        rows = torch.nonzero(model.row_groups == group, as_tuple=True)[0]
        chosen = rows[torch.linspace(0, max(len(rows)-1, 0), min(32, len(rows)), device=device).long()]
        half = len(chosen)//2
        if half:
            pairs.extend(zip(chosen[:half].tolist(), chosen[half:half*2].tolist()))
    pairs = torch.tensor(pairs, device=device, dtype=torch.long).reshape(-1, 2)
    rest_lengths = ((rest[pairs[:, 0], :3]-rest[pairs[:, 1], :3]).norm(dim=-1)
                    if len(pairs) else rest.new_empty(0))
    joint_max = torch.zeros(len(model.joint_groups), device=device, dtype=dtype)
    pair_max = mapping_max = quaternion_norm_max = group_norm_max = 0.
    limit = min(model.joint_tolerance, 1e-5)
    for index in range(model.frame_count):
        baked = tensor(index)
        after = model(rest, baked, index)
        quaternion, translation = model.group_transforms(index)
        gaps = model._joint_errors(quaternion, translation)
        joint_max = torch.maximum(joint_max, gaps)
        if (after.shape != rest.shape or not torch.isfinite(after).all()
                or torch.any(after[:, 3:].norm(dim=-1) < 1e-8)):
            raise ValueError(f'Invalid/deleted Gaussian rows at frame {index}')
        if index == 0 and not torch.equal(after, rest):
            raise ValueError('Canonical first frame changed')
        if not torch.equal(after[bands], baked[bands]):
            raise ValueError(f'Elastic-band rows changed at frame {index}')
        if not torch.equal(after[root_rows], rest[root_rows]):
            raise ValueError(f'Fixed-root rows changed at frame {index}')
        predicted = torch.cat((qrotate(quaternion[rigid_groups], rest[rigid, :3])+translation[rigid_groups],
                               qmul(quaternion[rigid_groups], rest[rigid, 3:])), -1)
        mapping = float((after[rigid]-predicted).abs().max()) if len(rigid) else 0.
        norms = float((after[rigid, 3:].norm(dim=-1)-rest[rigid, 3:].norm(dim=-1)).abs().max()) if len(rigid) else 0.
        group_norm = float((quaternion.norm(dim=-1)-1).abs().max())
        if len(pairs):
            current = (after[pairs[:, 0], :3]-after[pairs[:, 1], :3]).norm(dim=-1)
            pair = float((current-rest_lengths).abs().max())
        else:
            pair = 0.
        mapping_max, quaternion_norm_max = max(mapping_max, mapping), max(quaternion_norm_max, norms)
        pair_max, group_norm_max = max(pair_max, pair), max(group_norm_max, group_norm)
        if torch.any(gaps > limit) or max(mapping, pair, norms, group_norm) > 1e-5:
            raise ValueError(f'Hard-joint/rigidity/row-mapping tolerance exceeded at frame {index}')
        if progress is not None and index % 100 == 0:
            progress(index, {'max_joint_gap_scene': float(joint_max.max()) if len(joint_max) else 0.,
                             'max_group_pair_distance_error_scene': pair_max})
    effective_angles = (model.group_angles.tanh()*model.trainable_mask*model.envelopes.max(0).values*model.max_angle_degrees)
    trainable = torch.nonzero(model.trainable_mask > 0, as_tuple=True)[0].tolist()
    affected = torch.nonzero(effective_angles.abs() > 1e-8, as_tuple=True)[0].tolist()
    return {'format': 'hard_joint_pose_check_v1', 'geometry_invariants_passed': True,
            'frames_checked': model.frame_count, 'dynamic_rows_checked_per_frame': len(rest),
            'rigid_rows_checked_per_frame': len(rigid), 'unchanged_band_rows_checked_per_frame': len(bands),
            'fixed_root_rows_checked_per_frame': len(root_rows), 'motion_groups': model.group_count,
            'joints_checked_per_frame': len(model.joint_groups), 'joint_tolerance_scene': limit,
            'max_joint_gap_scene': float(joint_max.max()) if len(joint_max) else 0.,
            'max_gap_per_joint_scene': joint_max.tolist(), 'joint_group_pairs': model.joint_groups.tolist(),
            'all_poses_finite': True, 'canonical_first_frame_exact': True, 'band_rows_exactly_preserved': True,
            'fixed_root_rows_exactly_preserved': True, 'stable_row_count_and_order': True,
            'all_rigid_rows_match_group_se3': True, 'max_rigid_row_mapping_error': mapping_max,
            'rigid_pairs_sampled_per_frame': len(pairs), 'max_group_pair_distance_error_scene': pair_max,
            'max_rigid_quaternion_norm_change': quaternion_norm_max,
            'max_group_unit_quaternion_error': group_norm_max,
            'locked_groups': model.locked_groups, 'trainable_groups': trainable, 'affected_groups': affected,
            'signed_peak_effective_angle_degrees_per_group': effective_angles.tolist(),
            'loop_closure_edge_indices': model.loop_edges,
            'limitations': ['Closed-loop path nodes and all their ancestors have no learned angle residual; only off-cycle branches can be event-corrected.',
                            'Closure is checked to source/float precision, not claimed mathematically exact in stored floating point.',
                            'Elastic bands preserve the baked source approximation; their attachment or stretch is not newly solved.',
                            'This is not an image-completeness, hidden-surface, physical-force or full-sequence event-acceptance test.']}


@torch.no_grad()
def inspect_completion(model, completion, progress=None):
    """Check every compact suffix row at every time, without baking its poses."""
    if not isinstance(model, HardJointEventCorrection) or not isinstance(completion, CompactCompletion):
        raise ValueError('Expected inherited hard-joint motion and a compact completion')
    rest, groups = completion.rest, completion.groups
    if (rest.ndim != 2 or rest.shape[1] != 7 or not len(rest) or groups.shape != (len(rest),)
            or rest.device != model.row_groups.device or rest.dtype != model.source_quaternions.dtype
            or torch.any(groups < 0) or torch.any(groups >= model.group_count)):
        raise ValueError('Invalid compact canonical suffix rows')
    roots = torch.zeros(model.group_count, dtype=torch.bool, device=rest.device)
    roots[model.roots] = True
    fixed = roots[groups]
    pairs = []
    for group in range(model.group_count):
        rows = torch.nonzero(groups == group, as_tuple=True)[0]
        selected = rows[torch.linspace(0, max(len(rows)-1, 0), min(32, len(rows)), device=rest.device).long()]
        half = len(selected)//2
        pairs.extend(zip(selected[:half].tolist(), selected[half:half*2].tolist()))
    pairs = torch.as_tensor(pairs, dtype=torch.long, device=rest.device).reshape(-1, 2)
    lengths = ((rest[pairs[:, 0], :3]-rest[pairs[:, 1], :3]).norm(dim=-1)
               if len(pairs) else rest.new_empty(0))
    maximum_mapping = maximum_norm = maximum_pair = 0.
    for frame in range(model.frame_count):
        actual = completion.posed(frame, model)
        if actual.shape != rest.shape or not torch.isfinite(actual).all():
            raise ValueError(f'Invalid/deleted compact Gaussian suffix at frame {frame}')
        if frame == 0 and not torch.equal(actual, rest):
            raise ValueError('Compact canonical first frame changed')
        if not torch.equal(actual[fixed], rest[fixed]):
            raise ValueError(f'Compact fixed-root rows changed at frame {frame}')
        quaternion, translation = model.group_transforms(frame)
        expected = torch.cat((qrotate(quaternion[groups], rest[:, :3])+translation[groups],
                              qmul(quaternion[groups], rest[:, 3:])), -1)
        mapping = float((actual-expected).abs().max())
        norm = float((actual[:, 3:].norm(dim=-1)-1).abs().max())
        if len(pairs):
            current = (actual[pairs[:, 0], :3]-actual[pairs[:, 1], :3]).norm(dim=-1)
            pair = float((current-lengths).abs().max())
        else:
            pair = 0.
        maximum_mapping, maximum_norm, maximum_pair = max(maximum_mapping, mapping), max(maximum_norm, norm), max(maximum_pair, pair)
        if max(mapping, norm, pair) > 1e-5:
            raise ValueError(f'Compact group mapping/rigidity/quaternion tolerance exceeded at frame {frame}')
        if progress is not None and frame % 100 == 0:
            progress(frame, {'compact_suffix_rows_checked': len(rest), 'max_compact_mapping_error': maximum_mapping,
                             'max_compact_pair_distance_error_scene': maximum_pair})
    return {'format': 'compact_completion_pose_check_v1', 'geometry_invariants_passed': True,
            'frames_checked': model.frame_count, 'suffix_rows_checked_per_frame': len(rest),
            'moving_suffix_rows': int((~fixed).sum()), 'fixed_root_suffix_rows': int(fixed.sum()),
            'all_suffix_poses_finite': True, 'canonical_first_frame_exact': True,
            'static_root_rows_exact': True, 'all_suffix_rows_match_inherited_group_se3': True,
            'stable_suffix_count_and_order': True, 'pair_samples_per_frame': len(pairs),
            'maximum_group_mapping_error': maximum_mapping, 'maximum_unit_quaternion_error': maximum_norm,
            'maximum_intragroup_pair_distance_error_scene': maximum_pair,
            'unobserved_geometry_is_prior': True,
            'limitations': 'Full-timeline geometry/motion check, not image completeness, back-side truth, '
                          'new force simulation or full-sequence event acceptance. No new elastic-band rows.'}


def source_row_counts(source_count, baked_count, fixed_baked_count, suffix_check=None):
    """Separate stored baked rows from all moving/static rendered source rows."""
    suffix_check = suffix_check or {}
    added = suffix_check.get('suffix_rows_checked_per_frame', 0)
    moving_added = suffix_check.get('moving_suffix_rows', 0)
    fixed_added = suffix_check.get('fixed_root_suffix_rows', 0)
    if (any(type(value) is not int or value < 0 for value in
            (source_count, baked_count, fixed_baked_count, added, moving_added, fixed_added))
            or moving_added+fixed_added != added or fixed_baked_count > baked_count
            or baked_count+added > source_count):
        raise ValueError('Inconsistent baked/compact/static source counts')
    moving = baked_count-fixed_baked_count+moving_added
    return {'baked_pose_rows': baked_count, 'baked_fixed_root_rows': fixed_baked_count,
            'compact_suffix_rows': added, 'compact_moving_rows': moving_added,
            'compact_fixed_root_rows': fixed_added,
            'source_rows_without_baked_poses': source_count-baked_count,
            'total_moving_source_rows': moving, 'total_static_source_rows': source_count-moving,
            'non_dynamic_source_rows': source_count-moving}


def check_candidate(baseline, candidate, *, parts=None, binding=None, joint_report=None, device=None, progress=None):
    from plyfile import PlyData
    from scripts.export_lego_event_preview import verify_candidate
    baseline, candidate = Path(baseline), Path(candidate)
    cache, record, audit = verify_candidate(baseline, candidate, allow_unapproved=True)
    if record['format'] != 'event_supervised_hard_joint_alignment_v4':
        raise ValueError('Expected an audited hard-joint v4 candidate')
    structure = record['joint_structure']
    provenance = structure['input_sha256']
    for key, path in (('baseline_cache_metadata', baseline/'cache.json'), ('dynamic_indices', baseline/'dynamic_indices.npy')):
        if sha256(path) != provenance.get(key):
            raise ValueError(f'Hard-joint structure checksum mismatch: {key}')
    optional = {'parts': parts, 'binding': binding, 'joint_report': joint_report}
    checked_sources = []
    for key, path in optional.items():
        if path is not None:
            digest = sha256(path)
            if digest != provenance.get(key) or digest != record.get('group_input_sha256', {}).get(key):
                raise ValueError(f'Hard-joint input checksum mismatch: {key}')
            checked_sources.append(key)
    poses = np.load(baseline/'poses.npy', mmap_mode='r', allow_pickle=False)
    indices = np.load(baseline/'dynamic_indices.npy', allow_pickle=False)
    vertices = PlyData.read(cache['source'])['vertex'].data
    if (len(vertices) != cache['gaussian_count'] or len(indices) != cache['dynamic_count']
            or indices.dtype.kind not in 'iu' or indices.ndim != 1 or not len(indices)
            or np.any(indices < 0) or np.any(indices >= len(vertices)) or np.any(indices[1:] <= indices[:-1])
            or poses.shape != (cache['frame_count'], len(indices), 7)):
        raise ValueError('Source count, cached rows or stable indices are invalid')
    source_pose = np.column_stack([vertices[key][indices] for key in ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
    if not np.array_equal(poses[0], source_pose):
        raise ValueError('Baseline canonical rows differ from the hashed source PLY')
    if record.get('rgb_appearance_parameters_frozen') is not True:
        raise ValueError('Candidate does not declare its source Gaussian appearance frozen')
    binding_mapping_checked = False
    if binding is not None:
        with np.load(binding, allow_pickle=False) as archive:
            if not np.array_equal(archive['dynamic_indices'], indices):
                raise ValueError('Binding rows differ from baseline rows')
            labels = archive['labels'][indices]
        groups_by_label = {label: group['group'] for group in structure['source_groups'] for label in group['objects']}
        expected = np.asarray([groups_by_label.get(int(label), -1) for label in labels], dtype=np.int64)
        if not np.array_equal(expected, np.asarray(structure['row_groups'])):
            raise ValueError('Binding source-object labels disagree with hard-joint row groups')
        if parts is not None:
            with np.load(parts, allow_pickle=False) as archive:
                source_meta = json.loads(str(archive['metadata_json']))
            if not np.array_equal(expected < 0, np.isin(labels, source_meta.get('deforming_objects', []))):
                raise ValueError('Nonrigid rows are not exactly the source deforming-band objects')
        binding_mapping_checked = True
    model = correction_from_record(record, device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    report = inspect_model(model, poses, progress)
    verified = verify_multiview_completion(cache, record, audit, baseline=baseline)
    suffix_check = None
    if verified is not None:
        completion = CompactCompletion(verified, model.row_groups.device)
        suffix_check = inspect_completion(model, completion, progress)
        report['compact_completion'] = suffix_check
        report['completion_report_sha256'] = audit['artifact_sha256']['completion_report']
        report['completion_extension_sha256'] = audit['artifact_sha256']['completion_extension']
    report.update(source_row_counts(len(vertices), len(indices), report['fixed_root_rows_checked_per_frame'], suffix_check))
    report.update(source_gaussian_count=len(vertices),
                  canonical_dynamic_rows_exactly_match_source_ply=True, candidate_artifact_hashes_verified=True,
                  source_inputs_hash_verified=checked_sources, source_binding_row_mapping_verified=binding_mapping_checked,
                  source_ply_sha256=audit['artifact_sha256']['source_ply'],
                  correction_sha256=audit['artifact_sha256']['correction'], event_audit_sha256=sha256(candidate/'event-audit.json'),
                  event_approved_for_export=audit['approved_for_export'], event_summary=audit['summary'],
                  appearance_scope='The hashed source PLY is unchanged; the correction returns only XYZ/quaternions and preserves all source row indices. '
                                   'No rendered-image completeness is inferred from this check.')
    if cache.get('gaussian_cleanup') is not None:
        report.update(gaussian_cleanup=cache['gaussian_cleanup'],
                      cleanup_report_sha256=audit['artifact_sha256']['cleanup_report'],
                      cleanup_overlay_sha256=audit['artifact_sha256']['cleanup_overlay'])
        report['appearance_scope'] += (' An explicit static attribute overlay is applied by the renderer and '
                                       'included in the fresh event audit; this pose test does not assess appearance quality.')
    if cache.get('static_surface_rebuild') is not None:
        report.update(static_surface_rebuild=cache['static_surface_rebuild'],
                      static_surface_report_sha256=audit['artifact_sha256']['static_surface_report'],
                      static_surface_extension_sha256=audit['artifact_sha256']['static_surface_extension'],
                      rendered_gaussian_count=cache['rendered_gaussian_count'])
        report['appearance_scope'] += (' The explicit roof extension retires static source opacity and appends '
                                       'fixed static rows after this unchanged inherited motion. This pose test '
                                       'does not assess new roof appearance or unseen-view ground truth.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--joint-report', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    report = check_candidate(args.baseline, args.candidate, parts=args.parts, binding=args.binding,
                             joint_report=args.joint_report, device=args.device,
                             progress=lambda index, values: print(json.dumps({'frame': index, **values}), flush=True))
    serialized = json.dumps(report, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(serialized)
    print(serialized, flush=True)


if __name__ == '__main__':
    main()
