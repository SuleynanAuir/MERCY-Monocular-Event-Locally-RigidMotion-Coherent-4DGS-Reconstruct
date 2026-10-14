"""Explicit mesh-prior Gaussian suffixes driven by existing hard-joint motion.

The parent cloud and its baked rows remain immutable. Unobserved surfaces are
known-geometry priors, not new monocular measurements or event-approved views.
"""
import copy
import json
from pathlib import Path
import re

import numpy as np
import torch

from utils.event_refinement import correction_from_record, qmul, qrotate
from utils.joint_refinement import HardJointEventCorrection
from utils.rigid_checkpoint import sha256_file


FORMAT = 'lego_multiview_prior_completion_v1'
MAX_ADDED = 600000
CONTRACT_FIELDS = {'format', 'report', 'report_sha256', 'extension', 'extension_sha256'}


def completion_contract(*records):
    """Absent legacy metadata is valid; partial/mismatched extensions are not."""
    contracts = [record.get('multiview_completion') for record in records]
    if not contracts or all(value is None for value in contracts):
        return None
    first = contracts[0]
    if (not isinstance(first, dict) or set(first) != CONTRACT_FIELDS
            or first.get('format') != FORMAT or any(value != first for value in contracts)):
        raise ValueError('Multiview completion contract mismatch')
    for key in ('report', 'extension'):
        if not isinstance(first[key], str) or not Path(first[key]).is_absolute():
            raise ValueError(f'Multiview completion requires an absolute {key} path')
        if not isinstance(first[key + '_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', first[key + '_sha256']):
            raise ValueError(f'Invalid multiview completion {key} checksum')
    return dict(first)


def require_no_completion(metadata, context):
    if completion_contract(metadata) is not None:
        raise ValueError(f'{context} does not support compact multiview completion motion; '
                         'use the audited comparison UI. Export would otherwise omit moving suffix rows.')


def _motion_record(record):
    if record.get('format') != 'event_supervised_hard_joint_alignment_v4':
        raise ValueError('Multiview completion requires hard-joint v4 motion')
    structure = copy.deepcopy(record['joint_structure'])
    structure.pop('input_sha256', None)
    return {key: record[key] for key in ('format', 'frame_count', 'group_angles')} | {
        'max_angle_degrees': record.get('max_angle_degrees', 2.), 'joint_structure': structure}


def validate_report_shape(report):
    counts = ('original_gaussian_count', 'prefix_count', 'added_gaussians', 'gaussian_count')
    if (report.get('format') != FORMAT or any(type(report.get(key)) is not int for key in counts)
            or report['original_gaussian_count'] <= 0 or report['prefix_count'] <= 0
            or not 1 <= report['added_gaussians'] <= MAX_ADDED
            or report['gaussian_count'] != report['prefix_count'] + report['added_gaussians']
            or report.get('unobserved_geometry_is_prior') is not True
            or not isinstance(report.get('scope'), str) or not report['scope']):
        raise ValueError('Invalid multiview completion report counts or prior disclosure')
    for field, hash_field in (('source', 'source_sha256'), ('completed_source', 'completed_sha256'),
                             ('parent_source', 'parent_source_sha256'), ('parent_report', 'parent_report_sha256'),
                             ('extension', 'extension_sha256'), ('parts', 'parts_sha256'),
                             ('parent_correction', 'parent_correction_sha256'), ('mesh_prior', 'mesh_prior_sha256')):
        if (not isinstance(report.get(field), str) or not Path(report[field]).is_absolute()
                or not isinstance(report.get(hash_field), str)
                or not re.fullmatch('[0-9a-f]{64}', report[hash_field])):
            raise ValueError(f'Invalid multiview completion report provenance: {field}')
    if not isinstance(report.get('parent_baseline'), str) or not Path(report['parent_baseline']).is_absolute():
        raise ValueError('Invalid multiview parent baseline path')
    for key in ('parent_poses_sha256', 'parent_dynamic_indices_sha256'):
        if not isinstance(report.get(key), str) or not re.fullmatch('[0-9a-f]{64}', report[key]):
            raise ValueError(f'Invalid multiview parent motion checksum: {key}')


def validate_extension_arrays(arrays, report, record, part_metadata=None):
    """Validate canonical suffix, integer identities and exact source-group map."""
    count = report['added_gaussians']
    required = {'canonical_xyz': (np.dtype('float32'), (count, 3)),
                'canonical_quaternions': (np.dtype('float32'), (count, 4)),
                'part_ids': (np.dtype('int32'), (count,)), 'group_ids': (np.dtype('int32'), (count,))}
    for key, (dtype, shape) in required.items():
        value = arrays.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid multiview extension array: {key}')
    if np.any(np.abs(np.linalg.norm(arrays['canonical_quaternions'], axis=1) - 1) > 1e-4):
        raise ValueError('Multiview suffix requires unit canonical quaternions')
    if 'metadata_json' not in arrays or arrays['metadata_json'].ndim != 0:
        raise ValueError('Missing multiview extension metadata')
    if not isinstance(json.loads(str(arrays['metadata_json'])), dict):
        raise ValueError('Invalid multiview extension metadata')
    structure = record['joint_structure']
    count_groups = len(structure['parents'])
    mapping = {}
    for group in structure['source_groups']:
        number = group['group']
        if type(number) is not int or not 0 <= number < count_groups:
            raise ValueError('Invalid multiview source group ID')
        for part in group['objects']:
            if type(part) is not int or part < 0 or part in mapping:
                raise ValueError('Invalid or repeated multiview source part ID')
            mapping[part] = number
    if np.any(arrays['group_ids'] < 0) or np.any(arrays['group_ids'] >= count_groups):
        raise ValueError('Multiview suffix permits rigid source groups only; no elastic bands')
    expected = np.asarray([mapping.get(int(part), -1) for part in arrays['part_ids']], np.int32)
    if not np.array_equal(expected, arrays['group_ids']):
        raise ValueError('Multiview part IDs disagree with hard-joint source groups')
    if part_metadata is not None:
        if (np.any(arrays['part_ids'] < 0) or np.any(arrays['part_ids'] >= len(part_metadata['objects']))
                or np.isin(arrays['part_ids'], part_metadata.get('deforming_objects', [])).any()):
            raise ValueError('Multiview suffix references unknown or deforming source parts')
        for group in structure['source_groups']:
            if 'object_names' in group and [part_metadata['objects'][part] for part in group['objects']] != group['object_names']:
                raise ValueError('Multiview source group object names disagree')


def verify_multiview_completion(cache, correction=None, audit=None, *, baseline=None):
    """Verify all extension artifacts and the immutable parent-cloud chain.

    An optional audit is checked against the same extension contract. Omitting
    it is allowed only for rendering a new candidate before its independent
    event audit exists; this function never creates or changes event approval.
    """
    records = [cache] + ([correction] if correction is not None else []) + ([audit] if audit is not None else [])
    contract = completion_contract(*records)
    if contract is None:
        return None
    from plyfile import PlyData
    from utils.comparison_renderer import verify_rebuild_provenance, vertex_rows_sha256
    from utils.render_sampling import require_same_sampling
    for field in ('report', 'extension'):
        if sha256_file(contract[field]) != contract[field + '_sha256']:
            raise ValueError(f'Multiview completion checksum mismatch: {field}')
    report = json.loads(Path(contract['report']).read_text())
    validate_report_shape(report)
    require_same_sampling(cache, report)
    if (report['completed_source'] != cache['source'] or report['gaussian_count'] != cache['gaussian_count']
            or report['extension'] != contract['extension'] or report['extension_sha256'] != contract['extension_sha256']):
        raise ValueError('Multiview report/cache/extension identity mismatch')
    for field, hash_field in (('source', 'source_sha256'), ('completed_source', 'completed_sha256'),
                             ('parent_source', 'parent_source_sha256'), ('parent_report', 'parent_report_sha256'),
                             ('parts', 'parts_sha256'), ('parent_correction', 'parent_correction_sha256'),
                             ('mesh_prior', 'mesh_prior_sha256')):
        if sha256_file(report[field]) != report[hash_field]:
            raise ValueError(f'Multiview provenance checksum mismatch: {field}')
    parent_report = json.loads(Path(report['parent_report']).read_text())
    if (parent_report.get('format') != 'lego_prior_surface_rebuild_v1'
            or parent_report['original_gaussian_count'] != report['original_gaussian_count']
            or parent_report['gaussian_count'] != report['prefix_count']):
        raise ValueError('Multiview completion parent rebuild identity mismatch')
    verify_rebuild_provenance(parent_report, report['source'], report['parent_source'])
    parent = PlyData.read(report['parent_source'])['vertex'].data
    completed = PlyData.read(report['completed_source'])['vertex'].data
    prefix = report['prefix_count']
    if (len(parent) != prefix or len(completed) != report['gaussian_count'] or parent.dtype != completed.dtype
            or vertex_rows_sha256(parent) != vertex_rows_sha256(completed[:prefix])):
        raise ValueError('Multiview completion modified the immutable parent prefix')
    tail = completed[prefix:]
    if any(not np.isfinite(tail[name]).all() for name in tail.dtype.names):
        raise ValueError('Multiview appended Gaussian has nonfinite fields')
    if any(np.any(tail[name] != 0) for name in tail.dtype.names if name.startswith('f_rest_')):
        raise ValueError('Multiview appended Gaussian must have DC-only appearance')
    parent_correction = json.loads(Path(report['parent_correction']).read_text())
    _motion_record(parent_correction)
    if correction is not None and _motion_record(correction) != _motion_record(parent_correction):
        raise ValueError('Multiview completion changed inherited hard-joint motion')
    if cache['frame_count'] != parent_correction['frame_count'] or cache['dynamic_count'] != len(parent_correction['joint_structure']['row_groups']):
        raise ValueError('Multiview completion changed the parent baked-row timeline')
    parent_baseline = Path(report['parent_baseline'])
    for filename, key, artifact in (('poses.npy', 'parent_poses_sha256', 'baseline_poses'),
                                    ('dynamic_indices.npy', 'parent_dynamic_indices_sha256', 'dynamic_indices')):
        if sha256_file(parent_baseline / filename) != report[key]:
            raise ValueError(f'Multiview parent motion checksum mismatch: {filename}')
        if audit is not None and audit.get('artifact_sha256', {}).get(artifact) != report[key]:
            raise ValueError(f'Multiview audit changed inherited motion: {filename}')
        if baseline is not None:
            child_file = Path(baseline) / filename
            if child_file.resolve() != (parent_baseline / filename).resolve() and sha256_file(child_file) != report[key]:
                raise ValueError(f'Multiview baseline changed inherited motion: {filename}')
    indices = np.load(parent_baseline / 'dynamic_indices.npy', allow_pickle=False)
    poses = np.load(parent_baseline / 'poses.npy', mmap_mode='r', allow_pickle=False)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) != cache['dynamic_count']
            or np.any(indices < 0) or np.any(indices >= prefix) or np.any(indices[1:] <= indices[:-1])
            or poses.dtype != np.float32 or poses.shape != (cache['frame_count'], len(indices), 7)):
        raise ValueError('Multiview parent baked motion arrays are invalid')
    expected_rest = np.column_stack([parent[name][indices] for name in ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')])
    if not np.array_equal(poses[0], expected_rest):
        raise ValueError('Multiview parent baked canonical rows differ from parent PLY')
    with np.load(contract['extension'], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(report['parts'], allow_pickle=False) as archive:
        part_metadata = json.loads(str(archive['metadata_json']))
    validate_extension_arrays(arrays, report, parent_correction, part_metadata)
    xyz = np.column_stack([tail[name] for name in ('x', 'y', 'z')])
    quaternion = np.column_stack([tail[name] for name in ('rot_0', 'rot_1', 'rot_2', 'rot_3')])
    if not np.array_equal(xyz, arrays['canonical_xyz']) or not np.array_equal(quaternion, arrays['canonical_quaternions']):
        raise ValueError('Multiview canonical extension differs from the exact PLY suffix')
    if audit is not None:
        for artifact, expected in (('completion_report', contract['report_sha256']),
                                   ('completion_extension', contract['extension_sha256'])):
            if audit.get('artifact_sha256', {}).get(artifact) != expected:
                raise ValueError(f'Multiview audit checksum mismatch: {artifact}')
    return {'report': report, 'arrays': arrays, 'parent_correction': parent_correction}


class CompactCompletion:
    """O(N) canonical storage; compute a suffix from the same corrected SE(3)."""
    def __init__(self, verified, device='cpu'):
        self.prefix_count = verified['report']['prefix_count']
        arrays = verified['arrays']
        self.rest = torch.from_numpy(np.concatenate((arrays['canonical_xyz'], arrays['canonical_quaternions']), 1)).to(device)
        self.groups = torch.from_numpy(arrays['group_ids'].astype(np.int64)).to(device)
        self.parent_correction = verified['parent_correction']
        self.expected_angles = torch.as_tensor(self.parent_correction['group_angles'], dtype=torch.float32, device=device)
        self._checked_model = None
        roots = self.parent_correction['joint_structure']['parents']
        static = np.flatnonzero(np.asarray(roots) == -1)
        self.moving_count = int(np.count_nonzero(~np.isin(arrays['group_ids'], static)))

    def _verify_model(self, correction):
        if not isinstance(correction, HardJointEventCorrection):
            raise ValueError('Compact completion requires the inherited hard-joint model')
        if not torch.equal(correction.group_angles, self.expected_angles):
            raise ValueError('Compact completion runtime changed inherited group angles')
        # Check once per model instance. These constants are not trainable in
        # completion rendering; only the already audited parent model is used.
        if self._checked_model is correction:
            return
        expected = correction_from_record(self.parent_correction, 'cpu')
        if (correction.parents != expected.parents or correction.parent_joints != expected.parent_joints
                or correction.max_angle_degrees != expected.max_angle_degrees):
            raise ValueError('Compact completion runtime differs from parent motion')
        actual = correction.state_dict()
        for name, value in expected.state_dict().items():
            if name not in actual or not torch.equal(actual[name].detach().cpu(), value):
                raise ValueError(f'Compact completion runtime differs from parent motion: {name}')
        self._checked_model = correction

    def posed(self, index, correction):
        self._verify_model(correction)
        index = correction._index(index)
        if index == 0:
            return self.rest
        quaternion, translation = correction.group_transforms(index)
        return torch.cat((qrotate(quaternion[self.groups], self.rest[:, :3]) + translation[self.groups],
                          qmul(quaternion[self.groups], self.rest[:, 3:])), -1)
