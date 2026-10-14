"""Hard-connected, axis-only event correction of an articulated source prior.

Every rigid Gaussian row is rebuilt with one SE(3) per source motion group.
Translations are consequences of shared joint points, never learned offsets.
Closed-loop nodes and all their ancestors are frozen at the source angles;
unmodelled elastic-band rows retain their original baked motion unchanged.
"""
import math
import operator

import torch

from utils.event_refinement import qconj, qmul, qrotate


def joint_forest(parents, parent_joints, joint_groups):
    """Validate the forest and derive the unavoidable closed-loop locks."""
    count = len(parents)
    if not count or len(parent_joints) != count:
        raise ValueError('Invalid joint parent arrays')
    roots, order, pending = [], [], set(range(count))
    for group, parent in enumerate(parents):
        if parent == -1:
            if parent_joints[group] != -1:
                raise ValueError('A root cannot have a parent joint')
            roots.append(group)
        elif parent < 0 or parent >= count or parent == group:
            raise ValueError('Invalid joint parent')
        elif not 0 <= parent_joints[group] < len(joint_groups):
            raise ValueError('Invalid parent joint index')
        elif set(joint_groups[parent_joints[group]]) != {group, parent}:
            raise ValueError('Parent joint does not connect its child and parent')
    while pending:
        ready = sorted(group for group in pending if parents[group] == -1 or parents[group] in order)
        if not ready:
            raise ValueError('Parent relations contain a cycle')
        order.extend(ready)
        pending.difference_update(ready)
    tree_edges = {value for value in parent_joints if value >= 0}
    if len(tree_edges) != count-len(roots):
        raise ValueError('A tree edge was reused')
    loops = sorted(set(range(len(joint_groups)))-tree_edges)
    locked = set(roots)
    for edge in loops:
        for group in joint_groups[edge]:
            while group != -1:
                locked.add(group)
                group = parents[group]
    return roots, order, loops, sorted(locked)


class HardJointEventCorrection(torch.nn.Module):
    def __init__(self, structure, max_angle_degrees=2.0):
        super().__init__()
        if structure.get('format') != 'source_hard_joint_structure_v1':
            raise ValueError('Unsupported hard-joint structure')
        if not math.isfinite(max_angle_degrees) or not 0 < max_angle_degrees <= 5:
            raise ValueError('Joint correction angle must be in (0,5] degrees')
        self.frame_count = structure['frame_count']
        self.group_count = len(structure['parents'])
        self.max_angle_degrees = float(max_angle_degrees)
        self.joint_tolerance = float(structure['joint_tolerance'])
        if (not isinstance(self.frame_count, int) or self.frame_count < 2 or not 1 <= self.group_count <= 64
                or not math.isfinite(self.joint_tolerance) or not 0 < self.joint_tolerance <= 1e-5):
            raise ValueError('Invalid hard-joint size/tolerance')

        def integer_vector(values, label):
            value = torch.as_tensor(values)
            if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f'Invalid {label}')
            return value.long()

        rows = integer_vector(structure['row_groups'], 'row groups')
        if not len(rows) or torch.any(rows < -1) or torch.any(rows >= self.group_count):
            raise ValueError('Invalid rigid/band row groups')
        parents = integer_vector(structure['parents'], 'parents').tolist()
        parent_joints = integer_vector(structure['parent_joints'], 'parent joints').tolist()
        joint_groups = torch.as_tensor(structure['joint_groups'])
        if joint_groups.numel() == 0:
            joint_groups = torch.empty((0, 2), dtype=torch.long)
        if (joint_groups.ndim != 2 or joint_groups.shape[1] != 2
                or joint_groups.dtype not in (torch.int32, torch.int64)
                or torch.any(joint_groups < 0) or torch.any(joint_groups >= self.group_count)
                or torch.any(joint_groups[:, 0] == joint_groups[:, 1])):
            raise ValueError('Invalid joint group pairs')
        pairs = [tuple(sorted(pair)) for pair in joint_groups.tolist()]
        if len(set(pairs)) != len(pairs):
            raise ValueError('Duplicate joint group pair')
        self.parents, self.parent_joints = parents, parent_joints
        self.roots, self.order, self.loop_edges, self.locked_groups = joint_forest(parents, parent_joints, joint_groups.tolist())
        if sorted(structure['locked_groups']) != self.locked_groups or sorted(structure['loop_edges']) != self.loop_edges:
            raise ValueError('Closed-loop/ancestor lock metadata is inconsistent')
        source_q = torch.as_tensor(structure['source_quaternions'], dtype=torch.float32)
        source_t = torch.as_tensor(structure['source_translations'], dtype=torch.float32)
        pivots = torch.as_tensor(structure['joint_pivots'], dtype=torch.float32).reshape(-1, 3)
        axes = torch.as_tensor(structure['joint_axes'], dtype=torch.float32).reshape(-1, 3)
        envelopes = torch.as_tensor(structure['envelopes'], dtype=torch.float32)
        if (source_q.shape != (self.frame_count, self.group_count, 4)
                or source_t.shape != (self.frame_count, self.group_count, 3)
                or pivots.shape != (len(joint_groups), 3) or axes.shape != pivots.shape
                or envelopes.shape != (self.frame_count, self.group_count)
                or any(not torch.isfinite(value).all() for value in (source_q, source_t, pivots, axes, envelopes))
                or torch.any((source_q.norm(dim=-1)-1).abs() > 1e-4)
                or torch.any((axes.norm(dim=-1)-1).abs() > 1e-4)
                or torch.any(envelopes < 0) or torch.any(envelopes > 1.00001)
                or torch.any(envelopes[0] != 0)):
            raise ValueError('Invalid hard-joint transforms/axes/envelopes')
        identity = source_q.new_tensor([1., 0., 0., 0.])
        if (torch.any((source_q[0]-identity).abs() > 1e-6) or torch.any(source_t[0].abs() > 1e-7)
                or torch.any((source_q[:, self.roots]-identity).abs() > 1e-6)
                or torch.any(source_t[:, self.roots].abs() > 1e-7)):
            raise ValueError('Canonical first frame and fixed roots must be identity')
        trainable = torch.ones(self.group_count, dtype=torch.float32)
        trainable[self.locked_groups] = 0
        for key, value in (('row_groups', rows), ('joint_groups', joint_groups.long()),
                           ('joint_pivots', pivots), ('joint_axes', torch.nn.functional.normalize(axes, dim=-1)),
                           ('source_quaternions', torch.nn.functional.normalize(source_q, dim=-1)),
                           ('source_translations', source_t), ('envelopes', envelopes), ('trainable_mask', trainable)):
            self.register_buffer(key, value)
        self.register_buffer('rigid_rows', torch.nonzero(rows >= 0, as_tuple=True)[0])
        self.group_angles = torch.nn.Parameter(torch.zeros(self.group_count))
        # Check every source frame once, including every non-tree loop edge.
        with torch.no_grad():
            for index in range(self.frame_count):
                self.group_transforms(index)

    def _index(self, index):
        if isinstance(index, bool):
            raise ValueError('Boolean is not a frame index')
        try:
            index = operator.index(index)
        except TypeError as error:
            raise ValueError('Hard-joint correction requires an integer frame index') from error
        if not 0 <= index < self.frame_count:
            raise ValueError('Time outside the hard-joint timeline')
        return index

    def _joint_errors(self, quaternion, translation):
        first, second = self.joint_groups[:, 0], self.joint_groups[:, 1]
        a = qrotate(quaternion[first], self.joint_pivots)+translation[first]
        b = qrotate(quaternion[second], self.joint_pivots)+translation[second]
        return (a-b).norm(dim=-1)

    def group_transforms(self, index):
        index = self._index(index)
        source_q, source_t = self.source_quaternions[index], self.source_translations[index]
        angles = self.group_angles.tanh()*self.trainable_mask*self.envelopes[index]*(self.max_angle_degrees*math.pi/180)
        rotations, translations = [None]*self.group_count, [None]*self.group_count
        for group in self.order:
            parent = self.parents[group]
            if parent == -1:
                # All root groups are stationary; make that exact, not an
                # approximate float32 identity from the source archive.
                rotations[group] = source_q.new_tensor([1., 0., 0., 0.])
                translations[group] = source_t.new_zeros(3)
                continue
            edge = self.parent_joints[group]
            half = angles[group]/2
            residual = torch.cat((half.cos()[None], half.sin()*self.joint_axes[edge]))
            relative = qmul(qconj(source_q[parent]), source_q[group])
            rotation = qmul(rotations[parent], qmul(residual, relative))
            rotation = torch.nn.functional.normalize(rotation, dim=-1)
            pivot = self.joint_pivots[edge]
            rotations[group] = rotation
            translations[group] = qrotate(rotations[parent], pivot)+translations[parent]-qrotate(rotation, pivot)
        quaternion, translation = torch.stack(rotations), torch.stack(translations)
        errors = self._joint_errors(quaternion, translation)
        if (not torch.isfinite(quaternion).all() or not torch.isfinite(translation).all()
                or torch.any(errors > self.joint_tolerance)):
            raise ValueError('Hard-joint closure tolerance exceeded; candidate is not renderable')
        return quaternion, translation

    def joint_errors(self, index):
        quaternion, translation = self.group_transforms(index)
        return self._joint_errors(quaternion, translation)

    def forward(self, rest, baked, index=None):
        index = self._index(index)
        expected = (len(self.row_groups), 7)
        if (rest.shape != expected or baked.shape != expected or rest.dtype != self.source_quaternions.dtype
                or baked.dtype != rest.dtype or rest.device != self.row_groups.device or baked.device != rest.device
                or not torch.isfinite(rest).all() or not torch.isfinite(baked).all()
                or torch.any(rest[:, 3:].norm(dim=-1) < 1e-8) or torch.any(baked[:, 3:].norm(dim=-1) < 1e-8)):
            raise ValueError('Invalid canonical/baked Gaussian rows for hard-joint correction')
        if index == 0:
            return rest
        quaternion, translation = self.group_transforms(index)
        rows = self.rigid_rows
        groups = self.row_groups[rows]
        rigid = torch.cat((qrotate(quaternion[groups], rest[rows, :3])+translation[groups],
                           qmul(quaternion[groups], rest[rows, 3:])), -1)
        # index_copy keeps band rows exactly intact and preserves autograd.
        return baked.index_copy(0, rows, rigid)

    def regularization(self):
        values = self.group_angles.tanh()*self.trainable_mask
        return .01*values.square().sum()/self.trainable_mask.sum().clamp_min(1)
