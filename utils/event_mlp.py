"""Small, anchored MLP corrections on an explicitly event-trained rigid graph.

The MLP predicts local node SE(3) residuals, composed BEFORE each base rigid
transform. Neighborhoods never cross base groups. Appearance is not a model
parameter. Static/unknown groups with exactly identity trajectories stay fixed.
This is a sparse deformation model, not an implementation of DeformableGS.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.event_only_motion import (
    EventOnlyMotion, HARD_RIGID_GROUP_KIND, build_graph,
    _axis_angle_quaternion, _quaternion_multiply, _rotate,
)

FORMAT = 'event_only_residual_mlp_v1'


def build_residual_graph(xyz, base, nodes_per_group=32):
    """Build G0-only local neighborhoods inside the event-inferred base groups."""
    if base.metadata['kind'] != HARD_RIGID_GROUP_KIND:
        raise ValueError('Residual MLP requires an event-inferred hard rigid baseline')
    if type(nodes_per_group) is not int or not 1 <= nodes_per_group <= 256:
        raise ValueError('nodes_per_group must be in [1,256]')
    xyz = np.asarray(xyz, dtype=np.float32)
    labels = base.knn_indices[:, 0].cpu().numpy()
    if xyz.shape != (len(labels), 3) or not np.isfinite(xyz).all():
        raise ValueError('G0 rows must match the baseline')
    frozen = ((base.translations.detach() == 0).all(dim=2).all(dim=0) &
              (base.rotation_vectors.detach() == 0).all(dim=2).all(dim=0)).cpu().numpy()
    indices = np.zeros((len(xyz), 4), np.int64)
    weights = np.zeros((len(xyz), 4), np.float32)
    centers, edges, groups = [], [], []
    offset = 0
    for group in range(len(base.centers)):
        rows = np.flatnonzero(labels == group)
        local = build_graph(xyz[rows], nodes=1 if frozen[group] else nodes_per_group, neighbors=4)
        k = local['knn_indices'].shape[1]
        indices[rows, :k] = local['knn_indices'] + offset
        indices[rows, k:] = offset
        weights[rows, :k] = local['knn_weights']
        centers.append(local['centers'])
        edges.append(local['edges'] + offset)
        groups.extend([group] * len(local['centers']))
        offset += len(local['centers'])
    return {'centers': np.concatenate(centers), 'edges': np.concatenate(edges),
            'knn_indices': indices, 'knn_weights': weights,
            'scene_scale': float(base.scene_scale),
            'metadata': {'kind': 'within_event_rigid_group_residual_graph', 'version': 1,
                         'node_source': 'explicit_G0_positions_only',
                         'base_group_ids': groups, 'frozen_groups': frozen.tolist(),
                         'nodes_per_group': nodes_per_group}}


class ResidualMLP(nn.Module):
    def __init__(self, width=64, depth=3, space_frequencies=2, time_frequencies=4):
        super().__init__()
        if (type(width) is not int or not 8 <= width <= 256 or
                type(depth) is not int or not 1 <= depth <= 8 or
                any(type(f) is not int or not 0 <= f <= 8 for f in (space_frequencies, time_frequencies))):
            raise ValueError('Invalid MLP architecture')
        self.space_frequencies, self.time_frequencies = space_frequencies, time_frequencies
        inputs = 3 * (1 + 2 * space_frequencies) + 1 + 2 * time_frequencies
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.SiLU()])
            inputs = width
        layers.append(nn.Linear(width, 6))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def encode(value, frequencies):
        pieces = [value]
        for i in range(frequencies):
            phase = value * (math.pi * 2**i)
            pieces.extend((phase.sin(), phase.cos()))
        return torch.cat(pieces, dim=-1)

    def forward(self, positions, time):
        space = self.encode(positions, self.space_frequencies)
        t = time.expand(len(positions), 1)
        # Structural t=0 anchor survives training and checkpoint reload.
        current = self.net(torch.cat((space, self.encode(t, self.time_frequencies)), -1))
        anchor = self.net(torch.cat((space, self.encode(t * 0, self.time_frequencies)), -1))
        return (current - anchor).tanh()


class EventResidualMLP(EventOnlyMotion):
    def __init__(self, base, graph, *, translation_fraction=.01, rotation_radians=.08,
                 width=64, depth=3, space_frequencies=2, time_frequencies=4):
        super().__init__(graph, base.times.cpu(), chunk_size=base.chunk_size)
        if base.metadata['kind'] != HARD_RIGID_GROUP_KIND:
            raise ValueError('Only explicit event-inferred hard rigid baselines are supported')
        if not all(math.isfinite(v) and 0 < v <= .25 for v in (translation_fraction, rotation_radians)):
            raise ValueError('Residual bounds must be finite in (0,.25]')
        self.settings = dict(translation_fraction=translation_fraction, rotation_radians=rotation_radians,
                             width=width, depth=depth, space_frequencies=space_frequencies,
                             time_frequencies=time_frequencies)
        self.translations.requires_grad_(False)
        self.rotation_vectors.requires_grad_(False)
        self.base = base.requires_grad_(False)
        group = torch.tensor(graph['metadata']['base_group_ids'], dtype=torch.long)
        frozen = torch.tensor(graph['metadata']['frozen_groups'], dtype=torch.bool)
        if (group.shape != (len(self.centers),) or frozen.shape != (len(base.centers),) or
                torch.any(group < 0) or torch.any(group >= len(base.centers))):
            raise ValueError('Invalid residual group membership')
        labels = base.knn_indices[:, 0].cpu()
        if (len(labels) != len(self.knn_indices) or
                not torch.equal(group[self.knn_indices], labels[:, None].expand_as(self.knn_indices)) or
                (len(self.edges) and torch.any(group[self.edges[:, 0]] != group[self.edges[:, 1]]))):
            raise ValueError('Residual neighborhoods must not cross base groups')
        actual_frozen = ((base.translations.detach() == 0).all(dim=2).all(dim=0) &
                         (base.rotation_vectors.detach() == 0).all(dim=2).all(dim=0)).cpu()
        if not torch.equal(frozen, actual_frozen):
            raise ValueError('Frozen groups must match exact identity base trajectories')
        self.register_buffer('group_ids', group)
        self.register_buffer('active_nodes', ~frozen[group])
        self.register_buffer('static_rows', frozen[labels])
        self.register_buffer('coordinate_origin', self.centers.mean(0))
        self.mlp = ResidualMLP(width, depth, space_frequencies, time_frequencies)
        self.metadata.update(kind=FORMAT, appearance_frozen=True, base_motion_frozen=True)

    def residual(self, time):
        # Validate in the timeline's float64 precision before MLP float32
        # conversion (e.g. float32(6.65) is greater than the 6.65 endpoint).
        value = torch.as_tensor(time, dtype=self.times.dtype, device=self.centers.device)
        if value.ndim != 0 or not torch.isfinite(value) or value < 0 or value > self.times[-1]:
            raise ValueError('Time outside the trained interval')
        normalized = (self.centers - self.coordinate_origin) / self.scene_scale
        output = self.mlp(normalized, (value / self.times[-1]).to(self.centers.dtype))
        output = output * self.active_nodes[:, None]
        translation = output[:, :3] * (self.settings['translation_fraction'] * self.scene_scale / math.sqrt(3))
        rotation = output[:, 3:] * (self.settings['rotation_radians'] / math.sqrt(3))
        return translation, rotation

    def at_time(self, time):
        delta, vector = self.residual(time)
        base_t, base_q = self.base.at_time(time)
        centers = self.base.centers[self.group_ids]
        q = base_q[self.group_ids]
        translation = (_rotate(q, self.centers + delta - centers) + centers +
                       base_t[self.group_ids] - self.centers)
        return translation, _quaternion_multiply(q, _axis_angle_quaternion(vector))

    def deform(self, xyz, rot, time, indices=None):
        if indices is None:
            if len(xyz) != len(self.static_rows):
                raise ValueError('Subset xyz requires original G0 row indices')
            rows = torch.arange(len(xyz), device=self.centers.device)
        else:
            rows = torch.as_tensor(indices, device=self.centers.device)
            if (rows.ndim != 1 or len(rows) != len(xyz) or rows.dtype not in (torch.int32, torch.int64) or
                    torch.any(rows < 0) or torch.any(rows >= len(self.static_rows))):
                raise ValueError('Invalid original G0 row indices')
        active = torch.nonzero(~self.static_rows[rows], as_tuple=True)[0]
        # Do not construct an autograd graph for millions of static neighbors.
        # index_copy also preserves every static source bit, including rotation.
        moved, oriented = super().deform(xyz[active], rot[active], time, rows[active])
        return xyz.index_copy(0, active, moved), rot.index_copy(0, active, oriented)

    def regularization(self, time, step):
        if not math.isfinite(step) or not 0 < step <= float(self.times[-1])/2:
            raise ValueError('Temporal regularization step must fit inside the motion interval')
        delta, vector = self.residual(time)
        scale = self.settings['translation_fraction'] * self.scene_scale
        magnitude = (delta / scale).square().mean() + (vector / self.settings['rotation_radians']).square().mean()
        arap = delta.sum() * 0
        if len(self.edges):
            a, b = self.edges.T
            rest = self.centers[b] - self.centers[a]
            moved = rest + delta[b] - delta[a]
            arap = ((_rotate(_axis_angle_quaternion(vector[a]), rest) - moved) / self.scene_scale).square().mean()
            arap += ((_rotate(_axis_angle_quaternion(vector[b]), rest) - moved) / self.scene_scale).square().mean()
        # Symmetric temporal prior in normalized motion units per second^2.
        midpoint = min(max(float(time), step), float(self.times[-1]) - step)
        before, middle, after = [self.residual(t) for t in (midpoint-step, midpoint, midpoint+step)]
        acceleration = ((after[0] - 2*middle[0] + before[0]) / self.scene_scale / step**2).square().mean()
        acceleration += ((after[1] - 2*middle[1] + before[1]) / step**2).square().mean()
        return magnitude, arap, acceleration


def save_mlp_checkpoint(path, motion, graph, base_payload, summary):
    from utils.event_only_checkpoint import _plain
    payload = {'format': FORMAT, 'base': base_payload,
               'graph': {k: _plain(v) if k == 'metadata' else torch.as_tensor(v).cpu() for k, v in graph.items()},
               'settings': motion.settings,
               'mlp_state': {k: v.detach().cpu() for k, v in motion.mlp.state_dict().items()},
               'training_summary': _plain(summary), 'times': motion.times.cpu(),
               'input_sha256': base_payload['input_sha256'], 'camera': base_payload['camera']}
    with Path(path).open('xb') as f:
        torch.save(payload, f)


def load_mlp_checkpoint(path, g0_path, device='cuda'):
    from utils.event_only_checkpoint import _validate_payload
    from utils.event_only_inputs import sha256_file
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Not an event residual MLP checkpoint')
    base = _validate_payload(payload['base'])
    if (payload['input_sha256'] != payload['base']['input_sha256'] or
            sha256_file(g0_path) != payload['input_sha256']['g0'] or
            payload['camera'] != payload['base']['camera'] or
            not torch.equal(payload['times'], base.times)):
        raise ValueError('MLP/base input identity mismatch')
    if any(not torch.isfinite(v).all() for v in payload['mlp_state'].values()):
        raise ValueError('Nonfinite MLP weights')
    motion = EventResidualMLP(base, payload['graph'], **payload['settings'])
    motion.mlp.load_state_dict(payload['mlp_state'], strict=True)
    return motion.to(device), payload['camera'], payload
