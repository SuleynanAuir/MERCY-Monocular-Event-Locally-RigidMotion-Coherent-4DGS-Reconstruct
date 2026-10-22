"""Bounded local deformation AFTER the complete, frozen Blender/V3 motion.

Unlike event_mlp, this model never estimates a replacement motion graph. It
uses V3's existing rigid labels (including completion rows), and preserves
static and elastic rows. Appearance and the hard-joint FK are not parameters.
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.event_mlp import ResidualMLP
from utils.event_only_motion import build_graph, _axis_angle_quaternion
from utils.event_refinement import qmul, qrotate

FORMAT = 'v3_blender_auxiliary_mlp_v1'


def event_objective(first, second, expected, floor):
    """The V3 audit's log-then-pool relative L1 and cosine, including sparse windows."""
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError('Expected matching square grayscale renders')
    factor = first.shape[0] // expected.shape[0]
    if factor < 1 or expected.shape != (first.shape[0]//factor, first.shape[1]//factor):
        raise ValueError('Incompatible event comparison scale')
    change = second.clamp_min(floor).log()-first.clamp_min(floor).log()
    change = F.avg_pool2d(change[None, None], factor)[0, 0]
    relative = (change-expected).abs().sum()/expected.abs().sum().clamp_min(1e-12)
    cosine = 1-(change*expected).sum()/(change.norm()*expected.norm()).clamp_min(1e-12)
    return relative+.2*cosine


def make_graph(xyz, groups, joint_pivots, joint_groups, nodes_per_group=32):
    xyz, groups = np.asarray(xyz, np.float32), np.asarray(groups, np.int64)
    if xyz.shape != (len(groups), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid canonical geometry')
    rows = np.flatnonzero(groups > 0)  # -1 elastic and 0 static stay exact V3.
    if not len(rows):
        raise ValueError('No moving rigid rows')
    scale = float(np.linalg.norm(np.ptp(xyz[rows], axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Degenerate moving geometry')
    pivots = np.asarray(joint_pivots, np.float32).reshape(-1, 3)
    pairs = np.asarray(joint_groups, np.int64).reshape(-1, 2)
    if len(pivots) != len(pairs):
        raise ValueError('Invalid joint pins')
    centers, edges, node_groups = [], [], []
    knn = np.zeros((len(rows), 4), np.int64)
    weights = np.zeros((len(rows), 4), np.float32)
    pins = np.ones(len(rows), np.float32)
    offset = 0
    for group in np.unique(groups[rows]):
        selected = np.flatnonzero(groups[rows] == group)
        local = build_graph(xyz[rows[selected]], nodes=nodes_per_group, neighbors=4)
        k = local['knn_indices'].shape[1]
        knn[selected, :k] = local['knn_indices'] + offset
        knn[selected, k:] = offset
        weights[selected, :k] = local['knn_weights']
        centers.append(local['centers'])
        edges.append(local['edges'] + offset)
        node_groups.extend([int(group)] * len(local['centers']))
        offset += len(local['centers'])
        attached = pivots[np.any(pairs == group, axis=1)]
        if len(attached):
            distance = np.linalg.norm(xyz[rows[selected], None] - attached[None], axis=-1).min(1)
            # Exact zero at each pivot, C1 smooth transition over 3% of extent.
            u = np.clip(distance / (scale * .03), 0, 1)
            pins[selected] = u*u*(3-2*u)
    return dict(rows=rows, groups=groups[rows], centers=np.concatenate(centers),
                edges=np.concatenate(edges), node_groups=np.asarray(node_groups, np.int64),
                knn=knn, weights=weights, pins=pins, scene_scale=np.asarray(scale, np.float32))


class V3AuxiliaryMLP(nn.Module):
    def __init__(self, graph, frame_count, translation_fraction=.005, rotation_radians=.025):
        super().__init__()
        if type(frame_count) is not int or frame_count < 2:
            raise ValueError('Invalid V3 timeline')
        if not all(math.isfinite(v) and 0 < v <= .05 for v in (translation_fraction, rotation_radians)):
            raise ValueError('Residual bounds must be in (0,.05]')
        self.settings = dict(frame_count=frame_count, translation_fraction=translation_fraction,
                             rotation_radians=rotation_radians)
        for name, value in graph.items():
            self.register_buffer(name, torch.as_tensor(value))
        if (self.weights.shape != self.knn.shape or self.knn.shape != (len(self.rows), 4)
                or not torch.allclose(self.weights.sum(1), torch.ones(len(self.rows)))
                or torch.any(self.node_groups[self.knn] != self.groups[:, None])
                or (len(self.edges) and torch.any(self.node_groups[self.edges[:, 0]] != self.node_groups[self.edges[:, 1]]))):
            raise ValueError('Residual neighborhoods must remain inside V3 groups')
        self.register_buffer('origin', self.centers.mean(0))
        self.mlp = ResidualMLP()

    def residual(self, frame):
        if not isinstance(frame, (int, float)) or not math.isfinite(frame) or not 0 <= frame < self.settings['frame_count']:
            raise ValueError('Frame outside V3 timeline')
        time = self.centers.new_tensor(frame / (self.settings['frame_count'] - 1))
        return self.mlp((self.centers-self.origin)/self.scene_scale, time)

    def deform(self, xyz, rotation, group_quaternion, frame):
        """Apply local residual in V3's moving group coordinates; keep all other rows exact."""
        output = self.residual(frame)
        blend = (output[self.knn] * self.weights[..., None]).sum(1) * self.pins[:, None]
        delta = blend[:, :3] * (self.settings['translation_fraction'] * self.scene_scale / math.sqrt(3))
        vector = blend[:, 3:] * (self.settings['rotation_radians'] / math.sqrt(3))
        q = group_quaternion[self.groups]
        # Transform the local small rotation into world space, then compose.
        world_vector = qrotate(q, vector)
        moved = xyz[self.rows] + qrotate(q, delta)
        oriented = qmul(_axis_angle_quaternion(world_vector), rotation[self.rows])
        return xyz.index_copy(0, self.rows, moved), rotation.index_copy(0, self.rows, oriented)

    def regularization(self, frame):
        output = self.residual(frame)
        magnitude = output.square().mean()
        smooth = output.sum()*0
        if len(self.edges):
            a, b = self.edges.T
            smooth = (output[a]-output[b]).square().mean()
        middle = min(max(frame, 1), self.settings['frame_count']-2)
        acceleration = (self.residual(middle-1)-2*self.residual(middle)+self.residual(middle+1)).square().mean()
        return magnitude, smooth, acceleration


def save_checkpoint(path, model, provenance):
    torch.save(dict(format=FORMAT, settings=model.settings, provenance=provenance,
                    graph={name: value.detach().cpu() for name, value in model.named_buffers()
                           if name != 'origin'},
                    mlp_state={k: v.detach().cpu() for k, v in model.mlp.state_dict().items()}), path)


def load_checkpoint(path, device='cuda'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format'] != FORMAT:
        raise ValueError('Not a V3 auxiliary checkpoint')
    model = V3AuxiliaryMLP(payload['graph'], **payload['settings']).to(device)
    model.mlp.load_state_dict(payload['mlp_state'], strict=True)
    return model, payload['provenance']
