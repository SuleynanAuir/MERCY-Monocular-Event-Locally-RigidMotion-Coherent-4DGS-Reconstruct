"""SC-GS-style sparse control field composed on the frozen V3 motion.

Adapted from ControlNodeWarp.cal_nn_weight/forward and DeformNetwork in
utils/time_utils.py: Gaussian node kernels, local-frame warps, a shared
position/time MLP, and separate translation/rotation/scale heads. This small
residual adapter uses pure PyTorch, without the full trainer's PyTorch3D import.
V3 nodes/part labels stay fixed; this is not a replacement reconstruction.
"""
import math

import numpy as np
import torch
from torch import nn

from utils.event_mlp import ResidualMLP
from utils.event_only_motion import _axis_angle_quaternion
from utils.event_refinement import qmul, qrotate

FORMAT = 'v4_scgs_pose_residual_field_v2'


def make_scgs_graph(renderer):
    graph = renderer.graph()
    with torch.no_grad():
        canonical = renderer.base_pose(0)[0][torch.as_tensor(graph['rows'], device='cuda')].cpu().numpy()
    graph['canonical'] = canonical
    # Use SC-GS's Gaussian-distance kernel, with K=3 inside each V3 part.
    graph['knn'] = graph['knn'][:, :3].copy()
    offsets = canonical[:, None]-graph['centers'][graph['knn']]
    graph['offsets'] = offsets
    radius = np.empty(len(graph['centers']), np.float32)
    for group in np.unique(graph['node_groups']):
        selected = graph['groups'] == group
        extent = float(np.linalg.norm(np.ptp(canonical[selected], axis=0)))
        radius[graph['node_groups'] == group] = max(extent*.1, float(graph['scene_scale'])*.002)
    distance2 = (offsets*offsets).sum(-1)
    weight = np.exp(-distance2/(2*radius[graph['knn']]**2))+1e-7
    graph['weights'] = (weight/weight.sum(1, keepdims=True)).astype(np.float32)
    graph['radius'] = radius
    # Condition time through the actual V3 articulation. Identical V3 poses
    # produce identical residuals, including at rests and repeated cycles.
    # Rotation matrices avoid quaternion sign ambiguity and angle wrapping.
    with torch.no_grad():
        axes = torch.eye(3, device='cuda')[None]
        motion = []
        for frame in range(renderer.meta['frame_count']):
            quaternion, translation = renderer.correction.group_transforms(frame)
            rotation = qrotate(quaternion[:, None], axes.expand(len(quaternion), -1, -1))
            motion.append(torch.cat((rotation.reshape(len(quaternion), 9),
                                     translation/float(graph['scene_scale'])), -1).cpu().numpy())
    graph['motion'] = np.asarray(motion, np.float32)
    return graph


class NodeFieldNetwork(nn.Module):
    """SC-GS's node MLP with V3-pose-conditioned time and separate geometry heads."""
    def __init__(self, width=128, depth=4):
        super().__init__()
        # Encode canonical position plus the frozen rotation/translation at t.
        inputs = 3*(1+2*4)+12*(1+2*2)
        self.layers = nn.ModuleList()
        self.skip = depth//2
        for i in range(depth):
            self.layers.append(nn.Linear(inputs if i == 0 else width+(inputs if i == self.skip else 0), width))
        self.translation = nn.Linear(width, 3)
        self.local_rotation = nn.Linear(width, 3)
        self.rotation = nn.Linear(width, 3)
        self.scaling = nn.Linear(width, 3)
        for head in (self.translation, self.local_rotation, self.rotation, self.scaling):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, position, motion):
        inputs = torch.cat((ResidualMLP.encode(position, 4),
                            ResidualMLP.encode(motion, 2)), -1)
        hidden = inputs
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                hidden = torch.cat((hidden, inputs), -1)
            hidden = torch.nn.functional.silu(layer(hidden))
        return torch.cat([head(hidden) for head in (self.translation, self.local_rotation, self.rotation, self.scaling)], -1)


class SCGSV4Field(nn.Module):
    def __init__(self, graph, frame_count, translation_fraction=.0005,
                 rotation_radians=.005, scale_fraction=.02, width=128, depth=4):
        super().__init__()
        if type(frame_count) is not int or frame_count < 2:
            raise ValueError('Invalid V3 timeline')
        if not all(math.isfinite(v) and 0 < v <= .05 for v in (translation_fraction, rotation_radians, scale_fraction)):
            raise ValueError('Expected small finite V4 residual bounds')
        self.settings = dict(frame_count=frame_count, translation_fraction=translation_fraction,
                             rotation_radians=rotation_radians, scale_fraction=scale_fraction,
                             width=width, depth=depth)
        for key, value in graph.items():
            self.register_buffer(key, torch.as_tensor(value))
        if (self.knn.shape != (len(self.rows), 3) or self.weights.shape != self.knn.shape
                or torch.any(self.node_groups[self.knn] != self.groups[:, None])
                or not torch.allclose(self.weights.sum(1), torch.ones(len(self.rows)))
                or torch.any(self.weights < 0) or torch.any(self.groups <= 0)
                or self.offsets.shape != (len(self.rows), 3, 3)
                or self.motion.ndim != 3 or self.motion.shape[0] != frame_count or self.motion.shape[2] != 12
                or int(self.node_groups.max()) >= self.motion.shape[1]
                or (len(self.edges) and torch.any(self.node_groups[self.edges[:, 0]] != self.node_groups[self.edges[:, 1]]))):
            raise ValueError('SC-GS bindings must stay within the existing moving V3 parts')
        self.register_buffer('origin', self.centers.mean(0))
        self.mlp = NodeFieldNetwork(width, depth)

    def residual(self, frame):
        if not isinstance(frame, (int, float)) or not math.isfinite(frame) or not 0 <= frame < self.settings['frame_count']:
            raise ValueError('Frame outside V3 timeline')
        position = (self.centers-self.origin)/self.scene_scale
        lower, upper = math.floor(frame), math.ceil(frame)
        motion = self.motion[lower]*(upper-frame)+self.motion[upper]*(frame-lower) if lower != upper else self.motion[lower]
        return (self.mlp(position, motion[self.node_groups])-
                self.mlp(position, self.motion[0, self.node_groups])).tanh()

    def local_values(self, frame):
        raw = self.residual(frame)
        translation = raw[:, :3]*(self.settings['translation_fraction']*self.scene_scale/math.sqrt(3))
        local_vector = raw[:, 3:6]*(self.settings['rotation_radians']/math.sqrt(3))
        vector = raw[:, 6:9]*(self.settings['rotation_radians']/math.sqrt(3))
        log_scale = raw[:, 9:]*math.log1p(self.settings['scale_fraction'])
        return translation, local_vector, vector, log_scale

    def deform(self, xyz, rotation, group_quaternion, frame):
        translation, local_vector, vector, _ = self.local_values(frame)
        local_q = _axis_angle_quaternion(local_vector)
        local_warp = (qrotate(local_q[self.knn], self.offsets)-self.offsets+translation[self.knn])
        delta = (local_warp*self.weights[..., None]).sum(1)*self.pins[:, None]
        # Local rotations have a lever arm; bound the FINAL field displacement.
        cap = self.settings['translation_fraction']*self.scene_scale
        delta = delta*(cap/delta.norm(dim=1, keepdim=True).clamp_min(cap)).clamp_max(1)
        q = group_quaternion[self.groups]
        moved = xyz[self.rows]+qrotate(q, delta)
        residual_vector = (vector[self.knn]*self.weights[..., None]).sum(1)*self.pins[:, None]
        oriented = qmul(_axis_angle_quaternion(qrotate(q, residual_vector)), rotation[self.rows])
        return xyz.index_copy(0, self.rows, moved), rotation.index_copy(0, self.rows, oriented)

    def deform_scaling(self, scaling, frame):
        log_scale = self.local_values(frame)[3]
        residual = (log_scale[self.knn]*self.weights[..., None]).sum(1)*self.pins[:, None]
        return scaling.index_copy(0, self.rows, scaling[self.rows]+residual)

    def regularization(self, frame):
        raw = self.residual(frame)
        magnitude = raw.square().mean()
        shape = raw.sum()*0
        if len(self.edges):
            a, b = self.edges.T
            translation, local_vector, _, _ = self.local_values(frame)
            edge = self.centers[b]-self.centers[a]
            changed = edge+translation[b]-translation[a]
            shape = ((qrotate(_axis_angle_quaternion(local_vector[a]), edge)-changed)/self.scene_scale).square().mean()
            shape += (raw[a]-raw[b]).square().mean()
        middle = min(max(frame, 1), self.settings['frame_count']-2)
        temporal = (self.residual(middle-1)-2*self.residual(middle)+self.residual(middle+1)).square().mean()
        return magnitude, shape, temporal

    def parameter_groups(self, learning_rate):
        # V3 supplies the positions. Learn fine local shape first; positional
        # and rotational residual heads receive much smaller steps.
        motion = list(self.mlp.translation.parameters())+list(self.mlp.local_rotation.parameters())+list(self.mlp.rotation.parameters())
        ids = {id(p) for p in motion}
        return [dict(params=[p for p in self.mlp.parameters() if id(p) not in ids], lr=learning_rate, lr_factor=1.),
                dict(params=motion, lr=learning_rate*.001, lr_factor=.001)]


def save_checkpoint(path, model, provenance):
    torch.save(dict(format=FORMAT, settings=model.settings, provenance=provenance,
                    graph={k: v.detach().cpu() for k, v in model.named_buffers() if k != 'origin'},
                    mlp_state={k: v.detach().cpu() for k, v in model.mlp.state_dict().items()}), path)


def load_checkpoint(path, device='cuda'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') == 'v4_local_joint_mlp_v1':
        from utils.v4_local_mlp import from_payload
        return from_payload(payload, device)
    if payload.get('format') == 'v4_joint_trajectory_residual_v1':
        from utils.v4_joint_trajectory import from_payload
        return from_payload(payload, device)
    if payload.get('format') != FORMAT:
        raise ValueError('Not a V4 SC-GS residual-field checkpoint')
    model = SCGSV4Field(payload['graph'], **payload['settings']).to(device)
    model.mlp.load_state_dict(payload['mlp_state'], strict=True)
    return model, payload['provenance']
