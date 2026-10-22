"""Pose-conditioned MLP for bounded LOCAL joint residuals on frozen V3.

The network mixes multiscale temporal correction bases, not global motion.
Root, closed loops, transmission gears, appearance and Gaussian scales stay
unchanged. Each affected part remains rigid under the existing joint FK.
"""
import numpy as np
import torch
from torch import nn
from utils.v4_joint_trajectory import JointTrajectoryField, hinge_angles, smooth_proposal

FORMAT = 'v4_local_joint_mlp_v1'
SOURCE_KEYS = ('rows', 'groups', 'canonical', 'base_q', 'base_t', 'pivots', 'axes', 'scene_scale')


def make_local_basis(data, active_groups, strengths=(1., 16., 256., 4096.)):
    angles = hinge_angles(data['base_q'], data['parents'], data['parent_joints'], data['axes'])
    active = list(active_groups)
    if not active or len(set(active)) != len(active) or any(g in data['locked'] or not 0 < g < angles.shape[1] for g in active):
        raise ValueError('Only unique unlocked local joints can be trained')
    protected = [g for g in range(angles.shape[1]) if g not in active]
    proposals, anchors = [], None
    for strength in strengths:
        delta, current = smooth_proposal(angles, protected, strength, np.deg2rad(.05))
        proposals.append(delta[:, active])
        if anchors is not None and anchors != current:
            raise ValueError('Temporal bases must share anchors')
        anchors = current
    basis = np.stack(proposals, -1)
    # Source-only motion descriptors. No image or held-out event features.
    a = angles[:, active]
    span = np.ptp(a, axis=0).clip(1e-8)
    normalized = (a-a.mean(0))/span
    velocity = np.gradient(normalized, axis=0)
    velocity /= np.max(np.abs(velocity), axis=0).clip(1e-8)
    pivot = np.asarray(data['pivots'])[np.asarray(data['parent_joints'])[active]] / float(data['scene_scale'])
    features = np.concatenate((normalized[..., None], np.sin(a)[..., None], np.cos(a)[..., None],
                               velocity[..., None], np.broadcast_to(pivot, (len(a), *pivot.shape)),
                               np.broadcast_to(np.eye(len(active)), (len(a), len(active), len(active)))), -1)
    return dict(basis=torch.tensor(basis, dtype=torch.float64),
                features=torch.tensor(features, dtype=torch.float32),
                active_groups=active, anchors=anchors, strengths=list(strengths)), angles


class LocalJointMLP(JointTrajectoryField):
    def __init__(self, data, local, width=32, gain=1., group_gains=None):
        source = dict(data)
        source['angles'] = torch.zeros(source['base_q'].shape[:2], dtype=torch.float32)
        super().__init__(source, np.deg2rad(.05))
        self.active_groups = list(local['active_groups'])
        group_gains = [1.]*len(self.active_groups) if group_gains is None else list(group_gains)
        self.local_settings = dict(width=width, gain=float(gain), group_gains=group_gains)
        self.anchors = local['anchors']; self.strengths = list(local['strengths'])
        if (not self.active_groups or len(set(self.active_groups)) != len(self.active_groups)
                or any(g in self.locked or not 0 < g < self.base_q.shape[1] for g in self.active_groups)
                or not np.isfinite(gain) or not 0 <= gain <= 2
                or len(group_gains) != len(self.active_groups)
                or any(not np.isfinite(v) or not 0 <= v <= 1 for v in group_gains)):
            raise ValueError('Invalid local MLP support or gain')
        self.register_buffer('basis', torch.as_tensor(local['basis'], dtype=torch.float64))
        self.register_buffer('features', torch.as_tensor(local['features'], dtype=torch.float32))
        expected = (len(self.base_q), len(self.active_groups), len(self.strengths))
        if self.basis.shape != expected or self.features.shape[:2] != expected[:2] or not torch.isfinite(self.basis).all() or not torch.isfinite(self.features).all():
            raise ValueError('Invalid local temporal basis')
        for i, group in enumerate(self.active_groups):
            if torch.any(self.basis[self.anchors[str(group)], i] != 0):
                raise ValueError('MLP bases must preserve turning anchors exactly')
        self.mlp = nn.Sequential(nn.Linear(self.features.shape[-1], width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, len(self.strengths)))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def predict_angles(self):
        # A convex-sized signed mixture keeps the auxiliary correction small.
        gates = self.mlp(self.features).tanh().double()*self.local_settings['gain']/len(self.strengths)
        local = (self.basis*gates).sum(-1)
        local = local*local.new_tensor(self.local_settings['group_gains'])
        angles = self.basis.new_zeros(self.base_q.shape[:2])
        return angles.index_copy(1, torch.tensor(self.active_groups, device=angles.device), local)

    @torch.no_grad()
    def refresh(self):
        predicted = self.predict_angles()
        if not torch.isfinite(predicted).all() or predicted.abs().max() > self.settings['max_angle_radians']:
            raise ValueError('Nonfinite or excessive MLP correction')
        self.angles.copy_(predicted.float())
        q, t = self.delta_transforms()
        self.delta_q.copy_(q); self.delta_t.copy_(t)
        return self

    def deform(self, xyz, rotation, group_quaternion, frame):
        if type(frame) is not int or not 0 <= frame < self.settings['frame_count']:
            raise ValueError('Invalid V3 frame')
        if frame == 0 or not torch.any(self.angles[frame]):
            return xyz, rotation
        # Do not even round-trip protected parts through identity quaternion
        # arithmetic: their positions/orientations must be bitwise unchanged.
        affected = torch.zeros_like(self.groups, dtype=torch.bool)
        affected_groups = set(self.active_groups)
        for g in self.order:
            if self.parents[g] in affected_groups:
                affected_groups.add(g)
        for group in affected_groups:
            affected |= self.groups == group
        rows, groups = self.rows[affected], self.groups[affected]
        from utils.event_refinement import qrotate, qmul
        return (xyz.index_copy(0, rows, qrotate(self.delta_q[frame, groups], xyz[rows])+self.delta_t[frame, groups]),
                rotation.index_copy(0, rows, qmul(self.delta_q[frame, groups], rotation[rows])))


def save_checkpoint(path, model, provenance):
    data = {k: getattr(model, k).detach().cpu() for k in SOURCE_KEYS}
    data.update(parents=model.parents, parent_joints=model.parent_joints, order=model.order, locked=model.locked)
    local = dict(basis=model.basis.cpu(), features=model.features.cpu(), active_groups=model.active_groups,
                 anchors=model.anchors, strengths=model.strengths)
    torch.save(dict(format=FORMAT, data=data, local=local, settings=model.local_settings,
                    mlp_state={k: v.detach().cpu() for k, v in model.mlp.state_dict().items()}, provenance=provenance), path)


def from_payload(payload, device='cuda'):
    model = LocalJointMLP(payload['data'], payload['local'], **payload['settings']).to(device)
    model.mlp.load_state_dict(payload['mlp_state'], strict=True)
    return model.refresh(), payload['provenance']
