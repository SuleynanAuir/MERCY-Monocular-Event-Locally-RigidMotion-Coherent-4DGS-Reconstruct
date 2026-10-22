"""Small joint-space temporal residuals on frozen V3; no per-point softening."""
import math
import numpy as np
import torch
from torch import nn
from utils.event_refinement import qconj, qmul, qrotate

FORMAT = 'v4_joint_trajectory_residual_v1'


def hinge_angles(quaternions, parents, parent_joints, axes):
    """Unwrap relative hinge angles in float64, avoiding float32 unwrap noise."""
    q = torch.as_tensor(quaternions, dtype=torch.float64)
    result = np.zeros(q.shape[:2], np.float64)
    for g, parent in enumerate(parents):
        if parent < 0:
            continue
        relative = qmul(qconj(q[:, parent]), q[:, g])
        axis = torch.as_tensor(axes[parent_joints[g]], dtype=torch.float64)
        projection = (relative[:, 1:]*axis).sum(-1)
        if float((relative[:, 1:]-projection[:, None]*axis).abs().max()) > 1e-4:
            raise ValueError('Expected an axis-aligned hinge; cannot smooth a ball joint as a hinge')
        result[:, g] = np.unwrap((2*torch.atan2(projection, relative[:, 0])).numpy())
    return result


def anchor_frames(angle):
    # Preserve the first/last poses and all turning points. Small numerical
    # plateaus do not introduce hundreds of anchors on otherwise static joints.
    velocity = np.diff(angle)
    threshold = max(float(np.ptp(angle))*1e-7, 1e-10)
    signs = np.sign(np.where(np.abs(velocity) > threshold, velocity, 0))
    anchors = {0, 1, len(angle)-2, len(angle)-1}
    previous = 0
    for index, sign in enumerate(signs):
        if sign and previous and sign != previous:
            anchors.update((index-1, index, index+1))
        if sign:
            previous = sign
    return sorted(i for i in anchors if 0 <= i < len(angle))


def smooth_proposal(angles, locked, strength, max_angle_radians, coupled=()):
    """Constrained least squares: proximity + final angular third differences.

    Coupled groups share an exactly proportional residual, retaining their
    inherited transmission ratio. Original V3 offsets are not overwritten.
    """
    from scipy.optimize import lsq_linear
    from scipy.sparse import csr_matrix, eye, vstack
    a = np.asarray(angles, np.float64)
    if a.ndim != 2 or len(a) < 8 or not np.isfinite(a).all() or not math.isfinite(strength) or strength <= 0 or not 0 < max_angle_radians <= .01:
        raise ValueError('Invalid constrained smoothing settings')
    count, groups = a.shape
    d3 = csr_matrix(np.diff(np.eye(count), n=3, axis=0))
    operator = vstack((eye(count), math.sqrt(strength)*d3)).tocsr()
    residual = np.zeros_like(a)
    clusters = [list(cluster) for cluster in coupled]
    covered = [g for cluster in clusters for g in cluster]
    if len(set(covered)) != len(covered) or any(g in locked or not 0 <= g < groups for g in covered):
        raise ValueError('Coupled groups must be unique unlocked joints')
    clusters += [[g] for g in range(groups) if g not in locked and g not in covered]
    anchors_by_group = {}
    for cluster in clusters:
        driver = max(cluster, key=lambda g: np.ptp(a[:, g]))
        reference = a[:, driver]
        if np.ptp(reference) < 1e-8:
            continue
        centered = reference-reference.mean()
        ratios = {g: float(np.dot(centered, a[:, g]-a[:, g].mean())/np.dot(centered, centered)) for g in cluster}
        anchors = sorted({i for g in cluster for i in anchor_frames(a[:, g])})
        free = np.ones(count, bool); free[anchors] = False
        limit = max_angle_radians/max(1., max(abs(v) for v in ratios.values()))
        lower = np.maximum(-limit, reference.min()-reference)
        upper = np.minimum(limit, reference.max()-reference)
        # Exact min/max values are fixed rather than fed as equal LSQ bounds.
        free &= upper-lower > 1e-14
        target = np.concatenate((np.zeros(count), -math.sqrt(strength)*(d3@reference)))
        if free.any():
            fit = lsq_linear(operator[:, free], target, bounds=(lower[free], upper[free]),
                             tol=1e-12, lsmr_tol=1e-12, max_iter=200)
            if not fit.success:
                raise RuntimeError('Constrained trajectory solve failed: '+fit.message)
            for g, ratio in ratios.items():
                residual[free, g] = fit.x*ratio
                anchors_by_group[str(g)] = np.flatnonzero(~free).tolist()
    return residual, anchors_by_group


class JointTrajectoryField(nn.Module):
    def __init__(self, data, max_angle_radians=.0017453292519943296):
        super().__init__()
        self.parents = list(data['parents']); self.parent_joints = list(data['parent_joints'])
        self.order = list(data['order']); self.locked = list(data['locked'])
        for key in ('rows', 'groups', 'canonical', 'base_q', 'base_t', 'pivots', 'axes', 'angles', 'scene_scale'):
            self.register_buffer(key, torch.as_tensor(data[key]))
        frames, groups = self.base_q.shape[:2]
        self.settings = dict(frame_count=frames, max_angle_radians=float(max_angle_radians))
        if (self.base_q.shape != (frames, groups, 4) or self.base_t.shape != (frames, groups, 3)
                or self.angles.shape != (frames, groups) or self.canonical.shape != (len(self.rows), 3)
                or self.groups.shape != self.rows.shape or torch.any(self.groups <= 0)
                or torch.any(self.groups >= groups) or not 0 < max_angle_radians <= .01
                or torch.any(self.angles[:, self.locked] != 0) or torch.any(self.angles[0] != 0)
                or torch.any(self.angles.abs() > max_angle_radians+1e-8)
                or any(not torch.isfinite(v).all() for v in (self.base_q, self.base_t, self.canonical, self.angles))):
            raise ValueError('Invalid V4 joint residual or protected V3 pose')
        self.register_buffer('centers', self.pivots.clone())
        dq, dt = self.delta_transforms()
        self.register_buffer('delta_q', dq)
        self.register_buffer('delta_t', dt)

    def delta_transforms(self):
        frames, groups = self.base_q.shape[:2]
        identity = self.base_q.new_tensor([1., 0., 0., 0.]).expand(frames, -1)
        rotation, translation = [None]*groups, [None]*groups
        for g in self.order:
            parent = self.parents[g]
            if parent < 0:
                rotation[g] = identity
                translation[g] = self.base_t.new_zeros((frames, 3))
                continue
            joint = self.parent_joints[g]
            world_axis = qrotate(self.base_q[:, parent], self.axes[joint])
            half = self.angles[:, g:g+1]/2
            own = torch.cat((half.cos(), half.sin()*world_axis), -1)
            rotation[g] = qmul(rotation[parent], own)
            pivot = qrotate(self.base_q[:, parent], self.pivots[joint])+self.base_t[:, parent]
            translation[g] = translation[parent]+(qrotate(rotation[parent], pivot)-qrotate(rotation[g], pivot))
        return torch.stack(rotation, 1), torch.stack(translation, 1)

    def deform(self, xyz, rotation, group_quaternion, frame):
        if type(frame) is not int or not 0 <= frame < self.settings['frame_count']:
            raise ValueError('Invalid V3 frame')
        if frame == 0 or not torch.any(self.angles[frame]):
            return xyz, rotation
        q, t = self.delta_q[frame, self.groups], self.delta_t[frame, self.groups]
        return (xyz.index_copy(0, self.rows, qrotate(q, xyz[self.rows])+t),
                rotation.index_copy(0, self.rows, qmul(q, rotation[self.rows])))

    def deform_scaling(self, scaling, frame):
        return scaling

    def trajectories(self, indices=None):
        indices = torch.arange(len(self.rows), device=self.rows.device) if indices is None else indices
        groups, canonical = self.groups[indices], self.canonical[indices]
        base = qrotate(self.base_q[:, groups], canonical[None])+self.base_t[:, groups]
        moved = qrotate(self.delta_q[:, groups], base)+self.delta_t[:, groups]
        return base, moved

    def validate_renderer(self, renderer):
        graph = renderer.graph()
        if not torch.equal(torch.as_tensor(graph['rows']), self.rows.cpu()) or not torch.equal(torch.as_tensor(graph['groups']), self.groups.cpu()):
            raise ValueError('V4 moving rows do not match V3')
        actual = renderer.base_pose(0)[0][self.rows.to('cuda')]
        if not torch.equal(actual.cpu(), self.canonical.cpu()):
            raise ValueError('V4 canonical geometry differs from V3')


def save_checkpoint(path, model, provenance):
    data = {k: v.detach().cpu() for k, v in model.named_buffers() if k not in ('centers', 'delta_q', 'delta_t')}
    data.update(parents=model.parents, parent_joints=model.parent_joints, order=model.order, locked=model.locked)
    torch.save(dict(format=FORMAT, data=data, max_angle_radians=model.settings['max_angle_radians'], provenance=provenance), path)


def from_payload(payload, device='cuda'):
    model = JointTrajectoryField(payload['data'], payload['max_angle_radians']).to(device)
    return model, payload['provenance']
