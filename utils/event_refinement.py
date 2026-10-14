"""Conservative differentiable correction of a rigid articulated motion prior.

One shared similarity conjugation changes the motion coordinate alignment, not
individual Gaussian colors/opacities or part rigidity. This cannot repair every
wrong part assignment; an independent event audit decides whether to accept it.
"""
import torch


def qmul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    av, bv = torch.broadcast_tensors(av, bv)
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def qconj(q):
    return q * q.new_tensor([1, -1, -1, -1])


def qrotate(q, points):
    # q must be unit length. torch.cross requires explicit broadcasting.
    vector, points = torch.broadcast_tensors(q[..., 1:], points)
    cross = 2 * torch.cross(vector, points, dim=-1)
    return points + q[..., :1] * cross + torch.cross(vector, cross, dim=-1)


class RigidMotionCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.zeros(7))

    def parameters_physical(self, index=None):
        angle = self.raw[:3].tanh() * (5 * torch.pi / 180)
        half = angle / 2
        zero = torch.zeros_like(half[0])
        x = torch.stack((half[0].cos(), half[0].sin(), zero, zero))
        y = torch.stack((half[1].cos(), zero, half[1].sin(), zero))
        z = torch.stack((half[2].cos(), zero, zero, half[2].sin()))
        q = qmul(z, qmul(y, x))
        translation = .03 * self.raw[3:6].tanh()
        scale = (.1 * self.raw[6].tanh()).exp()
        return q, translation, scale

    def forward(self, rest, baked, index=None):
        q0 = torch.nn.functional.normalize(rest[:, 3:], dim=-1)
        qt = torch.nn.functional.normalize(baked[:, 3:], dim=-1)
        motion = qmul(qt, qconj(q0))
        original_translation = baked[:, :3] - qrotate(motion, rest[:, :3])
        q, translation, scale = self.parameters_physical(index)
        corrected_motion = qmul(qmul(q, motion), qconj(q))
        xyz = (qrotate(corrected_motion, rest[:, :3]) + scale * qrotate(q, original_translation)
               + translation - qrotate(corrected_motion, translation))
        return torch.cat((xyz, qmul(corrected_motion, rest[:, 3:])), -1)


def cubic_controls(controls, time, frame_count):
    """Uniform Catmull-Rom controls; C1 continuous, shared by every rigid part."""
    if frame_count < 2 or len(controls) < 4:
        raise ValueError('Temporal correction needs >=2 frames and >=4 controls')
    x = torch.as_tensor(time, device=controls.device, dtype=controls.dtype) * ((len(controls) - 1) / (frame_count - 1))
    if torch.any((x < 0) | (x > len(controls) - 1)):
        raise ValueError('Time outside the correction timeline')
    knot = x.floor().long().clamp(max=len(controls) - 2)
    u = (x - knot)[..., None]
    p0, p1, p2, p3 = [controls[(knot + offset).clamp(0, len(controls)-1)] for offset in (-1, 0, 1, 2)]
    return .5 * (2*p1 + (-p0+p2)*u + (2*p0-5*p1+4*p2-p3)*u.square()
                 + (-p0+3*p1-3*p2+p3)*u.pow(3))


class TemporalRigidMotionCorrection(RigidMotionCorrection):
    """Low-dimensional smooth residual alignment, never per-Gaussian displacement.

    A shared time-dependent conjugation preserves each rigid group's pairwise
    distances. Articulation attachment quality is not guaranteed by that alone;
    coverage and RGB constraints and the unchanged audit still apply.
    """
    def __init__(self, frame_count, knots=17):
        super().__init__()
        if not isinstance(frame_count, int) or frame_count < 2 or not 4 <= knots <= 65:
            raise ValueError('Invalid temporal correction size')
        self.frame_count = frame_count
        self.offsets = torch.nn.Parameter(torch.zeros(knots, 7))

    def parameters_physical(self, index=None):
        if index is None:
            raise ValueError('Temporal correction requires a frame index')
        q, translation, scale = super().parameters_physical()
        residual = cubic_controls(self.offsets, index, self.frame_count).tanh()
        angle = residual[:3] * (2 * torch.pi / 180)
        half, zero = angle / 2, torch.zeros_like(angle[0])
        x = torch.stack((half[0].cos(), half[0].sin(), zero, zero))
        y = torch.stack((half[1].cos(), zero, half[1].sin(), zero))
        z = torch.stack((half[2].cos(), zero, zero, half[2].sin()))
        return qmul(qmul(z, qmul(y, x)), q), translation + .015 * residual[3:6], scale * (.03 * residual[6]).exp()

    def regularization(self):
        values = self.offsets.tanh()
        # Prevent a different noisy transform every instant. Controls are about
        # 25 source frames apart by default, including withheld event intervals.
        return .002 * values.square().mean() + .1 * (values[2:] - 2*values[1:-1] + values[:-2]).square().mean()


class ArticulatedEventCorrection(TemporalRigidMotionCorrection):
    """Bounded rigid residual per motion group, with prior-contact protection.

    Unlike similarity conjugation alone, this can correct a joint's rotation
    amplitude. No per-point offsets, appearance edits, or image warps are used.
    """
    def __init__(self, frame_count, knots, groups, representatives, centers, envelopes, contacts, anchors):
        super().__init__(frame_count, knots)
        groups = torch.as_tensor(groups)
        if groups.ndim != 1 or groups.dtype not in (torch.int32, torch.int64) or not len(groups):
            raise ValueError('Invalid motion group IDs')
        count = int(groups.max())+1
        if groups.min() < 0 or count > 64 or len(torch.unique(groups)) != count:
            raise ValueError('Motion groups must be contiguous, <=64')
        representatives = torch.as_tensor(representatives, dtype=torch.long)
        centers = torch.as_tensor(centers, dtype=torch.float32)
        envelopes = torch.as_tensor(envelopes, dtype=torch.float32)
        contacts = torch.as_tensor(contacts, dtype=torch.long).reshape(-1, 2)
        anchors = torch.as_tensor(anchors, dtype=torch.long)
        if (representatives.shape != (count,) or centers.shape != (count, 3)
                or envelopes.shape != (frame_count, count) or anchors.ndim != 1
                or not torch.isfinite(centers).all() or not torch.isfinite(envelopes).all()
                or torch.any(envelopes < 0) or torch.any(envelopes > 1.0001)
                or torch.any(envelopes[0] != 0)):
            raise ValueError('Invalid group pivots/envelopes')
        for rows in (representatives, contacts, anchors):
            if torch.any(rows < 0) or torch.any(rows >= len(groups)):
                raise ValueError('Contact/group row outside the Gaussian motion array')
        if not torch.equal(groups[representatives], torch.arange(count)):
            raise ValueError('Group representative mismatch')
        for key, value in (('groups', groups.long()), ('representatives', representatives), ('centers', centers),
                           ('envelopes', envelopes), ('contacts', contacts), ('anchors', anchors)):
            self.register_buffer(key, value)
        self.group_raw = torch.nn.Parameter(torch.zeros(count, 6))
        self.contact_loss = torch.zeros(())

    def forward(self, rest, baked, index=None):
        if len(rest) != len(self.groups):
            raise ValueError('Motion grouping and Gaussian rows disagree')
        original = super().forward(rest, baked, index)
        reps = self.representatives
        motion = qmul(F_normalize(original[reps, 3:]), qconj(F_normalize(rest[reps, 3:])))
        pivot = qrotate(motion, self.centers-rest[reps, :3]) + original[reps, :3]
        residual = self.group_raw.tanh() * self.envelopes[index, :, None]
        half = residual[:, :3] * (4 * torch.pi / 180) / 2
        zero = torch.zeros_like(half[:, 0])
        x = torch.stack((half[:, 0].cos(), half[:, 0].sin(), zero, zero), -1)
        y = torch.stack((half[:, 1].cos(), zero, half[:, 1].sin(), zero), -1)
        z = torch.stack((half[:, 2].cos(), zero, zero, half[:, 2].sin()), -1)
        q = qmul(z, qmul(y, x))[self.groups]
        pivot = pivot[self.groups]
        xyz = qrotate(q, original[:, :3]-pivot) + pivot + .012 * residual[self.groups, 3:]
        displacement = xyz-original[:, :3]
        contact = displacement.new_zeros(())
        if len(self.contacts):
            contact = contact + (displacement[self.contacts[:, 0]]-displacement[self.contacts[:, 1]]).square().mean() / .005**2
        if len(self.anchors):
            contact = contact + displacement[self.anchors].square().mean() / .005**2
        self.contact_loss = contact
        return torch.cat((xyz, qmul(q, original[:, 3:])), -1)

    def regularization(self):
        return super().regularization() + .005*self.group_raw.tanh().square().mean() + .1*self.contact_loss


def F_normalize(value):
    return torch.nn.functional.normalize(value, dim=-1)


def correction_from_record(record, device='cpu'):
    """One validated loader for optimization, viewer and portable export."""
    kind = record.get('format')
    if kind == 'event_supervised_hard_joint_alignment_v4':
        from utils.joint_refinement import HardJointEventCorrection
        model = HardJointEventCorrection(record['joint_structure'], record.get('max_angle_degrees', 2.)).to(device)
        if record.get('frame_count') != model.frame_count:
            raise ValueError('Hard-joint correction timeline mismatch')
        angles = torch.as_tensor(record['group_angles'], dtype=torch.float32, device=device)
        if angles.shape != model.group_angles.shape or not torch.isfinite(angles).all():
            raise ValueError('Invalid hard-joint correction coefficients')
        with torch.no_grad():
            model.group_angles.copy_(angles)
            for index in range(model.frame_count):
                model.group_transforms(index)
        return model
    if kind == 'event_supervised_rigid_motion_alignment_v1':
        model = RigidMotionCorrection().to(device)
    elif kind in ('event_supervised_temporal_rigid_alignment_v2', 'event_supervised_articulated_alignment_v3'):
        if kind == 'event_supervised_articulated_alignment_v3':
            model = ArticulatedEventCorrection(record['frame_count'], record['knot_count'],
                                               **record['group_structure']).to(device)
            raw_groups = torch.as_tensor(record['group_raw'], dtype=torch.float32, device=device)
            if raw_groups.shape != model.group_raw.shape or not torch.isfinite(raw_groups).all():
                raise ValueError('Invalid articulated correction coefficients')
            with torch.no_grad():
                model.group_raw.copy_(raw_groups)
        else:
            model = TemporalRigidMotionCorrection(record['frame_count'], record['knot_count']).to(device)
        offsets = torch.as_tensor(record['temporal_offsets'], dtype=torch.float32, device=device)
        if offsets.shape != model.offsets.shape or not torch.isfinite(offsets).all():
            raise ValueError('Invalid temporal controls')
        with torch.no_grad():
            model.offsets.copy_(offsets)
    else:
        raise ValueError('Unsupported motion correction format')
    raw = torch.as_tensor(record['raw'], dtype=torch.float32, device=device)
    if raw.shape != (7,) or not torch.isfinite(raw).all():
        raise ValueError('Invalid shared correction')
    with torch.no_grad():
        model.raw.copy_(raw)
    return model
