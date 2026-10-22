import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from utils.event_refinement import qmul, qrotate
from utils.v4_joint_trajectory import JointTrajectoryField, smooth_proposal, save_checkpoint
from utils.scgs_v4_field import load_checkpoint


def fixture():
    frames = 64
    phase = torch.linspace(0, 2*np.pi, frames)
    angle = torch.stack((phase*0, .3*phase.sin(), .5*phase.sin()), 1)
    zero = torch.zeros_like(angle)
    q = torch.stack(((angle/2).cos(), zero, zero, (angle/2).sin()), -1)
    pivots = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    t = torch.zeros(frames, 3, 3)
    t[:, 1] = qrotate(q[:, 0], pivots[0])-qrotate(q[:, 1], pivots[0])
    t[:, 2] = qrotate(q[:, 1], pivots[1])+t[:, 1]-qrotate(q[:, 2], pivots[1])
    return dict(rows=torch.arange(1, 7), groups=torch.tensor([1, 1, 1, 2, 2, 2]),
                canonical=torch.tensor([[.1, .2, 0], [.2, .3, 0], [.3, .2, 0], [1.1, .1, 0], [1.2, .2, 0], [1.3, .1, 0]]),
                base_q=q, base_t=t, pivots=pivots, axes=torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
                parents=[-1, 0, 1], parent_joints=[-1, 0, 1], order=[0, 1, 2], locked=[0],
                angles=torch.zeros(frames, 3), scene_scale=torch.tensor(2.))


class JointTrajectoryTests(unittest.TestCase):
    def test_zero_residual_exact_and_scaling_is_frozen(self):
        m = JointTrajectoryField(fixture())
        x = torch.randn(7, 3); q = torch.randn(7, 4); s = torch.randn(7, 3)
        a, b = m.deform(x, q, None, 17)
        self.assertIs(a, x); self.assertIs(b, q)
        self.assertIs(m.deform_scaling(s, 17), s)
        self.assertTrue(torch.equal(m.delta_t, torch.zeros_like(m.delta_t)))

    def nonzero(self):
        data = fixture()
        data['angles'][:, 1] = .001*torch.sin(torch.linspace(0, np.pi, 64))
        data['angles'][:, 2] = -.0007*torch.sin(torch.linspace(0, np.pi, 64))
        data['angles'][[0, -1]] = 0
        return JointTrajectoryField(data)

    def test_hierarchy_keeps_shared_joints_closed(self):
        m = self.nonzero(); q = qmul(m.delta_q, m.base_q); t = qrotate(m.delta_q, m.base_t)+m.delta_t
        for child in (1, 2):
            parent, pivot = m.parents[child], m.pivots[m.parent_joints[child]]
            a = qrotate(q[:, parent], pivot)+t[:, parent]
            b = qrotate(q[:, child], pivot)+t[:, child]
            self.assertLess(float((a-b).norm(dim=-1).max()), 5e-7)

    def test_rigid_part_lengths_static_rows_and_anchors_preserved(self):
        m = self.nonzero(); before, after = m.trajectories()
        for start in (0, 3):
            a = torch.cdist(before[:, start:start+3], before[:, start:start+3])
            b = torch.cdist(after[:, start:start+3], after[:, start:start+3])
            torch.testing.assert_close(a, b, atol=4e-7, rtol=1e-6)
        x = torch.cat((torch.tensor([[8., 9., 10.]]), before[17]))
        rot = torch.tensor([[1., 0, 0, 0.]]).repeat(7, 1)
        y, _ = m.deform(x, rot, None, 17)
        self.assertTrue(torch.equal(y[0], x[0]))
        self.assertIs(m.deform(x, rot, None, 0)[0], x)

    def test_invalid_protected_or_excessive_residual_rejected(self):
        for group, frame, value in ((0, 20, .001), (1, 0, .001), (1, 20, .1)):
            data = fixture(); data['angles'][frame, group] = value
            with self.assertRaises(ValueError): JointTrajectoryField(data)

    def test_checkpoint_roundtrip_through_shared_loader(self):
        m = self.nonzero()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pth'; save_checkpoint(path, m, {'method': 'test'})
            other, p = load_checkpoint(path, 'cpu')
            self.assertEqual(p, {'method': 'test'})
            self.assertTrue(torch.equal(m.trajectories()[1], other.trajectories()[1]))

    def test_constrained_solve_reduces_jerk_keeps_anchors_and_gear_ratio(self):
        t = np.linspace(0, 4*np.pi, 120)
        angle = np.sin(t)+.0003*np.sin(17*t)
        a = np.stack((t*0, angle, angle/3), 1)
        delta, anchors = smooth_proposal(a, [0], 16, .002, [[1, 2]])
        self.assertLess(np.mean(np.diff(a+delta, 3, axis=0)**2), np.mean(np.diff(a, 3, axis=0)**2))
        np.testing.assert_allclose(delta[:, 1]/3, delta[:, 2], atol=1e-14)
        self.assertTrue(np.all(delta[:, 0] == 0)); self.assertLessEqual(np.abs(delta).max(), .002+1e-12)
        for group, frames in anchors.items(): self.assertTrue(np.all(delta[frames, int(group)] == 0))
        np.testing.assert_allclose(np.ptp(a+delta, axis=0), np.ptp(a, axis=0), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
