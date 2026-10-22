import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from utils.v3_auxiliary_mlp import make_graph
from utils.scgs_v4_field import SCGSV4Field, save_checkpoint, load_checkpoint


class SCGSV4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        canonical = np.random.default_rng(43).normal(size=(80, 3)).astype(np.float32)
        canonical[20] = 0
        groups = np.repeat([0, 1, 2], [20, 30, 30])
        graph = make_graph(canonical, groups, [[0, 0, 0]], [[1, 2]], 8)
        graph['knn'] = graph['knn'][:, :3].copy()
        graph['canonical'] = canonical[graph['rows']]
        graph['offsets'] = graph['canonical'][:, None]-graph['centers'][graph['knn']]
        graph['radius'] = np.full(len(graph['centers']), .5, np.float32)
        weights = np.exp(-(graph['offsets']**2).sum(-1)/(2*.5**2))+1e-7
        graph['weights'] = weights/weights.sum(1, keepdims=True)
        phase = np.linspace(0, 2*np.pi, 400, dtype=np.float32)
        graph['motion'] = np.zeros((400, 3, 12), np.float32)
        graph['motion'][:, :, 0] = np.sin(phase[:, None])
        graph['motion'][:, :, 1] = np.cos(phase[:, None])
        self.model = SCGSV4Field(graph, 400, width=32)
        self.xyz = torch.from_numpy(canonical)
        self.rot = torch.tensor([[1., 0., 0., 0.]]).repeat(80, 1)
        self.scale = torch.full((80, 3), -5.)
        self.group_q = self.rot[:3]

    def nonzero(self):
        with torch.no_grad():
            for head in (self.model.mlp.translation, self.model.mlp.local_rotation,
                         self.model.mlp.rotation, self.model.mlp.scaling):
                head.weight.normal_(0, 2)

    def test_zero_field_is_exact_v3_and_anchor_survives_learning(self):
        for frame in (0, 91, 399):
            x, q = self.model.deform(self.xyz, self.rot, self.group_q, frame)
            s = self.model.deform_scaling(self.scale, frame)
            self.assertTrue(torch.equal(x, self.xyz) and torch.equal(q, self.rot))
            self.assertTrue(torch.equal(s, self.scale))
        self.nonzero()
        x, q = self.model.deform(self.xyz, self.rot, self.group_q, 0)
        self.assertTrue(torch.equal(x, self.xyz) and torch.equal(q, self.rot))
        self.assertTrue(torch.equal(self.model.deform_scaling(self.scale, 0), self.scale))

    def test_final_motion_shape_bounds_static_rows_and_pins(self):
        self.nonzero()
        x, q = self.model.deform(self.xyz, self.rot, self.group_q, 399)
        s = self.model.deform_scaling(self.scale, 399)
        self.assertTrue(torch.equal(x[:21], self.xyz[:21]))
        self.assertTrue(torch.equal(q[:21], self.rot[:21]))
        self.assertTrue(torch.equal(s[:21], self.scale[:21]))
        self.assertLessEqual(float((x-self.xyz).norm(dim=1).max()), float(self.model.scene_scale)*.0005+1e-6)
        self.assertLessEqual(float(((s-self.scale).exp()-1).abs().max()), .02+1e-6)
        self.assertLessEqual(float(2*torch.atan2(q[:, 1:].norm(dim=1), q[:, 0].abs()).max()), .005+1e-6)

    def test_geometry_heads_receive_gradients_and_regularizers_are_finite(self):
        self.nonzero()
        x, q = self.model.deform(self.xyz, self.rot, self.group_q, 123)
        s = self.model.deform_scaling(self.scale, 123)
        loss = x.square().mean()+q[:, 1:].sum()+s.square().mean()+sum(self.model.regularization(123))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for head in (self.model.mlp.translation, self.model.mlp.local_rotation,
                     self.model.mlp.rotation, self.model.mlp.scaling):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_checkpoint_round_trip(self):
        self.nonzero()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'field.pth'
            save_checkpoint(path, self.model, {'base': 'frozen V3'})
            other, provenance = load_checkpoint(path, 'cpu')
            self.assertEqual(provenance, {'base': 'frozen V3'})
            for frame in (0, 123, 399):
                a = self.model.deform(self.xyz, self.rot, self.group_q, frame)
                b = other.deform(self.xyz, self.rot, self.group_q, frame)
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                self.assertTrue(torch.equal(self.model.deform_scaling(self.scale, frame), other.deform_scaling(self.scale, frame)))

    def test_motion_heads_have_conservative_learning_rate(self):
        groups = self.model.parameter_groups(.001)
        parameters = [p for group in groups for p in group['params']]
        self.assertEqual(len(parameters), len({id(p) for p in self.model.parameters()}))
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertLess(groups[1]['lr'], groups[0]['lr'])

    def test_same_v3_pose_has_same_field_independent_of_clock(self):
        self.nonzero()
        self.model.motion[71].copy_(self.model.motion[42])
        self.assertTrue(torch.equal(self.model.residual(42), self.model.residual(71)))
        self.assertTrue(torch.equal(self.model.deform_scaling(self.scale, 42),
                                    self.model.deform_scaling(self.scale, 71)))


if __name__ == '__main__':
    unittest.main()
