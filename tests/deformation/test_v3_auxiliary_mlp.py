"""Auxiliary deformation must never replace or modify its frozen V3 base."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from utils.v3_auxiliary_mlp import make_graph, V3AuxiliaryMLP, save_checkpoint, load_checkpoint, event_objective


class V3AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        xyz = np.random.default_rng(42).normal(size=(100, 3)).astype(np.float32)
        xyz[30] = 0
        self.xyz = torch.from_numpy(xyz)
        self.rotation = torch.tensor([1., 0., 0., 0.]).repeat(100, 1)
        groups = np.repeat([-1, 0, 1, 2], [10, 20, 30, 40])
        self.graph = make_graph(xyz, groups, [[0, 0, 0]], [[1, 2]], nodes_per_group=8)
        self.model = V3AuxiliaryMLP(self.graph, 400)
        self.q = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)

    def nonzero(self):
        with torch.no_grad():
            self.model.mlp.net[-1].weight.normal_(0, .2)

    def test_zero_is_exact_v3_and_rest_stays_exact_after_learning(self):
        for frame in (0, 42, 399):
            xyz, q = self.model.deform(self.xyz, self.rotation, self.q, frame)
            self.assertTrue(torch.equal(xyz, self.xyz))
            self.assertTrue(torch.equal(q, self.rotation))
        self.nonzero()
        xyz, q = self.model.deform(self.xyz, self.rotation, self.q, 0)
        self.assertTrue(torch.equal(xyz, self.xyz))
        self.assertTrue(torch.equal(q, self.rotation))

    def test_static_elastic_and_joint_pins_preserved_with_bounded_motion(self):
        self.nonzero()
        xyz, q = self.model.deform(self.xyz, self.rotation, self.q, 399)
        self.assertTrue(torch.equal(xyz[:31], self.xyz[:31]))
        self.assertTrue(torch.equal(q[:31], self.rotation[:31]))
        self.assertGreater(float((xyz-self.xyz).norm()), 0)
        self.assertLessEqual(float((xyz-self.xyz).norm(dim=1).max()), float(self.model.scene_scale)*.005+1e-6)
        angle = 2*torch.atan2(q[:, 1:].norm(dim=1), q[:, 0].abs())
        self.assertLessEqual(float(angle.max()), .025+1e-6)

    def test_residual_moves_with_blender_frame_and_gradients_reach_network(self):
        self.nonzero()
        first, _ = self.model.deform(self.xyz, self.rotation, self.q, 123)
        flipped = torch.tensor([[0., 0., 0., 1.]]).repeat(3, 1)
        second, _ = self.model.deform(self.xyz, self.rotation, flipped, 123)
        a, b = first-self.xyz, second-self.xyz
        self.assertTrue(torch.allclose(a[:, :2], -b[:, :2], atol=1e-6))
        self.assertTrue(torch.allclose(a[:, 2], b[:, 2], atol=1e-6))
        loss = first.square().mean()+sum(self.model.regularization(123))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters()), 0)

    def test_checkpoint_reproduces_deformation(self):
        self.nonzero()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'aux.pth'
            save_checkpoint(path, self.model, {'base_sha256': 'example'})
            loaded, provenance = load_checkpoint(path, 'cpu')
            self.assertEqual(provenance, {'base_sha256': 'example'})
            for frame in (0, 71, 399):
                a = self.model.deform(self.xyz, self.rotation, self.q, frame)
                b = loaded.deform(self.xyz, self.rotation, self.q, frame)
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_cross_group_edges_and_out_of_range_frames_rejected(self):
        graph = dict(self.graph)
        graph['edges'] = np.asarray([[0, 8]])
        with self.assertRaisesRegex(ValueError, 'inside V3 groups'):
            V3AuxiliaryMLP(graph, 400)
        for frame in (-1, 400, float('nan')):
            with self.assertRaises(ValueError):
                self.model.residual(frame)

    def test_sparse_event_loss_matches_existing_audit(self):
        from utils.event_supervision import consistency_metrics, log_intensity
        from scripts.events.audit_lego_event_motion import pool
        first = torch.full((16, 16), .3)
        second = first.clone()
        second[4:12, 4:12] += .01
        expected = torch.tensor([[.02, -.01], [.01, .03]])
        change = pool(log_intensity(second.numpy(), .03)-log_intensity(first.numpy(), .03), 8)
        metrics = consistency_metrics(change, expected.numpy(), tolerance=1, minimum_change=.01)
        self.assertLess(float(expected.abs().sum()), 1)
        self.assertAlmostEqual(float(event_objective(first, second, expected, .03)),
                               metrics['relative_l1']+.2*(1-metrics['cosine']), places=5)


if __name__ == '__main__':
    unittest.main()
