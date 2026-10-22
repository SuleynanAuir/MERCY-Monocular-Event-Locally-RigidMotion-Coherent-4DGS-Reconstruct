"""Motion/gradient/checkpoint invariants for optional event-only MLP residuals."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.event_only_motion import EventOnlyMotion, HARD_RIGID_GROUP_KIND
from utils.event_only_checkpoint import save_event_only_checkpoint
from utils.event_mlp import build_residual_graph, EventResidualMLP, save_mlp_checkpoint, load_mlp_checkpoint


class EventMLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.xyz = torch.tensor([[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.], [.1, .1, 1.],
                                 [1., 0., 1.], [1.1, 0., 1.], [1., .1, 1.], [1.1, .1, 1.]])
        self.rot = torch.tensor([[1., 0., 0., 0.]]).repeat(8, 1)
        self.hashes = {k: str(i)*64 for i, k in enumerate(('g0', 'events', 'intrinsics', 'extrinsics'))}
        self.base_graph = {'centers': torch.stack((self.xyz[:4].mean(0), self.xyz[4:].mean(0))),
                           'knn_indices': torch.tensor([[0]]*4+[[1]]*4), 'knn_weights': torch.ones(8, 1),
                           'edges': torch.empty((0, 2), dtype=torch.long), 'scene_scale': 2.,
                           'metadata': {'kind': HARD_RIGID_GROUP_KIND, 'version': 1,
                                        'binding_source': 'explicit_G0_raw_events_and_camera',
                                        'node_source': 'G0_group_member_centers', 'blender_parts': False,
                                        'input_sha256': self.hashes}}
        self.base = EventOnlyMotion(self.base_graph, [0., .5, 1.])
        with torch.no_grad():
            self.base.translations[:, 1, 0] = torch.tensor([.1, .2])
            self.base.rotation_vectors[:, 1, 2] = torch.tensor([.2, .4])
        self.graph = build_residual_graph(self.xyz.numpy(), self.base, 4)
        self.model = EventResidualMLP(self.base, self.graph)

    def test_zero_network_matches_base_at_fractional_times(self):
        for t in (0., .13, .6, 1.):
            for got, expected in zip(self.model.deform(self.xyz, self.rot, t), self.base.deform(self.xyz, self.rot, t)):
                torch.testing.assert_close(got, expected, atol=2e-7, rtol=0)

    def test_gradient_trains_network_without_base_or_static_drift(self):
        before = {k: v.clone() for k, v in self.base.state_dict().items()}
        opt = torch.optim.Adam(self.model.mlp.parameters(), lr=.005)
        target = self.base.deform(self.xyz, self.rot, .7)[0].detach().clone()
        target[4:, 1] += .005
        losses = []
        for _ in range(60):
            opt.zero_grad()
            moved, _ = self.model.deform(self.xyz, self.rot, .7)
            loss = (moved-target).square().sum()
            losses.append(float(loss.detach()))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
            opt.step()
        self.assertLess(losses[-1], losses[0]*.05)
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        for k, value in before.items():
            self.assertTrue(torch.equal(value, self.base.state_dict()[k]))
        for t in (0., .3, 1.):
            x, q = self.model.deform(self.xyz, self.rot, t)
            self.assertTrue(torch.equal(x[:4], self.xyz[:4]))
            self.assertTrue(torch.equal(q[:4], self.rot[:4]))
            if t == 0:
                self.assertTrue(torch.equal(x, self.xyz))
                self.assertTrue(torch.equal(q, self.rot))
            dt, dr = self.model.residual(t)
            self.assertLessEqual(float(dt.norm(dim=1).max()), .020001)
            self.assertLessEqual(float(dr.norm(dim=1).max()), .080001)

    def test_group_boundary_and_time_validation(self):
        broken = copy.deepcopy(self.graph)
        broken['knn_indices'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'cross'):
            EventResidualMLP(self.base, broken)
        for t in (-.01, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                self.model.residual(t)
        terms = self.model.regularization(.5, .025)
        self.assertTrue(all(torch.isfinite(v) for v in terms))

    def test_nonrepresentable_float32_endpoint_is_valid(self):
        base = EventOnlyMotion(self.base_graph, [0., .5, 6.65])
        with torch.no_grad():
            base.translations.copy_(self.base.translations)
            base.rotation_vectors.copy_(self.base.rotation_vectors)
        model = EventResidualMLP(base, self.graph)
        for value in model.deform(self.xyz, self.rot, 6.65):
            self.assertTrue(torch.isfinite(value).all())
        for value in model.regularization(6.65, .025):
            self.assertTrue(torch.isfinite(value))
        with self.assertRaises(ValueError):
            model.residual(6.6501)

    def test_checkpoint_reload_reproduces_motion_and_rejects_wrong_g0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ['x', 'y', 'z', 'opacity'] + [f'f_dc_{i}' for i in range(3)]
            fields += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
            raw = np.zeros(8, dtype=[(name, 'f4') for name in fields])
            for i, name in enumerate(('x', 'y', 'z')):
                raw[name] = self.xyz[:, i].numpy()
            raw['rot_0'] = 1
            g0 = root/'g0.ply'
            PlyData([PlyElement.describe(raw, 'vertex')]).write(str(g0))
            self.hashes['g0'] = hashlib.sha256(g0.read_bytes()).hexdigest()
            # The constructor copied the hashes; explicitly rebuild the fixture.
            base = EventOnlyMotion(self.base_graph, self.base.times)
            with torch.no_grad():
                base.translations.copy_(self.base.translations)
                base.rotation_vectors.copy_(self.base.rotation_vectors)
            camera = {'width': 32, 'height': 32, 'fx': 25., 'fy': 25., 'cx': 16., 'cy': 16.,
                      'w2c': np.eye(4).tolist()}
            payload = save_event_only_checkpoint(root/'base.pth', base, self.base_graph, camera,
                                                  {'allowed_modalities': ['g0', 'events', 'camera'],
                                                   'input_sha256': self.hashes}, {})
            model = EventResidualMLP(base, self.graph)
            with torch.no_grad():
                model.mlp.net[-1].weight.normal_(0, .1)
            save_mlp_checkpoint(root/'mlp.pth', model, self.graph, payload, {})
            loaded, _, _ = load_mlp_checkpoint(root/'mlp.pth', g0, 'cpu')
            for t in (0., .37, 1.):
                for a, b in zip(model.deform(self.xyz, self.rot, t), loaded.deform(self.xyz, self.rot, t)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            with g0.open('ab') as f:
                f.write(b'wrong')
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_mlp_checkpoint(root/'mlp.pth', g0, 'cpu')


if __name__ == '__main__':
    unittest.main()
