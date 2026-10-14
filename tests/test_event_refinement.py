import unittest

import torch

from utils.event_refinement import (RigidMotionCorrection, TemporalRigidMotionCorrection, ArticulatedEventCorrection,
                                    correction_from_record, cubic_controls, qmul, qconj, qrotate)


class RigidCorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        xyz = torch.randn(20, 3)
        quat = torch.nn.functional.normalize(torch.randn(20, 4), dim=-1)
        self.rest = torch.cat((xyz, quat), -1)
        q = torch.tensor([.9238795, 0., .3826834, 0.])
        self.baked = torch.cat((qrotate(q, xyz) + torch.tensor([.1, -.2, .3]),
                                qmul(q, quat)), -1)

    def test_identity_correction_reproduces_motion(self):
        actual = RigidMotionCorrection()(self.rest, self.baked)
        torch.testing.assert_close(actual, self.baked, atol=1e-6, rtol=1e-5)

    def test_rest_state_is_fixed_for_nonzero_correction(self):
        model = RigidMotionCorrection()
        with torch.no_grad():
            model.raw.copy_(torch.linspace(-1, 1, 7))
        torch.testing.assert_close(model(self.rest, self.rest), self.rest, atol=1e-6, rtol=1e-5)

    def test_shared_rigid_part_retains_distances(self):
        model = RigidMotionCorrection()
        with torch.no_grad():
            model.raw.copy_(torch.linspace(-2, 2, 7))
        actual = model(self.rest, self.baked)
        torch.testing.assert_close(torch.cdist(actual[:, :3], actual[:, :3]),
                                   torch.cdist(self.rest[:, :3], self.rest[:, :3]), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(actual[:, 3:].norm(dim=-1), torch.ones(20))

    def test_gradients_are_finite_and_nonzero(self):
        model = RigidMotionCorrection()
        actual = model(self.rest, self.baked)
        (actual * torch.randn_like(actual)).sum().backward()
        self.assertTrue(torch.isfinite(model.raw.grad).all())
        self.assertGreater(model.raw.grad.abs().sum().item(), 0)

    def test_quaternion_inverse_and_broadcast(self):
        q = self.rest[:, 3:]
        identity = qmul(q, qconj(q))
        torch.testing.assert_close(identity, torch.tensor([1., 0., 0., 0.]).expand_as(q))
        torch.testing.assert_close(qrotate(qconj(q), qrotate(q, self.rest[:, :3])),
                                   self.rest[:, :3], atol=2e-6, rtol=1e-5)

    def test_temporal_model_initially_equals_global_model(self):
        old, new = RigidMotionCorrection(), TemporalRigidMotionCorrection(400)
        with torch.no_grad():
            old.raw.normal_(); new.raw.copy_(old.raw)
        for index in (0, 100, 203, 399):
            torch.testing.assert_close(new(self.rest, self.baked, index), old(self.rest, self.baked))

    def test_temporal_rigidity_and_fixed_rest(self):
        model = TemporalRigidMotionCorrection(400)
        with torch.no_grad():
            model.offsets.normal_(); model.raw.normal_()
        for index in (0, 37, 171, 399):
            actual = model(self.rest, self.baked, index)
            torch.testing.assert_close(torch.cdist(actual[:, :3], actual[:, :3]),
                                       torch.cdist(self.rest[:, :3], self.rest[:, :3]), atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(model(self.rest, self.rest, index), self.rest, atol=2e-6, rtol=1e-5)

    def test_temporal_controls_are_smooth_and_receive_gradients(self):
        model = TemporalRigidMotionCorrection(401)
        with torch.no_grad():
            model.offsets.normal_(std=.1)
        left, center, right = [cubic_controls(model.offsets, t, 401) for t in (99.999, 100., 100.001)]
        torch.testing.assert_close((center-left)/.001, (right-center)/.001, atol=1e-4, rtol=.1)
        (model(self.rest, self.baked, 109).square().mean()+model.regularization()).backward()
        self.assertTrue(torch.isfinite(model.offsets.grad).all())
        self.assertGreater(model.offsets.grad.abs().sum().item(), 0)

    def test_temporal_serialization_and_invalid_records(self):
        model = TemporalRigidMotionCorrection(400, 9)
        with torch.no_grad():
            model.offsets.normal_(); model.raw.normal_()
        record = {'format': 'event_supervised_temporal_rigid_alignment_v2', 'frame_count': 400,
                  'knot_count': 9, 'raw': model.raw.tolist(), 'temporal_offsets': model.offsets.tolist()}
        reloaded = correction_from_record(record)
        torch.testing.assert_close(reloaded(self.rest, self.baked, 201), model(self.rest, self.baked, 201))
        with self.assertRaises(ValueError):
            reloaded(self.rest, self.baked)
        with self.assertRaises(ValueError):
            reloaded(self.rest, self.baked, 400)
        record['temporal_offsets'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            correction_from_record(record)

    def test_articulated_residuals_are_rigid_per_group_and_fixed_at_start(self):
        structure = {'groups': [0]*10+[1]*10, 'representatives': [0,10],
                     'centers': [self.rest[:10, :3].mean(0).tolist(), self.rest[10:, :3].mean(0).tolist()],
                     'envelopes': [[0,0], [.25,.25], [1,1], [.25,.25], [0,0]],
                     'contacts': [[0,10]], 'anchors': [1]}
        model = ArticulatedEventCorrection(5, 4, **structure)
        with torch.no_grad():
            model.group_raw.normal_(std=.3)
        torch.testing.assert_close(model(self.rest, self.rest, 0), self.rest, atol=2e-6, rtol=1e-5)
        actual = model(self.rest, self.baked, 2)
        for rows in (slice(0,10), slice(10,20)):
            torch.testing.assert_close(torch.cdist(actual[rows, :3], actual[rows, :3]),
                                       torch.cdist(self.rest[rows, :3], self.rest[rows, :3]), atol=3e-6, rtol=1e-5)
        (actual.square().mean()+model.regularization()).backward()
        self.assertTrue(torch.isfinite(model.group_raw.grad).all())
        self.assertGreater(model.group_raw.grad.abs().sum().item(), 0)
        record = {'format': 'event_supervised_articulated_alignment_v3', 'frame_count': 5,
                  'knot_count': 4, 'raw': model.raw.tolist(), 'temporal_offsets': model.offsets.tolist(),
                  'group_raw': model.group_raw.tolist(), 'group_structure': structure}
        loaded = correction_from_record(record)
        torch.testing.assert_close(loaded(self.rest, self.baked, 2), actual)


if __name__ == '__main__':
    unittest.main()
