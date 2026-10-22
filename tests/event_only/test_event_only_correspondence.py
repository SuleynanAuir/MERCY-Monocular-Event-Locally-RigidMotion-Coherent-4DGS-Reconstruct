import unittest

import numpy as np
import torch

from utils.event_only_correspondence import make_feature_evidence, feature_reprojection_loss
from utils.event_only_motion import EventOnlyMotion


class FeatureCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        graph = {'centers': [[0., 0., 2.], [1., 0., 2.]],
                 'knn_indices': np.array([[0], [1]], np.int32),
                 'knn_weights': np.ones((2, 1), np.float32), 'edges': np.empty((0, 2), np.int64),
                 'scene_scale': 2., 'metadata': {}}
        self.motion = EventOnlyMotion(graph, [0, 1, 2])
        self.camera = {'w2c': np.eye(4), 'fx': 100., 'fy': 100., 'cx': 50., 'cy': 50.}
        self.features = {'points': np.array([[0., 0., 2.], [1., 0., 2.]]),
                         'tracks': np.array([[[50., 50.], [100., 50.]], [[60., 50.], [100., 50.]],
                                             [[70., 50.], [100., 50.]]]),
                         'accepted': np.array([[False, False], [True, True], [True, True]])}

    def evidence(self):
        return make_feature_evidence(self.features, np.array([0, 1]), np.array([False, True]),
                                     self.camera, [0, 1, 2], 'cpu')

    def test_exact_and_interpolated_rigid_translation_match_boundary_pixels(self):
        with torch.no_grad():
            self.motion.translations[:, 0, 0] = torch.tensor([.2, .4])
        evidence = self.evidence()
        self.assertEqual(len(evidence['points']), 1)
        for time in (1., 1.5, 2.):
            self.assertLess(float(feature_reprojection_loss(self.motion, time, evidence)), 1e-6)

    def test_wrong_motion_has_finite_correction_gradient(self):
        loss = feature_reprojection_loss(self.motion, 1, self.evidence())
        loss.backward()
        self.assertGreater(float(loss), .9)
        self.assertTrue(torch.isfinite(self.motion.translations.grad).all())
        self.assertLess(float(self.motion.translations.grad[0, 0, 0]), 0)
        self.assertEqual(float(self.motion.translations.grad[:, 1].abs().sum()), 0)

    def test_unsupported_interpolated_endpoint_does_not_invent_observation(self):
        self.assertEqual(float(feature_reprojection_loss(self.motion, .5, self.evidence())), 0)
        self.assertGreater(float(feature_reprojection_loss(self.motion, 1, self.evidence())), 0)

    def test_missing_or_empty_evidence_is_differentiable_zero(self):
        zero = feature_reprojection_loss(self.motion, 1, None)
        zero.backward()
        self.assertEqual(float(self.motion.translations.grad.abs().sum()), 0)

    def test_malformed_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            make_feature_evidence(self.features, np.array([0, 9]), np.array([False, True]),
                                  self.camera, [0, 1, 2], 'cpu')


if __name__ == '__main__':
    unittest.main()
