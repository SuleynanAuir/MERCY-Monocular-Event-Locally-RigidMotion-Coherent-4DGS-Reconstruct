"""CPU synthetic tests only; no real inputs, old outputs or GPU work."""

import copy
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.event_only_motion import EventOnlyMotion
from utils.event_only_planar_constraint import (camera_plane_prior_metadata,
                                                enforce_camera_plane_motion)


class CameraPlaneConstraintTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(31)
        angles = [.3, .7, -.4]
        c, s = [math.cos(value) for value in angles], [math.sin(value) for value in angles]
        rx = torch.tensor([[1., 0., 0.], [0., c[0], -s[0]], [0., s[0], c[0]]], dtype=torch.float64)
        ry = torch.tensor([[c[1], 0., s[1]], [0., 1., 0.], [-s[1], 0., c[1]]], dtype=torch.float64)
        rz = torch.tensor([[c[2], -s[2], 0.], [s[2], c[2], 0.], [0., 0., 1.]], dtype=torch.float64)
        self.camera = torch.eye(4, dtype=torch.float64)
        self.camera[:3, :3] = rx @ ry @ rz
        self.camera[:3, 3] = torch.tensor([.7, -2., 4.], dtype=torch.float64)
        self.axis = self.camera[2, :3]
        self.motion = self.make_motion()

    def random(self, shape):
        return torch.randn(shape, dtype=torch.float64, generator=self.generator)

    def make_motion(self, soft=False):
        count, nodes = 48, 4
        graph = {
            'centers': self.random((nodes, 3)),
            'knn_indices': (torch.arange(nodes).repeat(count, 1) if soft
                            else (torch.arange(count) % nodes)[:, None]),
            'knn_weights': torch.full((count, nodes if soft else 1), .25 if soft else 1.),
            'edges': torch.empty((0, 2), dtype=torch.int64),
            'scene_scale': 3., 'metadata': {},
        }
        motion = EventOnlyMotion(graph, [0., .5, 1.4, 2.]).double()
        with torch.no_grad():
            motion.translations.copy_(self.random(motion.translations.shape))
            motion.rotation_vectors.copy_(self.random(motion.rotation_vectors.shape) * 2)
        return motion

    def assert_in_plane(self, translation, rotation, atol=2e-15):
        torch.testing.assert_close((translation * self.axis).sum(-1), torch.zeros(translation.shape[:-1], dtype=translation.dtype),
                                   atol=atol, rtol=0)
        torch.testing.assert_close(torch.cross(rotation, self.axis.expand_as(rotation), dim=-1),
                                   torch.zeros_like(rotation), atol=atol, rtol=0)

    def test_random_world_points_preserve_camera_z_at_knots_and_interpolated_times(self):
        for soft in (False, True):
            with self.subTest(soft_blend=soft):
                motion = self.make_motion(soft=soft)
                xyz = self.random((48, 3))
                quaternions = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(48, 1)
                expected_z = xyz @ self.axis + self.camera[2, 3]
                enforce_camera_plane_motion(motion, self.camera)
                self.assert_in_plane(motion.translations, motion.rotation_vectors)
                for time in (0., .2, .5, .8, 1.4, 1.7, 2.):
                    moved, _ = motion.deform(xyz, quaternions, time)
                    actual_z = moved @ self.axis + self.camera[2, 3]
                    torch.testing.assert_close(actual_z, expected_z, atol=3e-14, rtol=0)

    def test_parameters_and_existing_gradients_use_the_same_orthogonal_projection(self):
        original_translation = self.motion.translations.detach().clone()
        original_rotation = self.motion.rotation_vectors.detach().clone()
        gt, gr = self.random(original_translation.shape), self.random(original_rotation.shape)
        self.motion.translations.grad, self.motion.rotation_vectors.grad = gt.clone(), gr.clone()
        translation_id, rotation_id = id(self.motion.translations), id(self.motion.rotation_vectors)
        enforce_camera_plane_motion(self.motion, self.camera)
        expected_t = original_translation - (original_translation * self.axis).sum(-1, keepdim=True) * self.axis
        expected_r = (original_rotation * self.axis).sum(-1, keepdim=True) * self.axis
        torch.testing.assert_close(self.motion.translations, expected_t, atol=2e-15, rtol=0)
        torch.testing.assert_close(self.motion.rotation_vectors, expected_r, atol=2e-15, rtol=0)
        self.assert_in_plane(self.motion.translations.grad, self.motion.rotation_vectors.grad)
        self.assertEqual(id(self.motion.translations), translation_id)
        self.assertEqual(id(self.motion.rotation_vectors), rotation_id)
        self.assertTrue(self.motion.translations.is_leaf)
        self.assertTrue(self.motion.rotation_vectors.requires_grad)

    def test_gradient_only_leaves_every_state_tensor_exactly_unchanged(self):
        before = {name: value.clone() for name, value in self.motion.state_dict().items()}
        self.motion.translations.grad = self.random(self.motion.translations.shape)
        self.motion.rotation_vectors.grad = self.random(self.motion.rotation_vectors.shape)
        enforce_camera_plane_motion(self.motion, self.camera, gradients_only=True)
        for name, value in self.motion.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assert_in_plane(self.motion.translations.grad, self.motion.rotation_vectors.grad)

    def test_projection_is_idempotent_and_t0_remains_structurally_absent(self):
        self.motion.translations.grad = self.random(self.motion.translations.shape)
        self.motion.rotation_vectors.grad = self.random(self.motion.rotation_vectors.shape)
        buffers_before = {name: value.clone() for name, value in self.motion.named_buffers()}
        shape_before = tuple(self.motion.translations.shape)
        enforce_camera_plane_motion(self.motion, self.camera)
        projected = [value.detach().clone() for value in (self.motion.translations, self.motion.rotation_vectors,
                                                         self.motion.translations.grad, self.motion.rotation_vectors.grad)]
        enforce_camera_plane_motion(self.motion, self.camera)
        for actual, expected in zip((self.motion.translations, self.motion.rotation_vectors,
                                      self.motion.translations.grad, self.motion.rotation_vectors.grad), projected):
            torch.testing.assert_close(actual, expected, atol=3e-15, rtol=0)
        self.assertEqual(tuple(self.motion.translations.shape), shape_before)
        self.assertEqual(shape_before[0], len(self.motion.times) - 1)
        for name, value in self.motion.named_buffers():
            self.assertTrue(torch.equal(value, buffers_before[name]), name)
        translation, quaternion = self.motion.at_time(0.)
        self.assertTrue(torch.equal(translation, torch.zeros_like(translation)))
        torch.testing.assert_close(quaternion, torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(4, 1),
                                   atol=0, rtol=0)

    def test_absent_and_zero_gradients_are_safe_and_no_gradient_is_fabricated(self):
        self.assertIsNone(self.motion.translations.grad)
        enforce_camera_plane_motion(self.motion, self.camera, gradients_only=True)
        self.assertIsNone(self.motion.translations.grad)
        self.assertIsNone(self.motion.rotation_vectors.grad)
        for parameter in self.motion.parameters():
            parameter.grad = torch.zeros_like(parameter)
        enforce_camera_plane_motion(self.motion, self.camera)
        for parameter in self.motion.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))

    def test_identity_camera_removes_only_z_translation_and_xy_rotation(self):
        t, r = self.motion.translations.detach().clone(), self.motion.rotation_vectors.detach().clone()
        enforce_camera_plane_motion(self.motion, np.eye(4))
        torch.testing.assert_close(self.motion.translations[..., :2], t[..., :2], atol=0, rtol=0)
        self.assertTrue(torch.equal(self.motion.translations[..., 2], torch.zeros_like(t[..., 2])))
        self.assertTrue(torch.equal(self.motion.rotation_vectors[..., :2], torch.zeros_like(r[..., :2])))
        torch.testing.assert_close(self.motion.rotation_vectors[..., 2], r[..., 2], atol=0, rtol=0)

    def test_float32_motion_keeps_dtype_device_and_camera_depth_with_roundoff(self):
        motion = self.make_motion(soft=True).float()
        xyz = self.random((48, 3)).float()
        rotations = torch.tensor([[1., 0., 0., 0.]]).repeat(48, 1)
        axis = self.axis.float()
        enforce_camera_plane_motion(motion, self.camera)
        self.assertEqual(motion.translations.dtype, torch.float32)
        self.assertEqual(motion.rotation_vectors.device, xyz.device)
        moved, _ = motion.deform(xyz, rotations, .8)
        torch.testing.assert_close(moved @ axis, xyz @ axis, atol=2e-6, rtol=0)

    def test_improper_nonfinite_or_nonrigid_camera_is_rejected_without_mutation(self):
        bad = [torch.eye(3), torch.full((4, 4), float('nan')), [[True] * 4] * 4, 'old/camera.json']
        scaled, reflected, bad_bottom, shear = [self.camera.clone() for _ in range(4)]
        scaled[:3, :3] *= 2
        reflected[:3, 0] *= -1
        bad_bottom[3, 0] = .1
        shear[0, 1] += .3
        bad.extend((scaled, reflected, bad_bottom, shear))
        before = {name: value.clone() for name, value in self.motion.state_dict().items()}
        for camera in bad:
            with self.subTest(camera_type=type(camera)), self.assertRaises(ValueError):
                enforce_camera_plane_motion(self.motion, camera)
            for name, value in self.motion.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)

    def test_invalid_shape_or_anchor_contract_is_rejected(self):
        base = dict(translations=torch.zeros(2, 4, 3), rotation_vectors=torch.zeros(2, 4, 3),
                    centers=torch.zeros(4, 3), times=torch.tensor([0., .5, 1.]))
        invalid = [('translations', torch.zeros(2, 4)), ('rotation_vectors', torch.zeros(3, 4, 3)),
                   ('rotation_vectors', torch.zeros(2, 4, 3, dtype=torch.float64)),
                   ('translations', torch.zeros(2, 4, 3, dtype=torch.int64)),
                   ('centers', torch.zeros(5, 3)), ('times', torch.tensor([.1, .5, 1.])),
                   ('times', torch.tensor([0., .5, .5])), ('times', torch.tensor([0., .5])),
                   ('times', torch.tensor([0., .5, float('inf')])),
                   ('centers', torch.full((4, 3), float('nan')))]
        for name, value in invalid:
            with self.subTest(name=name, shape=value.shape), self.assertRaises(ValueError):
                enforce_camera_plane_motion(SimpleNamespace(**dict(base, **{name: value})), self.camera)

    def test_nonfinite_parameter_or_gradient_rejects_all_updates_atomically(self):
        for gradient in (False, True):
            with self.subTest(gradient=gradient):
                motion = self.make_motion()
                before_t = motion.translations.detach().clone()
                if gradient:
                    motion.translations.grad = self.random(motion.translations.shape)
                    motion.rotation_vectors.grad = torch.full_like(motion.rotation_vectors, float('nan'))
                    before_gt = motion.translations.grad.clone()
                else:
                    with torch.no_grad():
                        motion.rotation_vectors[0, 0, 0] = float('inf')
                with self.assertRaises(ValueError):
                    enforce_camera_plane_motion(motion, self.camera)
                self.assertTrue(torch.equal(motion.translations, before_t))
                if gradient:
                    self.assertTrue(torch.equal(motion.translations.grad, before_gt))

    def test_finite_inputs_whose_projection_overflows_are_rejected_atomically(self):
        motion = self.make_motion().float()
        camera = [[.6, -.8, 0., 0.], [0., 0., -1., 0.], [.8, .6, 0., 0.], [0., 0., 0., 1.]]
        with torch.no_grad():
            motion.rotation_vectors[..., :2] = torch.finfo(torch.float32).max
            motion.rotation_vectors[..., 2] = 0.
        before_t, before_r = motion.translations.clone(), motion.rotation_vectors.clone()
        with self.assertRaisesRegex(ValueError, 'finite motion dtype range'):
            enforce_camera_plane_motion(motion, camera)
        self.assertTrue(torch.equal(motion.translations, before_t))
        self.assertTrue(torch.equal(motion.rotation_vectors, before_r))

    def test_metadata_is_json_safe_explicit_prior_not_event_approval(self):
        camera = {'w2c': self.camera.tolist(), 'unused': 'never/read/legacy.json'}
        before = copy.deepcopy(camera)
        with patch('builtins.open', side_effect=AssertionError('No file inputs')):
            metadata = camera_plane_prior_metadata(camera)
            enforce_camera_plane_motion(self.motion, camera['w2c'])
        self.assertEqual(camera, before)
        np.testing.assert_allclose(metadata['world_optical_axis'], self.axis.tolist(), atol=1e-15, rtol=0)
        self.assertTrue(metadata['native_event_audit_required'])
        self.assertFalse(metadata['native_event_gates_relaxed'])
        self.assertFalse(metadata['physical_motion_validated'])
        self.assertFalse(metadata['approved_for_export'])
        self.assertEqual(metadata['independent_observations_added'], 0)
        json.dumps(metadata, allow_nan=False)
        for camera in ({}, self.camera):
            with self.assertRaises(ValueError):
                camera_plane_prior_metadata(camera)

    def test_gradient_projection_does_not_replace_post_adam_parameter_projection(self):
        optimizer = torch.optim.Adam(self.motion.parameters(), lr=.05)
        enforce_camera_plane_motion(self.motion, self.camera)
        self.motion.translations.grad = self.random(self.motion.translations.shape)
        self.motion.rotation_vectors.grad = self.random(self.motion.rotation_vectors.shape)
        enforce_camera_plane_motion(self.motion, self.camera, gradients_only=True)
        optimizer.step()
        # Diagonal adaptive scaling need not retain the plane's linear subspace.
        self.assertGreater(float((self.motion.translations * self.axis).sum(-1).abs().max()), 1e-5)
        enforce_camera_plane_motion(self.motion, self.camera)
        self.assert_in_plane(self.motion.translations, self.motion.rotation_vectors)

    def test_gradients_only_requires_an_explicit_boolean(self):
        for value in (1, 0, None, 'true', np.bool_(True)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                enforce_camera_plane_motion(self.motion, self.camera, gradients_only=value)


if __name__ == '__main__':
    unittest.main()
