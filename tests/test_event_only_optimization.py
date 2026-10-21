"""Synthetic optimization checks without any scene/video/cache inputs."""
from types import SimpleNamespace
import unittest

import torch

from scripts.reconstruct_lego_event_only import (
    difference_loss, integral_loss, pool_log, restrict_gradient_to_knot,
    seed_chronological_knot,
    observation_log,
    enforce_frozen_groups,
)


class EventOnlyOptimizationTests(unittest.TestCase):
    def test_frozen_unknown_and_static_groups_cannot_drift(self):
        motion = SimpleNamespace(translations=torch.nn.Parameter(torch.ones(3, 2, 3)),
                                 rotation_vectors=torch.nn.Parameter(torch.ones(3, 2, 3)))
        for parameter in (motion.translations, motion.rotation_vectors):
            parameter.grad = torch.ones_like(parameter)
        enforce_frozen_groups(motion, torch.tensor([True, False]), gradients_only=True)
        self.assertEqual(float(motion.translations[:, 0].sum()), 9)
        self.assertEqual(float(motion.translations.grad[:, 0].sum()), 0)
        enforce_frozen_groups(motion, torch.tensor([True, False]))
        for parameter in (motion.translations, motion.rotation_vectors):
            self.assertEqual(float(parameter[:, 0].sum()), 0)
            self.assertEqual(float(parameter[:, 1].sum()), 9)
        with self.assertRaises(ValueError):
            enforce_frozen_groups(motion, [1, 0])

    def test_static_model_is_not_a_solution_for_active_events(self):
        target = torch.zeros(32, 32)
        target[8:16, 8:16] = .4
        target[16:24, 16:24] = -.4
        self.assertGreater(float(difference_loss(torch.zeros_like(target), target)), .5)
        self.assertLess(float(difference_loss(target, target)), .001)

    def test_zero_initial_motion_receives_finite_nonzero_gradient(self):
        predicted = torch.zeros(32, 32, requires_grad=True)
        expected = torch.zeros_like(predicted)
        expected[2:7, 2:7] = .5
        difference_loss(predicted, expected).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad.abs().sum()), 0)

    def test_exact_zero_change_does_not_invent_polarity(self):
        zeros = torch.zeros(32, 32)
        self.assertEqual(float(difference_loss(zeros, zeros)), 0)

    def test_integral_proxy_is_log_domain(self):
        intensity = torch.tensor([[.01, 1.], [.01, 1.]])
        pooled_log = pool_log(intensity.log(), 2)
        self.assertFalse(torch.allclose(pooled_log, pool_log(intensity, 2).log()))
        self.assertEqual(float(integral_loss(intensity.log(), intensity.log())), 0)

    def test_chronological_updates_cannot_change_other_knots(self):
        motion = SimpleNamespace(translations=torch.nn.Parameter(torch.zeros(4, 2, 3)),
                                 rotation_vectors=torch.nn.Parameter(torch.zeros(4, 2, 3)))
        for parameter in (motion.translations, motion.rotation_vectors):
            parameter.grad = torch.ones_like(parameter)
        restrict_gradient_to_knot(motion, 2)
        for parameter in (motion.translations, motion.rotation_vectors):
            self.assertEqual(float(parameter.grad[2].sum()), 6)
            self.assertEqual(float(parameter.grad[:2].sum()), 0)
            self.assertEqual(float(parameter.grad[3:].sum()), 0)

    def test_event_initializer_is_not_replaced_by_previous_knot(self):
        initial = {name: torch.arange(3.).view(3, 1, 1).expand(3, 2, 3).clone()
                   for name in ('translations', 'rotation_vectors')}
        motion = SimpleNamespace(**{name: torch.nn.Parameter(value.clone())
                                    for name, value in initial.items()})
        with torch.no_grad():
            motion.translations[0].add_(.25)
        seed_chronological_knot(motion, 1, initial)
        self.assertTrue(torch.allclose(motion.translations[1], torch.full((2, 3), 1.25)))
        self.assertTrue(torch.equal(motion.rotation_vectors[1], initial['rotation_vectors'][1]))
        self.assertTrue(torch.equal(motion.translations[2], initial['translations'][2]))

    def test_identity_initializer_carries_previous_fitted_motion(self):
        motion = SimpleNamespace(translations=torch.nn.Parameter(torch.ones(3, 2, 3)),
                                 rotation_vectors=torch.nn.Parameter(torch.ones(3, 2, 3)))
        with torch.no_grad():
            motion.translations[0].mul_(2)
        seed_chronological_knot(motion, 1)
        self.assertTrue(torch.equal(motion.translations[1], motion.translations[0]))

    def test_event_background_is_composited_before_log_not_a_log_clamp(self):
        gray = torch.tensor([[.4, 0.]], requires_grad=True)
        alpha = torch.tensor([[[1., .5]]], requires_grad=True)
        result = observation_log({'gray': gray, 'alpha': alpha}, .04, .0001)
        self.assertTrue(torch.allclose(result.exp(), torch.tensor([[.4, .02]])))
        self.assertTrue(bool((result < 0).all()))
        result.sum().backward()
        self.assertGreater(float(gray.grad.min()), 0.)
        self.assertLess(float(alpha.grad.max()), 0.)


if __name__ == '__main__':
    unittest.main()
