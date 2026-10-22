import unittest
import numpy as np
import torch
from utils.residual_appearance import (BoundedColorResidual, split_timeline,
    detail_terms, multiscale_detail_terms, local_detail_loss, normalized_event_loss, temporal_nonregression_loss, C0)


class ResidualAppearanceTests(unittest.TestCase):
    def test_holdouts_are_never_training_rgb_or_events(self):
        train, events, frames = split_timeline(400)
        self.assertEqual((len(train), len(events), len(frames)), (240, 79, 80))
        self.assertFalse(np.intersect1d(train, events).size)
        self.assertFalse(np.intersect1d(np.r_[train, train+1], frames).size)
        self.assertTrue(np.array_equal(events, np.arange(4, 399, 5)))
        for bad in (True, 10, 400.):
            with self.assertRaises(ValueError):
                split_timeline(bad)

    def test_zero_is_exact_identity_and_unselected_rows_cannot_change(self):
        source = torch.linspace(-1., 1., 18).reshape(6, 1, 3)
        model = BoundedColorResidual(source, torch.tensor([1, 4]), .08)
        self.assertTrue(torch.equal(source, model()))
        with torch.no_grad():
            model.delta[:] = torch.tensor([.2, -.2])
        model.project_()
        result = model()
        self.assertTrue(torch.equal(source[[0, 2, 3, 5]], result[[0, 2, 3, 5]]))
        torch.testing.assert_close((result-source)[[1, 4], 0, 0]*C0, torch.tensor([.08, -.08]))
        torch.testing.assert_close(result[:, :, 1]-result[:, :, 0], source[:, :, 1]-source[:, :, 0])
        self.assertTrue(torch.equal(source, model.source_dc))
        result.sum().backward()
        self.assertTrue(torch.isfinite(model.delta.grad).all())

    def test_bad_indices_and_bounds_fail(self):
        source = torch.zeros(6, 1, 3)
        for indices in (torch.tensor([2, 1]), torch.tensor([1, 1]), torch.tensor([-1]),
                        torch.tensor([6]), torch.tensor([], dtype=torch.int64), torch.tensor([1.])):
            with self.assertRaises(ValueError):
                BoundedColorResidual(source, indices)
        for bound in (0, -.1, .16, float('nan')):
            with self.assertRaises(ValueError):
                BoundedColorResidual(source, torch.tensor([1]), bound)

    def test_losses_reward_reference_detail_not_synthetic_sharpness(self):
        target = torch.zeros(24, 24); target[:, 12:] = .7
        weights = torch.ones_like(target)
        for fn in (detail_terms, multiscale_detail_terms):
            self.assertTrue(all(v.item() == 0 for v in fn(target, target, weights)))
            self.assertGreater(sum(fn(1-target, target, weights)).item(), 0)
        rois = {'detail': (0, 0, 24, 24)}
        self.assertEqual(local_detail_loss(target, target, rois).item(), 0)
        smooth = torch.full_like(target, .35)
        self.assertGreater(local_detail_loss(smooth, target, rois).item(), 0)

    def test_event_polarity_inversion_is_penalized(self):
        event = torch.tensor([[.1, -.1], [.2, -.2]])
        self.assertAlmostEqual(normalized_event_loss(event, event).item(), 0, places=6)
        self.assertGreater(normalized_event_loss(-event, event).item(), 1)

    def test_temporal_guard_only_penalizes_regression(self):
        target = torch.tensor([[.1, -.1], [.2, -.2]])
        parent = target*.8
        self.assertEqual(temporal_nonregression_loss(parent, target, parent).item(), 0)
        self.assertEqual(temporal_nonregression_loss(target, target, parent).item(), 0)
        bad = (-target).requires_grad_()
        loss = temporal_nonregression_loss(bad, target, parent)
        self.assertGreater(loss.item(), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())


if __name__ == '__main__':
    unittest.main()
