import unittest

import torch

from utils.render_sampling import (render_sampling_contract, require_same_sampling,
                                   resolve_rgb, validate_supersample)


class RenderSamplingTests(unittest.TestCase):
    def test_legacy_factor_one_is_exact_clamped_identity(self):
        rgb = torch.linspace(-.1, 1.1, 3 * 4 * 6).reshape(3, 4, 6)
        self.assertTrue(torch.equal(resolve_rgb(rgb, 1), rgb.clamp(0, 1)))
        self.assertEqual(render_sampling_contract({}),
                         {'render_supersample': 1, 'render_downsample_filter': 'area_after_clamp'})

    def test_factor_two_exact_area_after_clamp_and_gradients(self):
        rgb = torch.linspace(-.2, 1.2, 3 * 4 * 6).reshape(3, 4, 6).requires_grad_()
        result = resolve_rgb(rgb, 2)
        expected = rgb.clamp(0, 1).reshape(3, 2, 2, 3, 2).mean((2, 4))
        torch.testing.assert_close(result, expected, atol=1e-7, rtol=0)
        self.assertEqual(result.shape, (3, 2, 3))
        result.sum().backward()
        torch.testing.assert_close(rgb.grad, ((rgb >= 0) & (rgb <= 1)).float() / 4)
        out_of_range = torch.tensor([[[-2., 2.], [-2., 2.]]] * 3)
        self.assertTrue(torch.equal(resolve_rgb(out_of_range, 2), torch.full((3, 1, 1), .5)))

    def test_invalid_factors_shapes_and_filters_are_rejected(self):
        for factor in (True, False, 0, -1, 3, 2., '2', None):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                validate_supersample(factor)
        for shape in ((3, 3, 4), (3, 4, 3), (4, 4, 4), (1, 3, 4, 4), (3, 0, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                resolve_rgb(torch.zeros(shape), 2)
        for metadata in ({'render_supersample': 2}, {'render_supersample': True},
                         {'render_downsample_filter': 'bilinear'},
                         {'render_supersample': 2, 'render_downsample_filter': 'clamp_after_area'}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                render_sampling_contract(metadata)

    def test_override_is_for_new_training_and_records_must_agree(self):
        two = render_sampling_contract({}, supersample=2)
        self.assertEqual(two, {'render_supersample': 2, 'render_downsample_filter': 'area_after_clamp'})
        self.assertEqual(require_same_sampling(two, dict(two)), two)
        self.assertEqual(require_same_sampling({}, {'render_supersample': 1}), render_sampling_contract({}))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            require_same_sampling({}, two)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            require_same_sampling()

    def test_live_viewer_sampling_checks_each_side_without_count_assumptions(self):
        from scripts.check_event_viewer import verify_live_sampling
        two = render_sampling_contract({}, supersample=2)
        scene = {'comparison': True, 'right_render_sampling': two,
                 'left_gaussian_count': 7, 'gaussian_count': 4}
        verify_live_sampling(scene, two, {})
        verify_live_sampling({'comparison': True}, {}, {})
        for changed_scene, audit, left in ((scene, {}, {}), ({'comparison': True}, two, {}),
                                           (scene, two, two)):
            with self.subTest(scene=changed_scene), self.assertRaisesRegex(ValueError, 'mismatch'):
                verify_live_sampling(changed_scene, audit, left)

    def test_live_surface_and_attribute_layers_require_their_own_audited_identity(self):
        import copy
        from scripts.check_event_viewer import verify_live_cleanup
        verify_live_cleanup({}, {})
        for key, fields in (('gaussian_cleanup', {'report': 'cleanup_report', 'overlay': 'cleanup_overlay'}),
                            ('static_surface_rebuild', {'report': 'static_surface_report',
                                                        'extension': 'static_surface_extension'})):
            contract = {field+'_sha256': str(index)*64 for index, field in enumerate(fields, 1)}
            audit = {key: contract, 'artifact_sha256':
                     {artifact: contract[field+'_sha256'] for field, artifact in fields.items()}}
            scene = {key: dict(contract)}
            verify_live_cleanup(scene, audit)
            with self.assertRaises(ValueError):
                verify_live_cleanup(scene, {})
            with self.assertRaises(ValueError):
                verify_live_cleanup({}, audit)
            for field in fields:
                bad = copy.deepcopy(scene)
                bad[key][field+'_sha256'] = 'f'*64
                with self.assertRaises(ValueError):
                    verify_live_cleanup(bad, audit)


if __name__ == '__main__':
    unittest.main()
