import copy
import json
import unittest
import numpy as np
from scripts.appearance.check_lego_surface_detail_release import check_observed_quality, view_checks, require_fresh_views


class SurfaceReleaseTests(unittest.TestCase):
    def setUp(self):
        row = {'passed': 382, 'intervals': 399, 'cosine': .944, 'polarity_agreement': .937,
               'relative_l1': .375, 'unsupported_change_fraction': .013}
        self.event = {'failed_intervals': [0, 398], 'summary': {'all': row, 'validation_events': dict(row)}}
        self.errors = np.full(400, .005)

    def test_unapproved_decision_is_not_promoted(self):
        result = check_observed_quality(self.event, self.event, self.errors, self.errors)
        self.assertEqual(result['remaining_failed_intervals'], [0, 398])
        self.assertNotIn('approved_for_export', result)

    def test_any_new_interval_failure_is_rejected(self):
        event = copy.deepcopy(self.event); event['failed_intervals'] = [0, 4]
        with self.assertRaisesRegex(ValueError, 'New failed'):
            check_observed_quality(self.event, event, self.errors, self.errors)

    def test_metric_and_local_frame_regressions_are_rejected(self):
        event = copy.deepcopy(self.event); event['summary']['all']['cosine'] -= .001
        with self.assertRaisesRegex(ValueError, 'Event metric'):
            check_observed_quality(self.event, event, self.errors, self.errors)
        errors = self.errors.copy(); errors[100] += .0001
        with self.assertRaisesRegex(ValueError, 'RGB error'):
            check_observed_quality(self.event, self.event, self.errors, errors)
        with self.assertRaisesRegex(ValueError, '400 finite'):
            check_observed_quality(self.event, self.event, self.errors[:40], self.errors[:40])

    def view_fixture(self):
        meta = json.dumps({'views': [{'frame': 150, 'yaw': 225, 'pitch': 70}]})
        image = np.zeros((1, 12, 12), np.float32)
        image[:, 2:10, 2:10] = .5
        render = {'metadata_json': meta, 'gray': image, 'alpha': np.ones_like(image)}
        parts = np.full(image.shape, -1, np.int32)
        parts[:, 2:10, 2:10] = 11
        masks = {'metadata_json': meta, 'part_ids': parts, 'depth': np.ones_like(image)}
        return render, copy.deepcopy(render), masks

    def test_reloaded_interior_hole_is_rejected_even_for_one_pixel(self):
        old, new, masks = self.view_fixture()
        new['alpha'][0, 5, 5] = .49
        with self.assertRaisesRegex(ValueError, 'interior holes at view 0: 1'):
            view_checks(old, new, masks, [11])

    def test_reloaded_boundary_change_is_reported_not_hidden(self):
        old, new, masks = self.view_fixture()
        new['alpha'][0, 2, 2] = .49
        row = view_checks(old, new, masks, [11])[0]
        self.assertEqual(row['new_target_interior_holes'], 0)
        self.assertEqual(row['new_alpha_holes_on_target']['pixels'], 1)

    def test_reloaded_camera_mismatch_is_rejected(self):
        old, new, masks = self.view_fixture()
        new['metadata_json'] = json.dumps({'views': [{'frame': 150, 'yaw': 226, 'pitch': 70}]})
        with self.assertRaisesRegex(ValueError, 'plans disagree'):
            view_checks(old, new, masks, [11])

    def test_previously_inspected_views_are_not_fresh_validation(self):
        views = [{'frame': 250, 'yaw': n*10, 'pitch': 65} for n in range(12)]
        require_fresh_views(views, [{'frame': 150, 'yaw': 0, 'pitch': 65}])
        with self.assertRaisesRegex(ValueError, 'development cameras'):
            require_fresh_views(views, [views[4]])
        with self.assertRaises(ValueError):
            require_fresh_views(views[:-1]+[views[0]], [])


if __name__ == '__main__':
    unittest.main()
