import copy
import unittest
from scripts.appearance.check_lego_residual_release import validate_quality


class ResidualReleaseTests(unittest.TestCase):
    def fixture(self):
        metrics = {'passed': 380, 'intervals': 399, 'cosine': .94, 'polarity_agreement': .93,
                   'relative_l1': .38, 'unsupported_change_fraction': .01}
        parent = {'failed_intervals': [1, 2], 'summary': {'all': metrics, 'validation_events': dict(metrics)}}
        child = copy.deepcopy(parent); child['failed_intervals'] = [1]
        child['summary']['all']['passed'] = 381
        detail = {'full_image': {'parent': .01, 'candidate': .007},
                  'rois': {'cabin': {'parent': {'rgb_l1': .02, 'edge_f1': .9},
                                    'candidate': {'rgb_l1': .015, 'edge_f1': .92}}}}
        return parent, child, detail

    def test_preview_can_keep_known_failures_without_calling_them_approved(self):
        a, b, d = self.fixture()
        result = validate_quality(a, b, d)
        self.assertEqual(result['fixed_event_intervals'], [2])
        self.assertEqual(result['new_event_failures'], [])
        self.assertNotIn('approved_for_export', result)

    def test_unchanged_pass_count_cannot_hide_different_new_failure(self):
        a, b, d = self.fixture(); b['failed_intervals'] = [3]
        with self.assertRaisesRegex(ValueError, 'New event failures'):
            validate_quality(a, b, d)

    def test_local_or_validation_regression_is_rejected_despite_global_gain(self):
        a, b, d = self.fixture(); d['rois']['cabin']['candidate']['edge_f1'] = .89
        with self.assertRaises(ValueError):
            validate_quality(a, b, d)
        a, b, d = self.fixture(); b['summary']['validation_events']['cosine'] = .93
        with self.assertRaises(ValueError):
            validate_quality(a, b, d)


if __name__ == '__main__':
    unittest.main()
