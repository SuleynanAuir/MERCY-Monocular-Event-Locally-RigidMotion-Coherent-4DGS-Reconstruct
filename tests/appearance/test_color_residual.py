import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from utils.color_residual import (color_residual_contract, validate_residual_arrays,
    validate_training_split, apply_color_residual, FORMAT)
from utils.residual_appearance import split_timeline


class ColorResidualContractTests(unittest.TestCase):
    def test_contract_requires_every_record_and_both_artifact_hashes(self):
        self.assertIsNone(color_residual_contract({}, {}))
        c = {'format': FORMAT, 'report': '/tmp/report.json', 'report_sha256': 'a'*64,
             'residual': '/tmp/residual.npz', 'residual_sha256': 'b'*64}
        self.assertEqual(color_residual_contract({'color_residual': c}, {'color_residual': c}), c)
        with self.assertRaises(ValueError):
            color_residual_contract({'color_residual': c}, {})
        for field, value in (('format', 'other'), ('report', 'relative.json'), ('residual_sha256', 'bad')):
            bad = dict(c); bad[field] = value
            with self.assertRaises(ValueError):
                color_residual_contract({'color_residual': bad})

    def fixture(self):
        arrays = {'indices': np.array([0, 2], np.int32), 'delta_gray': np.array([.01, -.02], np.float32),
                  'maximum_training_visibility': np.array([.1, .2], np.float32)}
        report = {'optimized_count': 2, 'gaussian_count': 6, 'prefix_count': 4,
                  'maximum_delta': .08, 'visibility_threshold': .001}
        return arrays, report, np.array([0, 142, 5, 143], np.int32)

    def test_array_checks_protect_elastic_suffix_bounds_visibility_and_rows(self):
        arrays, report, labels = self.fixture()
        validate_residual_arrays(arrays, report, 6, 4, labels, [142, 143])
        for key, value in [('indices', np.array([1, 2], np.int32)),
                           ('indices', np.array([0, 4], np.int32)),
                           ('indices', np.array([2, 0], np.int32)),
                           ('delta_gray', np.array([.081, 0], np.float32)),
                           ('maximum_training_visibility', np.array([0, .1], np.float32))]:
            bad = dict(arrays); bad[key] = value
            with self.assertRaises(ValueError):
                validate_residual_arrays(bad, report, 6, 4, labels, [142, 143])

    def test_holdout_and_visibility_records_are_checked(self):
        train, events, rgb = split_timeline(400)
        report = {'training_intervals': train.tolist(), 'training_probe_intervals': train[:12].tolist(),
                  'rgb_heldout_frames': rgb.tolist(), 'event_heldout_intervals': events.tolist(),
                  'visibility_probe_frames': [0, 1, 2]}
        validate_training_split(report, 400)
        for field, value in [('training_intervals', [0, 3]), ('training_probe_intervals', [4]),
                             ('visibility_probe_frames', [4]), ('rgb_heldout_frames', [])]:
            bad = dict(report); bad[field] = value
            with self.assertRaises(ValueError):
                validate_training_split(bad, 400)

    def test_application_changes_only_named_dc_rows_and_refuses_stacking(self):
        arrays, report, _ = self.fixture()
        source = np.arange(18, dtype=np.float32).reshape(6, 3)/100
        verified = {'arrays': arrays, 'source_dc': source, 'report': report}
        model = SimpleNamespace(_features_dc=torch.from_numpy(source.copy())[:, None],
                                _xyz=torch.arange(18.).reshape(6, 3), _opacity=torch.ones(6, 1))
        before = copy.deepcopy(model)
        apply_color_residual(model, verified)
        self.assertTrue(torch.equal(model._xyz, before._xyz))
        self.assertTrue(torch.equal(model._opacity, before._opacity))
        self.assertTrue(torch.equal(model._features_dc[[1, 3, 4, 5]], before._features_dc[[1, 3, 4, 5]]))
        with self.assertRaises(ValueError):
            apply_color_residual(model, verified)


if __name__ == '__main__':
    unittest.main()
