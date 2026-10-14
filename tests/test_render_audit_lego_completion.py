import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.render_audit_lego_completion import (attach_provenance, comparison_metrics,
    output_paths, prepare_inputs, validate_supervision_contract, write_new_json)
from utils.multiview_completion import FORMAT
from utils.rigid_checkpoint import sha256_file


class CompletionRenderAuditTests(unittest.TestCase):
    def test_full_original_timeline_is_mandatory(self):
        meta = {'frame_count': 400, 'fps': 60}
        images, counts = (400, 640, 640), (399, 2, 640, 640)
        validate_supervision_contract(meta, images, counts, meta)
        for bad_meta, bad_images, bad_counts, cache in (
            ({**meta, 'frame_count': 40}, images, counts, meta),
            (meta, (40, 640, 640), counts, meta),
            (meta, (400, 320, 320), counts, meta),
            (meta, images, (398, 2, 640, 640), meta),
            (meta, images, counts, {**meta, 'fps': 30}),
            ({**meta, 'fps': float('nan')}, images, counts, meta),
            (meta, images, counts, {**meta, 'frame_count': 399})):
            with self.subTest(meta=bad_meta, shape=bad_images, cache=cache), self.assertRaisesRegex(ValueError, 'complete 400-frame'):
                validate_supervision_contract(bad_meta, bad_images, bad_counts, cache)

    def test_image_checks_detect_nonfinite_range_missing_pixels_and_new_brightness(self):
        old = np.zeros((640, 640), np.float32); old[10, 10] = .5
        same = comparison_metrics(old, old)
        self.assertEqual(same['mean_absolute_change'], 0.)
        changed = old.copy(); changed[10, 10] = 0.; changed[20, 20] = .4
        actual = comparison_metrics(changed, old)
        self.assertEqual(actual['parent_bright_pixels_below_coverage_threshold'], 1)
        self.assertGreater(actual['newly_bright_background_fraction'], 0.)
        self.assertEqual(actual['parent_foreground_mean_absolute_change'], .5)
        for value in (np.nan, np.inf, -.01, 1.01):
            broken = old.copy(); broken[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite 640x640'):
                comparison_metrics(broken, old)
        with self.assertRaises(ValueError):
            comparison_metrics(old[:320], old[:320])

    def artifact_fixture(self, root):
        baseline, candidate = root/'baseline', root/'candidate'
        baseline.mkdir(); candidate.mkdir()
        source, report, extension, supervision = (root / name for name in ('source.ply', 'report.json', 'extension.npz', 'supervision.npz'))
        for path in (source, report, extension, supervision, baseline/'cache.json', baseline/'poses.npy',
                     baseline/'dynamic_indices.npy', candidate/'correction.json', candidate/'rendered_gray.npy'):
            path.write_bytes(('fixture:' + path.name).encode())
        contract = {'format': FORMAT, 'report': str(report), 'report_sha256': sha256_file(report),
                    'extension': str(extension), 'extension_sha256': sha256_file(extension)}
        cache = {'source': str(source), 'multiview_completion': contract}
        return baseline, candidate, supervision, cache, copy.deepcopy(cache)

    def test_audit_provenance_keeps_failed_decision_and_discloses_inherited_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline, candidate, supervision, cache, record = self.artifact_fixture(Path(folder))
            original = {'approved_for_export': False, 'failed_intervals': [10, 11],
                        'intervals': [{'interval': 10, 'passed': False, 'reasons': ['direction']}],
                        'summary': {'all': {'passed': 397, 'intervals': 399}}}
            before = copy.deepcopy(original)
            actual = attach_provenance(original, cache, record, baseline, candidate, supervision)
            self.assertEqual(original, before)
            for key in before:
                self.assertEqual(actual[key], before[key])
            self.assertTrue(actual['event_optimization_inherited'])
            self.assertTrue(actual['unobserved_geometry_is_prior'])
            self.assertFalse(actual['completion_motion_reoptimized'])
            self.assertFalse(actual['completion_appearance_event_optimized'])
            self.assertEqual(actual['artifact_sha256']['baseline_poses'], sha256_file(baseline/'poses.npy'))
            self.assertEqual(actual['artifact_sha256']['completion_report'], cache['multiview_completion']['report_sha256'])
            self.assertEqual(actual['artifact_sha256']['completion_extension'], cache['multiview_completion']['extension_sha256'])
            self.assertEqual(actual['artifact_sha256']['supervision'], sha256_file(supervision))

    def test_changed_completion_artifacts_cannot_receive_matching_audit_identity(self):
        for field in ('report', 'extension'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                baseline, candidate, supervision, cache, record = self.artifact_fixture(Path(folder))
                Path(cache['multiview_completion'][field]).write_bytes(b'changed during rendering')
                with self.assertRaisesRegex(ValueError, 'changed while rendering'):
                    attach_provenance({}, cache, record, baseline, candidate, supervision)

    def test_overwrite_and_partial_render_are_refused_before_input_load(self):
        for filename in ('rendered_gray.npy', 'rendered_gray.npy.partial', 'event-audit.json',
                         'completion-render-check.json', 'frame_0100.png'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                candidate = Path(folder)/'candidate'; candidate.mkdir()
                (candidate/filename).write_bytes(b'existing user artifact')
                with self.assertRaises(FileExistsError):
                    prepare_inputs(Path(folder)/'absent', candidate, Path(folder)/'absent.npz')
                self.assertEqual((candidate/filename).read_bytes(), b'existing user artifact')
        self.assertEqual(len(output_paths(Path('/tmp/example'))), 11)

    def test_json_outputs_are_created_exclusively(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'audit.json'
            write_new_json(path, {'approved_for_export': False})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {'approved_for_export': True})
            self.assertIs(json.loads(path.read_text())['approved_for_export'], False)


if __name__ == '__main__':
    unittest.main()
