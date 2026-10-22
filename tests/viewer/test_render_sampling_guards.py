import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from utils.render_sampling import require_native_sampling, render_sampling_contract
from utils.rigid_checkpoint import sha256_file


class RenderSamplingGuardTests(unittest.TestCase):
    def test_legacy_native_guard_accepts_only_one(self):
        require_native_sampling({}, 'test')
        require_native_sampling(render_sampling_contract({}), 'test')
        with self.assertRaisesRegex(ValueError, 'comparison UI'):
            require_native_sampling(render_sampling_contract({}, 2), 'test')

    def test_motion_main_guards_run_before_data_load_gpu_or_output_creation(self):
        from scripts.events import refine_lego_events, refine_lego_temporal
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); baseline = root/'baseline'; baseline.mkdir()
            (baseline/'cache.json').write_text(json.dumps(render_sampling_contract({}, 2)))
            for module in (refine_lego_events, refine_lego_temporal):
                output = root/module.__name__
                argv = ['test', '--baseline', str(baseline), '--supervision', str(root/'absent.npz'),
                        '--output', str(output)]
                with self.subTest(module=module.__name__), patch.object(sys, 'argv', argv), \
                        patch.object(module.np, 'load') as data_load, \
                        patch.object(module, 'MotionRenderer') as renderer:
                    with self.assertRaisesRegex(ValueError, 'render_supersample=1 only'):
                        module.main()
                    data_load.assert_not_called(); renderer.assert_not_called()
                    self.assertFalse(output.exists())

    def test_baked_export_rejects_two_before_reading_vertices_or_writing(self):
        from scripts.events import export_lego_event_preview as module
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root/'output'
            argv = ['test', '--baseline', str(root/'baseline'), '--candidate', str(root/'candidate'),
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(module, 'verify_candidate', return_value=(render_sampling_contract({}, 2), {}, {})), \
                    patch.object(module.PlyData, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'comparison UI'):
                    module.main()
                read.assert_not_called(); self.assertFalse(output.exists())


class AppearanceRecordHashTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.attributes = root/'attributes.npz'
        self.attributes.write_bytes(b'fixture attributes')
        self.optimization_path = root/'optimization.json'
        self.optimization = {'attributes_sha256': sha256_file(self.attributes),
            'seed_count': 10, 'retained_count': 20,
            'input_sha256': {'source_ply': 'a'*64, 'rebuild_report': 'b'*64},
            **render_sampling_contract({}, 2)}
        self.optimization_path.write_text(json.dumps(self.optimization))
        appearance = {'format': 'lego_rigid_seed_appearance_refinement_v1',
            'optimization_sha256': sha256_file(self.optimization_path),
            'attributes_sha256': sha256_file(self.attributes),
            'parent_source_sha256': 'a'*64, 'parent_rebuild_report_sha256': 'b'*64,
            'optimized_rows': 10, 'row_start': 20, **render_sampling_contract({}, 2)}
        self.cache = {'appearance_refinement': appearance, **render_sampling_contract({}, 2)}
        self.record = copy.deepcopy(self.cache)

    def verify(self):
        from scripts.appearance.render_lego_appearance_candidate import verify_appearance_record
        return verify_appearance_record(self.cache, self.record, self.optimization_path, self.attributes)

    def test_exact_hashes_and_sampling_are_required(self):
        optimization, sampling = self.verify()
        self.assertEqual(optimization, self.optimization)
        self.assertEqual(sampling, render_sampling_contract({}, 2))

    def test_wrong_cache_correction_provenance_is_rejected(self):
        self.record['appearance_refinement']['optimized_rows'] += 1
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            self.verify()

    def test_optimization_file_changed_after_packaging_is_rejected(self):
        self.optimization_path.write_text(json.dumps(self.optimization)+'\n')
        with self.assertRaisesRegex(ValueError, 'optimization checksum'):
            self.verify()

    def test_attributes_file_changed_after_packaging_is_rejected(self):
        self.attributes.write_bytes(b'changed attributes')
        with self.assertRaisesRegex(ValueError, 'attributes checksum'):
            self.verify()

    def test_optimization_attributes_hash_must_match_packaged_attributes(self):
        self.optimization['attributes_sha256'] = 'c'*64
        self.optimization_path.write_text(json.dumps(self.optimization))
        for record in (self.cache, self.record):
            record['appearance_refinement']['optimization_sha256'] = sha256_file(self.optimization_path)
        with self.assertRaisesRegex(ValueError, 'attributes checksum'):
            self.verify()

    def test_parent_and_rows_must_match_optimization(self):
        for key, value in [('parent_source_sha256', 'c'*64), ('parent_rebuild_report_sha256', 'c'*64),
                           ('optimized_rows', 11), ('row_start', 21), ('row_start', None)]:
            original_cache, original_record = copy.deepcopy(self.cache), copy.deepcopy(self.record)
            for record in (self.cache, self.record):
                record['appearance_refinement'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'parent or row counts'):
                self.verify()
            self.cache, self.record = original_cache, original_record

    def test_sampling_must_match_optimization(self):
        self.cache.update(render_sampling_contract({})); self.record.update(render_sampling_contract({}))
        with self.assertRaisesRegex(ValueError, 'sampling contract mismatch'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
