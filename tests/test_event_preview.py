import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_lego_event_preview import verify_candidate
from utils.rigid_checkpoint import sha256_file


class EventPreviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.baseline, self.candidate = root / 'baseline', root / 'candidate'
        self.baseline.mkdir(); self.candidate.mkdir()
        self.source = root / 'source.ply'
        self.source.write_bytes(b'original source fixture')
        cache = {'source': str(self.source), 'frame_count': 2, 'fps': 60}
        (self.baseline / 'cache.json').write_text(json.dumps(cache))
        (self.baseline / 'poses.npy').write_bytes(b'poses fixture')
        (self.baseline / 'dynamic_indices.npy').write_bytes(b'indices fixture')
        (self.candidate / 'rendered_gray.npy').write_bytes(b'render fixture')
        (self.candidate / 'correction.json').write_text(json.dumps({
            'format': 'event_supervised_rigid_motion_alignment_v1', 'raw': [0] * 7}))
        artifacts = {'source_ply': self.source, 'baseline_cache_metadata': self.baseline / 'cache.json',
                     'baseline_poses': self.baseline / 'poses.npy', 'dynamic_indices': self.baseline / 'dynamic_indices.npy',
                     'rendered_gray': self.candidate / 'rendered_gray.npy', 'correction': self.candidate / 'correction.json'}
        self.report = {'format': 'lego_event_motion_audit_v1', 'event_optimization_performed': True,
                       'approved_for_export': False, 'frame_count': 2, 'fps': 60, 'intervals': [{'interval': 0}],
                       'artifact_sha256': {key: sha256_file(path) for key, path in artifacts.items()}}
        self.write_report()

    def write_report(self):
        (self.candidate / 'event-audit.json').write_text(json.dumps(self.report))

    def test_unapproved_preview_is_explicit_and_stays_unapproved(self):
        with self.assertRaisesRegex(ValueError, 'explicit'):
            verify_candidate(self.baseline, self.candidate)
        _, _, report = verify_candidate(self.baseline, self.candidate, True)
        self.assertIs(report['approved_for_export'], False)

    def test_changed_source_is_rejected_even_for_preview(self):
        self.source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch: source_ply'):
            verify_candidate(self.baseline, self.candidate, True)

    def test_non_event_model_cannot_be_labeled_as_event_corrected(self):
        self.report['event_optimization_performed'] = False
        self.write_report()
        with self.assertRaises(ValueError):
            verify_candidate(self.baseline, self.candidate, True)

    def test_incomplete_audit_timeline_is_rejected(self):
        self.report['intervals'] = []
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'timeline'):
            verify_candidate(self.baseline, self.candidate, True)

    def test_temporal_candidate_is_verified_without_approving_it(self):
        correction = {'format': 'event_supervised_temporal_rigid_alignment_v2', 'frame_count': 2,
                      'knot_count': 4, 'raw': [0]*7, 'temporal_offsets': [[0]*7]*4}
        path = self.candidate / 'correction.json'
        path.write_text(json.dumps(correction))
        self.report['artifact_sha256']['correction'] = sha256_file(path)
        self.write_report()
        _, _, report = verify_candidate(self.baseline, self.candidate, True)
        self.assertIs(report['approved_for_export'], False)
        correction['frame_count'] = 3
        path.write_text(json.dumps(correction))
        with self.assertRaisesRegex(ValueError, 'timeline'):
            verify_candidate(self.baseline, self.candidate, True)

    def test_sampling_contract_must_match_cache_correction_and_audit(self):
        sampling = {'render_supersample': 2, 'render_downsample_filter': 'area_after_clamp'}
        paths = {'baseline_cache_metadata': self.baseline / 'cache.json',
                 'correction': self.candidate / 'correction.json'}
        for key, path in paths.items():
            record = json.loads(path.read_text()); record.update(sampling)
            path.write_text(json.dumps(record))
            self.report['artifact_sha256'][key] = sha256_file(path)
        # Correct file hashes alone cannot excuse a mismatched image operator.
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'sampling contract mismatch'):
            verify_candidate(self.baseline, self.candidate, True)
        self.report.update(sampling); self.write_report()
        verify_candidate(self.baseline, self.candidate, True)
        self.report['render_downsample_filter'] = 'bilinear'; self.write_report()
        with self.assertRaisesRegex(ValueError, 'render_downsample_filter'):
            verify_candidate(self.baseline, self.candidate, True)


if __name__ == '__main__':
    unittest.main()
