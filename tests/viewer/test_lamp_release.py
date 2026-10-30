import unittest
import tempfile
import gzip
import json
from pathlib import Path

from scripts.lamp.verify_release import DIRECTIONS, quality_gate, sha, verify


class LampReleaseTests(unittest.TestCase):
    def rows(self):
        return [dict(version='V2', **{key: 1. for key in DIRECTIONS}),
                dict(version='V3', **{key: 1.+direction*.1 for key, direction in DIRECTIONS.items()})]

    def test_all_metrics_improve(self):
        self.assertEqual(len(quality_gate(self.rows())), 12)

    def test_event_regression_rejected_despite_rgb_improvement(self):
        rows = self.rows()
        rows[1]['event_cosine'] = .5
        with self.assertRaisesRegex(ValueError, 'event_cosine'):
            quality_gate(rows)

    def test_missing_nan_and_infinite_cannot_approve(self):
        for value in (None, float('nan'), float('inf')):
            rows = self.rows()
            rows[1]['lpips'] = value
            with self.assertRaisesRegex(ValueError, 'lpips'):
                quality_gate(rows)

    def test_identical_baseline_is_not_an_improvement(self):
        rows = self.rows()
        rows[1] = dict(rows[0], version='V3')
        with self.assertRaisesRegex(ValueError, 'no measured improvement'):
            quality_gate(rows)

    def test_report_binds_exact_candidate_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model.npz'
            model.write_bytes(b'checkpoint')
            pair = root/'pair.json'
            pair.write_text(json.dumps(dict(right=dict(path=model.name, sha256=sha(model)))))
            data = root/'data'
            data.mkdir()
            (data/'analysis.json').write_text(json.dumps(dict(summary=self.rows())))
            manifest = dict(scene='lamp', samples=[dict(render_mode='gaussian')],
                            sources=[dict(source=str(p), sha256=sha(p)) for p in (model,pair)])
            (data/'input_manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest).encode()))
            (root/'artifact_manifest.json').write_text(json.dumps(dict(files={
                str(p.relative_to(root)): dict(sha256=sha(p)) for p in data.iterdir()})))
            self.assertTrue(verify(pair,root)['passed'])
            model.write_bytes(b'other checkpoint')
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                verify(pair,root)
            model.write_bytes(b'checkpoint')
            pair.write_text(pair.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'different asset'):
                verify(pair,root)
