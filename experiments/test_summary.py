import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import build_summary as summary


class QuantitativeArchiveTests(unittest.TestCase):
    def test_original_bytes_and_selected_values(self):
        result = summary.verify()
        self.assertTrue(result['passed'])
        self.assertEqual(result['selected_objects'], 9)

    def test_decimal_tokens_are_not_rounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'original.json'
            p.write_text('{"x":0.10000000000000001,"y":1.2300e-09,"n":null}')
            self.assertEqual(summary.read(p, exact=True),
                             dict(x='0.10000000000000001', y='1.2300e-09', n=None))

    def test_missing_is_not_zero(self):
        self.assertNotEqual(summary.fmt(None), summary.fmt('0'))
        self.assertIn('缺失', summary.fmt(None))

    def test_one_recommendation_per_object(self):
        items = summary.read(summary.OUT/'summary.json')['objects']
        self.assertEqual({x['object'] for x in items},
                         {'lego','fan','lamp','metronome','microphone','arm','robot','gears','rotation'})
        self.assertEqual(len(items), 9)

    def test_robot_uses_one_protocol_without_old_metrics(self):
        item = summary.read(summary.OUT/'best/robot.json')
        self.assertEqual(item['json_pointer'], '/summary/original_all/candidate')
        self.assertEqual(item['sample_count'], 24)
        self.assertNotIn('foreground_ssim', item['metrics_exact_decimal_strings'])
        self.assertNotIn('latency_ms', item['metrics_exact_decimal_strings'])

    def test_tampered_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); raw=b'{"psnr":25.0}'
            (out/'source.json').write_bytes(b'{"psnr":95.0}')
            summary.write(out/'source_manifest.json', [dict(original='nonexistent-input.json',
                snapshot='source.json', sha256=summary.digest(raw), bytes=len(raw))])
            with patch.object(summary,'OUT',out), self.assertRaises(AssertionError):
                summary.verify()


if __name__ == '__main__':
    unittest.main()
