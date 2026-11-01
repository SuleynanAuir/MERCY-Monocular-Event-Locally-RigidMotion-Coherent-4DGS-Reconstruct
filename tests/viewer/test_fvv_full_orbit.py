"""Resource recovery and isolation for the long-running orbit pipeline."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scripts.fvv import train_full_orbit as pipeline


class PipelineRecovery(unittest.TestCase):
    def test_dense_arc_repeats_training_only_and_reuses_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['wide','arc']:
                source=root/name;source.mkdir()
                views=[]
                for index,split in enumerate(['train','validation']):
                    filename=f'{index}.png';(source/filename).write_bytes(b'image')
                    views.append(dict(file=filename,split=split,time=index))
                (source/'views.json').write_text(json.dumps(dict(expected_views=2,views=views)))
            pipeline.merge_showcase_views(root/'wide',root/'arc',root/'mixed')
            data=json.loads((root/'mixed/views.json').read_text())
            self.assertEqual(data['expected_views'],5)
            self.assertEqual(sum(v['split']=='validation' for v in data['views']),2)
            self.assertEqual(len(list((root/'mixed').glob('*.png'))),4)
            self.assertTrue((root/'mixed/arc-0.png').is_symlink())

    def exercise(self, failure, restore_failure=False, normal_constraint=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'models' / 'current'
            model.mkdir(parents=True)
            (model / 'model.pth').write_bytes(b'unchanged original')
            argv = ['train_full_orbit', '--root', str(root / 'job')]
            for key, value in dict(repo=root, model=model, blender=root/'blender',
                                   blend=root/'scene.blend', binding=root/'binding', video=root/'video').items():
                argv += ['--' + key, str(value)]
            if normal_constraint:
                argv += ['--geometry',str(root/'prior.npz'),'--surface-normal-weight','5']
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if failure and len(calls) == 2:
                    raise subprocess.CalledProcessError(1, command)
            with patch('sys.argv', argv), patch.object(pipeline, 'pause') as pause, \
                 patch.object(pipeline, 'resume') as resume, patch.object(pipeline.subprocess, 'run', side_effect=run), \
                 patch.object(pipeline.signal, 'signal'):
                if restore_failure:
                    resume.side_effect=RuntimeError('renderer restart failed')
                    with self.assertRaisesRegex(RuntimeError, 'renderer restart failed'): pipeline.main()
                elif failure:
                    with self.assertRaises(subprocess.CalledProcessError): pipeline.main()
                else:
                    pipeline.main()
                pause.assert_called_once()
                resume.assert_called_once()
            self.assertFalse((model.parent / '.fvv-writer.lock').exists())
            self.assertEqual((model / 'model.pth').read_bytes(), b'unchanged original')
            state = json.loads((root/'job/status.json').read_text())
            self.assertEqual(state['stage'], 'restore_failed' if restore_failure else 'failed' if failure else 'evaluated')
            self.assertEqual(len(calls), 2 if failure else 4)
            if normal_constraint:
                self.assertIn('--surface-normal-weight',calls[1])
                self.assertEqual(calls[1][calls[1].index('--surface-normal-weight')+1],'5.0')

    def test_training_failure_restores_renderers_and_preserves_model(self):
        self.exercise(True)

    def test_evaluates_without_activating_candidate(self):
        self.exercise(False)

    def test_optional_normal_constraint_reaches_isolated_worker(self):
        self.exercise(False,normal_constraint=True)

    def test_restore_failure_is_reported_and_lock_is_released(self):
        self.exercise(False, restore_failure=True)


if __name__ == '__main__':
    unittest.main()
