import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from scripts.fvv import train_ray_quality as pipeline
from scripts.fvv.training_kernel import patch_backward


class RayPipeline(unittest.TestCase):
    def test_failed_child_restores_services_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'models/current';model.mkdir(parents=True)
            (model/'model.pth').write_bytes(b'original')
            argv=['train_ray_quality','--root',str(root/'run')]
            for name in ['repo','video','dataset','multiview','geometry','rays','wide-validation','novel-validation','arc-validation']:
                argv.extend(['--'+name,str(root/name)])
            argv.extend(['--model',str(model)])
            child=Mock();child.wait.return_value=1
            with patch('sys.argv',argv),patch.object(pipeline.signal,'signal'),patch.object(pipeline,'pause') as pause,patch.object(pipeline,'resume') as resume,patch.object(pipeline.subprocess,'Popen',return_value=child):
                with self.assertRaises(pipeline.subprocess.CalledProcessError):pipeline.main()
            pause.assert_called_once();resume.assert_called_once();child.terminate.assert_called_once()
            self.assertFalse((model.parent/'.fvv-writer.lock').exists())
            self.assertEqual((model/'model.pth').read_bytes(),b'original')
            self.assertEqual(json.loads((root/'run/status.json').read_text())['stage'],'failed')

    def test_kernel_patch_only_replaces_backward_dispatch_and_static_depth_gather(self):
        source='''collected_conic_opacity[block.thread_rank()] = conic_opacity_static[coll_id-P];
void BACKWARD::render(int P) { PerGaussianRenderCUDA<NUM_CHANNELS>(); }
// unchanged trailing source'''
        patched=patch_backward(source)
        self.assertIn('renderCUDA<NUM_CHANNELS><<<grid, block>>>',patched)
        self.assertIn('collected_depths[block.thread_rank()] = depths[coll_id];',patched)
        self.assertTrue(patched.endswith('// unchanged trailing source'))
        with self.assertRaises(ValueError):patch_backward(source.replace('PerGaussianRenderCUDA','unknown'))


if __name__=='__main__':unittest.main()
