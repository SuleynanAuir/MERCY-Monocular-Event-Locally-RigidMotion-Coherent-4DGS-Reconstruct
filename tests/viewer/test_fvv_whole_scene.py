"""Whole-scene failures must restore managed viewers and preserve the live model."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from scripts.fvv import train_whole_scene as pipeline
from scripts.fvv import train_detail_scene as detail_pipeline


class WholeScenePipeline(unittest.TestCase):
    def test_failed_training_restores_services_and_releases_only_its_lock(self):
        self.check_pipeline(pipeline)

    def test_failed_joint_detail_training_restores_services(self):
        self.check_pipeline(detail_pipeline)

    def check_pipeline(self,pipeline):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'models/current';model.mkdir(parents=True)
            (model/'model.pth').write_bytes(b'base');(model/'rigid-bucket.npz').write_bytes(b'articulated')
            argv=['train_whole_scene','--root',str(root/'run'),'--model',str(model)]
            for key in ['repo','video','dataset','views','rays','references','audit-rays']:argv+=['--'+key,str(root/key)]
            child=Mock();child.wait.return_value=1
            with patch('sys.argv',argv),patch.object(pipeline.signal,'signal'),patch.object(pipeline,'pause') as pause,patch.object(pipeline,'resume') as resume,patch.object(pipeline.subprocess,'Popen',return_value=child):
                with self.assertRaises(pipeline.subprocess.CalledProcessError):pipeline.main()
            pause.assert_called_once();resume.assert_called_once();child.terminate.assert_called_once()
            self.assertFalse((model.parent/'.fvv-writer.lock').exists())
            self.assertEqual((model/'model.pth').read_bytes(),b'base');self.assertEqual((model/'rigid-bucket.npz').read_bytes(),b'articulated')
            command=json.loads((root/'run/commands.jsonl').read_text().splitlines()[0])
            for flag in ['--whole-scene','--preserve-rigid','--ray-spread-weight']:self.assertIn(flag,command)
            self.assertEqual(json.loads((root/'run/status.json').read_text())['stage'],'failed')


if __name__=='__main__':unittest.main()
