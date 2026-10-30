import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.fvv import training_resources as resources


class TrainingResourceTests(unittest.TestCase):
    def test_exact_argv_replaces_stale_pid_and_excludes_hybrid(self):
        with tempfile.TemporaryDirectory() as directory:
            registry=Path(directory)/'processes.json'
            cmd=['/env/python','viewer.py','--scenes_path','scene.json','--port','8895','--lazy']
            registry.write_text(json.dumps([dict(pid=999,port=8895,command=cmd),
                dict(pid=44,port=8897,command=['/env/python','hybrid.py','serve'])]))
            running={321:cmd,999:['unrelated'],44:['/env/python','hybrid.py','serve']}
            def kill(pid, sig):running.pop(pid)
            records=[]
            with patch.object(resources,'command',side_effect=lambda pid:running.get(pid,[])), \
                 patch.object(Path,'iterdir',return_value=[Path('/proc/321'),Path('/proc/999'),Path('/proc/44')]), \
                 patch.object(Path,'read_bytes',return_value=b'PATH=/bin\0'), \
                 patch.object(resources.os,'readlink',return_value=directory), \
                 patch.object(resources.os,'kill',side_effect=kill) as terminate:
                resources.pause([str(registry)],records)
            self.assertEqual([r['pid'] for r in records],[321])
            self.assertEqual(terminate.call_count,1)
            self.assertIn(999,running);self.assertIn(44,running)
            with patch.object(resources,'command',return_value=[]), \
                 patch.object(resources.subprocess,'Popen',return_value=SimpleNamespace(pid=777)) as launch:
                resources.resume(records)
            self.assertEqual(json.loads(registry.read_text())[0]['pid'],777)
            self.assertEqual(launch.call_args.args[0],cmd)

    def test_resume_attempts_all_workers_even_if_one_fails(self):
        records=[dict(registry='/missing/registry',port=p,command=['python','viewer.py'],pid=p,env={},cwd='/tmp') for p in (1,2)]
        with patch.object(resources,'command',return_value=[]), \
             patch.object(Path,'open',side_effect=OSError('unavailable')) as opened:
            with self.assertRaisesRegex(RuntimeError,'port 1.*port 2'):
                resources.resume(records)
            self.assertEqual(opened.call_count,2)
