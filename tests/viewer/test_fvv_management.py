import asyncio
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock
import zipfile
import shutil

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from scripts.fvv import common

spec=importlib.util.spec_from_file_location('fvv_management_test',Path(__file__).resolve().parents[2]/'scripts/fvv/management.py')
management=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'common':common}):spec.loader.exec_module(management)


class ManagementTests(unittest.IsolatedAsyncioTestCase):
    def test_training_error_preserves_cause_instead_of_supervisor_traceback(self):
        output='RuntimeError: CUDA error: an illegal memory access was encountered\nRuntimeError: Offline worker failed; inspect its console output'
        self.assertIn('非法内存访问',management.training_error(output))
        self.assertNotIn('Offline worker',management.training_error(output))
        self.assertIn('尚未开始训练',management.training_error(output+'\nGPU health check failed'))
        oom=management.training_error('CUDA error: out of memory\nGPU health check failed')
        self.assertIn('显存不足',oom)
        self.assertNotIn('修复 GPU',oom)
        self.assertIn('训练期间',management.training_error('torch.cuda.OutOfMemoryError: CUDA out of memory'))
        self.assertEqual(management.training_error('ValueError: invalid dataset\nRuntimeError: Offline worker failed'),'ValueError: invalid dataset')

    async def test_failed_training_keeps_bounded_log_and_preserves_current_model(self):
        folder=self.root/'failed-job';folder.mkdir()
        self.manager.job=dict(id=folder.name,iterations=6000,state='running')
        stdout=asyncio.StreamReader()
        stdout.feed_data(b'x'*(management.TRAINING_LOG_LIMIT*2)+b'\nRuntimeError: CUDA error: an illegal memory access was encountered')
        stdout.feed_eof()
        process=SimpleNamespace(stdout=stdout,wait=AsyncMock(return_value=1))
        profile=dict(config='config.yaml',source='dataset',scene='lego',position=[0,0,0])
        with patch.object(management.asyncio,'create_subprocess_exec',AsyncMock(return_value=process)) as launch:
            await self.manager.run_training(profile,folder.name)
        self.assertEqual(self.manager.job['state'],'failed')
        self.assertEqual((folder/'training-console.log').stat().st_size,management.TRAINING_LOG_LIMIT)
        self.assertIn('非法内存访问',self.manager.job['error'])
        self.assertEqual(common.digest(self.old/'model.pth'),self.meta['sha256'])
        self.assertFalse((folder/'manifest.json').exists())
        self.assertEqual(self.loaded,[])

    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=self.root/'existing';self.old.mkdir()
        (self.old/'model.pth').write_bytes(b'tensor-only-test-fixture');(self.old/'effective.yaml').write_text('gaussian_dim: 4')
        self.meta=dict(schema=1,revision='a'*40,scene='lego',checkpoint='model.pth',config='effective.yaml',
                       position=[0,0,0],yaw=0,pitch=0,time_duration=[0,1],sha256=common.digest(self.old/'model.pth'),
                       config_sha256=common.digest(self.old/'effective.yaml'))
        (self.old/'manifest.json').write_text(json.dumps(self.meta));self.loaded=[]
        async def load(path):self.loaded.append(path)
        self.manager=management.ModelManager(SimpleNamespace(model=self.old,repo=self.root,profiles=None),'a'*40,lambda:self.old,load)
        app=web.Application();self.manager.install(app);self.client=TestClient(TestServer(app));await self.client.start_server()
    async def asyncTearDown(self):
        await self.client.close();self.temp.cleanup()
    def bundle(self,overrides=None):
        content={p.name:p.read_bytes() for p in self.old.iterdir()}
        content.update(overrides or {});stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            for name,value in content.items():z.writestr(name,value)
        return stream.getvalue()
    async def test_import_deduplicates_and_cleans_temporary_files(self):
        r=await self.client.post('/models/import?scene=lego',data=self.bundle());self.assertEqual(r.status,200)
        self.assertEqual((await r.json())['id'],'existing');self.assertEqual(list(self.root.iterdir()),[self.old])
    async def test_import_rejects_traversal_bad_hash_and_wrong_scene(self):
        for content,scene in [(self.bundle({'../escape':b'x'}),'lego'),(self.bundle({'model.pth':b'changed'}),'lego'),(self.bundle(),'fan')]:
            r=await self.client.post('/models/import?scene='+scene,data=content);self.assertEqual(r.status,400)
            self.assertEqual(list(self.root.iterdir()),[self.old]);self.assertEqual(common.digest(self.old/'model.pth'),self.meta['sha256'])
    async def test_import_rejects_symlink(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            for p in self.old.iterdir():
                info=zipfile.ZipInfo(p.name)
                if p.name=='model.pth':info.external_attr=(0o120777<<16)
                z.writestr(info,p.read_bytes())
        r=await self.client.post('/models/import?scene=lego',data=stream.getvalue());self.assertEqual(r.status,400)
    async def test_select_is_scene_scoped_and_cannot_traverse(self):
        for key,scene in [('../existing','lego'),('existing','fan')]:
            r=await self.client.post('/models/load?scene='+scene,json={'id':key});self.assertEqual(r.status,400)
        r=await self.client.post('/models/load?scene=lego',json={'id':'existing'});self.assertEqual(r.status,202)
        await asyncio.sleep(.01);self.assertEqual(self.loaded,[self.old.resolve()])
    async def test_real_metrics_drive_progress_and_saving_never_claims_complete(self):
        self.manager.job=dict(id='job',state='running',scene='lego',iterations=6000,iteration=0,progress=0)
        folder=self.root/'job';folder.mkdir();(folder/'training-metrics.jsonl').write_text(json.dumps({'iteration':3000})+'\n')
        self.assertEqual(self.manager.progress()['progress'],50)
        (folder/'training-metrics.jsonl').write_text(json.dumps({'iteration':6000})+'\n')
        self.assertEqual(self.manager.progress()['progress'],99)
        r=await self.client.get('/models?scene=fan');self.assertIsNone((await r.json())['job'])
        r=await self.client.post('/models/import?scene=lego',data=self.bundle());self.assertEqual(r.status,409)
    async def test_train_rejects_unconfigured_profiles(self):
        r=await self.client.post('/training?scene=lego',json={'profile':'arbitrary-command'});self.assertEqual(r.status,400)

    async def test_delete_rejects_active_wrong_scene_traversal_and_stale_hash(self):
        other=self.root/'other';shutil.copytree(self.old,other)
        for name,scene,sha,status in [('existing','lego',self.meta['sha256'],409),
            ('../other','lego',self.meta['sha256'],400),('other','fan',self.meta['sha256'],400),
            ('other','lego','stale',409)]:
            r=await self.client.post('/models/delete?scene='+scene,json={'id':name,'sha256':sha})
            self.assertEqual(r.status,status);self.assertTrue(other.exists());self.assertTrue(self.old.exists())

    async def test_delete_removes_only_selected_model_and_clears_completed_job(self):
        other=self.root/'other';shutil.copytree(self.old,other)
        self.manager.job=dict(id='other',state='complete',scene='lego')
        r=await self.client.post('/models/delete?scene=lego',json={'id':'other','sha256':self.meta['sha256']})
        self.assertEqual(r.status,200);self.assertFalse(other.exists());self.assertTrue(self.old.exists())
        self.assertIsNone(self.manager.job);self.assertEqual(list(self.root.iterdir()),[self.old])

    async def test_delete_rejects_loading_or_writer_lock_and_exposes_timestamp(self):
        other=self.root/'other';shutil.copytree(self.old,other)
        for loading in [True,False]:
            self.manager.loading=loading
            if not loading:(self.root/'.fvv-writer.lock').touch()
            r=await self.client.post('/models/delete?scene=lego',json={'id':'other','sha256':self.meta['sha256']})
            self.assertEqual(r.status,409);self.assertTrue(other.exists())
        rows=self.manager.models('lego');self.assertIn('trained_at',rows[0])
