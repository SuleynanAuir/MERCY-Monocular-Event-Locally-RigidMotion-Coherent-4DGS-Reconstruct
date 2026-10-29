import asyncio
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from scripts.fvv import common

spec=importlib.util.spec_from_file_location('fvv_management_test',Path(__file__).resolve().parents[2]/'scripts/fvv/management.py')
management=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'common':common}):spec.loader.exec_module(management)


class ManagementTests(unittest.IsolatedAsyncioTestCase):
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
