import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from aiohttp.test_utils import TestClient,TestServer
from aiohttp import web
from scripts.fvv.orbit_export import trajectory,OrbitExporter

META=dict(scene='lego',position=[0,0,0],yaw=0,pitch=0,fov=40,time_duration=[0,2],sha256='abc',revision='rev')
class Trajectory(unittest.TestCase):
    def test_overview_covers_all_sides_without_dolly_or_large_height_changes(self):
        plan=trajectory(dict(seconds=24,fps=60,yaw=170,position=[9,9,9]),META)
        self.assertEqual(plan['path'],'overview')
        self.assertEqual(plan['target'],[0,0,1.5])
        self.assertAlmostEqual(plan['poses'][720]['yaw'],180)
        self.assertGreater(plan['poses'][-1]['yaw'],359.99)
        for pose in plan['poses']:
            self.assertTrue(1<=pose['pitch']<=3)
            self.assertAlmostEqual(math.dist(pose['position'],plan['target']),1.8)
            direction=[b-a for a,b in zip(pose['position'],plan['target'])]
            self.assertAlmostEqual(direction[1]/1.8,math.sin(math.radians(pose['pitch'])))

    def test_full_orbit_looks_at_fixed_target_and_wraps_animation(self):
        p=trajectory(dict(path='full',seconds=4,fps=24,width=640),META)
        self.assertEqual(p['frames'],96);self.assertEqual(p['height'],360)
        self.assertEqual(p['poses'][0]['position'],META['position'])
        self.assertLess(min(x['pitch'] for x in p['poses']),-14)
        self.assertGreater(max(x['pitch'] for x in p['poses']),14)
        for pose in p['poses']:
            delta=[b-a for a,b in zip(pose['position'],p['target'])];length=math.sqrt(sum(v*v for v in delta))
            self.assertAlmostEqual(length,1.5)
            self.assertAlmostEqual(delta[1]/length,math.sin(math.radians(pose['pitch'])))
            self.assertTrue(0<=pose['time']<2)
        self.assertAlmostEqual(p['poses'][48]['time'],0)
        self.assertAlmostEqual(p['poses'][24]['yaw'],90)
    def test_showcase_anchors_front_and_stays_in_shallow_arc(self):
        p=trajectory(dict(path='showcase',yaw=160,pitch=89,position=[9,9,9],radius=0),META)
        self.assertEqual(p['path'],'showcase')
        self.assertEqual(p['target'],[0,0,1.5])
        self.assertAlmostEqual(p['poses'][0]['yaw'],0)
        for pose in p['poses']:
            self.assertTrue(-3<=pose['yaw']<=0)
            self.assertTrue(0<=pose['pitch']<=.5)
            self.assertEqual(pose['fov'],40)
            self.assertAlmostEqual(math.dist(pose['position'],p['target']),1.65)
        self.assertLess(abs(p['poses'][-1]['yaw']-p['poses'][0]['yaw']),.002)
        with self.assertRaises(ValueError):trajectory({'path':'unknown'},META)
        self.assertEqual(trajectory({},dict(META,scene='fan'))['path'],'full')
    def test_rejects_unbounded_and_nonfinite_inputs(self):
        for d in [dict(seconds=100),dict(width=1000),dict(fps=25),dict(radius=0),dict(elevation=float('nan')),dict(position=[0,float('inf'),0]),dict(temporal_samples=2),dict(seconds=True)]:
            with self.assertRaises(ValueError):trajectory(dict(d,path='full'),META)

class Jobs(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)/'models';root.mkdir()
        r=SimpleNamespace(meta=META,root=root/'lego',temporal_sigma=1/60,torch=SimpleNamespace(__version__='test',cuda=SimpleNamespace(get_device_name=lambda:'fake')))
        self.pool=ThreadPoolExecutor(1);self.exporter=OrbitExporter(root,lambda:r,self.pool,lambda:False)
        app=web.Application();self.exporter.install(app);self.client=TestClient(TestServer(app));await self.client.start_server()
    async def asyncTearDown(self):
        await self.client.close();self.pool.shutdown();self.temp.cleanup()
    async def test_failure_releases_lock_and_does_not_publish_partial_video(self):
        with patch('scripts.fvv.orbit_export.shutil.which',return_value='/ffmpeg'),patch('scripts.fvv.orbit_export.asyncio.create_subprocess_exec',side_effect=OSError('encoder unavailable')):
            response=await self.client.post('/orbit-export?scene=lego',json={'seconds':4,'fps':24,'width':640})
            self.assertEqual(response.status,202);await self.exporter.task
        self.assertEqual(self.exporter.job['state'],'failed')
        self.assertFalse((self.exporter.model_root/'.fvv-writer.lock').exists())
        self.assertEqual((await (await self.client.get('/orbit-export?scene=lego')).json())['exports'],[])
    async def test_writer_lock_and_invalid_downloads(self):
        lock=self.exporter.model_root/'.fvv-writer.lock';lock.write_text('training')
        with patch('scripts.fvv.orbit_export.shutil.which',return_value='/ffmpeg'):
            self.assertEqual((await self.client.post('/orbit-export?scene=lego',json={})).status,409)
        self.assertTrue(lock.exists())
        self.assertEqual((await self.client.get('/orbit-export/not-an-id/video')).status,404)
    async def test_completed_files_are_scene_scoped_and_deletable(self):
        folder=self.exporter.root/('a'*32);folder.mkdir(parents=True)
        (folder/'video.mp4').write_bytes(b'video');(folder/'parameters.json').write_text(json.dumps(dict(scene='lego',created_at='today')))
        response=await self.client.get('/orbit-export/'+'a'*32+'/video?scene=lego')
        self.assertEqual(await response.read(),b'video');self.assertIn('attachment',response.headers['Content-Disposition'])
        self.assertEqual((await self.client.get('/orbit-export/'+'a'*32+'/video?scene=fan')).status,404)
        self.assertEqual((await self.client.post('/orbit-export/'+'a'*32+'/delete?scene=lego',json={})).status,200)
        self.assertFalse(folder.exists())

    async def test_cancel_releases_writer_lock(self):
        entered=asyncio.Event()
        async def stalled_encoder(*args,**kwargs):
            entered.set();await asyncio.Event().wait()
        with patch('scripts.fvv.orbit_export.shutil.which',return_value='/ffmpeg'),patch('scripts.fvv.orbit_export.asyncio.create_subprocess_exec',side_effect=stalled_encoder):
            self.assertEqual((await self.client.post('/orbit-export?scene=lego',json={})).status,202)
            await entered.wait()
            self.assertEqual((await self.client.post('/orbit-export/cancel?scene=lego',json={})).status,200)
        self.assertFalse(self.exporter.running)
        self.assertFalse((self.exporter.model_root/'.fvv-writer.lock').exists())
        self.assertEqual(self.exporter.job['state'],'failed')

    async def test_storage_quota_rejects_before_creating_a_job(self):
        folder=self.exporter.root/('b'*32);folder.mkdir(parents=True)
        with (folder/'video.mp4').open('wb') as f:f.truncate(self.exporter.STORE_LIMIT)
        with patch('scripts.fvv.orbit_export.shutil.which',return_value='/ffmpeg'):
            response=await self.client.post('/orbit-export?scene=lego',json={})
        self.assertEqual(response.status,507)
        self.assertFalse((self.exporter.model_root/'.fvv-writer.lock').exists())
