"""FVV contracts and gateway isolation, runnable without CUDA."""
import io
import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.fvv import common
import json
import unittest
from aiohttp import web, WSMsgType
from aiohttp.test_utils import TestClient, TestServer
from scripts.fvv.common import validate_pose, LimitedWriter
from utils.fvv_gateway import install_fvv


class Contracts(unittest.TestCase):
    def test_camera_validation(self):
        valid = dict(position=[1,2,3], yaw=45, pitch=20, time=.5, width=640, height=480)
        self.assertEqual(validate_pose(valid, [0,1])['position'], [1,2,3])
        for key, value in [('position',[1,2]), ('yaw',float('nan')), ('time',2),
                           ('width',4096), ('pitch',90), ('height',True), ('jpeg_quality',91), ('jpeg_quality',99), ('temporal_samples',2), ('temporal_samples',True), ('temporal_samples',3.0)]:
            with self.assertRaises(ValueError):
                validate_pose({**valid,key:value}, [0,1])

    def test_serialization_stops_before_budget(self):
        output = io.BytesIO()
        writer = LimitedWriter(output, 4)
        writer.write(b'123')
        with self.assertRaises(OSError):
            writer.write(b'45')
        self.assertEqual(output.getvalue(), b'123')

    def test_reject_remote_upstream(self):
        for url in ('https://example.com','http://example.com','http://localhost/path','http://a@localhost'):
            with self.assertRaises(ValueError):
                install_fvv(web.Application(),url)


class Gateway(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients=[]
        self.servers=[]
    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        for server in self.servers:
            await server.close()
    async def client(self, app):
        client=TestClient(TestServer(app)); await client.start_server(); self.clients.append(client)
        return client
    async def test_unconfigured_and_assets(self):
        app=web.Application(); install_fvv(app,''); client=await self.client(app)
        status=await (await client.get('/fvv/status')).json()
        self.assertFalse(status['ready'])
        for path in ('/fvv','/fvv/fvv.js','/fvv/fvv.css','/fvv/fvv-loop-cache.js'):
            self.assertEqual((await client.get(path)).status,200)
        self.assertEqual((await client.get('/fvv/secret')).status,404)
        self.assertEqual((await client.get('/fvv/ws')).status,503)
    async def test_proxy_keeps_scene_and_binary_frames(self):
        upstream=web.Application()
        async def status(request):
            return web.json_response({'ready':True,'scene':request.query['scene']})
        async def socket(request):
            ws=web.WebSocketResponse(); await ws.prepare(request)
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await ws.send_bytes(request.query['scene'].encode()+message.data.encode())
            return ws
        upstream.router.add_get('/status',status); upstream.router.add_get('/ws',socket)
        server=TestServer(upstream); await server.start_server(); self.servers.append(server)
        app=web.Application(); install_fvv(app,str(server.make_url('')).rstrip('/'))
        client=await self.client(app)
        self.assertEqual((await (await client.get('/fvv/status?scene=lego')).json())['scene'],'lego')
        ws=await client.ws_connect('/fvv/ws?scene=lego')
        await ws.send_str('camera')
        self.assertEqual((await ws.receive()).data,b'legocamera')
        await ws.close()
        self.assertEqual((await client.get('/fvv/ws',headers={'Origin':'http://evil.example'})).status,403)
    async def test_management_streams_upload_and_rejects_foreign_origin(self):
        upstream=web.Application()
        async def upload(request):
            count=0
            async for chunk in request.content.iter_chunked(65536):count+=len(chunk)
            return web.json_response({'bytes':count,'scene':request.query['scene']},status=201)
        upstream.router.add_post('/models/import',upload)
        upstream.router.add_post('/models/delete',upload)
        server=TestServer(upstream);await server.start_server();self.servers.append(server)
        app=web.Application();install_fvv(app,str(server.make_url('')).rstrip('/'))
        client=await self.client(app)
        response=await client.post('/fvv/api/models/import?scene=lego',data=b'x'*(9*1024**2))
        self.assertEqual(response.status,201)
        self.assertEqual(await response.json(),{'bytes':9*1024**2,'scene':'lego'})
        response=await client.post('/fvv/api/models/import',data=b'x',headers={'Origin':'http://evil.example'})
        self.assertEqual(response.status,403)
        self.assertEqual((await client.post('/fvv/api/arbitrary')).status,404)
        response=await client.post('/fvv/api/models/delete?scene=lego',json={'id':'old','sha256':'known'})
        self.assertEqual(response.status,201)
        self.assertEqual((await response.json())['scene'],'lego')
        self.assertEqual((await client.post('/fvv/api/models/delete',headers={'Origin':'http://evil.example'})).status,403)

    async def test_offline_upstream_reports_not_ready(self):
        app=web.Application(); install_fvv(app,'http://127.0.0.1:1')
        client=await self.client(app)
        self.assertFalse((await (await client.get('/fvv/status')).json())['ready'])



class OfflineSafety(unittest.TestCase):
    def adapter(self):
        spec = importlib.util.spec_from_file_location('fvv_adapter_test', Path(__file__).resolve().parents[2]/'scripts/fvv/hybrid.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'common': common}):
            spec.loader.exec_module(module)
        return module

    def test_compaction_preserves_inference_fields(self):
        module = self.adapter()
        sentinel = object()
        torch = SimpleNamespace(empty=lambda _: sentinel, is_tensor=lambda _: False)
        state = tuple(range(28))
        with patch.dict(sys.modules, {'torch': torch}):
            result, step = module.compact((state,6000))
        dropped = {7,8,9,10,11,25,26,27}
        for index in set(range(28))-dropped:
            self.assertEqual(result[index], state[index])
        self.assertIsNone(result[11])
        self.assertEqual(step,6000)
        self.assertEqual(state,tuple(range(28)))

    def test_torch21_load_keeps_weights_only(self):
        module=self.adapter()
        calls=[]
        fake=SimpleNamespace(serialization=SimpleNamespace(),
                             load=lambda *a,**kw:calls.append((a,kw)))
        with patch.dict(sys.modules, {'torch':fake}):
            module.load_checkpoint('model.pth','cpu')
        self.assertEqual(calls[0][1],dict(map_location='cpu',weights_only=True))

    def test_archive_revision_is_validated(self):
        module=self.adapter()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); marker=root/'UPSTREAM_REVISION'
            marker.write_text('9528828d79148f032c924ccc840bd81374ab800c')
            self.assertEqual(module.revision(root),marker.read_text())
            marker.write_text('unverified-main')
            with self.assertRaises(ValueError):module.revision(root)

    def test_existing_model_is_never_removed(self):
        module=self.adapter()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'model';root.mkdir()
            marker=root/'manifest.json';marker.write_text('existing')
            args=SimpleNamespace(output=root,total_budget_gib=32,budget_gib=8)
            with self.assertRaises(FileExistsError):module.supervise(args)
            self.assertEqual(marker.read_text(),'existing')
            self.assertFalse((root.parent/'.fvv-writer.lock').exists())

    def test_store_budget_refuses_new_run(self):
        module=self.adapter()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new'
            args=SimpleNamespace(output=root,total_budget_gib=1,budget_gib=8)
            with self.assertRaises(OSError):module.supervise(args)
            self.assertFalse(root.exists())


class CudaPreflightTests(unittest.TestCase):
    def test_training_priority_restores_viewers_when_health_fails(self):
        from unittest.mock import Mock
        module=OfflineSafety().adapter()
        records=[]
        def pause(registries,paused):
            self.assertEqual(registries,['trusted.json']);paused.append('viewer');records.append('paused')
        helper=SimpleNamespace(pause=pause,resume=Mock(side_effect=lambda paused:records.append('resumed')))
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory)/'new',total_budget_gib=1,budget_gib=.01,command='train')
            with patch.dict(sys.modules,{'training_resources':helper}), \
                 patch.dict(module.os.environ,{'MERCY_FVV_RENDER_REGISTRIES':'["trusted.json"]'}), \
                 patch.object(module.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout=b'CUDA out of memory')), \
                 patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'GPU health check'):module.supervise(args)
            self.assertEqual(records,['paused','resumed'])
            helper.resume.assert_called_once_with(['viewer'])
            self.assertFalse((args.output.parent/'.fvv-writer.lock').exists())

    def test_failed_or_timed_out_health_check_never_starts_training(self):
        module=OfflineSafety().adapter()
        cases=[SimpleNamespace(returncode=1,stdout=b'CUDA failure'*10000),
               module.subprocess.TimeoutExpired('health',90,output=b'timed out')]
        for result in cases:
            with self.subTest(result=type(result).__name__),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'new'
                args=SimpleNamespace(output=root,total_budget_gib=1,budget_gib=.01,command='train')
                options={'side_effect':result} if isinstance(result,Exception) else {'return_value':result}
                with patch.object(module.subprocess,'run',**options),patch.object(module.subprocess,'Popen') as worker,patch('builtins.print'):
                    with self.assertRaisesRegex(RuntimeError,'GPU health check'):module.supervise(args)
                worker.assert_not_called()
                self.assertFalse((root/'manifest.json').exists())
                self.assertFalse((root.parent/'.fvv-writer.lock').exists())
                self.assertLessEqual((root/'cuda-health.log').stat().st_size,65536)


class FramePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_is_ordered_and_bounds_render_ahead(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        module=OfflineSafety().adapter()
        gate=threading.Event(); rendered=[]
        class Renderer:
            def frame(self,pose,pixels=False):
                rendered.append(pose['time']);return pose['time']
            def encode(self,pixels,quality):
                gate.wait(2);return str(pixels).encode()
        class Socket:
            def __init__(self):self.incoming=asyncio.Queue();self.output=[]
            def __aiter__(self):return self
            async def __anext__(self):
                message=await self.incoming.get()
                if message is None:raise StopAsyncIteration
                return message
            async def send_bytes(self,value):
                self.output.append(value)
                if len(self.output)==10:await self.incoming.put(None)
        ws=Socket()
        for i in range(10):
            pose=dict(position=[0,0,0],yaw=0,pitch=0,time=i/10,width=640,height=640)
            await ws.incoming.put(SimpleNamespace(type=__import__('aiohttp').WSMsgType.TEXT,data=json.dumps(pose)))
        with ThreadPoolExecutor(max_workers=1) as render_pool,ThreadPoolExecutor(max_workers=1) as encode_pool:
            task=asyncio.create_task(module.stream_frames(ws,Renderer(),[0,1],render_pool,encode_pool))
            try:
                await asyncio.sleep(.1);self.assertLessEqual(len(rendered),4)
                gate.set();await asyncio.wait_for(task,3)
            finally:
                gate.set();task.cancel();await asyncio.gather(task,return_exceptions=True)
        self.assertEqual(ws.output,[str(i/10).encode() for i in range(10)])

    async def test_identified_live_camera_refresh_preserves_frame_credits(self):
        from concurrent.futures import ThreadPoolExecutor
        module=OfflineSafety().adapter()
        for coalesce in (True,False):
            class Renderer:
                def frame(self,pose,pixels=False):return pose['yaw']
                def encode(self,pixels,quality):return str(pixels).encode()
            class Socket:
                def __init__(self):self.output=[];self.index=0
                def __aiter__(self):return self
                async def __anext__(self):
                    if self.index==10:raise StopAsyncIteration
                    i=self.index;self.index+=1
                    return SimpleNamespace(type=WSMsgType.TEXT,data=json.dumps(dict(
                        position=[0,0,0],yaw=i,pitch=0,time=i/10,width=640,height=640,
                        request_id=i,coalesce=coalesce)))
                async def send_bytes(self,value):self.output.append(value)
            ws=Socket()
            with ThreadPoolExecutor(1) as rp,ThreadPoolExecutor(1) as ep:
                await asyncio.wait_for(module.stream_frames(ws,Renderer(),[0,1],rp,ep),3)
            ids=[int.from_bytes(frame[4:8],'big') for frame in ws.output]
            self.assertEqual(ids,list(range(10)))
            yaws=[float(frame[8:]) for frame in ws.output]
            if coalesce:
                self.assertGreater(yaws[0],0)
                self.assertEqual(yaws[-1],9)
            else:self.assertEqual(yaws,list(range(10)))
            for frame in ws.output:self.assertEqual(frame[:4],b'FVV1')


if __name__ == "__main__":
    unittest.main()
