"""Verify original UI and render bytes, merged selection and local-only access."""
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web, WSMsgType, FormData
from aiohttp.test_utils import TestClient, TestServer
from scripts.viewer.serve_multi_object_ui import create_app, read_config


class MultiObjectUITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b'original-jpeg-byte-payload'
        upstream = web.Application()
        async def status(request):
            self.status_scene = request.query['scene']
            return web.json_response({'ready': True, 'scene_id': request.query['scene'],
                'scene': {'left_label': 'V1', 'right_label': 'V3',
                          'clay': {'available': True, 'right_label': 'V3 · 白模'}}})
        self.html = b'<html>original UI</html>'
        async def html(request):
            return web.Response(body=self.html, content_type='text/html')
        async def ws(request):
            self.ws_scene = request.query['scene']
            socket = web.WebSocketResponse(); await socket.prepare(request)
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    await socket.send_bytes(self.payload)
            return socket
        async def experiment(request):
            if request.method == 'POST':
                return web.json_response({'scene':request.query['scene'], 'received':await request.json()},status=202)
            return web.Response(body=b'lossless-experiment-zip',headers={
                'Content-Type':'application/zip','Content-Disposition':'attachment; filename="experiment.zip"'})
        async def comparison(request):
            if request.method == 'POST':
                if request.content_type.startswith('multipart/'):
                    reader=await request.multipart();part=await reader.next();size=0
                    while chunk:=await part.read_chunk(1024*1024):size+=len(chunk)
                    return web.json_response({'field':part.name,'filename':part.filename,'bytes':size},status=202)
                return web.json_response({'received':await request.json(),'scene':request.query.get('scene')},status=202)
            return web.Response(body=b'comparison-content',headers={
                'Content-Type':'text/html','Content-Security-Policy':"frame-ancestors 'self'",
                'Content-Disposition':'inline','Cache-Control':'no-store'})
        for path in ('/comparison','/comparison.js','/comparison.css','/api/comparisons',
                     '/api/comparisons/job','/api/comparisons/job/files/report.html',
                     '/api/comparisons/job/files/figures/01_quality.png',
                     '/api/comparisons/job/files/results.zip'):
            upstream.router.add_get(path,comparison)
        upstream.router.add_post('/api/comparisons',comparison)
        upstream.router.add_post('/api/experiments', experiment)
        upstream.router.add_post('/api/experiments/plan', experiment)
        upstream.router.add_get('/api/experiments/job/download', experiment)
        upstream.router.add_get('/api/status', status)
        upstream.router.add_get('/', html)
        upstream.router.add_get('/ws', ws)
        self.remote = TestServer(upstream); await self.remote.start_server()
        url = str(self.remote.make_url('')).rstrip('/')
        self.config = self.root/'catalog.json'
        self.config.write_text(json.dumps({'default':'lego','asset_upstream':url,'objects':[
            {'id':'lego','name':'Lego','note':'Original assets missing'},
            {'id':'lamp','name':'Lamp','live':[url], 'revision_label':'重建版本'}]}))
        self.client = TestClient(TestServer(create_app(self.config))); await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close(); await self.remote.close(); self.temp.cleanup()

    async def test_optional_fvv_entry_keeps_scene_and_original_html(self):
        self.html=b'<html><body><aside>original controls</aside><canvas id="original"></canvas></body></html>'
        cfg=json.loads(self.config.read_text());cfg['fvv_entry']=True;self.config.write_text(json.dumps(cfg))
        response=await self.client.get('/?scene=lamp');html=await response.text()
        self.assertIn('href="/fvv?scene=lamp"',html)
        self.assertIn('/fvv/fvv-entry.js',html)
        self.assertIn('original controls',html)
        self.assertIn('<canvas id="original"></canvas>',html)
        self.assertEqual(html.count('id="realtimeGaussian"'),1)

    async def test_original_ui_and_missing_asset(self):
        r = await self.client.get('/api/status?scene=lego'); value = await r.json()
        self.assertTrue(value['error']); self.assertFalse(value['ready'])
        self.assertFalse(value['objects'][0]['ready']); self.assertTrue(value['objects'][1]['ready'])
        self.assertNotIn(self.temp.name, json.dumps(value))
        r = await self.client.get('/?scene=lamp')
        self.assertEqual(await r.read(), b'<html>original UI</html>')
        r = await self.client.get('/', allow_redirects=False)
        self.assertEqual(r.status, 302); self.assertEqual(r.headers['Location'], '/?scene=lego')
        for route in ('/media/lego/clay','/secret.json','/api/status?scene=missing'):
            self.assertEqual((await self.client.get(route)).status, 404)

    async def test_live_bytes_and_security(self):
        r = await self.client.get('/api/status?scene=lamp')
        data = await r.json()
        self.assertEqual(data['scene_id'], 'lamp')
        self.assertEqual(data['scene']['left_label'], 'V1')
        self.assertEqual(data['scene']['right_label'], 'V3（重建版本）')
        self.assertEqual(data['scene']['clay']['right_label'], 'V3 · 白模（重建版本）')
        async with self.client.ws_connect('/ws?scene=lamp') as socket:
            await socket.send_json({'type':'frame','frame':0})
            message = await socket.receive(timeout=3)
            self.assertEqual(message.type, WSMsgType.BINARY)
            self.assertEqual(message.data, self.payload)
        for headers in ({'Origin':'https://example.com'}, {'Host':'untrusted.example'}):
            self.assertEqual((await self.client.get('/api/status?scene=lamp', headers=headers)).status, 403)

    async def test_external_upstream_rejected(self):
        c = json.loads(self.config.read_text()); c['asset_upstream']='https://example.com'
        self.config.write_text(json.dumps(c))
        with self.assertRaises(ValueError): read_config(self.config)

    async def test_scene_alias_routes_status_frames_and_exports(self):
        cfg = json.loads(self.config.read_text())
        cfg['objects'][1]['upstream_scene'] = 'quality'
        self.config.write_text(json.dumps(cfg))
        response = await self.client.get('/api/status?scene=lamp')
        self.assertEqual((await response.json())['scene_id'], 'lamp')
        self.assertEqual(self.status_scene, 'quality')
        async with self.client.ws_connect('/ws?scene=lamp') as socket:
            await socket.send_json({'frame': 0})
            self.assertEqual((await socket.receive(timeout=3)).data, self.payload)
            self.assertEqual(self.ws_scene, 'quality')
        response = await self.client.post('/api/experiments?scene=lamp', json={})
        self.assertEqual((await response.json())['scene'], 'quality')

    async def test_export_posts_and_streamed_archive_use_selected_upstream(self):
        for endpoint in ('/api/experiments','/api/experiments/plan'):
            response=await self.client.post(endpoint+'?scene=lamp',json={'samples':[1,2,3]})
            self.assertEqual(response.status,202)
            self.assertEqual(await response.json(),{'scene':'lamp','received':{'samples':[1,2,3]}})
        response=await self.client.get('/api/experiments/job/download?scene=lamp')
        self.assertEqual(await response.read(),b'lossless-experiment-zip')
        self.assertIn('attachment',response.headers['Content-Disposition'])
        self.assertEqual((await self.client.post('/api/experiments?scene=lego',json={})).status,503)

    async def test_comparison_page_jobs_and_nested_assets_without_loaded_scene(self):
        for path in ('/comparison','/comparison/','/comparison.js','/comparison.css','/api/comparisons',
                     '/api/comparisons/job','/api/comparisons/job/files/report.html',
                     '/api/comparisons/job/files/figures/01_quality.png',
                     '/api/comparisons/job/files/results.zip'):
            response=await self.client.get(path)
            self.assertEqual(response.status,200,path)
            self.assertEqual(await response.read(),b'comparison-content')
            self.assertEqual(response.headers['Content-Security-Policy'],"frame-ancestors 'self'")
        response=await self.client.post('/api/comparisons',json={'path':'exports/run.zip'})
        self.assertEqual(response.status,202)
        self.assertEqual(await response.json(),{'received':{'path':'exports/run.zip'},'scene':None})
        self.assertEqual((await self.client.post('/api/comparisons',json={},headers={'Origin':'https://example.com'})).status,403)

    async def test_comparison_multipart_stream_exceeds_old_eight_mib_limit(self):
        form=FormData();size=9*1024*1024
        form.add_field('archive',b'x'*size,filename='experiment.zip',content_type='application/zip')
        response=await self.client.post('/api/comparisons',data=form)
        self.assertEqual(response.status,202,await response.text())
        self.assertEqual(await response.json(),{'field':'archive','filename':'experiment.zip','bytes':size})




    async def test_comparison_service_can_be_separate_from_gpu_renderer(self):
        dedicated=web.Application()
        async def page(request):return web.Response(text='dedicated comparison')
        for path in ('/comparison','/comparison.js','/comparison.css','/api/comparisons'):
            dedicated.router.add_get(path,page)
        server=TestServer(dedicated);await server.start_server()
        try:
            cfg=json.loads(self.config.read_text());cfg['comparison_upstream']=str(server.make_url('')).rstrip('/')
            self.config.write_text(json.dumps(cfg))
            for path in ('/comparison','/comparison.js','/comparison.css','/api/comparisons'):
                response=await self.client.get(path);self.assertEqual(await response.text(),'dedicated comparison')
            response=await self.client.get('/?scene=lamp');self.assertEqual(await response.read(),b'<html>original UI</html>')
        finally:await server.close()


if __name__ == '__main__': unittest.main()
