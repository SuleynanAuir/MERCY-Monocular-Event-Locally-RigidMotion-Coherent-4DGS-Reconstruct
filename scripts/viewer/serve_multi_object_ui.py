"""Expose the original UI with one object menu across local GPU forwards.

The original HTML, JS, CSS and render packets are forwarded unchanged. Only the
status object's scene catalog and explicit revision labels are merged. Missing assets are explicitly reported;
there is no video replay or substitute reconstruction fallback.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

ASSETS = {'style.css', 'app.js', 'playback.js', 'references.js',
          'comparison.css', 'comparison.js'}
HTTP = web.AppKey('http', aiohttp.ClientSession)


def read_config(path):
    config = json.loads(path.read_text())
    seen = set()
    sources = [config['asset_upstream'],config.get('comparison_upstream',config['asset_upstream'])]
    for scene in config['objects']:
        key = scene['id']
        if not key or key in seen or not all(c.isascii() and (c.isalnum() or c in '-_') for c in key):
            raise ValueError('Scene identifiers must be unique ASCII names')
        seen.add(key)
        sources.extend(scene.get('live', []))
        sources.extend(scene.get('observations', []))
    # localhost and 127.0.0.1 name the same local forward. A mode must
    # never silently share a worker with the other mode, even across objects.
    live_ports = {urlsplit(u).port or 80 for s in config['objects'] for u in s.get('live', [])}
    observation_ports = {urlsplit(u).port or 80 for s in config['objects'] for u in s.get('observations', [])}
    if live_ports & observation_ports:
        raise ValueError('Live and observation renderers require separate processes and ports')
    if config['default'] not in seen:
        raise ValueError('Default scene is missing')
    for source in sources:
        u = urlsplit(source)
        if (u.scheme != 'http' or u.hostname not in {'127.0.0.1', 'localhost'}
                or u.path not in {'', '/'} or u.username or u.password or u.query or u.fragment):
            raise ValueError('Upstreams must be local SSH forwards')
    return config


@web.middleware
async def local_only(request, handler):
    if request.url.host not in {'localhost', '127.0.0.1', '::1'}:
        raise web.HTTPForbidden()
    if request.headers.get('Origin') not in (None, f'{request.scheme}://{request.host}'):
        raise web.HTTPForbidden()
    response = await handler(request)
    if not response.prepared:
        response.headers.update({'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
            'Cache-Control': 'no-store', 'Content-Security-Policy':
            "default-src 'self'; connect-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "style-src 'self'; script-src 'self'; frame-src 'self'; frame-ancestors 'self'; base-uri 'none'"})
    return response


def create_app(config_path):
    app = web.Application(middlewares=[local_only], client_max_size=8*1024*1024)

    def config():
        return read_config(config_path)

    def selected(request):
        key = request.query.get('scene', config()['default'])
        return next((s for s in config()['objects'] if s['id'] == key), None)

    def upstream(request):
        if config().get('require_pipeline') and 'pipeline' not in request.query:
            raise web.HTTPConflict(text='界面已更新，请刷新页面以选择独立渲染模式')
        scene = selected(request)
        channel = request.query.get('pipeline', 'live')
        if channel not in {'live', 'observations'}:
            raise web.HTTPBadRequest(text='Unknown rendering pipeline')
        if not scene or not scene.get(channel):
            raise web.HTTPServiceUnavailable(text='该模式的独立渲染服务尚未配置')
        return scene[channel][0].rstrip('/')

    async def lifecycle(app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), auto_decompress=False) as session:
            app[HTTP] = session
            yield
    app.cleanup_ctx.append(lifecycle)

    async def status(request):
        if config().get('require_pipeline') and 'pipeline' not in request.query:
            return web.json_response(dict(ready=False, error=True, progress=0,
                message='渲染模式已拆分，请刷新页面', detail='需要加载新版界面'))
        scene = selected(request)
        if scene is None:
            raise web.HTTPNotFound()
        channel = request.query.get('pipeline', 'live')
        if channel not in {'live', 'observations'}:
            raise web.HTTPBadRequest(text='Unknown rendering pipeline')
        if not scene.get(channel):
            data = dict(ready=False, error=not scene.get('pending', False), progress=0, message=scene.get('note', '原版资产待恢复'),
                        detail='尚未接入实时三维资产')
            if channel == 'observations':
                data.update(error=True, message='六面观测独立服务尚未配置；自由视角可继续使用')
        else:
            try:
                async with app[HTTP].get(upstream(request)+'/api/status', params={'scene':scene.get('upstream_scene', scene['id']), 'pipeline':channel}) as r:
                    r.raise_for_status()
                    data = await r.json()
                    if channel == 'observations' and data.get('render_pipeline') != channel:
                        data = dict(ready=False, error=True, progress=0, message='六面观测服务未隔离，已拒绝使用自由视角进程')
            except (aiohttp.ClientError, asyncio.TimeoutError):
                data = dict(ready=False, progress=0, message='正在连接渲染服务，请保持 SSH 隧道运行')
                fvv = os.environ.get('MERCY_FVV_URL')
                if fvv:
                    try:
                        async with app[HTTP].get(fvv.rstrip('/')+'/models', timeout=aiohttp.ClientTimeout(total=2)) as job:
                            if (await job.json()).get('gpu_training_active'):
                                data['message']='独立训练优先：原场景渲染暂时暂停，训练结束后自动恢复'
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                        pass
        data['scene_id'] = scene['id']
        data['render_pipeline'] = channel
        # Identify regenerated checkpoints in the existing UI's labels, without
        # changing its renderer, reference frames or any binary render packets.
        if scene.get('revision_label') and isinstance(data.get('scene'), dict):
            info = data['scene']
            suffix = '（' + scene['revision_label'] + '）'
            for labels in (info, *(info.get(mode) for mode in ('clay', 'pseudocolor', 'mesh', 'hybrid'))):
                if isinstance(labels, dict) and labels.get('right_label'):
                    labels['right_label'] += suffix
        data['objects'] = [dict(id=s['id'], name=s['name'], ready=bool(s.get(channel)))
                           for s in config()['objects']]
        return web.json_response(data)

    async def proxy(request):
        comparison = (request.path in {'/comparison', '/comparison/'}
                      or request.path == '/api/comparisons'
                      or request.path.startswith('/api/comparisons/'))
        if request.path == '/':
            if 'scene' not in request.query:
                query = request.rel_url.with_query({**request.query, 'scene':config()['default']})
                raise web.HTTPFound(str(query))
            path = '/'
        elif request.path.removeprefix('/') in ASSETS:
            path = request.path
        elif comparison:
            path = '/comparison' if request.path == '/comparison/' else request.path
        elif request.path.startswith('/api/reference/') or request.path == '/api/experiments' or request.path.startswith('/api/experiments/'):
            path = request.path
        else:
            raise web.HTTPNotFound()
        if config().get('local_assets') and (path == '/' or path.lstrip('/') in ASSETS - {'comparison.js', 'comparison.css'}):
            # Keep navigation available while GPU processes release memory.
            asset = Path(__file__).resolve().parents[2]/'viewer'/('index.html' if path == '/' else path.lstrip('/'))
            if path == '/':
                html = asset.read_text()
                if config().get('fvv_entry') and '/fvv/fvv-entry.js' not in html:
                    html = html.replace('</body>', '<script src="/fvv/fvv-entry.js" defer></script></body>', 1)
                return web.Response(text=html, content_type='text/html')
            return web.FileResponse(asset)
        # Comparisons operate on exported data, independent of the selected
        # renderer. Keep page, jobs and result assets on one stable service.
        comparison_asset=request.path in ('/comparison.js','/comparison.css')
        base = (config().get('comparison_upstream',config()['asset_upstream']).rstrip('/') if comparison or comparison_asset
                else config()['asset_upstream'].rstrip('/') if request.path == '/' or request.path.removeprefix('/') in ASSETS
                else upstream(request))
        try:
            params = dict(request.query)
            if path.startswith('/api/') and not comparison:
                scene = selected(request)
                params['scene'] = scene.get('upstream_scene', scene['id'])
            streaming_upload = path == '/api/comparisons' and request.method == 'POST'
            if streaming_upload and request.content_length and request.content_length > 2*1024**3 + 16*1024**2:
                raise web.HTTPRequestEntityTooLarge(max_size=2*1024**3, actual_size=request.content_length)
            body = request.content.iter_chunked(1024*1024) if streaming_upload else await request.read()
            async with app[HTTP].request(request.method, base+path, params=params,
                    data=body, headers={'Content-Type': request.headers.get('Content-Type', 'application/octet-stream')},
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)) as remote:
                headers = {key: remote.headers[key] for key in (
                    'Content-Type', 'Content-Disposition', 'Content-Length', 'Content-Encoding',
                    'Content-Security-Policy', 'Cache-Control', 'X-Content-Type-Options',
                    'Referrer-Policy') if key in remote.headers}
                if path == '/' and remote.status == 200 and config().get('fvv_entry', False):
                    content = await remote.read()
                    if remote.headers.get('Content-Encoding') == 'gzip':
                        import gzip
                        content = gzip.decompress(content)
                    html = content.decode('utf-8')
                    if 'id="realtimeGaussian"' not in html:
                        scene_id = selected(request)['id']
                        entry = ('<section class="panel"><h2>自由视角</h2>'
                                 f'<a id="realtimeGaussian" class="secondary-button" href="/fvv?scene={scene_id}">'
                                 '实时动态浏览高斯 ↗</a><p class="setting-note">独立训练、模型导入与自由视角浏览</p></section>')
                        marker = '</aside>' if '</aside>' in html else '</body>'
                        html = html.replace(marker, entry+marker, 1)
                    if '/fvv/fvv-entry.js' not in html:
                        html = html.replace('</body>', '<script src="/fvv/fvv-entry.js" defer></script></body>', 1)
                    # Only the optional entry HTML changes; original assets and
                    # binary render/experiment packets retain the existing path.
                    return web.Response(text=html,content_type='text/html')
                response = web.StreamResponse(status=remote.status, headers=headers)
                await response.prepare(request)
                async for chunk in remote.content.iter_chunked(1024*1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise web.HTTPBadGateway(text='渲染连接暂时不可用，请检查 SSH 隧道')

    async def websocket(request):
        base = upstream(request)
        try:
            scene = selected(request)
            if request.query.get('pipeline') == 'observations':
                async with app[HTTP].get(base+'/api/status', params={
                        'scene':scene.get('upstream_scene', scene['id']), 'pipeline':'observations'}) as check:
                    check.raise_for_status()
                    if (await check.json()).get('render_pipeline') != 'observations':
                        raise web.HTTPServiceUnavailable(text='六面观测必须使用独立模式进程')
            remote = await app[HTTP].ws_connect(base+'/ws', params={**request.query, 'scene':scene.get('upstream_scene', scene['id'])},
                                                  max_msg_size=32*1024**2, heartbeat=20)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise web.HTTPServiceUnavailable(text='渲染器正在准备或连接暂时不可用')
        client = web.WebSocketResponse(max_msg_size=4096, heartbeat=20, compress=False)
        await client.prepare(request)

        async def pipe(source, target):
            async for message in source:
                if message.type == aiohttp.WSMsgType.BINARY:
                    await target.send_bytes(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    await target.send_str(message.data)
                else:
                    break
        jobs = [asyncio.create_task(pipe(client, remote)), asyncio.create_task(pipe(remote, client))]
        try:
            await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            await remote.close()
            await client.close()
        return client

    from utils.fvv_gateway import install_fvv
    install_fvv(app)
    app.router.add_get('/', proxy)
    app.router.add_get('/api/status', status)
    app.router.add_get('/api/reference/{frame:\\d+}', proxy)
    app.router.add_post('/api/experiments', proxy)
    app.router.add_post('/api/experiments/plan', proxy)
    app.router.add_get('/api/experiments/{job}', proxy)
    app.router.add_get('/api/experiments/{job}/download', proxy)
    app.router.add_get('/comparison', proxy)
    app.router.add_get('/comparison/', proxy)
    app.router.add_get('/api/comparisons', proxy)
    app.router.add_post('/api/comparisons', proxy)
    app.router.add_get('/api/comparisons/{job}', proxy)
    app.router.add_get('/api/comparisons/{job}/files/{path:.*}', proxy)
    app.router.add_get('/ws', websocket)
    app.router.add_get('/{file}', proxy)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18896)
    args = parser.parse_args()
    read_config(args.catalog)
    web.run_app(create_app(args.catalog), host='127.0.0.1', port=args.port)
