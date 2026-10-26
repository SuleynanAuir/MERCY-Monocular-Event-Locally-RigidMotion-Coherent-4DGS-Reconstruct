"""Expose the original UI with one object menu across local GPU forwards.

The original HTML, JS, CSS and render packets are forwarded unchanged. Only the
status object's scene catalog and explicit revision labels are merged. Missing assets are explicitly reported;
there is no video replay or substitute reconstruction fallback.
"""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

ASSETS = {'style.css', 'app.js', 'playback.js', 'references.js'}
HTTP = web.AppKey('http', aiohttp.ClientSession)


def read_config(path):
    config = json.loads(path.read_text())
    seen = set()
    sources = [config['asset_upstream']]
    for scene in config['objects']:
        key = scene['id']
        if not key or key in seen or not all(c.isascii() and (c.isalnum() or c in '-_') for c in key):
            raise ValueError('Scene identifiers must be unique ASCII names')
        seen.add(key)
        sources.extend(scene.get('live', []))
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
        scene = selected(request)
        if not scene or not scene.get('live'):
            raise web.HTTPServiceUnavailable(text='该物件的原版三维资产尚未恢复')
        return scene['live'][0].rstrip('/')

    async def lifecycle(app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            app[HTTP] = session
            yield
    app.cleanup_ctx.append(lifecycle)

    async def status(request):
        scene = selected(request)
        if scene is None:
            raise web.HTTPNotFound()
        if not scene.get('live'):
            data = dict(ready=False, error=not scene.get('pending', False), progress=0, message=scene.get('note', '原版资产待恢复'),
                        detail='尚未接入实时三维资产')
        else:
            try:
                async with app[HTTP].get(upstream(request)+'/api/status', params={'scene':scene['id']}) as r:
                    r.raise_for_status()
                    data = await r.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                data = dict(ready=False, progress=0, message='正在连接渲染服务，请保持 SSH 隧道运行')
        data['scene_id'] = scene['id']
        # Identify regenerated checkpoints in the existing UI's labels, without
        # changing its renderer, reference frames or any binary render packets.
        if scene.get('revision_label') and isinstance(data.get('scene'), dict):
            info = data['scene']
            suffix = '（' + scene['revision_label'] + '）'
            for labels in (info, *(info.get(mode) for mode in ('clay', 'pseudocolor', 'mesh', 'hybrid'))):
                if isinstance(labels, dict) and labels.get('right_label'):
                    labels['right_label'] += suffix
        data['objects'] = [dict(id=s['id'], name=s['name'], ready=bool(s.get('live')))
                           for s in config()['objects']]
        return web.json_response(data)

    async def proxy(request):
        if request.path == '/':
            if 'scene' not in request.query:
                query = request.rel_url.with_query({**request.query, 'scene':config()['default']})
                raise web.HTTPFound(str(query))
            path = '/'
        elif request.path.removeprefix('/') in ASSETS:
            path = request.path
        elif request.path.startswith('/api/reference/') or request.path == '/api/experiments' or request.path.startswith('/api/experiments/'):
            path = request.path
        else:
            raise web.HTTPNotFound()
        base = (config()['asset_upstream'].rstrip('/') if request.path == '/' or request.path.removeprefix('/') in ASSETS
                else upstream(request))
        try:
            params = dict(request.query)
            if path.startswith('/api/'):
                params['scene'] = selected(request)['id']
            async with app[HTTP].request(request.method, base+path, params=params,
                    data=await request.read(), headers={'Content-Type': request.content_type},
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)) as remote:
                headers = {key: remote.headers[key] for key in ('Content-Type', 'Content-Disposition', 'Content-Length') if key in remote.headers}
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
            remote = await app[HTTP].ws_connect(base+'/ws', params={**request.query, 'scene':selected(request)['id']},
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

    app.router.add_get('/', proxy)
    app.router.add_get('/api/status', status)
    app.router.add_get('/api/reference/{frame:\\d+}', proxy)
    app.router.add_post('/api/experiments', proxy)
    app.router.add_post('/api/experiments/plan', proxy)
    app.router.add_get('/api/experiments/{job}', proxy)
    app.router.add_get('/api/experiments/{job}/download', proxy)
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
