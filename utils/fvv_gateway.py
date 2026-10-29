"""Optional FVV routes. Never import Hybrid/CUDA into the original renderer."""
import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web


def install_fvv(app, upstream=None):
    base = upstream if upstream is not None else os.environ.get('MERCY_FVV_URL', '')
    if base:
        u = urlsplit(base)
        if u.scheme != 'http' or u.hostname not in {'localhost', '127.0.0.1', '::1'} or u.path not in ('', '/') or u.query or u.fragment or u.username or u.password:
            raise ValueError('MERCY_FVV_URL must be a loopback HTTP service')
        base = base.rstrip('/')
    static = Path(__file__).resolve().parents[1] / 'viewer'

    async def asset(request):
        name = request.match_info.get('asset', 'fvv.html')
        if name not in {'fvv.html', 'fvv.js', 'fvv.css', 'fvv-manager.js', 'fvv-entry.js'}:
            raise web.HTTPNotFound()
        return web.FileResponse(static / name, headers={'Cache-Control': 'no-store'})

    async def status(request):
        if not base:
            return web.json_response({'ready': False, 'message': '尚未配置 Hybrid 4DGS 服务。请按集成说明离线训练或加载模型，然后设置 MERCY_FVV_URL。'})
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base + '/status', params=request.query, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return web.json_response(await r.json(), status=r.status)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return web.json_response({'ready': False, 'message': 'Hybrid 服务未连接；请检查模型加载状态和 SSH 隧道。'})

    async def management(request):
        if not base:
            raise web.HTTPServiceUnavailable(text='尚未配置独立 FVV 服务')
        if request.headers.get('Origin') not in (None, f'{request.scheme}://{request.host}'):
            raise web.HTTPForbidden()
        endpoint = request.match_info['endpoint']
        allowed = {('GET','models'),('POST','models/load'),('POST','models/import'),('POST','training')}
        if (request.method,endpoint) not in allowed:
            raise web.HTTPNotFound()
        limit = 512*1024**2 if endpoint == 'models/import' else 4096
        if request.content_length and request.content_length > limit:
            raise web.HTTPRequestEntityTooLarge(max_size=limit,actual_size=request.content_length)
        async def body():
            count = 0
            async for chunk in request.content.iter_chunked(1024**2):
                count += len(chunk)
                if count > limit:
                    raise web.HTTPRequestEntityTooLarge(max_size=limit,actual_size=count)
                yield chunk
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
                async with session.request(request.method,base+'/'+endpoint,params=request.query,
                    data=body() if request.method=='POST' else None,
                    headers={'Content-Type':request.headers.get('Content-Type','application/json')}) as remote:
                    return web.Response(body=await remote.read(),status=remote.status,
                        headers={'Content-Type':remote.headers.get('Content-Type','text/plain'),'Cache-Control':'no-store'})
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise web.HTTPServiceUnavailable(text='训练／模型服务未连接，请检查 SSH 隧道')

    async def socket(request):
        if not base:
            raise web.HTTPServiceUnavailable()
        origin = request.headers.get('Origin')
        if origin not in (None, f'{request.scheme}://{request.host}'):
            raise web.HTTPForbidden()
        async with aiohttp.ClientSession() as session:
            try:
                remote = await session.ws_connect(base + '/ws', params=request.query, max_msg_size=8*1024**2, heartbeat=20)
            except aiohttp.ClientError:
                raise web.HTTPServiceUnavailable(text='Hybrid 服务不可用或正在使用中')
            client = web.WebSocketResponse(max_msg_size=4096, heartbeat=20, compress=False)
            await client.prepare(request)
            async def pipe(source, target):
                async for msg in source:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await target.send_str(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await target.send_bytes(msg.data)
                    else:
                        break
            jobs = [asyncio.create_task(pipe(client, remote)), asyncio.create_task(pipe(remote, client))]
            try:
                await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in jobs:
                    task.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
                await remote.close()
                await client.close()
        return client

    app.router.add_get('/fvv', asset)
    app.router.add_get('/fvv/status', status)
    app.router.add_get('/fvv/ws', socket)
    app.router.add_route('*','/fvv/api/{endpoint:.*}', management)
    app.router.add_get('/fvv/{asset}', asset)
