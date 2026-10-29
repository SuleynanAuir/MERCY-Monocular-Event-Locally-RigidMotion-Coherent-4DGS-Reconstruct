"""Run the isolated FVV page without restarting an existing viewer.

Usage: MERCY_FVV_URL=http://127.0.0.1:21897 python -m scripts.fvv.preview_ui
"""
import argparse

from aiohttp import web
from utils.fvv_gateway import install_fvv


def create_app():
    app = web.Application()
    install_fvv(app)

    async def index(request):
        return web.Response(text='''<!doctype html><html lang="zh-CN">
<meta charset="utf-8"><title>Hybrid FVV</title>
<h1>真实 Lego · Hybrid 3D-4DGS</h1>
<p>原始 RGB 帧离线训练；单固定视角监督，新视角质量尚未独立验收。</p>
<a href="/fvv?scene=lego">实时动态浏览高斯</a>
</html>''', content_type='text/html')

    app.router.add_get('/', index)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=21896)
    args = parser.parse_args()
    web.run_app(create_app(), host='127.0.0.1', port=args.port)
