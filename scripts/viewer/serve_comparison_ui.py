"""Serve comparison jobs without restarting or loading GPU renderers."""
import argparse
from pathlib import Path
from aiohttp import web
from scripts.viewer.serve_multi_object_ui import local_only
from utils.viewer_comparisons import install_comparisons


def create_app(repo, export_root=None):
    repo=Path(repo).resolve()
    app=web.Application(middlewares=[local_only],client_max_size=8*1024**2)
    install_comparisons(app,repo,export_root or repo/'outputs/experiment-exports')
    async def asset(request):
        name=request.match_info['file']
        if name not in ('comparison.js','comparison.css'):raise web.HTTPNotFound()
        return web.FileResponse(repo/'viewer'/name)
    app.router.add_get('/{file}',asset)
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--port',type=int,default=8896)
    args=parser.parse_args()
    web.run_app(create_app(args.repo,args.export_root),host='127.0.0.1',port=args.port)
