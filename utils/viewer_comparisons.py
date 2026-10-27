"""Comparison jobs run in isolated CPU processes, independently of GPU loading."""
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid
from urllib.parse import quote
from aiohttp import web
from utils.comparison_io import MAX_BYTES,MAX_FILES,relative_name,locate_manifest
from utils.comparison_outputs import allocate, identity, object_name, archive_name


def install_comparisons(app, repo, export_root, output_root=None):
    repo=Path(repo).resolve();export_root=Path(export_root).resolve()
    outputs=Path(output_root or repo/'experiment/output').resolve();processes={};tasks=set();busy=False;locations={}
    def folder(job):
        if not re.fullmatch(r'[a-f0-9]{32}',job):raise web.HTTPNotFound()
        if job in locations:return locations[job]
        for path in (outputs/job,outputs/'comparisons'/job):
            if path.is_dir() and path.resolve().is_relative_to(outputs):locations[job]=path;return path
        for metadata in outputs.glob('*/job.json'):
            if metadata.parent.resolve().is_relative_to(outputs) and identity(metadata.parent).get('id')==job:
                locations[job]=metadata.parent;return metadata.parent
        raise web.HTTPNotFound()
    def state(job):
        p=folder(job)/'progress.json'
        if not p.is_file():raise web.HTTPNotFound()
        data=json.loads(p.read_text());base=f'/api/comparisons/{job}'
        data.update(identity(folder(job)))
        return dict(data,id=job,output_directory=str(folder(job)),report_url=base+'/files/report.html',download_url=base+'/files/'+quote(archive_name(folder(job))))
    def save_state(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False))
    async def execute(job,source,temporary=None):
        nonlocal busy
        target=folder(job);log=(target/'worker.log').open('wb')
        try:
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg',MPLCONFIGDIR=str(Path(tempfile.gettempdir())/'scgs-comparison-mpl'))
            process=await asyncio.create_subprocess_exec(sys.executable,'-m','experiment.comparison','--input',str(source),'--output',str(target),'--progress',str(target/'progress.json'),cwd=repo,env=env,stdout=log,stderr=log)
            processes[job]=process;code=await process.wait()
            current=state(job)
            if code and current.get('status')=='running':save_state(target/'progress.json',dict(status='failed',stage=current.get('stage','queued'),percent=current.get('percent',0),detail='计算进程退出；请检查该结果目录的 worker.log'))
        except asyncio.CancelledError:
            process=processes.get(job)
            if process and process.returncode is None:process.terminate();await process.wait()
            save_state(target/'progress.json',dict(status='failed',stage='interrupted',percent=0,detail='服务器停止，任务已中断；原始实验数据未改变'))
            raise
        except Exception as error:save_state(target/'progress.json',dict(status='failed',stage='failed',percent=0,detail=str(error)))
        finally:
            log.close();processes.pop(job,None);busy=False
            if temporary:shutil.rmtree(temporary,ignore_errors=True)
    async def start(request):
        nonlocal busy
        if busy:raise web.HTTPConflict(text='本服务已有对照试验运行中，请等待完成')
        busy=True;temporary=None
        try:
            if request.content_type.startswith('multipart/'):
                temporary=Path(tempfile.mkdtemp(prefix='scgs-comparison-upload-'));reader=await request.multipart();total=0;count=0;archive=None;name=None
                while part:=await reader.next():
                    if not part.filename:
                        if part.name!='object_name' or name is not None:raise ValueError('无效或重复的物品名称字段')
                        raw=await part.read_chunk(1024)
                        if len(raw)>512 or not part.at_eof():raise ValueError('物品名称过长')
                        name=object_name(raw.decode('utf-8'));continue
                    if archive is not None:raise ValueError('ZIP 必须单独上传，不能与文件夹文件混合')
                    count+=1
                    if count>MAX_FILES:raise ValueError('文件数量超过上限')
                    path=temporary/relative_name(part.filename)
                    if part.name=='archive':
                        if count!=1 or path.suffix.lower()!='.zip':raise ValueError('请选择一个实验 ZIP 文件')
                        archive=path
                    if path.exists():raise ValueError('重复文件名')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as stream:
                        while chunk:=await part.read_chunk(1024**2):
                            total+=len(chunk)
                            if total>MAX_BYTES:raise ValueError('上传数据超过 2 GiB')
                            stream.write(chunk)
                source=archive if archive is not None else locate_manifest(temporary)
            else:
                if request.content_length and request.content_length>8192:raise ValueError('输入路径过长')
                body=await request.json()
                if not isinstance(body,dict):raise ValueError('请求应为 JSON 对象')
                name=object_name(body.get('object_name'));value=body.get('path','')
                if not isinstance(value,str) or not value.strip():raise ValueError('请输入实验文件夹或 ZIP 的服务器路径')
                source=Path(value.strip()).expanduser();source=(repo/source).resolve() if not source.is_absolute() else source.resolve()
                if not any(source.is_relative_to(p) for p in (repo,export_root)):raise ValueError('路径须位于本项目或配置的导出目录内')
                if source.is_relative_to(outputs):raise ValueError('请输入原始实验数据，不能选择已有分析结果')
                if source.is_file():
                    if source.suffix.lower()!='.zip' or source.stat().st_size>MAX_BYTES:raise ValueError('仅支持不超过 2 GiB 的实验 ZIP')
                elif source.is_dir():source=locate_manifest(source)
                else:raise ValueError('服务器上找不到此路径；电脑上的文件夹请使用本地上传')
            name=object_name(name)
            job=uuid.uuid4().hex;target,metadata=allocate(outputs,name,job);locations[job]=target
            save_state(target/'progress.json',dict(status='running',stage='queued',percent=0,detail='等待独立计算进程',source_name=source.name))
            task=asyncio.create_task(execute(job,source,temporary));tasks.add(task);task.add_done_callback(tasks.discard)
            return web.json_response(state(job),status=202)
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
            busy=False
            if temporary:shutil.rmtree(temporary,ignore_errors=True)
            raise web.HTTPBadRequest(text=str(error))
        except BaseException:
            busy=False
            if temporary:shutil.rmtree(temporary,ignore_errors=True)
            raise
    async def status(request):return web.json_response(state(request.match_info['job']),headers={'Cache-Control':'no-store'})
    async def files(request):
        job=request.match_info['job'];current=state(job)
        if current['status']!='complete':raise web.HTTPConflict(text='报告尚未完成')
        root=folder(job);name=request.match_info['path'];record=json.loads((root/'artifact_manifest.json').read_text())
        if name=='results.zip':name=archive_name(root)
        if name not in record['files'] and name not in ('artifact_manifest.json',archive_name(root)):raise web.HTTPNotFound()
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise web.HTTPNotFound()
        headers={'Cache-Control':'private, max-age=0, must-revalidate','X-Content-Type-Options':'nosniff'}
        if name=='report.html':headers['Content-Security-Policy']="default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'self'; base-uri 'none'"
        if path.suffix in ('.zip','.csv','.npz','.gz','.py'):
            fallback='results.zip' if path.suffix=='.zip' else 'download'+path.suffix
            headers['Content-Disposition']=f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quote(path.name)}'
        return web.FileResponse(path,headers=headers)
    async def options(request):
        candidates=[]
        for parent in (repo/'experiment/exports',export_root):
            if parent.exists():
                for p in sorted(parent.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)[:60]:
                    if (p.is_dir() and (p/'manifest.json').is_file()) or (p.is_file() and p.suffix=='.zip'):
                        candidates.append(dict(path=str(p),name=p.name))
        history=[]
        if outputs.exists():
            records=[*outputs.glob('*/progress.json'),*(outputs/'comparisons').glob('*/progress.json')]
            for p in sorted(records,key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
                job=identity(p.parent).get('id',p.parent.name)
                if re.fullmatch(r'[a-f0-9]{32}',job) and p.parent.resolve().is_relative_to(outputs):
                    locations[job]=p.parent;item=state(job)
                    if item['status']=='running' and item.get('worker_pid'):
                        try:os.kill(item['worker_pid'],0)
                        except ProcessLookupError:
                            item.update(status='failed',detail='计算进程已退出，旧任务未完成；可重新运行')
                            save_state(p,item)
                    history.append(item)
        return web.json_response(dict(candidates=candidates,history=history,output_root=str(outputs),timezone='Asia/Shanghai',limits=dict(max_bytes=MAX_BYTES,max_files=MAX_FILES)))
    async def page(request):return web.FileResponse(repo/'viewer/comparison.html')
    async def shutdown(app):
        for task in tasks:task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
    app.on_shutdown.append(shutdown)
    app.router.add_get('/comparison',page)
    app.router.add_get('/api/comparisons',options)
    app.router.add_post('/api/comparisons',start)
    app.router.add_get('/api/comparisons/{job}',status)
    app.router.add_get('/api/comparisons/{job}/files/{path:.*}',files)
