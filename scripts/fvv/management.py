"""Bounded model imports and offline jobs, only in the isolated FVV process."""
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import uuid
import zipfile
from datetime import datetime, timezone

from aiohttp import web
from common import digest, usage, validate_pose, model_name, model_timestamp

UPLOAD_LIMIT = 512 * 1024**2
UNPACK_LIMIT = 1024**3
STORE_LIMIT = 8 * 1024**3
FILES = {'manifest.json', 'effective.yaml', 'model.pth'}
TRAINING_LOG_LIMIT = 64 * 1024


def training_error(output):
    """Keep the causal CUDA error instead of the supervisor's final traceback."""
    if 'out of memory' in output.lower() or 'OutOfMemoryError' in output:
        return ('GPU 显存不足，尚未开始训练；请释放其他渲染进程占用的显存后重试。原模型未变。'
                if 'GPU health check' in output else
                '训练期间 GPU 显存不足；本次结果未发布，原模型未变。请释放其他渲染进程占用后重试。')
    if 'GPU health check' in output:
        return 'GPU 基础运算自检未通过，尚未开始训练；原模型未变。请先修复 GPU/CUDA 运行环境。'
    if 'an illegal memory access was encountered' in output:
        return '训练 CUDA 内核发生非法内存访问；本次结果未发布，原模型未变。诊断日志已保留。'
    lines = [line.strip() for line in output.splitlines() if re.match(
        r'^(?:RuntimeError|ValueError|OSError|TimeoutError|torch\..*Error):', line.strip())]
    return (lines[0] if lines else output[-2000:]) or '训练未生成有效模型'


def unpack_model(archive, target, revision, scene):
    """Extract exactly three regular files with quotas and fixed paths."""
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) != 3 or {x.filename for x in entries} != FILES:
            raise ValueError('模型包根目录必须且只能包含 manifest.json、effective.yaml、model.pth')
        if sum(x.file_size for x in entries) > UNPACK_LIMIT:
            raise ValueError('模型解压后不能超过 1 GiB')
        for entry in entries:
            if entry.is_dir() or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('不接受目录或符号链接')
            limit = UNPACK_LIMIT if entry.filename == 'model.pth' else 128 * 1024
            if entry.file_size > limit:
                raise ValueError('模型元数据或检查点超过限额')
            count = 0
            with bundle.open(entry) as src, (target/entry.filename).open('xb') as dst:
                while block := src.read(1024**2):
                    count += len(block)
                    if count > limit:
                        raise ValueError('模型解压超限')
                    dst.write(block)
    meta = json.loads((target/'manifest.json').read_text())
    if meta.get('schema') != 1 or meta.get('revision') != revision or meta.get('scene') != scene:
        raise ValueError('模型版本或场景不匹配')
    if meta.get('checkpoint') != 'model.pth' or meta.get('config') != 'effective.yaml':
        raise ValueError('模型文件名称不匹配')
    validate_pose(dict(position=meta['position'],yaw=meta['yaw'],pitch=meta['pitch'],time=meta['time_duration'][0],
                       width=640,height=480,fov=meta.get('fov',60)),meta['time_duration'])
    for name, key in [('model.pth','sha256'), ('effective.yaml','config_sha256')]:
        if digest(target/name) != meta[key]:
            raise ValueError('模型校验失败：'+name)
    return meta


class ModelManager:
    def __init__(self, args, revision, current, load):
        self.args, self.revision, self.current, self.load = args, revision, current, load
        self.root = args.model.resolve().parent
        self.profiles = json.loads(args.profiles.read_text()) if args.profiles else {}
        self.job = None
        self.loading = False
        self.load_error = ''
        self.lock = asyncio.Lock()
        self.tasks = set()
        self.process = None

    def launch(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def models(self, scene):
        result = []
        for path in sorted(self.root.iterdir()):
            if path.is_symlink() or not path.is_dir() or path.name.startswith('.'):
                continue
            try:
                meta = json.loads((path/'manifest.json').read_text())
                if meta.get('scene') != scene or not (path/'model.pth').is_file():
                    continue
                result.append(dict(id=path.name,scene=scene,bytes=(path/'model.pth').stat().st_size,
                                   trained_at=meta.get('trained_at') or (None if meta.get('imported_at') else datetime.fromtimestamp((path/'model.pth').stat().st_mtime,timezone.utc).isoformat()),
                                   sha256=meta['sha256'],active=self.current() is not None and path.resolve() == Path(self.current()).resolve()))
            except (OSError, ValueError, KeyError):
                continue
        return result

    def progress(self):
        if not self.job:
            return None
        job = dict(self.job)
        if job['state'] == 'running':
            metrics = self.root/job['id']/'training-metrics.jsonl'
            try:
                with metrics.open('rb') as stream:
                    stream.seek(max(0,metrics.stat().st_size-4096))
                    lines = stream.read().decode().splitlines()
                row = json.loads(lines[-1])
                job['iteration'] = row['iteration']
                job['progress'] = min(99,100*row['iteration']/job['iterations'])
                job['stage'] = '训练与最终验证'
            except (OSError, ValueError, IndexError):
                pass  # Keep resource preparation/self-check progress visible.
        return job

    async def list_models(self, request):
        scene = request.query.get('scene','lego')
        profiles = [dict(id=k,label=v['label'],iterations=v['iterations']) for k,v in self.profiles.items() if v['scene']==scene]
        job = self.progress()
        return web.json_response(dict(models=self.models(scene),profiles=profiles,job=job if job and job['scene']==scene else None,
                                      gpu_training_active=bool(self.job and self.job['state']=='running'),
                                      loading=self.loading,load_error=self.load_error,upload_limit=UPLOAD_LIMIT))

    async def select(self, request):
        data = await request.json()
        scene = request.query.get('scene','lego')
        selected = next((x for x in self.models(scene) if x['id']==data.get('id')),None)
        if not selected:
            raise web.HTTPBadRequest(text='模型不存在或场景不匹配')
        if self.loading or self.current() is None:
            raise web.HTTPConflict(text='正在加载模型，请等待')
        self.loading, self.load_error = True, ''
        async def run():
            try:
                await self.load(self.root/selected['id'])
            except Exception as exc:
                self.load_error = str(exc)
            finally:
                self.loading = False
        self.launch(run())
        return web.json_response({'loading':True},status=202)

    async def train(self, request):
        data = await request.json()
        profile = self.profiles.get(data.get('profile'))
        if not profile or profile['scene'] != request.query.get('scene','lego'):
            raise web.HTTPBadRequest(text='该场景没有配置真实训练数据')
        async with self.lock:
            if self.job and self.job['state']=='running' or (self.root/'.fvv-writer.lock').exists():
                raise web.HTTPConflict(text='已有离线任务或模型导入正在进行')
            if usage(self.root)+4*1024**3 > STORE_LIMIT or shutil.disk_usage(self.root).free < 5*1024**3:
                raise web.HTTPInsufficientStorage(text='模型存储预算或磁盘空间不足')
            name=model_name(profile['scene'])
            self.job=dict(id=name,scene=profile['scene'],state='running',progress=0,iteration=0,
                          iterations=profile['iterations'],stage='GPU 自检与数据准备',error='')
            self.launch(self.run_training(profile,name))
        return web.json_response(self.job,status=202)

    async def run_training(self, profile, name):
        command=[sys.executable,str(Path(__file__).with_name('hybrid.py')),'--repo',str(self.args.repo),'train',
            '--config',profile['config'],'--source',profile['source'],'--output',str(self.root/name),
            '--scene',profile['scene'],'--budget-gib','4','--total-budget-gib','8','--max-minutes','45',
            '--position',*[str(v) for v in profile['position']], '--yaw',str(profile.get('yaw',0)),
            '--pitch',str(profile.get('pitch',0))]
        tail=b''
        try:
            env = dict(os.environ)
            # Only trusted server profiles can opt into pausing registered
            # viewers. Never accept process lists from the training request.
            env['MERCY_FVV_RENDER_REGISTRIES'] = json.dumps(profile.get('render_process_registries', []))
            self.process=await asyncio.create_subprocess_exec(*command,stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,start_new_session=True,env=env)
            while block := await self.process.stdout.read(4096):
                tail=(tail+block)[-TRAINING_LOG_LIMIT:]
                if b'pausing original viewer' in tail:
                    self.job['stage']='训练优先：原场景渲染暂时暂停'
                if b'Restoring original viewer' in tail:
                    self.job['stage']='正在恢复原场景渲染服务'
            code=await self.process.wait()
            folder=self.root/name
            if folder.is_dir():
                (folder/'training-console.log').write_bytes(tail)
            if code or not (self.root/name/'manifest.json').is_file():
                raise RuntimeError(training_error(tail.decode(errors='replace')))
            self.job.update(state='complete',progress=100,iteration=self.job['iterations'],stage='训练完成，可加载浏览')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.job.update(state='failed',stage='训练失败，原模型保留',error=str(exc))
        finally:
            self.process=None

    async def upload(self, request):
        scene=request.query.get('scene','lego')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',scene):
            raise web.HTTPBadRequest(text='场景名称无效')
        async with self.lock:
            if self.job and self.job['state']=='running':
                raise web.HTTPConflict(text='请等待当前训练完成后再导入')
            try:
                fd=os.open(self.root/'.fvv-writer.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:
                raise web.HTTPConflict(text='已有离线任务正在写入')
            name=model_name(scene)
            archive=self.root/('.'+name+'.zip');target=self.root/('.'+name)
            try:
                if usage(self.root)+UPLOAD_LIMIT+UNPACK_LIMIT > STORE_LIMIT or shutil.disk_usage(self.root).free < UPLOAD_LIMIT+UNPACK_LIMIT+1024**3:
                    raise web.HTTPInsufficientStorage(text='导入所需的存储预算不足')
                count=0
                with archive.open('xb') as stream:
                    async for block in request.content.iter_chunked(1024**2):
                        count+=len(block)
                        if count > UPLOAD_LIMIT:
                            raise web.HTTPRequestEntityTooLarge(max_size=UPLOAD_LIMIT,actual_size=count)
                        stream.write(block)
                target.mkdir()
                # Extraction completes before cleanup even if the client disconnects.
                meta=unpack_model(archive,target,self.revision,scene)
                for existing in self.models(scene):
                    existing_meta=json.loads((self.root/existing['id']/'manifest.json').read_text())
                    if existing['sha256']==meta['sha256'] and existing_meta['config_sha256']==meta['config_sha256']:
                        return web.json_response({'id':existing['id'],'deduplicated':True})
                meta['imported_at']=model_timestamp()
                (target/'manifest.json').write_text(json.dumps(meta,indent=2))
                target.rename(self.root/name)
                return web.json_response({'id':name,'deduplicated':False},status=201)
            except (ValueError,KeyError,zipfile.BadZipFile,RuntimeError) as exc:
                raise web.HTTPBadRequest(text=str(exc))
            finally:
                archive.unlink(missing_ok=True)
                shutil.rmtree(target,ignore_errors=True)
                os.close(fd)
                (self.root/'.fvv-writer.lock').unlink(missing_ok=True)

    async def delete(self, request):
        data=await request.json()
        scene=request.query.get('scene','lego')
        async with self.lock:
            if self.loading or (self.job and self.job['state']=='running'):
                raise web.HTTPConflict(text='请等待模型加载或训练完成')
            selected=next((x for x in self.models(scene) if x['id']==data.get('id')),None)
            if not selected:
                raise web.HTTPBadRequest(text='模型不存在或场景不匹配')
            if selected['active']:
                raise web.HTTPConflict(text='不能删除正在浏览的模型，请先加载另一个模型')
            if data.get('sha256')!=selected['sha256']:
                raise web.HTTPConflict(text='模型已变化，请刷新列表后重试')
            try:
                fd=os.open(self.root/'.fvv-writer.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:
                raise web.HTTPConflict(text='已有离线任务正在写入')
            try:
                target=self.root/selected['id']
                tombstone=self.root/('.deleting-'+uuid.uuid4().hex)
                target.rename(tombstone)
                await asyncio.to_thread(shutil.rmtree,tombstone)
                if self.job and self.job['id']==selected['id']:self.job=None
                return web.json_response({'deleted':selected['id'],'freed_model_bytes':selected['bytes']})
            finally:
                os.close(fd)
                (self.root/'.fvv-writer.lock').unlink(missing_ok=True)

    async def cleanup(self, app):
        process = self.process
        if process and process.returncode is None:
            process.send_signal(signal.SIGINT)  # supervisor terminates its worker group
            try:
                await asyncio.wait_for(process.wait(),15)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)

    def install(self, app):
        app.router.add_get('/models',self.list_models)
        app.router.add_post('/models/load',self.select)
        app.router.add_post('/models/import',self.upload)
        app.router.add_post('/models/delete',self.delete)
        app.router.add_post('/training',self.train)
        app.on_shutdown.append(self.cleanup)
