"""Bounded offline orbit export, sharing the isolated FVV GPU worker."""
import asyncio
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid
from datetime import datetime, timezone

from aiohttp import web


def trajectory(data, meta):
    if not isinstance(data,dict):raise ValueError('参数必须是 JSON 对象')
    path=data.get('path','overview' if meta.get('scene')=='lego' else 'full')
    if path not in ('overview','showcase','full'):raise ValueError('未知的镜头路线')
    if path in ('overview','showcase'):
        if meta.get('scene')!='lego':raise ValueError('精选路线仅适用于 Lego')
        # Disabled custom-camera controls and an arbitrary browser pose must
        # not invalidate a route that uses the calibrated camera instead.
        data={k:v for k,v in data.items() if k not in ('position','yaw','pitch','fov','radius','elevation')}
    def number(key, default, lo, hi):
        value=data.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('参数超出范围：'+key)
        return float(value)
    seconds=number('seconds',12,4,24);fps=number('fps',30,24,60)
    width=number('width',960,640,1280)
    if fps not in (24,30,60) or width not in (640,960,1280):raise ValueError('不支持的帧率或分辨率')
    height=int(width*9/16);radius=number('radius',1.5,.2,20)
    elevation=number('elevation',15,0,35)
    yaw=number('yaw',meta['yaw'],-1e6,1e6);pitch=number('pitch',meta['pitch'],-75,75)
    fov=number('fov',meta.get('fov',60),20,100)
    start=number('time',meta['time_duration'][0],*meta['time_duration'])
    samples=number('temporal_samples',3,1,3)
    if samples not in (1,3):raise ValueError('采样次数必须为 1 或 3')
    position=data.get('position',meta['position'])
    if not isinstance(position,list) or len(position)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1e6 for x in position):raise ValueError('无效的相机位置')
    # This route is calibrated to the Lego training camera, not whichever
    # arbitrary rear view happens to be open when export is clicked.
    if path in ('overview','showcase'):
        yaw=float(meta['yaw']);pitch=float(meta['pitch']);position=list(meta['position'])
        fov=float(meta['fov']);radius=1.5;elevation=1. if path=='overview' else .5
    def direction(y,p):
        y,p=math.radians(y),math.radians(p)
        return [math.sin(y)*math.cos(p),math.sin(p),math.cos(y)*math.cos(p)]
    target=[p+radius*d for p,d in zip(position,direction(yaw,pitch))]
    count=int(round(seconds*fps));lo,hi=meta['time_duration'];poses=[]
    if hi<=lo:raise ValueError('模型时间范围无效')
    for i in range(count):
        phase=2*math.pi*i/count
        if path=='overview':
            # Full azimuth coverage, small height variation, no push/pull.
            u=i/count; eased=u*u*u*(10+u*(-15+6*u))
            y=yaw+360*eased;p=pitch+2+elevation*math.sin(2*math.pi*eased)
            distance=1.8
        elif path=='showcase':
            # Ease to a shallow left oblique and back. Do not pass through
            # poorly constrained rear/bucket views or below the baseplate.
            blend=(1-math.cos(phase))*.5
            y=yaw-3*blend;p=pitch+elevation*blend
            distance=radius+.15
        else:
            y=yaw+360*i/count;p=max(-75,min(75,pitch+elevation*math.sin(2*phase)))
            distance=radius
        pos=[v-distance*d for v,d in zip(target,direction(y,p))]
        poses.append(dict(position=pos,yaw=y,pitch=p,time=lo+(start-lo+i/fps)%(hi-lo),width=int(width),height=height,fov=fov,temporal_samples=int(samples)))
    distances=[math.sqrt(sum((a-b)**2 for a,b in zip(pose['position'],target))) for pose in poses]
    return dict(seconds=count/fps,fps=int(fps),width=int(width),height=height,frames=count,radius=radius,elevation_degrees=elevation,target=target,
                camera_distance_range=[min(distances),max(distances)],yaw_range=[min(p['yaw'] for p in poses),max(p['yaw'] for p in poses)],
                path=path,motion=('Calibrated 360 degree overview, eased motion, pitch 1..3 degrees, fixed distance 1.8' if path=='overview' else 'Lego training-camera anchored 3 degree eased arc, 0.5 degree rise, fixed distance; not a full orbit' if path=='showcase' else '360 degree orbit, two sinusoidal elevation cycles, always looking at target'),coordinate_system='FVV x right, y down, z forward; yaw/pitch degrees',
                animation='model time advances 1 second per video second, loops at model boundary',poses=poses)


class OrbitExporter:
    FILE_LIMIT=256*1024**2
    STORE_LIMIT=1024**3
    def __init__(self, model_root, current, pool, blocked):
        self.root=Path(model_root).parent/'orbit-exports'
        self.model_root=Path(model_root)
        self.current=current;self.pool=pool;self.blocked=blocked
        self.job=None;self.task=None;self.process=None

    @property
    def running(self):return self.task is not None and not self.task.done()

    async def start(self,request):
        if self.running or self.blocked():raise web.HTTPConflict(text='请等待当前离线任务或模型加载完成')
        renderer=self.current()
        if renderer is None or request.query.get('scene','lego')!=renderer.meta['scene']:raise web.HTTPConflict(text='场景尚未就绪')
        try:plan=trajectory(await request.json(),renderer.meta)
        except (ValueError,TypeError,KeyError) as exc:raise web.HTTPBadRequest(text=str(exc))
        if self.running or self.blocked():raise web.HTTPConflict(text='请等待当前离线任务或模型加载完成')
        if not shutil.which('ffmpeg'):raise web.HTTPServiceUnavailable(text='服务器未安装 ffmpeg')
        self.root.mkdir(parents=True,exist_ok=True)
        used=sum(p.stat().st_size for p in self.root.glob('*/*') if p.is_file())
        if used+self.FILE_LIMIT+4*1024**2>self.STORE_LIMIT or shutil.disk_usage(self.root).free<2*self.FILE_LIMIT:
            raise web.HTTPInsufficientStorage(text='视频导出空间不足，请下载后删除旧导出文件')
        try:fd=os.open(self.model_root/'.fvv-writer.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError:raise web.HTTPConflict(text='已有训练或导入任务')
        ident=uuid.uuid4().hex
        self.job=dict(id=ident,state='running',progress=0,frame=0,frames=plan['frames'],scene=renderer.meta['scene'],error='')
        self.task=asyncio.create_task(self.run(renderer,plan,fd))
        return web.json_response(self.job,status=202)

    async def run(self,renderer,plan,fd):
        folder=self.root/self.job['id'];started=time.monotonic()
        try:
            folder.mkdir()
            summary=dict(schema=1,created_at=datetime.now(timezone.utc).isoformat(),scene=renderer.meta['scene'],model_sha256=renderer.meta['sha256'],
                         model_directory=renderer.root.name,model_trained_at=renderer.meta.get('trained_at'),config_sha256=renderer.meta.get('config_sha256'),training_iteration=renderer.meta.get('training_iteration'),upstream_revision=renderer.meta['revision'],
                         temporal_sigma_seconds=renderer.temporal_sigma,video=dict(codec='H.264',pixel_format='yuv420p',crf=18,preset='fast'),
                         gpu=renderer.torch.cuda.get_device_name(),torch=renderer.torch.__version__,trajectory={k:v for k,v in plan.items() if k!='poses'})
            if renderer.meta.get('rigid_bucket'):summary['rigid_bucket']=renderer.meta['rigid_bucket']
            temp=folder/'video.partial.mp4'
            command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f"{plan['width']}x{plan['height']}",
                     '-r',str(plan['fps']),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                     '-metadata','comment='+json.dumps(summary,ensure_ascii=False),'-movflags','+faststart',str(temp)]
            with (folder/'encoder.log').open('wb') as log:
                self.process=await asyncio.create_subprocess_exec(*command,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.DEVNULL,stderr=log)
                async def encode_frames():
                    for i,pose in enumerate(plan['poses']):
                        pixels=await asyncio.get_running_loop().run_in_executor(self.pool,lambda p=pose:renderer.frame(p,pixels=True))
                        self.process.stdin.write(pixels.tobytes());await self.process.stdin.drain()
                        if temp.exists() and temp.stat().st_size>self.FILE_LIMIT:raise OSError('单次视频超过 256 MiB 上限')
                        self.job.update(frame=i+1,progress=99*(i+1)/plan['frames'])
                        await asyncio.sleep(0)
                    self.process.stdin.close()
                    if await self.process.wait():raise RuntimeError('视频编码失败，请检查 ffmpeg 日志')
                await asyncio.wait_for(encode_frames(),1200)
            if temp.stat().st_size>self.FILE_LIMIT:raise OSError('单次视频超过 256 MiB 上限')
            summary.update(elapsed_seconds=time.monotonic()-started,video_bytes=temp.stat().st_size,frames=[dict(index=i,video_time=i/plan['fps'],**p) for i,p in enumerate(plan['poses'])])
            (folder/'parameters.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
            temp.rename(folder/'video.mp4')
            self.job.update(state='complete',progress=100)
        except (Exception,asyncio.CancelledError) as exc:
            self.job.update(state='failed',error=str(exc) or '任务已取消')
            if self.process and self.process.returncode is None:
                self.process.kill();await self.process.wait()
            for name in ('video.partial.mp4','video.mp4','parameters.json'):(folder/name).unlink(missing_ok=True)
        finally:
            self.process=None;os.close(fd);(self.model_root/'.fvv-writer.lock').unlink(missing_ok=True)

    async def status(self,request):
        records=[]
        if self.root.exists():
            for p in sorted(self.root.glob('*/parameters.json'),key=lambda p:p.stat().st_mtime,reverse=True):
                if not (p.parent/'video.mp4').is_file():continue
                try:
                    m=json.loads(p.read_text())
                    if m['scene']==request.query.get('scene','lego'):records.append(dict(id=p.parent.name,created_at=m['created_at'],scene=m['scene']))
                except (OSError,ValueError,KeyError):continue
        return web.json_response(dict(job=self.job if self.job and self.job['scene']==request.query.get('scene','lego') else None,exports=records))

    def folder(self,request):
        ident=request.match_info['id']
        if not re.fullmatch('[a-f0-9]{32}',ident):raise web.HTTPNotFound()
        folder=self.root/ident
        if not (folder/'video.mp4').is_file():raise web.HTTPNotFound()
        meta=json.loads((folder/'parameters.json').read_text())
        if meta['scene']!=request.query.get('scene','lego'):raise web.HTTPNotFound()
        return folder

    async def download(self,request):
        name={'video':'video.mp4','parameters':'parameters.json'}.get(request.match_info['kind'])
        if not name:raise web.HTTPNotFound()
        folder=self.folder(request)
        return web.FileResponse(folder/name,headers={'Content-Disposition':f'attachment; filename="orbit-{folder.name}-{name}"'})

    async def delete(self,request):
        folder=self.folder(request)
        if self.running and self.job['id']==folder.name:raise web.HTTPConflict()
        shutil.rmtree(folder)
        return web.json_response({'deleted':folder.name})

    async def cancel(self,request):
        if self.job and self.job['scene']!=request.query.get('scene','lego'):raise web.HTTPNotFound()
        if self.running:
            self.task.cancel();await self.task
        return web.json_response({'cancelled':True})

    async def cleanup(self,app):
        if self.running:
            self.task.cancel();await self.task

    def install(self,app):
        app.router.add_get('/orbit-export',self.status)
        app.router.add_post('/orbit-export',self.start)
        app.router.add_post('/orbit-export/cancel',self.cancel)
        app.router.add_get('/orbit-export/{id}/{kind}',self.download)
        app.router.add_post('/orbit-export/{id}/delete',self.delete)
        app.on_shutdown.append(self.cleanup)
