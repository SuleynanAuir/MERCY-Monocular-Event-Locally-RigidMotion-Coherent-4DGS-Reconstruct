"""Bounded local export jobs, sharing the viewer's serial GPU executor."""
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import uuid
import zipfile

from aiohttp import web

from utils.experiment_export import camera_record, run_export, validate_plan, reference_plan


def install_exports(app, selected, scenes, references, executor, output_root, loading):
    jobs, tasks = {}, set()

    def job_for(request):
        key, _ = selected(request)
        job = jobs.get(request.match_info['job'])
        if not job or job['scene'] != key:
            raise web.HTTPNotFound()
        return job

    async def start(request):
        key, renderer = selected(request)
        if not renderer.status.get('ready') or loading['id'] is not None:
            raise web.HTTPConflict(text='模型尚未就绪')
        if any(job['status'] == 'running' for job in jobs.values()):
            raise web.HTTPConflict(text='已有导出正在运行，请等待完成')
        try:
            plan = validate_plan(await request.json(), renderer.info)
            plan = reference_plan(plan, renderer.info, scenes[key].get('evaluation_path'))
        except (ValueError, TypeError, KeyError) as error:
            raise web.HTTPBadRequest(text=str(error))
        identifier = uuid.uuid4().hex
        folder = (f'{key}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{identifier[:8]}'
                  if plan['operation'] == 'complete' else identifier)
        destination = (Path(output_root)/folder).resolve()
        job = dict(id=identifier, scene=key, status='running', stage='queued', message='等待导出',
                   completed=0, total=1, percent=0, stage_percent=0, detail='', unit='items')
        jobs[identifier] = job
        # Keep completed jobs addressable without permitting browser-controlled paths.
        if len(jobs) > 100:
            jobs.pop(next(iter(jobs)))

        async def execute():
            try:
                archive = await asyncio.get_running_loop().run_in_executor(executor, lambda: run_export(
                    renderer, plan, destination, key,
                    reference_path=scenes[key].get('evaluation_path'), preview=references.get(key),
                    progress=lambda update: job.update(update)))
                with zipfile.ZipFile(archive) as bundle:
                    prefix = destination.name+'/' if plan['operation'] == 'complete' else ''
                    job['missing_data'] = json.loads(bundle.read(prefix+'data_availability.json'))['missing_data']
                job.update(status='complete', message='导出完成', archive=archive,
                           output_archive=str(archive), output_bytes=archive.stat().st_size,
                           sampled_frames=len(plan['samples']), sample_step=plan.get('sample_step'),
                           storage_mode='archive_only', stage='complete', percent=100,
                           stage_percent=100, completed=1, total=1, unit='items', detail='ZIP 已生成，可以下载')
            except Exception as error:
                logging.getLogger('gaussian-viewer').exception('Experiment export failed')
                job.update(status='failed', message=f'{job["message"]}失败：{error}')
        task = asyncio.create_task(execute())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return web.json_response({k: v for k, v in job.items() if k != 'archive'}, status=202)

    async def freeze_plan(request):
        key, renderer = selected(request)
        if not renderer.status.get('ready') or loading['id'] is not None:
            raise web.HTTPConflict(text='模型尚未就绪')
        try:
            plan = validate_plan(await request.json(), renderer.info)
            plan['samples'] = [{**sample, 'camera': camera_record(renderer, sample)} for sample in plan['samples']]
        except (ValueError, TypeError, KeyError) as error:
            raise web.HTTPBadRequest(text=str(error))
        return web.json_response(dict(plan, format='scgs_experiment_plan_v1', scene=key))

    async def status(request):
        return web.json_response({k: v for k, v in job_for(request).items() if k != 'archive'})

    async def download(request):
        job = job_for(request)
        if job['status'] != 'complete':
            raise web.HTTPConflict(text='导出尚未完成')
        return web.FileResponse(job['archive'], headers={
            'Content-Disposition': f'attachment; filename="{job["archive"].name}"'})

    async def shutdown(app):
        if tasks: await asyncio.gather(*tasks)
    app.on_shutdown.append(shutdown)
    app.router.add_post('/api/experiments', start)
    app.router.add_post('/api/experiments/plan', freeze_plan)
    app.router.add_get('/api/experiments/{job}', status)
    app.router.add_get('/api/experiments/{job}/download', download)
