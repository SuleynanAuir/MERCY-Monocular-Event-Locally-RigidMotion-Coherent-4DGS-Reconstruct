"""Finish a rebuilt catalog, capture the real UI, and produce audited reports."""
import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from .protocol import save_json, sha256

SCENES=('lego','fan','lamp','metronome','microphone')


async def views(url, catalog, output, deployment_identity=None):
    import aiohttp
    from .ui_capture import decode_packet
    presets=[('front',0,0),('oblique',35,20),('side',80,0),('back',180,0),('top',0,35),('bottom',0,-35)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for entry in catalog['scenes']:
            dest=output/entry['id']
            if (dest/'manifest.json').exists():continue
            dest.mkdir(parents=True,exist_ok=True)
            async with session.get(url+'/api/status',params={'scene':entry['id']}) as response:
                status=await response.json()
            info=status['scene'];frames=[0,info['frame_count']//2,info['frame_count']-1]
            identity=info.get('asset_identity') or deployment_identity
            if not identity:
                raise ValueError('Six-view capture requires a renderer identity or a verified deployment lock')
            records=[]
            async with session.ws_connect(url+'/ws',params={'scene':entry['id']},max_msg_size=32*1024**2) as ws:
                for view,yaw,pitch in presets:
                    for frame in frames:
                        settings=dict(type='frame',id=len(records),frame=frame,width=640,height=640,
                            yaw=yaw,pitch=pitch,zoom=1,pan_x=0,pan_y=0,fovy=info['fovy'],
                            background='light',render_mode='clay',clay_style='solid',quality=95)
                        await ws.send_json(settings)
                        message=await ws.receive(timeout=300)
                        if message.type!=aiohttp.WSMsgType.BINARY:raise ValueError(message.data)
                        meta,jpeg,_=decode_packet(message.data,frame,640)
                        filename=f'{view}-{frame:04d}.jpg';(dest/filename).write_bytes(jpeg)
                        records.append(dict(file=filename,request=settings,sha256=hashlib.sha256(jpeg).hexdigest()))
            save_json(dest/'manifest.json',dict(asset_identity=identity,views=records,
                scope='Actual UI six presets at fixed first/middle/last frames; no independent nonfront RGB reference'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--port',type=int,default=8895)
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--scenes',nargs='+',choices=SCENES,default=list(SCENES))
    p.add_argument('--reuse-running-server',action='store_true',help='Resume audits/captures using this run\'s already launched UI')
    args=p.parse_args();root=args.root.resolve();py=sys.executable
    deadline=time.monotonic()+args.timeout
    statuspath=root/'finish-status.json'
    while True:
        missing=[str(root/s/f'pair-v1-{v}.json') for s in args.scenes for v in ('v2','v3')
                 if not (root/s/f'pair-v1-{v}.json').is_file()]
        if not missing:break
        save_json(statuspath,dict(status='waiting_for_rebuilt_assets',missing=missing))
        if time.monotonic()>deadline:raise TimeoutError(missing)
        time.sleep(5)
    catalog=dict(default=args.scenes[0]+'-v1-v3',scenes=[])
    for scene in args.scenes:
        for version in ('v2','v3'):
            catalog['scenes'].append(dict(id=f'{scene}-v1-{version}',name=f'{scene.title()} · V1 / {version.upper()}',
                type='gaussian_pair',cache='cpu',path=str(root/scene/f'pair-v1-{version}.json'),
                reference_path=str(root/scene/'references/reference.json')))
    catalog_path=root/'scenes.json';save_json(catalog_path,catalog)
    def run(name,cmd):
        save_json(statuspath,dict(status='running',step=name))
        print('Final validation:',name,flush=True)
        with (root/(name+'.log')).open('a') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    # Audits use the exact serialized coordinates; no older adapter output.
    for scene in args.scenes:
        for version in ('v1','v2','v3'):
            dest=root/'native-audit'/scene/version
            if (dest/'evaluation.json').exists():
                previous=json.loads((dest/'evaluation.json').read_text())
                identity=previous.get('ui_asset_identity',{})
                asset=root/scene/('supplied-pair/left.npz' if version=='v1' else version+'.npz')
                prior=root/scene/('v3-prior' if version=='v3' else 'v1-prior')/'prior.json'
                if identity.get('sequence_sha256')!=sha256(asset) or identity.get('prior_sha256')!=sha256(prior):
                    raise ValueError('Cannot reuse a native audit from different UI assets')
                if version!='v1' and identity.get('checkpoint_sha256')!=sha256(root/scene/(version+'-training/checkpoint.pt')):
                    raise ValueError('Cannot reuse a native audit from a different checkpoint')
                if not previous.get('common_trajectory_probes'):
                    raise ValueError('Native audit lacks the shared V1 trajectory probes')
                continue
            run(f'native-{scene}-{version}',[py,'-m','experiment.ui_native_audit','--root',str(root),
                '--scene',scene,'--version',version,'--output',str(dest)])
    server=None
    if args.reuse_running_server:
        previous=json.loads((root/'viewer-process.json').read_text())
        if previous['port']!=args.port or previous['catalog']!=str(catalog_path):
            raise ValueError('Cannot reuse a different UI server')
    else:
        server_log=(root/'viewer-final.log').open('a')
        server=subprocess.Popen([py,'-u','viewer.py','--scenes_path',str(catalog_path),
            '--host','127.0.0.1','--port',str(args.port)],stdout=server_log,stderr=subprocess.STDOUT,
            start_new_session=True,stdin=subprocess.DEVNULL)
        save_json(root/'viewer-process.json',dict(pid=server.pid,port=args.port,catalog=str(catalog_path)))
    url=f'http://127.0.0.1:{args.port}'
    import urllib.request
    deadline=time.monotonic()+args.timeout
    while time.monotonic()<deadline:
        if server is not None and server.poll() is not None:raise RuntimeError('Final UI exited; inspect viewer-final.log')
        try:
            with urllib.request.urlopen(url+'/api/status',timeout=5) as response:
                status=json.load(response)
            if status.get('error'):raise RuntimeError(status)
            if {o['id'] for o in status['objects']}!={o['id'] for o in catalog['scenes']}:
                raise ValueError('Running UI has a different scene catalog')
            save_json(statuspath,dict(status='initializing_ui',message=status.get('message'),
                completed=sum(o['ready'] for o in status['objects']),total=len(catalog['scenes'])))
            if status.get('ready'):break
        except (OSError,TimeoutError):pass
        time.sleep(2)
    else:raise TimeoutError('Final UI initialization')
    run('actual-ui-clay',[py,'-m','experiment.ui_evaluate','--catalog',str(catalog_path),
        '--url',url,'--output',str(root/'ui-evaluation/full'),'--mode','clay'])
    save_json(statuspath,dict(status='running',step='six_view_capture'))
    asyncio.run(views(url,catalog,root/'ui-evaluation/six-views'))
    run('ui-report',[py,'-m','experiment.ui_report','--root',str(root),
        '--output',str(root/'report/01_main_comparison')])
    save_json(statuspath,dict(status='complete' if len(args.scenes)==5 else 'selected_scenes_complete',scene_count=len(args.scenes),versions=['V1','V2','V3'],
        primary_mode='clay',main_comparison=len(args.scenes)==5,full_new_ablation=False,
        url=url,catalog=str(catalog_path)))


if __name__=='__main__':main()
