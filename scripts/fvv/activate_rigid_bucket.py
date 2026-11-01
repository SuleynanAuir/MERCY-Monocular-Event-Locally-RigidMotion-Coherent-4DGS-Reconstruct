"""Publish an evaluated bucket candidate with a managed-service rollback."""
import argparse,json,os,signal,socket,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen
from common import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','candidate','evaluation','registry']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();source=a.candidate.resolve();meta=json.loads((source/'manifest.json').read_text())
    if json.loads((a.evaluation/'status.json').read_text())['stage']!='evaluated':raise ValueError('Candidate evaluation incomplete')
    reviewed=json.loads((a.evaluation/'novel/results.json').read_text())['candidate']
    checked=reviewed['sha256']
    if reviewed.get('sidecar_sha256')!=meta['rigid_bucket']['sha256']:raise ValueError('Unverified articulated appearance')
    if meta['scene']!='lego' or checked!=meta['sha256'] or digest(source/'model.pth')!=meta['sha256']:raise ValueError('Unverified model')
    if not meta['rigid_bucket']['trained'] or digest(source/'rigid-bucket.npz')!=meta['rigid_bucket']['sha256']:raise ValueError('Unverified bucket')
    rows=json.loads(a.registry.read_text());row=next(x for x in rows if x['port']==8897);oldcmd=row['command']
    if Path(oldcmd[1]).name!='hybrid.py' or 'serve' not in oldcmd or oldcmd[oldcmd.index('--port')+1]!='8897':raise ValueError('Unexpected service registry')
    def command(pid):
        try:return Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except OSError:return []
    matching=[int(x.name) for x in Path('/proc').iterdir() if x.name.isdigit() and command(int(x.name))==oldcmd]
    if len(matching)>1:raise ValueError('Multiple matching FVV services')
    with socket.socket() as sock:
        if not matching and sock.connect_ex(('127.0.0.1',8897))==0:raise ValueError('Unmanaged process occupies FVV port')
    env=dict(os.environ);env.update(TORCH_EXTENSIONS_DIR='/root/autodl-tmp/hybrid-fvv-validation/torch-extensions',TORCH_CUDA_ARCH_LIST='8.6+PTX',MAX_JOBS='2')
    env['PATH']=str(Path(oldcmd[0]).parent)+':'+env.get('PATH','')
    def stop(pid,expected):
        if command(pid)!=expected:return
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if command(pid)!=expected:return
            time.sleep(.1)
        raise TimeoutError('Old FVV service did not stop')
    def launch(cmd):
        with (base/'bucket-service.log').open('ab') as log:return subprocess.Popen(cmd,cwd=base,env=env,stdout=log,stderr=log,start_new_session=True)
    def ready(process,expected):
        for _ in range(120):
            if process.poll() is not None:raise RuntimeError('FVV service exited')
            try:
                with urlopen('http://127.0.0.1:8897/status',timeout=2) as reply:status=json.load(reply)
                if status.get('ready') and status['sha256']==expected['sha256'] and status.get('rigid_bucket',{}).get('sha256')==expected.get('rigid_bucket',{}).get('sha256'):return status
            except OSError:pass
            time.sleep(.5)
        raise TimeoutError('FVV readiness timeout')
    def record(process,cmd,model,info):
        row.update(pid=process.pid,command=cmd);tmp=a.registry.with_suffix('.bucket.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(a.registry)
        tmp=base/'active-model.bucket.tmp';tmp.write_text(json.dumps(dict(model=str(model),pid=process.pid,sha256=info['sha256']),indent=2));tmp.replace(base/'active-model.json')
    name='lego-'+datetime.fromisoformat(meta['trained_at']).astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');target=base/'models'/name
    if target.exists():raise FileExistsError(target)
    lock=base/'models/.fvv-writer.lock'
    with lock.open('x') as f:f.write('reviewed rigid bucket activation')
    try:
        source.rename(target)
        try:source.symlink_to(target,target_is_directory=True)
        except BaseException:target.rename(source);raise
        if matching:stop(matching[0],oldcmd)
        cmd=list(oldcmd);cmd[cmd.index('--model')+1]=str(target);process=None
        try:
            process=launch(cmd);status=ready(process,meta);record(process,cmd,target,meta)
        except BaseException:
            if process is not None and process.poll() is None:stop(process.pid,cmd)
            oldmodel=Path(oldcmd[oldcmd.index('--model')+1]);oldmeta=json.loads((oldmodel/'manifest.json').read_text());old=launch(oldcmd);ready(old,oldmeta);record(old,oldcmd,oldmodel,oldmeta)
            raise
        result=dict(model=str(target),pid=process.pid,sha256=meta['sha256'],bucket_sha256=meta['rigid_bucket']['sha256'],ready=status['ready'],previous_model=oldcmd[oldcmd.index('--model')+1],previous_service_running=bool(matching))
        (a.evaluation/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
