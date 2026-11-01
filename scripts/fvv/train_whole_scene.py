"""Bounded whole-Lego refinement retaining the complete articulated sidecar.

Uses corrected composite depth moments, full-frame RGB and temporal losses.
Never activates a candidate. Pauses only explicitly managed original viewers.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
try:
    from .training_resources import pause,resume
    from .common import digest,usage
except ImportError:
    from training_resources import pause,resume
    from common import digest,usage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','root','video','dataset','views','rays','references','audit-rays']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--registry',type=Path,action='append',default=[]);p.add_argument('--steps',type=int,choices=[6000,12000],default=12000)
    a=p.parse_args();a.root.mkdir(exist_ok=False);s=Path(__file__).resolve().parent;out=a.root/'candidate';records=[];state=dict(stage='preparing',started=time.time(),steps=a.steps)
    def update(stage):state['stage']=stage;(a.root/'status.json').write_text(json.dumps(state,indent=2))
    def run(stage,args,timeout=7200):
        update(stage);command=[sys.executable,*map(str,args)]
        with (a.root/'commands.jsonl').open('a') as f:f.write(json.dumps(command)+'\n')
        with (a.root/(stage+'.log')).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=log)
            try:
                code=process.wait(timeout=timeout)
                if code:raise subprocess.CalledProcessError(code,command)
            except BaseException:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    lock=a.model.parent/'.fvv-writer.lock'
    with lock.open('x') as f:f.write('whole-scene training '+str(os.getpid()))
    try:
        pause(a.registry,records)
        run('training',[s/'refine_lego.py','--repo',a.repo,'--model',a.model,'--video',a.video,'--dataset',a.dataset,'--output',out,
            '--preserve-rigid','--whole-scene','--all-frames','--steps',a.steps,'--max-minutes','120',
            '--multiview',a.views,'--multiview-weight','3','--multiview-batch','2','--train-static','--freeze-static-geometry',
            '--position-lr','3e-6','--appearance-lr','.001','--opacity-lr','.001','--edge-weight','.5',
            '--appearance-weight','8','--temporal-weight','8','--midpoint-weight','1',
            '--ray-prior',a.rays,'--ray-weight','1','--ray-batch','1','--ray-render-width','320','--ray-alpha-weight','.25','--ray-spread-weight','5'])
        if digest(out/'rigid-bucket.npz')!=digest(a.model/'rigid-bucket.npz'):raise AssertionError('Protected articulated sidecar changed')
        if usage(a.root)>2*1024**3:raise OSError('Whole-scene experiment exceeds 2 GiB')
        (a.root/'preservation.json').write_text(json.dumps(dict(rigid_sidecar_exact=True,sha256=digest(out/'rigid-bucket.npz'),config_exact=digest(out/'effective.yaml')==digest(a.model/'effective.yaml')),indent=2))
        run('evaluation',[s/'evaluate_rigid_bucket.py','--repo',a.repo,'--previous',a.model,'--model',out,'--references',a.references,'--video',a.video,'--rays',a.audit_rays,'--output',a.root/'evaluation'],1800)
        update('ready_for_review')
    except BaseException as e:state['error']=str(e);update('failed');raise
    finally:
        try:resume(records)
        finally:lock.unlink(missing_ok=True);state['finished']=time.time();(a.root/'status.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
