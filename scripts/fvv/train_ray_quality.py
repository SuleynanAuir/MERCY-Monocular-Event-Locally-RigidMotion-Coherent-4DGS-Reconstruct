"""Isolated long quality refinement using existing RGB and bounded mesh rays.

Never activates a candidate. Restores paused viewers on success and failure.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
try:
    from .training_resources import pause,resume
except ImportError:
    from training_resources import pause,resume


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','repo','model','video','dataset','multiview','geometry','rays','wide-validation','novel-validation','arc-validation']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registry',type=Path,action='append',default=[])
    parser.add_argument('--steps',type=int,default=12000,choices=[3000,6000,12000,18000,24000])
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent;candidate=args.root/'candidate';records=[]
    state={'stage':'preparing','steps':args.steps,'started_at':time.time(),'parent':str(args.model)}
    def update():(args.root/'status.json').write_text(json.dumps(state,indent=2))
    def stage(name,command):
        state['stage']=name;update()
        with (args.root/(name+'.log')).open('w') as log:
            process=subprocess.Popen([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=125*60)
                if code:raise subprocess.CalledProcessError(code,command)
            except BaseException:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
    def interrupted(signum,frame):raise KeyboardInterrupt('Offline quality training interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    lock=args.model.parent/'.fvv-writer.lock'
    with lock.open('x') as stream:stream.write(f'quality ray training pid={os.getpid()}')
    try:
        pause(args.registry,records)
        stage('training',[sys.executable,scripts/'refine_lego.py','--repo',args.repo,'--model',args.model,'--video',args.video,
            '--dataset',args.dataset,'--output',candidate,'--steps',args.steps,'--max-minutes','120','--all-frames',
            '--multiview',args.multiview,'--multiview-weight','3','--multiview-batch','2','--train-static',
            '--position-lr','3e-5','--opacity-lr','.001','--edge-weight','.5','--temporal-weight','8','--appearance-weight','8',
            '--midpoint-weight','2','--subframe-samples','2','--geometry',args.geometry,'--geometry-weight','.5','--surface-normal-weight','1',
            '--ray-prior',args.rays,'--ray-weight','2','--ray-batch','2'])
        base=['--repo',args.repo,'--previous',args.model,'--model',candidate]
        for tag,views in [('orbit',args.wide_validation),('novel',args.novel_validation),('arc',args.arc_validation)]:
            stage(tag,[sys.executable,scripts/'check_multiview.py',*base,'--views',views,'--output',args.root/(tag+'-evaluation')])
        stage('motion',[sys.executable,scripts/'check_refinement.py',*base,'--video',args.video,'--output',args.root/'motion-evaluation'])
        stage('overview',[sys.executable,scripts/'render_roi_comparison.py',*base,'--overview','--output',args.root/'overview-comparison.mp4'])
        state['stage']='evaluated'
    except BaseException as error:state.update(stage='failed',error=str(error));raise
    finally:
        try:resume(records)
        except BaseException as error:state.update(stage='restore_failed',restore_error=str(error));raise
        finally:lock.unlink(missing_ok=True);state['finished_at']=time.time();update()


if __name__=='__main__':main()
