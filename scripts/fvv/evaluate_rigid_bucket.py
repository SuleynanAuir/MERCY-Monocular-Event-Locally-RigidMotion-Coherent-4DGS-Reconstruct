"""Bounded real GPU comparison of a candidate; never activates it."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','previous','model','references','video','rays','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);scripts=Path(__file__).resolve().parent
    state=dict(stage='starting',started_at=time.time());commands=[]
    def stage(name,command):
        state['stage']=name;(a.output/'status.json').write_text(json.dumps(state,indent=2))
        commands.append([str(x) for x in command]);(a.output/'commands.json').write_text(json.dumps(commands,indent=2))
        with (a.output/(name+'.log')).open('w') as log:subprocess.run(commands[-1],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    common=['--repo',a.repo,'--previous',a.previous,'--model',a.model]
    try:
        for name,views in [('orbit','views'),('novel','novel-angle-validation'),('arc','showcase-views')]:
            stage(name,[sys.executable,scripts/'check_multiview.py',*common,'--views',a.references/views,'--output',a.output/name])
        stage('motion',[sys.executable,scripts/'check_refinement.py',*common,'--video',a.video,'--output',a.output/'motion'])
        stage('parts',[sys.executable,scripts/'check_ray_quality.py',*common,'--rays',a.rays,'--rgb-views',a.references/'views','--output',a.output/'parts.json'])
        stage('overview',[sys.executable,scripts/'render_roi_comparison.py',*common,'--overview','--panel-width','640','--output',a.output/'overview.mp4'])
        stage('bucket',[sys.executable,scripts/'render_roi_comparison.py',*common,'--output',a.output/'bucket.mp4'])
        state['stage']='evaluated'
    except BaseException as e:state.update(stage='failed',error=str(e));raise
    finally:state['finished_at']=time.time();(a.output/'status.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
