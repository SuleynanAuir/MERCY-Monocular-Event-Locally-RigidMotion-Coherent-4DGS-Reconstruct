"""Sequential GPU queue. Existing complete runs are verified, never overwritten."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ROOT,save_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--scenes',nargs='+',default=['lego','fan','lamp','metronome','microphone'])
    p.add_argument('--section',choices=['comparison','ablation'],default='comparison')
    args=p.parse_args();jobs=[]
    if args.section=='comparison':
        for scene in args.scenes:
            jobs.append((scene,'C1',args.seeds[0],'full'))
            for seed in args.seeds:
                jobs.extend((scene,method,seed,'full') for method in ('C2','C3'))
        folder=ROOT/'01_main_comparison'
    else:
        folder=ROOT/'02_ablation'
        for scene in args.scenes:
            for variant in ('no_mlp','no_event','no_joint','no_jerk'):
                if scene=='fan' and variant=='no_event':continue
                jobs.append((scene,'C3',args.seeds[0],variant))
    folder.mkdir(exist_ok=True,parents=True);(folder/'logs').mkdir(exist_ok=True)
    config=dict(steps=args.steps,seeds=args.seeds,scenes=args.scenes,jobs=jobs,
        scope='fixed-budget five-scene adapters; not original full Lego pipeline',
        code_sha256={p.name:sha256(p) for p in ROOT.glob('*.py')})
    frozen=folder/'queue_protocol.json'
    if frozen.exists():
        old=json.loads(frozen.read_text())
        if old!=json.loads(json.dumps(config)):raise ValueError('Queue protocol changed; choose a fresh experiment revision')
    else:save_json(frozen,config)
    records=[];environment=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for scene,method,seed,variant in jobs:
        name=f'{scene}_{method}_s{seed}_{variant}';output=folder/'runs'/name
        if (output/'evaluation.json').exists():
            records.append(dict(run=name,status='previously_complete'));continue
        if output.exists():
            records.append(dict(run=name,status='incomplete_existing_run'));save_json(folder/'queue_status.json',records);continue
        command=[sys.executable,'-m','experiment.run','train','--prepared',str(ROOT/'01_main_comparison/prepared'/scene),
                 '--method',method,'--seed',str(seed),'--steps',str(args.steps),'--variant',variant,'--output',str(output)]
        start=time.time();print('Starting',name,flush=True)
        save_json(folder/'active_run.json',dict(run=name,command=command,started=start))
        with (folder/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT.parent,env=environment,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(run=name,status='complete' if result.returncode==0 else 'failed',
                            returncode=result.returncode,wall_seconds=time.time()-start))
        save_json(folder/'queue_status.json',records)
        print('Finished',name,records[-1]['status'],round(time.time()-start,1),'s',flush=True)
    failed=[r for r in records if r['status'] in ('failed','incomplete_existing_run')]
    if failed:raise RuntimeError(f'{len(failed)} runs failed or incomplete; inspect recorded logs')


if __name__=='__main__':main()
