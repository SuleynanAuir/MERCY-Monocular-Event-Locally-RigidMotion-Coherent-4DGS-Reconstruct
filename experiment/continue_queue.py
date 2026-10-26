"""Wait for a complete comparison, then run the remaining registered studies."""
import json
import os
import subprocess
import sys
import time
from .protocol import ROOT,save_json


def main():
    start=time.monotonic();status=ROOT/'01_main_comparison/queue_status.json'
    expected=35  # five fixed C1 + five scenes × two learned methods × three seeds
    while True:
        if status.exists():
            rows=json.loads(status.read_text())
            if any(r['status'] in ('failed','incomplete_existing_run') for r in rows):
                raise RuntimeError('Comparison has a failed run; refusing to hide it in subsequent reports')
            if len(rows)==expected:break
        if time.monotonic()-start>12*3600:raise TimeoutError('Comparison queue not complete after 12 hours')
        time.sleep(30)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    commands=[['experiment.batch','--section','ablation','--steps','200','--seeds','17'],
              ['experiment.followups'],['experiment.run','report']]
    records=[]
    for command in commands:
        print('Starting study',command,flush=True);begin=time.time()
        result=subprocess.run([sys.executable,'-m',*command],cwd=ROOT.parent,env=env)
        records.append(dict(command=command,returncode=result.returncode,wall_seconds=time.time()-begin))
        save_json(ROOT/'study_status.json',records)
        if result.returncode:raise RuntimeError('Study failed: '+command[0])
    save_json(ROOT/'completed.json',dict(status='complete',comparison_runs=expected,
        ablation_runs=19,sensitivity='five validation-only gains per scene',efficiency='five warmed timing repetitions'))


if __name__=='__main__':main()
