"""Validation-only inference sensitivity and repeated warmed GPU measurements."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .protocol import ROOT,save_json,sha256
from .model import Adapter,SceneRenderer
from .training import Supervision,restore,evaluate


def load(scene,method,seed=17):
    run=ROOT/'01_main_comparison/runs'/f'{scene}_{method}_s{seed}_full'
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    prepared=ROOT/'01_main_comparison/prepared'/scene
    for key,digest in checkpoint['config']['inputs'].items():
        if sha256(prepared/key)!=digest:raise ValueError('Checkpoint input identity mismatch: '+key)
    prior=json.loads((prepared/'prior.json').read_text())
    if sha256(prior['source'])!=prior['source_sha256']:
        raise ValueError('Original G0 changed since preparation')
    if sha256(prepared/'motion.npz')!=prior['motion_sha256']:
        raise ValueError('Prepared motion changed')
    adapter=Adapter(prepared,method,seed).cuda();restore(adapter,checkpoint['weights'])
    return SceneRenderer(prepared,adapter),Supervision(prepared),run


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenes',nargs='+',default=['lego','fan','lamp','metronome','microphone'])
    args=p.parse_args();efficiency=[]
    for scene in args.scenes:
        destination=ROOT/'03_sensitivity/runs'/scene
        if not (destination/'results.json').exists():
            renderer,sup,run=load(scene,'C3');rows=[]
            for gain in (0.,.25,.5,.75,1.):
                renderer.adapter.gain=gain
                result=evaluate(renderer,sup,split='validation')
                rows.append(dict(gain=gain,metrics=result['metrics']))
                print('Sensitivity',scene,gain,flush=True)
            save_json(destination/'results.json',dict(scene=scene,seed=17,checkpoint_sha256=sha256(run/'checkpoint.pt'),
                scope='Post-training inference gain sensitivity on validation only. No retraining and no test-set parameter selection.',
                rows=rows))
            del renderer,sup;torch.cuda.empty_cache()
        for method in ('C1','C2','C3'):
            renderer,sup,run=load(scene,method)
            frames=sup.meta['split']['frames']['validation']
            frames=np.asarray(frames)[np.linspace(0,len(frames)-1,min(10,len(frames))).astype(int)].tolist()
            for warmup in range(20):renderer.render(frames[warmup%len(frames)])
            torch.cuda.synchronize();samples=[]
            for repeat in range(5):
                for frame in frames:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();renderer.render(frame);end.record();torch.cuda.synchronize()
                    samples.append(dict(repeat=repeat,frame=frame,gpu_ms=start.elapsed_time(end)))
            runtime=json.loads((run/'runtime.json').read_text());config=json.loads((run/'config.json').read_text())
            training_runs=[]
            for other in sorted((ROOT/'01_main_comparison/runs').glob(f'{scene}_{method}_s*_full')):
                if not (other/'evaluation.json').exists():continue
                timing=json.loads((other/'runtime.json').read_text())
                identity=json.loads((other/'config.json').read_text())
                training_runs.append(dict(seed=identity['seed'],wall_seconds=timing['training_wall_seconds'],
                                          peak_gpu_bytes=timing['peak_gpu_bytes']))
            efficiency.append(dict(scene=scene,method=method,gpu=config['gpu'],resolution=config['resolution'],
                seed=17,samples=samples,median_gpu_ms=float(np.median([x['gpu_ms'] for x in samples])),
                p95_gpu_ms=float(np.quantile([x['gpu_ms'] for x in samples],.95)),
                training_wall_seconds=runtime['training_wall_seconds'],peak_gpu_bytes=runtime['peak_gpu_bytes'],
                training_seed_runs=training_runs,
                mean_training_wall_seconds=float(np.mean([r['wall_seconds'] for r in training_runs])),
                sd_training_wall_seconds=float(np.std([r['wall_seconds'] for r in training_runs],ddof=1)) if len(training_runs)>1 else None,
                maximum_training_peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in training_runs),
                gaussians=renderer.adapter.metadata['gaussians'],
                trainable_parameters=config['parameter_count'],warmup_frames=20,
                scope='CUDA-event elapsed time around the motion + GS render call. Includes in-call staging/transfers and submission gaps, including baked source bands in Lego. Excludes supervision video decode, SSIM and result-file writes. Not a pure kernel benchmark. Repeats are timing samples, not independent training seeds.'))
            save_json(ROOT/'04_efficiency/results.json',dict(runs=efficiency))
            print('Efficiency',scene,method,efficiency[-1]['median_gpu_ms'],flush=True)
            del renderer,sup;torch.cuda.empty_cache()


if __name__=='__main__':main()
