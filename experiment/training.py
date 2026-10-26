"""Matched-budget training with validation selection and one held-out audit."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model import Adapter,SceneRenderer
from .protocol import save_json,sha256,measured,unavailable
from .metrics import image_metrics,missing_fraction,jerk
from .background import training_background,foreground
from utils.event_supervision import consistency_metrics,acceptance_gate,signed_event_change


class Supervision:
    def __init__(self,prepared):
        directory=Path(prepared)/'supervision'
        self.meta=json.loads((directory/'metadata.json').read_text())
        self.rgb=np.load(directory/'rgb.npy',mmap_mode='r')
        self.background=training_background(prepared)
        self.background_tensor=torch.tensor(self.background['rgb'],dtype=torch.float32,device='cuda')[:,None,None]
        self.counts=None
        if self.meta['status']=='ready':
            with np.load(directory/'events.npz') as z:self.counts=z['counts'].copy()
            c=self.meta['calibration'];self.floor=c['log_floor']
            self.observed=self.counts[:,0]*c['thresholds'][0]-self.counts[:,1]*c['thresholds'][1]
        self.factor=self.meta['event_pool_factor']

    def image(self,i):return torch.tensor(np.asarray(self.rgb[i]).transpose(2,0,1)/255,dtype=torch.float32,device='cuda')

    def change(self,a,b):
        weights=a.new_tensor([.299,.587,.114])[:,None,None]
        log=lambda x:torch.log((x*weights).sum(0).clamp(min=self.floor))
        return F.avg_pool2d((log(b)-log(a))[None,None],self.factor)[0,0]

    def loss(self,renderer,i,use_event=True):
        first=renderer.render(i)['render'].clamp(0,1);gt=self.image(i)
        mask=(gt-self.background_tensor).abs().mean(0)>8/255
        rgb=(first-gt).abs().mean()+.5*(first[:,mask]-gt[:,mask]).abs().mean()
        event=first.new_zeros(())
        if use_event and self.counts is not None:
            second=renderer.render(i+1)['render'].clamp(0,1)
            target=torch.tensor(self.observed[i],device='cuda')
            event=(self.change(first,second)-target).abs().sum()/target.abs().sum().clamp(min=1e-4)
        if getattr(self, 'event_only', False):
            if not use_event or self.counts is None:
                raise ValueError('Event-only auxiliary training requires complete calibrated events')
            return .03*event, dict(rgb_diagnostic_only=float(rgb.detach()), event=float(event.detach()))
        return rgb+.03*event,dict(rgb=float(rgb.detach()),event=float(event.detach()))


def weights(adapter):
    return dict(alignment=copy.deepcopy(adapter.alignment.state_dict()),mlp=copy.deepcopy(adapter.mlp.state_dict()))


def restore(adapter,state):
    adapter.alignment.load_state_dict(state['alignment']);adapter.mlp.load_state_dict(state['mlp'])


def unwrap(angle):
    difference=torch.diff(angle,dim=0)
    difference=torch.remainder(difference+torch.pi,2*torch.pi)-torch.pi
    return torch.cat([angle[:1],angle[:1]+torch.cumsum(difference,dim=0)],0)


def regularizer(renderer,ids,baseline,variant):
    a=renderer.adapter;xyz,gaps,angles=renderer.trajectories(ids)
    d=a.diagonal;base_xyz,base_gap,base_angle=baseline
    loss=xyz.new_zeros(())
    if variant!='no_joint' and gaps.numel():
        # Soft connection protection in the five-scene adapter, explicitly not
        # the original Lego hard-FK constraint.
        loss=loss+5*((gaps-base_gap).relu()/d/.001).square().mean()
    if variant!='no_jerk':
        final=torch.diff(xyz,n=3,dim=0)/d;prior=torch.diff(base_xyz,n=3,dim=0)/d
        loss=loss+.002*final.square().mean()/prior.square().mean().clamp(min=1e-9)
        if angles.numel():
            final=torch.diff(unwrap(angles),n=3,dim=0);prior=torch.diff(unwrap(base_angle),n=3,dim=0)
            loss=loss+.002*final.square().mean()/prior.square().mean().clamp(min=1e-7)
        amplitude=torch.amax(xyz,dim=0)-torch.amin(xyz,dim=0)
        original=torch.amax(base_xyz,dim=0)-torch.amin(base_xyz,dim=0)
        loss=loss+.1*((amplitude-original)/d/.01).square().mean()
    return loss


@torch.no_grad()
def validate(renderer,supervision,ids,use_event):
    return float(np.mean([float(supervision.loss(renderer,int(i),use_event)[0]) for i in ids]))


@torch.no_grad()
def evaluate(renderer,supervision,split='test',save_to=None):
    from PIL import Image
    a=renderer.adapter;meta=supervision.meta;frame_ids=meta['split']['frames'][split]
    events=meta.get('event_intervals',{}).get(split,[])
    needed=sorted(set(frame_ids)|set(events)|{i+1 for i in events})
    changes={};rows=[];previews=[];timing=[]
    selected=set(np.asarray(frame_ids)[np.linspace(0,len(frame_ids)-1,min(3,len(frame_ids))).astype(int)])
    # GPU event timings measure device rendering plus motion, excluding CPU
    # image metrics, disk I/O and video decode. Those phases remain wall timed.
    for i in needed:
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();result=renderer.render(i);end.record();torch.cuda.synchronize()
        timing.append(start.elapsed_time(end))
        rgb=result['render'].clamp(0,1)
        if supervision.counts is not None:
            lum=(rgb*rgb.new_tensor([.299,.587,.114])[:,None,None]).sum(0)
            changes[i]=F.avg_pool2d(lum.clamp(min=supervision.floor).log()[None,None],supervision.factor)[0,0].cpu().numpy()
        if i in frame_ids:
            gt=np.asarray(supervision.rgb[i],dtype=np.float32)/255
            predicted=rgb.permute(1,2,0).cpu().numpy();mask=foreground(gt,supervision.background['rgb'])
            row=dict(frame=i,**image_metrics(gt,predicted,mask),
                missing=missing_fraction(mask,result['alpha'].squeeze().cpu().numpy()))
            rows.append(row)
            if i in selected and save_to:
                path=Path(save_to)/'renders';path.mkdir(exist_ok=True,parents=True)
                Image.fromarray((predicted*255).round().astype(np.uint8)).save(path/f'frame_{i:04d}.png')
                Image.fromarray(np.asarray(supervision.rgb[i])).save(path/f'reference_{i:04d}.png')
    event_rows=[]
    for i in events:
        c=supervision.counts[i];target=supervision.observed[i]
        metric=consistency_metrics(changes[i+1]-changes[i],target,c.sum(0),tolerance=1,minimum_change=.01)
        reference=meta['reference_self_check'][i]
        event_rows.append(dict(interval=i,metrics=metric,gate=acceptance_gate(metric,reference)))
    metrics={k:measured(np.mean([r[k] for r in rows])) for k in ('rgb_l1','ssim')}
    metrics['psnr']=measured(np.mean([r['psnr'] for r in rows])) if all(r['psnr'] is not None for r in rows) else unavailable('exact_match_infinite_psnr; see per-frame records')
    numerator=sum(r['missing']['numerator'] for r in rows);denominator=sum(r['missing']['denominator'] for r in rows)
    metrics['visible_missing']=measured(numerator/denominator,numerator=numerator,denominator=denominator,
        scope='RGB absolute distance from frozen training background >8/255; alpha <0.5. Image-mask coverage proxy, not geometric missing area.')
    metrics['back_missing']=unavailable('No independent registered rear-view reference. Source animation is a prior, not held-out reconstruction ground truth.')
    if event_rows:
        eligible=[r['metrics']['relative_l1'] for r in event_rows if r['metrics']['observable']]
        metrics['event_relative_l1']=measured(np.mean(eligible),observable=len(eligible),total=len(event_rows)) if eligible else unavailable('no_observable_event_intervals')
        passed=sum(r['gate']['passed'] for r in event_rows)
        metrics['event_pass']=measured(passed/len(event_rows),passed=passed,total=len(event_rows),scope='held-out intervals; unobservable windows fail closed')
    else:
        for key in ('event_relative_l1','event_pass'):metrics[key]=unavailable(meta.get('reason','no_event_supervision'))
    ids=torch.tensor(frame_ids,device='cuda');xyz,gaps,angle=renderer.trajectories(ids)
    times=np.array(frame_ids)/meta['video']['fps']
    spatial=jerk(xyz.cpu().numpy(),times)
    metrics['position_jerk']=measured(spatial['value'],unit=spatial['unit'],samples=spatial['samples'])
    if gaps.numel():
        metrics['joint_gap_max']=measured(gaps.max().item(),unit='canonical scene units',scope=a.metadata['joint_scope'])
        angular=jerk(angle.cpu().numpy(),times,angular=True)
        metrics['angular_jerk']=measured(angular['value'],unit=angular['unit'])
    else:
        angular=None
        for key in ('joint_gap_max','angular_jerk'):metrics[key]=unavailable('no_validated_source_revolute_joint_candidates')
    result=dict(scene=a.metadata['scene'],method=a.method,split=split,metrics=metrics,
        frame_metrics=rows,event_metrics=event_rows,trajectory=dict(spatial=spatial,angular=angular),
        timing=dict(gpu_ms=timing,median_gpu_ms=float(np.median(timing)),p95_gpu_ms=float(np.quantile(timing,.95))),
        observation_group='RGB+raw_events' if supervision.counts is not None else 'RGB_only',
        interpretation='Conditional on supplied G0 and source motion. Not unseen-scene or unknown-motion generalization.')
    if save_to:save_json(Path(save_to)/'evaluation.json',result)
    return result


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    prepared=Path(args.prepared);sup=Supervision(prepared)
    adapter=Adapter(prepared,args.method,args.seed).cuda()
    parent = getattr(args, 'initial_alignment', None)
    if parent:
        checkpoint = torch.load(parent, map_location='cpu')
        if checkpoint['config']['method'] != 'C2':
            raise ValueError('Auxiliary stage must start from a frozen C2 alignment')
        adapter.alignment.load_state_dict(checkpoint['weights']['alignment'])
    frozen_alignment = bool(getattr(args, 'freeze_alignment', False))
    if frozen_alignment:
        if args.method != 'C3' or not parent:
            raise ValueError('Frozen alignment requires C3 and its C2 parent')
        adapter.alignment.requires_grad_(False)
    sup.event_only = bool(getattr(args, 'event_only', False))
    if args.variant=='no_mlp':adapter.gain=0.;adapter.mlp.requires_grad_(False)
    renderer=SceneRenderer(prepared,adapter)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    use_event=sup.counts is not None and args.variant!='no_event'
    # Keep the eligible frame population identical when ablating the event
    # objective. Only its loss/selection term is removed, not boundary guards.
    available_events=sup.counts is not None
    ids=sup.meta['event_intervals']['train'] if available_events else sup.meta['split']['intervals']['train']
    valid=sup.meta['event_intervals']['validation'] if available_events else sup.meta['split']['intervals']['validation']
    validation=np.asarray(valid)[np.linspace(0,len(valid)-1,min(5,len(valid))).astype(int)]
    config=dict(method=args.method,variant=args.variant,seed=args.seed,steps=0 if args.method=='C1' else args.steps,
        learning_rate=.01,mlp_learning_rate=.003,validation_every=40,validation_intervals=validation.tolist(),
        selection='minimum validation RGB+0.03 event objective; no test selection',resolution=[640,640],
        inputs={k:sha256(prepared/k) for k in ('prior.json','motion.npz','supervision/metadata.json')},
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,numpy_version=np.__version__,
        parameter_count=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        background=renderer.background_record,
        method_scope='Unified motion adapter; frozen appearance/density. C3 local coherent rigid-group MLP with soft joints and trajectory regularization.',
        determinism='Seeded initialization and interval order; CUDA rasterizer atomic reductions may be nondeterministic.')
    save_json(output/'config.json',config)
    if parent or frozen_alignment or sup.event_only:
        config.update(initial_alignment_sha256=sha256(parent) if parent else None,
            freeze_alignment=frozen_alignment, auxiliary_event_only=sup.event_only,
            selection='minimum validation event objective; no RGB objective' if sup.event_only else config['selection'],
            method_scope='Auxiliary motion stage on the explicitly recorded UI canonical asset; geometry and display finish evaluated separately through UI')
        save_json(output/'config.json',config)
    start=time.perf_counter();best=weights(adapter);best_step=0
    best_loss=validate(renderer,sup,validation,use_event);logs=[dict(step=0,validation=best_loss)]
    print(adapter.metadata['scene'],args.method,args.seed,'initial validation',best_loss,flush=True)
    if args.method!='C1':
        parameters=[] if frozen_alignment else [dict(params=adapter.alignment.parameters(),lr=.01)]
        if args.method=='C3' and args.variant!='no_mlp':parameters.append(dict(params=adapter.mlp.parameters(),lr=.003))
        optimizer=torch.optim.Adam(parameters)
        generator=np.random.default_rng(args.seed);order=[]
        train_frames=torch.tensor(sup.meta['split']['frames']['train'],device='cuda')
        with torch.no_grad():baseline=tuple(x.detach() for x in renderer.trajectories(train_frames))
        for step in range(1,args.steps+1):
            if not order:order=generator.permutation(ids).tolist()
            i=int(order.pop());optimizer.zero_grad(set_to_none=True)
            loss,detail=sup.loss(renderer,i,use_event)
            regularization=loss.new_zeros(())
            if args.method=='C3':regularization=regularizer(renderer,train_frames,baseline,args.variant)
            loss=loss+regularization+.0001*adapter.alignment.raw.square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step()
            entry=dict(step=step,interval=i,loss=float(loss.detach()),regularization=float(regularization.detach()),**detail)
            if step%40==0 or step==args.steps:
                value=validate(renderer,sup,validation,use_event);entry['validation']=value
                if value<best_loss:best_loss=value;best_step=step;best=weights(adapter)
                print(adapter.metadata['scene'],args.method,args.seed,'step',step,'validation',value,'best',best_step,flush=True)
            logs.append(entry)
            if step%40==0:save_json(output/'training.json',logs)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    restore(adapter,best)
    torch.save(dict(weights={k:{name:value.cpu() for name,value in state.items()} for k,state in best.items()},
                    config=config,best_step=best_step),output/'checkpoint.pt')
    save_json(output/'training.json',logs)
    save_json(output/'runtime.json',dict(training_wall_seconds=elapsed,best_step=best_step,
        validation_objective=best_loss,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256=sha256(output/'checkpoint.pt'),includes_validation=True))
    result=evaluate(renderer,sup,save_to=output)
    print('Completed',adapter.metadata['scene'],args.method,args.seed,
          {k:v['value'] for k,v in result['metrics'].items()},flush=True)
