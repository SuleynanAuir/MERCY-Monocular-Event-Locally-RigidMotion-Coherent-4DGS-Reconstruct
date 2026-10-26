"""Check result provenance, disjoint splits and denominators before reporting."""
import json
import math
from pathlib import Path
from .protocol import ROOT,save_json,sha256


def audit(root=ROOT):
    root=Path(root);records=[];identities={};sampling={};checked_scenes=set()
    for section in ('01_main_comparison','02_ablation'):
        for path in sorted((root/section/'runs').glob('*/evaluation.json')):
            directory=path.parent
            result=json.loads(path.read_text());config=json.loads((directory/'config.json').read_text())
            runtime=json.loads((directory/'runtime.json').read_text());training=json.loads((directory/'training.json').read_text())
            prepared=root/'01_main_comparison/prepared'/result['scene']
            for key,digest in config['inputs'].items():
                if sha256(prepared/key)!=digest:raise ValueError(f'Input changed: {directory.name}/{key}')
            source=json.loads((prepared/'prior.json').read_text());sup=json.loads((prepared/'supervision/metadata.json').read_text())
            if result['scene'] not in checked_scenes:
                checked_scenes.add(result['scene'])
                split=sup['split']['frames']
                flattened=[i for ids in split.values() for i in ids]
                if len(flattened)!=len(set(flattened)):raise ValueError('Overlapping temporal partitions')
                if sup['calibration_used_test']:raise ValueError('Event calibration used test data')
                train_frames=set(split['train'])
                for i in sup.get('calibration_ids',[]):
                    if any(i+delta not in train_frames or i+delta+1 not in train_frames for delta in (-1,0,1)):
                        raise ValueError('Event calibration search crossed training boundary')
                for filename,key in [('rgb.npy','rgb_array_sha256'),('events.npz','event_counts_sha256')]:
                    array=prepared/'supervision'/filename
                    if array.exists() and sha256(array)!=sup[key]:raise ValueError('Prepared supervision checksum mismatch')
            if (prepared/'supervision/rgb.npy').exists():
                from .background import training_background
                expected_background=training_background(prepared)
            else:
                expected_background=json.loads((root/'01_main_comparison/background_audit.json').read_text())['scenes'][result['scene']]
            if 'background' in config:
                if config['background']!=expected_background:raise ValueError('Training background identity mismatch')
            elif expected_background['rgb']!=[0.,0.,0.]:
                raise ValueError('Legacy black-background run is invalid for this scene')
            current=(source['source_sha256'],source['motion_sha256'],tuple(sorted(config['inputs'].items())))
            if result['scene'] in identities and identities[result['scene']]!=current:
                raise ValueError('Unequal method input identities')
            identities[result['scene']]=current
            if sha256(directory/'checkpoint.pt')!=runtime['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
            if [r['frame'] for r in result['frame_metrics']]!=sup['split']['frames']['test']:
                raise ValueError('Test RGB frame population mismatch')
            expected=sup.get('event_intervals',{}).get('test',[])
            if [r['interval'] for r in result['event_metrics']]!=expected:raise ValueError('Test event population mismatch')
            if expected:
                metric=result['metrics']['event_pass'];passed=sum(r['gate']['passed'] for r in result['event_metrics'])
                if metric['passed']!=passed or metric['total']!=len(expected) or metric['value']!=passed/len(expected):
                    raise ValueError('Invalid passing interval denominator')
            if config['method']!='C1':
                updates=[r for r in training if 'interval' in r]
                if len(updates)!=config['steps']:raise ValueError('Training budget mismatch')
                train=sup['split']['intervals']['train']
                if any(r['interval'] not in train for r in updates):raise ValueError('Training crossed split boundary')
                sample_key=(result['scene'],config['seed'])
                intervals=[r['interval'] for r in updates]
                if sample_key in sampling and sampling[sample_key]!=intervals:
                    raise ValueError('Methods or ablations used unmatched training sample order')
                sampling[sample_key]=intervals
            validation=[r for r in training if 'validation' in r]
            best=min(validation,key=lambda r:r['validation'])
            if best['step']!=runtime['best_step']:raise ValueError('Checkpoint not selected by frozen validation rule')
            visible=result['metrics']['visible_missing'];rows=result['frame_metrics']
            for key in ('rgb_l1','psnr','ssim'):
                values=[row[key] for row in rows]
                if all(value is not None for value in values):
                    expected_mean=sum(values)/len(values)
                    if not math.isclose(result['metrics'][key]['value'],expected_mean,rel_tol=1e-12,abs_tol=1e-12):
                        raise ValueError('Image aggregate differs from saved frame measurements')
            observable=[r['metrics']['relative_l1'] for r in result['event_metrics'] if r['metrics']['observable']]
            if observable and not math.isclose(result['metrics']['event_relative_l1']['value'],sum(observable)/len(observable),rel_tol=1e-12):
                raise ValueError('Event aggregate differs from saved interval measurements')
            if (visible['numerator']!=sum(r['missing']['numerator'] for r in rows)
                    or visible['denominator']!=sum(r['missing']['denominator'] for r in rows)):
                raise ValueError('Coverage denominator mismatch')
            records.append(dict(run=directory.name,section=section,status='passed',
                checkpoint_sha256=runtime['checkpoint_sha256'],test_frames=len(rows),test_intervals=len(expected)))
    save_json(root/'result_audit.json',dict(runs=records,checked_runs=len(records),
        checks=['same G0/motion/supervision per scene','checkpoint identities','training budget and split',
                'calibration search confined to training','prepared array checksums where available',
                'matched method/ablation sampling order','validation-only selection',
                'complete test frame/interval populations','per-frame and per-interval aggregation','event and coverage denominators']))
    print('Result audit passed:',len(records),'runs',flush=True)
    return records


if __name__=='__main__':audit()
