"""Auditable metronome reference repair; run from the server repository root."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

repo = Path.cwd()
sys.path.insert(0, str(repo))
from scripts.evaluation.prepare_experiment_reference import geometry_masks
from utils.event_supervision import EventStream
from utils.experiment_export import compact_counts
from utils.experiment_format import sha256, save_json
from utils.comparison_io import extract
from utils.comparison_outputs import allocate

work = repo/'outputs/metronome-event-repair'
old = Path('/root/autodl-tmp/scgs-evaluation-references-20261027-step2/metronome')
target = work/'reference'
data = repo/'data/metronome_data'

def prepare():
    fit = json.loads((data/'event-calibration.json').read_text())
    assert sha256(data/'metronome_event.npy') == fit['events_sha256']
    assert sha256(data/'metronome.mkv') == fit['video_sha256']
    assert fit['fps'] == 30
    shutil.copytree(old, target)
    r = json.loads((target/'reference.json').read_text())
    assert len(r['samples']) == 25
    assert all('part_mask' in s for s in r['samples'])
    c = fit['calibration']
    assert c['offset_frames'] == 0
    cal = dict(space='rgb_image',transfer='encoded',luminance='opencv_bt601_u8', log_floor=c['log_floor'],contrast_positive=c['thresholds'][0],contrast_negative=c['thresholds'][1],calibration_ids=fit['calibration_ids'], time_offset_seconds=0,coordinate_mapping='identity x/y pixels; timestamp seconds; interval (t0,t1]', source_calibration_sha256=sha256(data/'event-calibration.json'),provenance=fit['provenance'])
    r['event_calibration'] = cal
    stream = EventStream(data/'metronome_event.npy',640,640)
    assert abs(stream.times[-1]-49/30) < 2/30
    (target/'events').mkdir(exist_ok=True)
    for prev,row in zip(r['samples'],r['samples'][1:]):
        counts = stream.counts(prev['time_seconds'],row['time_seconds'])
        name = f"events/{row['frame']:06d}.npz"
        np.savez_compressed(target/name,positive=compact_counts(counts[0]),negative=compact_counts(counts[1]),t0=prev['time_seconds'],t1=row['time_seconds'])
        row['events'] = name
        row['event_split'] = 'overlaps_response_calibration' if any(prev['frame'] <= i < row['frame'] for i in cal['calibration_ids']) else 'response_validation'
    for key in ('events','event_response','part_mask'):r['unavailable'].pop(key,None)
    for key,filename in [('events','metronome_event.npy'),('calibration','event-calibration.json')]:
        r['sources'][key] = dict(name=filename,sha256=sha256(data/filename),bytes=(data/filename).stat().st_size)
    save_json(target/'reference.json',r)
    print('Prepared',len(r['samples']),'frames',len(r['part_labels']),'parts', flush=True)

def report():
    r=json.loads((target/'reference.json').read_text())
    root=extract(repo/'outputs/experiment-exports/metronome_20261103T155726Z_669a312d.zip',work/'repaired-export')
    m=json.loads((root/'manifest.json').read_text())
    parent=sha256(root/'manifest.json')
    assert parent == '12fe0684753b9d3ea787490dc52672f13497e98f6f1a170987465b79335fbe70'
    lookup={s['frame']:s for s in r['samples']}
    changed=[]
    for row in m['reference']['samples']:
        ref=lookup[row['frame']]
        assert row['aligned'] and row['background_matches'] and row['time_seconds']==ref['time_seconds']
        for key,suffix in [('part_mask','png'),('events','npz')]:
            if key in ref:
                name=f"reference/{row['sample']:06d}-{key}.{suffix}"
                shutil.copyfile(target/ref[key],root/name)
                row[key]=name;changed.append(name)
        row['event_calibration']=r['event_calibration']
        if 'event_split' in ref:row['event_split']=ref['event_split']
        row.pop('events_unavailable',None)
    for key in ('part_labels','part_mask_definition','part_provenance'):
        if key in r:m['reference'][key]=r[key]
    for key in ('events','event_response','part_mask'):
        m['reference'].get('unavailable',{}).pop(key,None)
    m['reference']['repair_provenance']=dict(parent_manifest_sha256=parent,source_reference_sha256=sha256(target/'reference.json'))
    save_json(root/'reference/source.json',r)
    shutil.copyfile(data/'event-calibration.json',root/'reference/event-calibration.json')
    changed += ['reference/source.json','reference/event-calibration.json']
    availability_path = root/'data_availability.json'
    availability = json.loads(availability_path.read_text())
    for key in ('part_mask','events'):
        rows = m['reference']['samples'][1:] if key == 'events' else m['reference']['samples']
        availability['reference'][key] = dict(saved=sum(key in s for s in rows),expected=len(rows),aligned=sum(key in s and s['aligned'] for s in rows))
    availability['missing_data'] = [v for v in availability['missing_data'] if not v.startswith(('固定部件掩码：','原始事件计数：'))]
    save_json(availability_path,availability)
    changed.append('data_availability.json')
    for name in changed:m['files'][name]=dict(bytes=(root/name).stat().st_size,sha256=sha256(root/name))
    save_json(root/'manifest.json',m)
    out,info=allocate(repo/'experiment/output','metronome_events_parts_fixed')
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR='/tmp/scgs-comparison-mpl')
    with (out/'worker.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','experiment.comparison','--input',str(root),'--output',str(out)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job=dict(output=str(out),source=str(root),id=info['id'],pid=proc.pid)
    save_json(Path('/tmp/metronome-events-repair-job.json'),job)
    print(json.dumps(job),flush=True)

if __name__ == '__main__':
    {'prepare':prepare,'report':report}[sys.argv[1]]()
