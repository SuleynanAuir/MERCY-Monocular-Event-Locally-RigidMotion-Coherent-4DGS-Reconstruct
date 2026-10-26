"""Calibrate raw events on training windows only; keep fixed held-out windows."""
import json
from pathlib import Path
import cv2
import numpy as np
from utils.event_supervision import EventStream, log_intensity, fit_thresholds, signed_event_change, consistency_metrics
from .protocol import save_json, sha256


def pool(image, factor):
    h,w=image.shape[-2:]
    if h%factor or w%factor:raise ValueError('Invalid pooling factor')
    return image.reshape(*image.shape[:-2],h//factor,factor,w//factor,factor).mean(axis=(-3,-1))


def read_rgb(path):
    cap=cv2.VideoCapture(str(path))
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            yield cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:cap.release()


def prepare(scene, destination, factor=8):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    video=scene['video'];split=scene['split'];files=scene['files']
    rgb=np.asarray(list(read_rgb(files['rgb']['path'])))
    if rgb.shape!=(video['frame_count'],video['height'],video['width'],3):
        raise ValueError('Decoded video disagrees with frozen inventory')
    np.save(destination/'rgb.npy',rgb,allow_pickle=False)
    gray=np.asarray([cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb])
    metadata=dict(scene=scene['scene'],split=split,video=video,event_source=scene['events'],
        rgb_sha256=files['rgb']['sha256'],event_sha256=files['events']['sha256'],
        rgb_array_sha256=sha256(destination/'rgb.npy'),color_space='stored RGB values in [0,1]; no gamma conversion',
        event_pool_factor=factor,calibration=None,calibration_used_test=False,
        parent_exposure=scene['prior_exposure'])
    if not scene['events']['valid']:
        metadata['status']='rgb_only';metadata['reason']=scene['events']['reason']
        save_json(destination/'metadata.json',metadata)
        return metadata
    events=EventStream(files['events']['path'],video['width'],video['height'])
    # Offset search uses interior training windows; shifts cannot cross the
    # training/guard boundary. All event targets use the same selected mapping.
    train=split['intervals']['train'][2:-2]
    ids=np.unique(np.asarray(train)[np.linspace(0,len(train)-1,min(12,len(train))).astype(int)])
    trials=[]
    for offset in (-1,0,1):
        counts=np.asarray([pool(events.counts((i+offset)/video['fps'],(i+1+offset)/video['fps']),factor) for i in ids])
        for floor in (1/255,.01,.03):
            changes=np.asarray([pool(log_intensity(gray[i+1]/255,floor)-log_intensity(gray[i]/255,floor),factor) for i in ids])
            try:
                threshold=fit_thresholds(counts,changes)
                scores=[consistency_metrics(d,signed_event_change(c,threshold),c.sum(0),minimum_change=.01)
                        for d,c in zip(changes,counts)]
                observed=[s['cosine'] for s in scores if s['observable']]
                score=float(np.mean(observed)) if observed else -1.
                trials.append(dict(offset_frames=offset,log_floor=floor,thresholds=threshold.tolist(),cosine=score))
            except ValueError as error:
                trials.append(dict(offset_frames=offset,log_floor=floor,cosine=-1.,reason=str(error)))
    best=max(trials,key=lambda trial:trial['cosine'])
    metadata.update(calibration_ids=ids.tolist(),calibration_trials=trials)
    if best['cosine']<.5:
        metadata.update(status='rgb_only',reason='unreliable_training_event_calibration')
        save_json(destination/'metadata.json',metadata);return metadata
    # Offset windows near the end of a split are excluded if any observed
    # event time would lie outside that split's declared frame span.
    valid_intervals={}
    for label,frame_ids in split['frames'].items():
        lo,hi=frame_ids[0],frame_ids[-1];offset=best['offset_frames']
        valid_intervals[label]=[i for i in split['intervals'][label]
            if i+offset>=lo and i+1+offset<=hi]
    counts=[];reference=[]
    for i in range(video['frame_count']-1):
        offset=best['offset_frames'];c=pool(events.counts((i+offset)/video['fps'],(i+1+offset)/video['fps']),factor)
        observed=signed_event_change(c,best['thresholds'])
        delta=pool(log_intensity(gray[i+1]/255,best['log_floor'])-log_intensity(gray[i]/255,best['log_floor']),factor)
        counts.append(c);reference.append(dict(interval=i,**consistency_metrics(delta,observed,c.sum(0),tolerance=1,minimum_change=.01)))
    np.savez_compressed(destination/'events.npz',counts=np.asarray(counts,dtype=np.float32))
    metadata.update(status='ready',calibration=best,event_intervals=valid_intervals,
        event_counts_sha256=sha256(destination/'events.npz'),reference_self_check=reference)
    save_json(destination/'metadata.json',metadata)
    print('Prepared supervision',scene['scene'],'training calibration cosine',round(best['cosine'],4),flush=True)
    return metadata
