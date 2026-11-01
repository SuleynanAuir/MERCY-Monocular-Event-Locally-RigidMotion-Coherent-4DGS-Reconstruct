"""Build an untrained, bounded Hybrid candidate from explicit source surfaces.

The checkpoint supplies schema/configuration. An optional static appearance
warm start requires exactly matching static geometry. Fit and evaluate before
publishing; this is known-mesh assistance, not image-only reconstruction.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import hybrid
from common import digest,usage,model_timestamp
from seed_covariance import factor_surface_motion,static_surface_rotation


def run(a):
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.source,allow_pickle=False) as data:
        arrays={key:data[key] for key in ['points','normals','velocities','times','static_points','static_normals']}
        source_metadata=json.loads(str(data['metadata_json']))
    dynamic=len(arrays['points']);static=len(arrays['static_points'])
    if not 1<=dynamic<=1048576 or not 1<=static<=98304 or dynamic+static>1200000:raise ValueError('Source model point budget exceeded')
    for key,value in arrays.items():
        count=static if key.startswith('static') else dynamic
        if value.shape!=(count,1 if key=='times' else 3) or not np.isfinite(value).all():raise ValueError('Invalid '+key)
    meta=json.loads((a.template/'manifest.json').read_text())
    if arrays['times'].min()<meta['time_duration'][0] or arrays['times'].max()>meta['time_duration'][1]+1e-5:raise ValueError('Source time outside model interval')
    state,_=hybrid.load_checkpoint(a.template/'model.pth','cpu');state=list(state)
    tangent,thickness=(.004,.0008) if dynamic>262144 else (.006,.0012)
    scales,left,right=factor_surface_motion(arrays['velocities'],arrays['normals'],tangent=tangent,thickness=thickness,temporal=.065)
    def features(count,index,dc):return np.full((count,*state[index].shape[1:]),(.55-.5)/.28209479177387814 if dc else 0,np.float32)
    replacements={1:arrays['points'],2:features(dynamic,2,True),3:features(dynamic,3,False),4:scales[:,:3],5:left,6:np.full((dynamic,1),np.log(.6/.4),np.float32),
        13:arrays['times'],14:scales[:,3:],15:right,19:arrays['static_points'],20:features(static,20,True),21:features(static,21,False),
        22:np.tile(np.log(np.array([.004,.004,.0008],np.float32)),(static,1)),23:static_surface_rotation(arrays['static_normals']),24:np.full((static,1),np.log(.6/.4),np.float32)}
    for key,value in replacements.items():state[key]=torch.from_numpy(np.ascontiguousarray(value)).to(dtype=state[key].dtype)
    static_parent=None
    if a.static_appearance_from:
        learned,_=hybrid.load_checkpoint(a.static_appearance_from/'model.pth','cpu')
        for key in [19,22,23]:
            if not torch.equal(state[key],learned[key]):raise ValueError('Static appearance warm start requires identical geometry')
        for key in [20,21,24]:state[key]=learned[key].detach().clone()
        static_parent=digest(a.static_appearance_from/'model.pth')
    hybrid.save_bounded(hybrid.compact((tuple(state),0)),a.output/'model.pth',512*1024**2)
    (a.output/'effective.yaml').write_bytes((a.template/'effective.yaml').read_bytes())
    for key in ['refinement','surface_repair','seed_initialization']:meta.pop(key,None)
    meta.update(sha256=digest(a.output/'model.pth'),bytes=usage(a.output),training_iteration=0,trained_at=None,
        source_surface_initialization=dict(template_sha256=json.loads((a.template/'manifest.json').read_text())['sha256'],source_sha256=digest(a.source),initialized_at=model_timestamp(),
            requires_training=True,dynamic_points=dynamic,static_points=static,moving_parts=source_metadata.get('moving_parts'),source='Explicit supplied animated Lego mesh; production initialization, not RGB-only reconstruction',
            static_appearance_sha256=static_parent,point_budget=1200000,dynamic_tangent_sigma=tangent,dynamic_normal_sigma=thickness,temporal_sigma=.065,static_tangent_sigma=.004,static_normal_sigma=.0008,initial_opacity=.6))
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta['source_surface_initialization'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['template','source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--static-appearance-from',type=Path)
    run(p.parse_args())
