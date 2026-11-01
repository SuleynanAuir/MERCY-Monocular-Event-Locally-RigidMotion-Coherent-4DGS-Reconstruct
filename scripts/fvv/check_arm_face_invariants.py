"""Check appearance-only refinement leaves complete geometry/base/bucket intact."""
import argparse,json
from pathlib import Path
import numpy as np
from common import digest
from rigid_bucket import read_arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','model','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();old=read_arrays(a.previous/'rigid-bucket.npz',2);new=read_arrays(a.model/'rigid-bucket.npz',2);arm=old['point_groups']>0;checks={}
    for key,value in old.items():
        checks[key]=bool(np.array_equal(value[:,~arm],new[key][:,~arm])) if key=='temporal_dc' else bool(np.array_equal(value[~arm],new[key][~arm])) if key in ['dc','rest','opacity'] else bool(np.array_equal(value,new[key]))
    checks['core_checkpoint']=digest(a.previous/'model.pth')==digest(a.model/'model.pth');checks['config']=digest(a.previous/'effective.yaml')==digest(a.model/'effective.yaml')
    if not all(checks.values()):raise AssertionError('Frozen original data changed')
    a.output.write_text(json.dumps(dict(checks=checks,model_sha256=digest(a.model/'model.pth'),sidecar_sha256=digest(a.model/'rigid-bucket.npz'),scope='Only arm appearance may differ; complete geometry, trajectories, bucket appearance, base checkpoint and config exactly preserved.'),indent=2))


if __name__=='__main__':main()
