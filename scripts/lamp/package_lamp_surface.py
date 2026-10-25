"""Package an approved Lamp surface candidate using the existing Blender motion."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
from scripts.lamp.prepare_lamp_viewer import bake, rigid_deltas
from scripts.motion.bind_lego_parts import transform
from utils.gaussian_pair_renderer import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','parts','binding-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=json.loads(args.baseline.read_text());report=json.loads((args.candidate/'report.json').read_text())
    binding=json.loads(args.binding_report.read_text())
    if (report['parts_sha256']!=file_sha(args.parts) or binding['parts_sha256']!=report['parts_sha256']
            or report['original_binding_sha256']!=file_sha(args.binding_report)
            or report['surface_sha256']!=file_sha(args.candidate/'surface.ply')
            or report['binding_sha256']!=file_sha(args.candidate/'binding.npz')):
        raise ValueError('Surface candidate or motion identity differs')
    paths=[(args.baseline.parent/config[k]['path']).resolve() for k in ('left','right')]
    for k,path in zip(('left','right'),paths):
        if config[k]['sha256']!=file_sha(path):raise ValueError('Baseline sequence checksum mismatch')
    with np.load(paths[0],allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']));base=z['base_vertices']
        xyz=np.column_stack([base[k] for k in ('x','y','z')])
    with np.load(args.parts,allow_pickle=False) as z:
        registration=np.array(binding['registration'])
        surface=transform(z['rest_points'],registration);parts=z['point_parts']
        delta=rigid_deltas(z['object_matrices'],registration)
    with np.load(args.candidate/'binding.npz',allow_pickle=False) as z:
        labels,parents=z['groups'],z['reference_parents']
    reference_groups=parts[cKDTree(surface).query(xyz,workers=4)[1]]
    if (parents.dtype.kind not in 'iu' or parents.shape!=labels.shape
            or len(parents)!=report['report']['output_count'] or parents.min()<0 or parents.max()>=len(base)
            or not np.array_equal(labels,reference_groups[parents])):
        raise ValueError('Refinement children cross their original rigid parts')
    args.output.mkdir(parents=True)
    target=args.output/'right.npz'
    provenance=dict(meta['provenance'],surface_refinement=report)
    stats=bake(args.candidate/'surface.ply',target,surface,parts,delta,meta['preview_camera'],
        dict(frame_count=meta['frame_count'],fps=meta['fps']),provenance,bound_labels=labels)
    config['left']['path']=str(paths[0])
    config['right']=dict(path=target.name,sha256=file_sha(target),label='V3 · 精细表面优化',render_supersample=2)
    config['material_reference']=dict(path=str(paths[1]),sha256=config.get('material_reference',{}).get('sha256',file_sha(paths[1])))
    config['surface_refinement']=dict(report['report'],render_supersample=2,source_report=str(args.candidate/'report.json'))
    config['description']='V3 以原始 GS 表面为几何约束，保留可靠的修正版外观细节，细分表面并改善抗锯齿；Blender 运动不变。属于几何先验优化，未重新训练事件模型。'
    (args.output/'manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (args.output/'package-report.json').write_text(json.dumps(dict(binding=stats,surface=report),indent=2)+'\n')
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
