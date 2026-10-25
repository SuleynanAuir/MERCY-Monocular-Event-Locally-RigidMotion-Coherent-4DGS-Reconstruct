"""Create a new V3 Gaussian surface constrained by the supplied Blender geometry."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData,PlyElement
from utils.gaussian_pair_renderer import file_sha
from utils.object_surface import resample_surface
from scripts.lamp.prepare_lamp_viewer import bake,rigid_deltas
from scripts.motion.bind_lego_parts import transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('surface','baseline','parts','binding-report','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--spacing',type=float,default=.0015)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=json.loads(args.baseline.read_text());binding=json.loads(args.binding_report.read_text())
    with np.load(args.surface,allow_pickle=False) as z:
        surface={k:z[k] for k in z.files};identity=json.loads(str(z['metadata_json']))
    if (identity['format']!='object_surface_prior_v1' or not identity['autoexec_disabled']
            or identity['parts_sha256']!=file_sha(args.parts)
            or identity['binding_sha256']!=file_sha(args.binding_report)
            or binding['parts_sha256']!=identity['parts_sha256']):
        raise ValueError('Geometry prior and motion binding differ')
    for side in ('left','right'):
        path=(args.baseline.parent/cfg[side]['path']).resolve()
        if cfg[side]['sha256']!=file_sha(path):raise ValueError('Baseline sequence changed')
        cfg[side]['path']=str(path)
    original_right=dict(cfg['right'])
    with np.load(original_right['path'],allow_pickle=False) as z:
        original=z['base_vertices'];meta=json.loads(str(z['metadata_json']))
    args.output.mkdir(parents=True)
    vertices,labels,local_normals,report=resample_surface(surface['triangles'],surface['normals'],
        surface['groups'],surface['materials'],original,spacing=args.spacing)
    ply=args.output/'surface.ply';PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(ply)
    with np.load(args.parts,allow_pickle=False) as z:
        reg=np.array(binding['registration']);points=transform(z['rest_points'],reg)
        groups=z['point_parts'];delta=rigid_deltas(z['object_matrices'],reg)
    sequence=args.output/'right.npz'
    provenance=dict(meta['provenance'],source_surface_sha256=file_sha(args.surface),surface_refinement=report)
    stats=bake(ply,sequence,points,groups,delta,meta['preview_camera'],
        dict(frame_count=meta['frame_count'],fps=meta['fps']),provenance,bound_labels=labels,name=cfg['name'])
    sequence_sha=file_sha(sequence)
    with np.load(sequence,allow_pickle=False) as z:output_meta=json.loads(str(z['metadata_json']))
    normal_meta=dict(format='lamp_detail_normals_v1',sequence_sha256=sequence_sha,
        canonical_sha256=output_meta['base_vertices_sha256'],normal_count=len(vertices),
        source_surface_sha256=file_sha(args.surface),method='Exact source interpolated normals in canonical Gaussian frames')
    normal_path=args.output/'normals.npz'
    np.savez_compressed(normal_path,local_normals=local_normals,metadata_json=np.asarray(json.dumps(normal_meta)))
    np.savez_compressed(args.output/'binding.npz',groups=labels)
    cfg['right']=dict(path=sequence.name,sha256=sequence_sha,label='V3 · 表面约束优化',render_supersample=2)
    # Preserve V1's original shared color chart when replacing V3.
    if cfg.get('material_reference'):
        cfg['material_reference']['path']=str((args.baseline.parent/cfg['material_reference']['path']).resolve())
    else:cfg['material_reference']=dict(path=original_right['path'],sha256=original_right['sha256'])
    cfg['surface_refinement']=dict(report,source_surface=str(args.surface),binding=stats)
    cfg['description']='V3 使用已有 Blender 表面约束重采样高斯，按材质与平面整理修正版外观，清除运动残影与杂点；保留原始部件运动。属于几何先验增强，未新增事件训练。'
    (args.output/'manifest.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    detailed=dict(cfg,detail_finish=dict(style='normal_detail_v1',path=normal_path.name,sha256=file_sha(normal_path)))
    (args.output/'manifest-detail.json').write_text(json.dumps(detailed,ensure_ascii=False,indent=2)+'\n')
    report.update(source_surface=identity,source_surface_sha256=file_sha(args.surface),binding=stats)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
