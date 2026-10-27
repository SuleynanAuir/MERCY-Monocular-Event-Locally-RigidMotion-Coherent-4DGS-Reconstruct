"""Build new viewer catalogs from independently validated appearance checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.experiment_format import sha256, save_json


def promote(catalog, checkpoints, output):
    output.mkdir(parents=True, exist_ok=True)
    scenes=json.loads(catalog.read_text())
    for scene in scenes['scenes']:
        checkpoint=checkpoints/(scene['id']+'-appearance')/'appearance.npz'
        with np.load(checkpoint,allow_pickle=False) as archive:meta=json.loads(str(archive['metadata_json']))
        if (meta['validation']['foreground_mse']>=meta['baseline']['foreground_mse']
                or meta['validation']['iou']<meta['baseline']['iou']
                or (meta.get('perceptual_weight',0)>0 and meta['validation']['lpips']>=meta['baseline']['lpips'])):
            raise ValueError('Checkpoint failed predefined validation gate')
        middle=json.loads(Path(scene['middle_path']).read_text())
        if sha256(middle['path'])!=meta['parent_sha256']:raise ValueError('Checkpoint does not refine this V2')
        pair_path=Path(scene['path']);pair=json.loads(pair_path.read_text())
        def absolute_paths(record):
            if isinstance(record,dict):
                for key,value in record.items():
                    if key=='path' and isinstance(value,str):record[key]=str((pair_path.parent/value).resolve())
                    else:absolute_paths(value)
            elif isinstance(record,list):
                for item in record:absolute_paths(item)
        absolute_paths(pair)
        for key in ('detail_finish','surface_refinement','material_reference'):pair.pop(key,None)
        if pair['format']=='gaussian_pair_v1':
            pair['right']=dict(path=middle['path'],sha256=meta['parent_sha256'],label='V3 · 光度与覆盖校正')
        elif pair['source_fingerprints']['right']['sequence_sha256']!=meta['parent_sha256']:
            raise ValueError('Fan parent mismatch')
        pair['appearance_refinement']=dict(path=str(checkpoint.resolve()),sha256=sha256(checkpoint),
            training_report=str(checkpoint.with_name('training.json').resolve()))
        description='V1 原始阶段；V2 已提供修正/绑定修复；V3 在 V2 上校正颜色、透明度及有界尺度，中心与运动冻结。训练/验证时间帧分离；非事件训练或跨视角泛化结论。'
        pair['description']=description;middle['description']=description
        new_pair=output/(scene['id']+'-pair.json');new_middle=output/(scene['id']+'-middle.json')
        for p in (new_pair,new_middle):
            if p.exists():raise FileExistsError(p)
        save_json(new_pair,pair);save_json(new_middle,middle)
        scene.update(path=str(new_pair.resolve()),middle_path=str(new_middle.resolve()))
    target=output/catalog.name
    if target.exists():raise FileExistsError(target)
    save_json(target,scenes)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(promote(a.catalog,a.checkpoints,a.output))
