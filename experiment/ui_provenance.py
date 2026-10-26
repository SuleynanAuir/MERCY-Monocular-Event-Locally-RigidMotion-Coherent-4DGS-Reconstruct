"""Snapshot code and version identities for the rebuilt UI experiment."""
import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from .protocol import save_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    args=p.parse_args();project=Path(__file__).resolve().parents[1]
    files=[project/'viewer.py']
    for folder in ('experiment','utils','scripts','scene','gaussian_renderer','viewer'):
        files.extend(f for f in (project/folder).rglob('*') if f.suffix in ('.py','.js','.html','.css') and '__pycache__' not in f.parts)
    packages={}
    for key in ('torch','numpy','scipy','scikit-image','Pillow','aiohttp','nvdiffrast','matplotlib','plyfile'):
        try:packages[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:packages[key]=None
    stages=[]
    for scene in ('lego','fan','lamp','metronome','microphone'):
        stage=dict(scene=scene,versions={})
        for v in ('v2','v3'):
            d=args.root/scene/(v+'-training')
            entry={}
            for name in ('config.json','checkpoint.json','history.json','evaluation.json'):
                f=d/name
                if f.exists():entry[name]=dict(path=str(f),sha256=sha256(f))
            checkpoint=d/'checkpoint.pt'
            if checkpoint.exists():
                import torch
                ck=torch.load(checkpoint,map_location='cpu')
                entry['checkpoint']=dict(sha256=sha256(checkpoint),best_step=ck.get('best_step'),config=ck.get('config'))
            stage['versions'][v]=entry
        stages.append(stage)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    save_json(args.root/'reproducibility.json',dict(format='rebuilt_ui_code_and_stage_snapshot_v1',
        code_sha256={str(f.relative_to(project)):sha256(f) for f in sorted(set(files))},packages=packages,gpu=gpu,stages=stages,
        input_inventory_sha256=sha256(args.root/'input-inventory.json'),
        comparison='Actual UI assets reconstructed from data; not historical full V3 checkpoint recovery'))


if __name__=='__main__':main()
