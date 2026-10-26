"""Reproducible CLI: python -m experiment.run --help."""
import argparse
import json
from pathlib import Path
from .protocol import ROOT,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit');audit.add_argument('--data',type=Path,required=True)
    audit.add_argument('--output',type=Path,default=ROOT/'01_main_comparison/input_inventory.json')
    audit.add_argument('--scenes',nargs='+',choices=['lego','fan','lamp','metronome','microphone'])
    prep=sub.add_parser('prepare');prep.add_argument('--inventory',type=Path,required=True)
    prep.add_argument('--blender',type=Path,required=True)
    prep.add_argument('--scenes',nargs='+',default=['lego','fan','lamp','metronome','microphone'])
    prep.add_argument('--output',type=Path,default=ROOT/'01_main_comparison/prepared')
    train=sub.add_parser('train');train.add_argument('--prepared',type=Path,required=True)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--method',choices=['C1','C2','C3'],required=True)
    train.add_argument('--seed',type=int,default=17);train.add_argument('--steps',type=int,default=200)
    train.add_argument('--variant',choices=['full','no_mlp','no_event','no_joint','no_jerk'],default='full')
    report=sub.add_parser('report');report.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    if args.command=='audit':
        from .datasets import inventory
        inventory(args.data,args.output,args.scenes)
    elif args.command=='prepare':
        from .prepare import prepare
        from .supervision import prepare as supervision
        data=json.loads(args.inventory.read_text())
        for scene in data['scenes']:
            if scene['scene'] not in args.scenes:continue
            dest=args.output/scene['scene']
            if not (dest/'supervision/metadata.json').exists():supervision(scene,dest/'supervision')
            if not (dest/'prior.json').exists():prepare(scene,dest,args.blender)
    elif args.command=='train':
        from .training import run
        run(args)
    else:
        from .report import generate
        generate(args.root)


if __name__=='__main__':main()
