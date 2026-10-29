"""Real CUDA smoke test with an UNTRAINED synthetic Hybrid fixture, not scene QA."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import hybrid
from common import digest, usage


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve()
    torch.manual_seed(1234)
    n = args.points // 2
    def tensor(*shape):
        return torch.zeros(*shape)
    def rotation():
        value = tensor(n, 4); value[:, 0] = 1
        return value
    xyz = torch.rand(n, 3) * 1.2 - .6
    static = xyz.clone(); static[:, 1] += .8
    left = rotation(); left[:, 0] = math.cos(.25); left[:, 1] = math.sin(.25)
    state = (0, xyz, torch.rand(n, 1, 3)*3-1.5, tensor(n, 0, 3),
             torch.full((n,3), math.log(.008)), left, torch.full((n,1), 1.),
             tensor(n), tensor(n,1), tensor(n,1), tensor(n,1), {}, 1.,
             torch.full((n,1), .5), torch.full((n,1), math.log(.3)), rotation(), True, None, 0,
             static, torch.rand(n,1,3)*3-1.5, tensor(n,0,3),
             torch.full((n,3), math.log(.008)), rotation(), torch.full((n,1), 1.),
             tensor(n), tensor(n,1), tensor(n,1))
    torch.save((state, 0), root/'source.pth')
    hybrid.save_bounded(hybrid.compact((state, 0)), root/'model.pth', 512*1024**2)
    cfg = OmegaConf.create(dict(gaussian_dim=4, time_duration=[0.,1.], rot_4d=True,
        force_sh_3d=True, ModelParams=dict(sh_degree=0, frame_ratio=1, white_background=False),
        PipelineParams=dict(eval_shfs_4d=False, env_map_res=0)))
    OmegaConf.save(cfg, root/'effective.yaml')
    manifest_args = SimpleNamespace(repo=repo, scene='synthetic-smoke', position=[0.,0.,-3.],
                                    yaw=0., pitch=0., budget_gib=.5)
    settings = SimpleNamespace(frame_ratio=1, time_duration=[0.,1.])
    hybrid.manifest(manifest_args, settings, root)
    start = time.monotonic()
    renderer = hybrid.Renderer(SimpleNamespace(repo=repo, model=root))
    load_seconds = time.monotonic()-start
    pose = dict(position=[0.,0.,-3.], yaw=0., pitch=0., time=.5, width=960, height=640, fov=60.)
    baseline = renderer.frame(pose, raw=True)
    temporal = renderer.frame({**pose, 'time':.1}, raw=True)
    novel_view = renderer.frame({**pose, 'yaw':12., 'position':[.15,0.,-3.]}, raw=True)
    assert torch.isfinite(baseline).all() and baseline.max() > .01
    temporal_error = float((baseline-temporal).abs().mean())
    camera_error = float((baseline-novel_view).abs().mean())
    assert temporal_error > 1e-5 and camera_error > 1e-5
    torch.cuda.synchronize()
    report = dict(test_asset='UNTRAINED synthetic fixture; NOT Lego/Fan or quality/training acceptance',
        points=2*n, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        revision=hybrid.revision(repo), load_seconds=load_seconds, finite_render=True,
        temporal_mean_abs_difference=temporal_error, camera_mean_abs_difference=camera_error,
        model_bytes=(root/'model.pth').stat().st_size,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), artifact_bytes=usage(root))
    (root/'smoke.json').write_text(json.dumps(report,indent=2))
    (root/'preview.jpg').write_bytes(renderer.frame(pose))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--points',type=int,default=40000)
    args=p.parse_args()
    if not 2 <= args.points <= 1000000:p.error('points must be in [2,1000000]')
    run(args)
