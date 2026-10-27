"""Isolated comparison worker, with durable progress and compact reusable outputs."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import tempfile
import time
import zipfile

from experiment.export_report_analysis import save, sha
from scripts.evaluation.evaluate_experiment import evaluate, verify
from utils.comparison_io import extract, locate_manifest
from utils.comparison_outputs import allocate, identity, archive_name, name_figures

SCHEMA='comparison-v2-grouped-parts'


def validate(manifest):
    samples=manifest.get('samples',[]);versions=manifest.get('versions',[])
    if not 1<=len(samples)<=5000 or not 2<=len(versions)<=6:raise ValueError('需要 1–5000 帧、2–6 个版本')
    ids=[v['id'] for v in versions]
    if len(set(ids))!=len(ids) or any(not isinstance(s,str) or not s.isascii() or not s.isdigit() for s in ids):raise ValueError('版本 ID 无效或重复')
    times=[s['time_seconds'] for s in samples]
    import numpy as np
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('采样时间必须有限且严格递增')
    files=manifest['files']
    for sid in ids:
        for i,s in enumerate(samples):
            if f'versions/{sid}/frames/{i:06d}.png' not in files:raise ValueError('渲染帧未记录在哈希清单')
    for r in manifest.get('reference',{}).get('samples',[]):
        for key in ('rgb','foreground_mask','part_mask','events'):
            if key in r and r[key] not in files:raise ValueError('参考数据未记录在哈希清单')
    timing=manifest.get('performance',{}).get('timings','timings.json')
    if timing not in files:raise ValueError('运行计时未记录在清单')


def extras(root,output,a):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from experiment.comparison_figures import COLORS
    from experiment.export_report_analysis import METRICS
    with np.load(output/'data/series.npz',allow_pickle=False) as z:z=dict(z)
    names=[r['version'] for r in a['summary']];m=json.loads((root/'manifest.json').read_text());n=len(m['samples'])
    from experiment.comparison_figures import temporal
    temporal(output,a)
    selected=sorted({0,n//2,n-1});fig,axes=plt.subplots(len(selected),len(names)+1,squeeze=False,figsize=(4*(len(names)+1),3.4*len(selected)+1))
    for row,i in enumerate(selected):
        ref=m.get('reference',{}).get('samples',[]);r=ref[i] if i<len(ref) else {};paths=[root/r['rgb'] if r.get('aligned') and r.get('rgb') else None]+[root/f'versions/{sid}/frames/{i:06d}.png' for sid in a['protocol']['source_version_ids']]
        for col,path in enumerate(paths):
            ax=axes[row,col]
            if path:ax.imshow(np.asarray(Image.open(path).convert('RGB')))
            else:ax.text(.5,.5,'No aligned RGB',ha='center',va='center')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(['RGB reference',*names][col])
            if col==0:ax.set_ylabel(f'Frame {m["samples"][i]["frame"]}\n{z["times"][i]:.3f} s')
    fig.suptitle('04 / Same-time image comparison · first / middle / last',x=.05,ha='left',fontsize=17,weight='bold');fig.tight_layout(rect=(0,.02,1,.94));fig.savefig(output/'figures/04_qualitative.png',dpi=160);plt.close(fig)
    return ['02_temporal','04_qualitative']


def run(source,output,progress_path):
    output=Path(output).resolve();source=Path(source).resolve();output.mkdir(parents=True,exist_ok=True);start=time.monotonic();last=0
    def progress(stage,percent,done=0,total=1,detail=''):
        nonlocal last
        last=max(last,float(percent));record=dict(status='running',worker_pid=os.getpid(),stage=stage,percent=round(last,2),completed=done,total=total,detail=str(detail),elapsed_seconds=round(time.monotonic()-start,1))
        tmp=progress_path.with_suffix('.tmp');save(tmp,record);tmp.replace(progress_path)
    warnings=[]
    # A global advisory lock prevents three UI ports from launching three CPU jobs.
    import fcntl
    lock=(Path(tempfile.gettempdir())/('scgs-comparison-'+hashlib.sha256(str(Path(__file__).resolve().parents[1]).encode()).hexdigest()[:12]+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('另一个对照试验正在计算，请等待完成')
    with tempfile.TemporaryDirectory(prefix='scgs-comparison-') as temporary:
        progress('import',1,detail='读取实验数据；ZIP 使用临时解压，结束后清理')
        root=extract(source,Path(temporary)/'source',lambda d,t,s:progress('import',1+4*d/t,d,t,s)) if source.is_file() else locate_manifest(source)
        manifest=json.loads((root/'manifest.json').read_text());validate(manifest)
        if any(s['render_mode']!='gaussian' for s in manifest['samples']):warnings.append('包含非 Gaussian 显示模式：图像分数仅为显示外观差异，不是原生 RGB 重建质量')
        verify(root,manifest,lambda d,t,s:progress('verify',5+5*d/max(t,1),d,t,s))
        data=output/'data';data.mkdir(exist_ok=True);fingerprint=sha(root/'manifest.json')
        cached=Path(__file__).resolve().parent/'output';base=None;lp=None;cache_provenance={}
        # Only the repository's previous report, whose artifact hashes verify, is trusted as a cache.
        index=cached/'artifact_manifest.json'
        if index.exists():
            record=json.loads(index.read_text())
            if record.get('source_manifest_sha256')==fingerprint:
                cache_provenance={'artifact_manifest_sha256':sha(index),'source_manifest_sha256':fingerprint,'accepted_files':{}}
                for key in ('data/base_metrics.json.gz','data/lpips.json'):
                    path=cached/key;expected=record.get('files',{}).get(key,{})
                    if path.is_file() and expected.get('sha256')==sha(path):
                        cache_provenance['accepted_files'][key]=expected['sha256']
                        if key.endswith('.gz'):
                            with gzip.open(path,'rt') as f:base=json.load(f)
                        else:lp=json.loads(path.read_text())
        if base is not None and base.get('manifest_sha256')!=fingerprint:base=None
        if lp is not None and lp.get('manifest_sha256')!=fingerprint:lp=None
        if base is None:
            def update(stage,d,t,detail):
                if stage=='metrics':progress('metrics',10+48*d/max(t,1),d,t,detail)
                elif stage=='motion':progress('metrics',last,d,t,'运动与刚体约束 · '+detail)
            base=evaluate(root,progress=update,verified=True)
        else:progress('metrics',58,detail='源文件与已有报告哈希已验证，复用逐帧／逐部件分数')
        with gzip.open(data/'base_metrics.json.gz','wt') as f:json.dump(base,f,allow_nan=False)
        if lp is None:
            progress('lpips',60,detail='初始化 LPIPS；官方权重首次使用会下载并共用缓存')
            refs=manifest.get('reference',{}).get('samples',[])
            if len(refs)==len(manifest['samples']) and all(r.get('aligned') and r.get('rgb') and r.get('background_matches',True) for r in refs):
                try:
                    from experiment.lpips_export import run as perceptual
                    previous_timeout=socket.getdefaulttimeout();socket.setdefaulttimeout(30)
                    try:perceptual(root,data/'lpips.json',Path.home()/'.cache/scgs-comparison/weights',lambda d,t:progress('lpips',60+15*d/t,d,t,'LPIPS 原始分辨率'))
                    finally:socket.setdefaulttimeout(previous_timeout)
                    lp=json.loads((data/'lpips.json').read_text())
                except Exception as error:warnings.append('LPIPS 不可用：'+str(error))
            else:warnings.append('LPIPS 不可用：缺少全部时刻对齐的 RGB 真值')
        elif lp:save(data/'lpips.json',lp);progress('lpips',75,detail='复用已验证的 LPIPS 分数与权重指纹')
        progress('analysis',77,detail='配对统计、部件分布、运动约束与性能分析')
        from experiment.comparison_analysis import build
        a=build(root,output,base,lp)
        if manifest.get('reference',{}).get('mask_quality')=='rgb_background_proxy':
            warnings.append('前景 PSNR / IoU 使用原始 RGB 的固定背景分割参考，属于光度前景代理掩码，不是人工或几何真值；阈值与定义随实验包保存。')
        progress('figures',82,detail='开始生成 PNG 图谱')
        from experiment.comparison_figures import generate
        figures=generate(root,output,lambda d,t,s:progress('figures',82+12*d/t,d,t,s));figures+=extras(root,output,a)
        name_figures(output,figures)
        progress('report',96,detail='生成交互报告、CSV、协议和哈希清单')
        from experiment.comparison_report import write
        write(output,a,figures,warnings)
        sources=['experiment/comparison.py','experiment/comparison_analysis.py','experiment/comparison_report.py','experiment/comparison_explorer.py','experiment/comparison_figures.py','experiment/export_report_analysis.py','experiment/lpips_export.py','scripts/evaluation/evaluate_experiment.py','utils/experiment_format.py','utils/mesh_motion.py','utils/version_presentation.py','utils/comparison_io.py','utils/comparison_outputs.py']
        repo=Path(__file__).resolve().parents[1]
        for name in sources:
            target=output/'reproduce/source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
        packages={}
        for name in ('numpy','scipy','Pillow','matplotlib','torch','torchvision','lpips'):
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        save(data/'environment.json',dict(schema=SCHEMA,python=platform.python_version(),packages=packages,source_code_sha256={p:sha(repo/p) for p in sources},warnings=warnings,verified_cache_provenance=cache_provenance,elapsed_seconds=time.monotonic()-start))
        (output/'reproduce/commands.txt').write_text('从项目根目录运行，输入原始实验文件夹或 ZIP；只输出比较结果，不运行训练或消融。\npython -m experiment.comparison --input /path/to/export --object-name ITEM --output experiment/output\n物品名和北京时间戳用于目录、PNG 和 ZIP 命名；每次运行创建独立目录。\n当前运行依赖及源代码哈希见 data/environment.json；原始导出不会被复制到报告中。\n')
        bundle=archive_name(output)
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','results.zip',bundle,'progress.json','worker.log')}
        save(output/'artifact_manifest.json',dict(source_manifest_sha256=fingerprint,files=files,total_bytes=sum(r['bytes'] for r in files.values())))
        with zipfile.ZipFile(output/bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
            for name in [*files,'artifact_manifest.json']:archive.write(output/name,name)
        save(progress_path,dict(status='complete',stage='complete',percent=100,completed=1,total=1,elapsed_seconds=round(time.monotonic()-start,1),detail='对照试验计算与可视化完成',warnings=warnings,summary=a['summary'],output_bytes=(output/bundle).stat().st_size))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',type=Path,default=Path('experiment/output'));p.add_argument('--object-name');p.add_argument('--progress',type=Path);args=p.parse_args()
    if args.object_name:args.output,_=allocate(args.output,args.object_name)
    elif not identity(args.output):p.error('请用 --object-name 指定物品名；结果自动保存到 --output/物品名_时间戳/')
    args.output.mkdir(parents=True,exist_ok=True);progress=args.progress or args.output/'progress.json'
    try:run(args.input,args.output,progress)
    except Exception as error:
        old=json.loads(progress.read_text()) if progress.exists() else {}
        save(progress,dict(old,status='failed',detail=str(error)))
        raise

if __name__=='__main__':main()
