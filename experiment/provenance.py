"""Capture executable source and package identities without environment secrets."""
import importlib
import importlib.metadata
import json
import platform
import subprocess
import tarfile
from .protocol import ROOT,save_json,sha256


def main():
    files=[]
    for directory in ('arguments','gaussian_renderer','scene','utils','scripts'):
        files.extend(p for p in (ROOT.parent/directory).rglob('*.py') if '__pycache__' not in p.parts)
    files.extend(ROOT.glob('*.py'));files.extend(ROOT.glob('*.txt'));files.extend(ROOT.glob('*.md'))
    bundle=ROOT/'source_bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for path in sorted(set(files)):archive.add(path,arcname=str(path.relative_to(ROOT.parent)),recursive=False)
    packages={}
    for name in ('torch','numpy','scipy','scikit-image','matplotlib','opencv-python','plyfile','Pillow'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='not found under this distribution name'
    extensions={}
    for name in ('diff_gaussian_rasterization._C','simple_knn._C'):
        module=importlib.import_module(name)
        extensions[name]=dict(path=module.__file__,sha256=sha256(module.__file__))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                 '--format=csv,noheader'],text=True).strip()
    import torch
    save_json(ROOT/'runtime_environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages=packages,gpu=gpu,torch_cuda=torch.version.cuda,extensions=extensions,
        source_bundle=dict(path=bundle.name,sha256=sha256(bundle),files={str(p.relative_to(ROOT.parent)):sha256(p) for p in files}),
        scope='Current reproducible runner plus shared Python sources. Original datasets and compiled CUDA extensions are not redistributed. Per-run code hashes remain authoritative for each executed revision.'))
    print('Captured runtime and source bundle',flush=True)


if __name__=='__main__':main()
