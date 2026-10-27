"""Bounded, traversal-safe comparison imports; never execute bundled code."""
from pathlib import Path, PurePosixPath
import shutil
import zipfile

MAX_BYTES=2*1024**3
MAX_FILES=30000


def relative_name(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or any(x in ('..','') for x in p.parts) or ':' in name or '\x00' in name:
        raise ValueError('无效的导入文件路径')
    return Path(*p.parts)


def locate_manifest(folder):
    folder=Path(folder)
    if (folder/'manifest.json').is_file():return folder
    choices=[p.parent for p in folder.glob('*/manifest.json')]
    if len(choices)!=1:raise ValueError('请选择包含一个 manifest.json 的实验文件夹或单个实验 ZIP')
    return choices[0]


def extract(archive, target, progress=None):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        if len(members)>MAX_FILES or sum(i.file_size for i in members)>MAX_BYTES:raise ValueError('ZIP 展开后超过 2 GiB 或文件数上限')
        seen=set();written=0
        for i,member in enumerate(members):
            relative=relative_name(member.filename)
            if relative in seen:raise ValueError('ZIP 包含重复文件名')
            seen.add(relative)
            if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP 不支持符号链接')
            dest=target/relative
            if member.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as source,dest.open('xb') as stream:
                while chunk:=source.read(1024**2):
                    written+=len(chunk)
                    if written>MAX_BYTES:raise ValueError('解压数据超过 2 GiB')
                    stream.write(chunk)
            if progress:progress(i+1,len(members),member.filename)
    return locate_manifest(target)
