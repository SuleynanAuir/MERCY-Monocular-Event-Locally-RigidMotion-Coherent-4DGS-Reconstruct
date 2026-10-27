"""Human-readable comparison runs with stable API identifiers."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import unicodedata
import uuid

CST = timezone(timedelta(hours=8))


def object_name(value):
    if not isinstance(value, str):
        raise ValueError('请输入物品名称，例如 lego 或 吊机')
    name = re.sub(r'\s+', '_', unicodedata.normalize('NFKC', value).strip())
    if not name or len(name) > 40 or not all(c.isalnum() or c in '_-' for c in name) or not any(c.isalnum() for c in name):
        raise ValueError('物品名须为 1–40 个汉字、字母、数字、空格、下划线或短横线')
    return name


def allocate(root, name, job=None, now=None):
    name = object_name(name)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    instant = (now or datetime.now(CST)).astimezone(CST)
    for attempt in range(1000):
        created = instant + timedelta(microseconds=attempt)
        stamp = created.strftime('%Y%m%d_%H%M%S_%f')
        stem = f'{name}_{stamp}'
        target = root / stem
        try:
            target.mkdir()
        except FileExistsError:
            continue
        info = dict(id=job or uuid.uuid4().hex, object_name=name, timestamp=stamp,
                    created_at=created.isoformat(), timezone='Asia/Shanghai',
                    output_stem=stem, download_name=stem+'.zip')
        (target/'job.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        return target, info
    raise ValueError('无法分配唯一的输出目录，请重试')


def identity(output):
    path = Path(output)/'job.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def archive_name(output):
    return identity(output).get('download_name', 'results.zip')


def figure_name(output, logical_name):
    stem = identity(output).get('output_stem')
    return f'{stem}_{logical_name}.png' if stem else logical_name+'.png'


def name_figures(output, figures):
    output = Path(output)
    for name in figures:
        source = output/'figures'/(name+'.png')
        target = output/'figures'/figure_name(output, name)
        if source != target:
            source.replace(target)
