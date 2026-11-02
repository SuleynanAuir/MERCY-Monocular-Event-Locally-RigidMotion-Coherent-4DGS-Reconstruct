"""Archive source bytes and exact JSON numeric tokens; never edit experiment inputs.

Run from any directory. --verify checks the archived files and each selected
metric against the source JSON pointer, without requiring the original server.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
UNITS = {
 'psnr': 'dB ↑', 'foreground_psnr': 'dB ↑', 'ssim': '↑', 'foreground_ssim': '↑',
 'lpips': '↓', 'iou': '↑', 'missing': '% ↓', 'part_missing': '% ↓',
 'event_l1': '↓', 'event_cosine': '↑', 'polarity': '% ↑', 'silent': '↓',
 'latency_ms': 'ms ↓', 'point_count': 'points',
}

def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path, exact=False):
    options = dict(parse_float=str, parse_int=str) if exact else {}
    return json.loads(Path(path).read_text(), **options)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def pointer(data, path):
    for key in path.strip('/').split('/'):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def fmt(value):
    if value is None:
        return '缺失（未填零）'
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(rows):
    keys = list(dict.fromkeys(k for row in rows for k in row if k != 'version'))
    lines = ['| 指标（源字段） | 单位/方向 | '+' | '.join(fmt(r.get('version', 'source')) for r in rows)+' |',
             '|---|---|'+'---|'*len(rows)]
    for key in keys:
        lines.append('| '+key+' | '+UNITS.get(key, '依源协议')+' | '+' | '.join(fmt(r.get(key)) for r in rows)+' |')
    return '\n'.join(lines)


def identify(path):
    text = str(path).lower()
    for name in ('metronome', 'microphone', 'rotation', 'gears', 'robot', 'lamp', 'lego', 'arm', 'fan'):
        if name in text:
            return name
    return None


def verify():
    manifest = read(OUT/'source_manifest.json')
    for entry in manifest:
        data = (OUT/entry['snapshot']).read_bytes()
        assert digest(data) == entry['sha256'], entry['snapshot']
        assert len(data) == entry['bytes'], entry['snapshot']
    originals_checked = 0
    for entry in manifest:
        if '!/' in entry['original']: continue
        original = Path(entry['original'])
        if not original.is_absolute(): original = ROOT/original
        if original.exists():
            assert digest(original.read_bytes()) == entry['sha256'], 'Source changed: '+str(original)
            originals_checked += 1
    summary = read(OUT/'summary.json')
    for item in summary['objects']:
        raw = read(OUT/item['source_snapshot'], exact=True)
        selected = pointer(raw, item['json_pointer'])
        assert selected == item['source_record'], item['object']
        archived = read(OUT/'best'/f"{item['object']}.json")
        assert archived == item, item['object']
        assert item['metrics_exact_decimal_strings'] == {k:v for k,v in selected.items() if k != 'version'}, item['object']
    return dict(passed=True, archived_files=len(manifest), original_files_unchanged=originals_checked, selected_objects=len(summary['objects']),
                verification='SHA256/byte count plus exact numeric token comparison at each source JSON pointer; no remeasurement')


def build():
    config = read(OUT/'selection.json')
    sources = {}
    def archive(path):
        path = path.resolve()
        if str(path) in sources:
            return sources[str(path)]
        data = path.read_bytes(); sha = digest(data)
        suffix = ''.join(path.suffixes)
        target = Path('sources')/(sha+suffix)
        dest = OUT/target; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == data, str(path)
        else:
            dest.write_bytes(data)
        try: original = str(path.relative_to(ROOT))
        except ValueError: original = str(path)
        entry = dict(original=original, snapshot=str(target), sha256=sha, bytes=len(data))
        sources[str(path)] = entry
        return entry

    paths = set((ROOT/'experiment/output').glob('*/data/analysis.json'))
    paths.update((ROOT/'outputs').rglob('analysis.json'))
    paths.update((ROOT/'outputs/six-object-preview').glob('*result.json'))
    paths.update((ROOT/'outputs/six-object-preview').rglob('report-result.json'))
    paths.update(Path('/tmp').glob('robot*result.json'))
    paths.update(ROOT/x['source'] for x in config['objects'])
    runs = []; run_sources = {}
    for path in sorted(paths):
        data = read(path, exact=True)
        if not isinstance(data.get('summary'), list):
            continue
        if not data['summary'] or not all(isinstance(r, dict) and 'version' in r for r in data['summary']):
            continue
        obj = identify(path)
        if obj is None:
            continue
        origin = archive(path); attachments = []
        if path.name == 'analysis.json':
            for name in ('protocol.json','availability.json','version_mapping.json','input_manifest.json.gz',
                         'base_metrics.json.gz','series.npz','lpips.json','visualization_protocol.json'):
                sibling = path.parent/name
                if sibling.exists(): attachments.append(archive(sibling))
        row = dict(object=obj, source=origin, summary=data['summary'], protocol=data.get('protocol'),
                   warnings=data.get('warnings', []), motion=data.get('motion'), efficiency=data.get('efficiency'),
                   performance=data.get('performance'), attachments=attachments)
        runs.append(row); run_sources[str(path.resolve())] = row
    selected = []
    for entry in config['objects']:
        obj = entry['object']; path = ROOT/entry['source']; data = read(path, exact=True)
        origin = archive(path); record = pointer(data, entry['pointer']); metrics = {k:v for k,v in record.items() if k != 'version'}
        run = run_sources.get(str(path.resolve()))
        attachments = list(run['attachments']) if run else []
        if obj in ('robot','gears','rotation'):
            for name in ('quality-gate.json','validation.json','spatial-validation.json','heldout-validation.json'):
                p = path.parent/name
                if p.exists(): attachments.append(archive(p))
        if obj == 'arm':
            for name in ('arm-rgb-validation-result.json','arm-rgb-spatial-audit.json','arm-rgb-optimized-export.json'):
                p = path.parent/name
                if p.exists(): attachments.append(archive(p))
        # Preserve model identity and evaluation protocol directly from archived
        # input manifests where available. These are not checkpoint backups.
        identity = []
        for a in attachments:
            if a['original'].endswith('input_manifest.json.gz'):
                import gzip
                m = json.loads(gzip.decompress((OUT/a['snapshot']).read_bytes()))
                identity.append(dict(source_snapshot=a['snapshot'],sources=m.get('sources'),
                                     appearance_refinement=m.get('scene_info',{}).get('appearance_refinement'),
                                     versions=m.get('versions'),renderer_config=m.get('renderer_config')))
        if entry.get('model_locator_reported'):
            doc = archive(ROOT/'assets/technical/16-four-object-animated-observations.md')
            attachments.append(doc)
            identity.append(dict(path_reported=entry['model_locator_reported'], context_source=doc['snapshot'],
                                 status='Reported deployment path, remote existence and checkpoint bytes not reverified'))
        if obj == 'robot':
            identity.append(dict(candidate_sha256=data['candidate_sha256'], reference_sha256=data['reference_sha256'],
                                 note='candidate is the selected V3; baseline is previous V3, not V1 or V2'))
        if obj in ('gears','rotation'):
            # Export ZIPs contain the model hashes/configuration but are temporary.
            # Archive only manifest/config JSON, never duplicate frames/checkpoints.
            import zipfile
            zpath = Path('/tmp')/('gears-compact-export.zip' if obj=='gears' else 'rotation-quality-export.zip')
            if zpath.exists():
                with zipfile.ZipFile(zpath) as z:
                    names = [n for n in z.namelist() if n.endswith('/manifest.json')]
                    if len(names)==1:
                        blob = z.read(names[0]); sha=digest(blob); dest=OUT/'sources'/(sha+'.json')
                        if dest.exists(): assert dest.read_bytes()==blob
                        else: dest.write_bytes(blob)
                        a=dict(original=str(zpath)+'!/'+names[0],snapshot=str(dest.relative_to(OUT)),sha256=sha,bytes=len(blob))
                        sources[a['original']]=a; attachments.append(a)
                        m=json.loads(blob);identity.append(dict(source_snapshot=a['snapshot'],sources=m.get('sources'),
                            appearance_refinement=m.get('scene_info',{}).get('appearance_refinement'),versions=m.get('versions')))
        item=dict(object=obj, recommended_version=entry['version'], sample_count=entry['sample_count'],
                  source_original=origin['original'],source_snapshot=origin['snapshot'],source_sha256=origin['sha256'],
                  json_pointer=entry['pointer'],source_record=record,metrics_exact_decimal_strings=metrics,
                  selection_reason=entry['selection_reason'],attachments=attachments,model_identity_records=identity,
                  model_retention='原模型未删除/改写；这里只归档实验报告、身份和引用，不声称远端模型已备份或当前可用。',
                  units=UNITS,missing_fields='缺失/不存在字段保持缺失；不从其他版本或协议拼接数值。')
        selected.append(item);write(OUT/'best'/f'{obj}.json',item)
        title = [f'# {obj} — 推荐保留 {entry["version"]}', '', entry['selection_reason'], '',
                 f'样本数：{entry["sample_count"]}。来源：`{origin["original"]}`。',
                 f'原始文件 SHA256：`{origin["sha256"]}`；JSON pointer：`{entry["pointer"]}`。', '',
                 f'[原始文件快照](../{origin["snapshot"]}) · [最佳版本登记](../best/{obj}.json)', '',
                 '以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。', '']
        if obj == 'robot':
            title += ['## 当前模型：原视频24帧', '',
                '这里的 baseline 是旧V3，candidate 是推荐的新V3，不能把它们称为V1/V2。', '',
                table([dict(version=k,**v) for k,v in data['summary']['original_all'].items()])]
            for group, group_rows in data['summary'].items():
                if group == 'original_all': continue
                title += ['',f'## 独立列示：{group}', '',table([dict(version=k,**v) for k,v in group_rows.items()])]
            title += ['', '合成视角与原视频指标不能混合平均；synthetic_test 为未用于选点的合成角度，不是新物件独立测试。缺少新V3完整标准报告的本地副本，因此前景SSIM、事件、速度等不从旧模型补填。']
        else:
            title += [table(data['summary'])]
            for key in ('motion','efficiency'):
                if isinstance(data.get(key),list) and data[key]:title += ['',f'## 源报告：{key}', '',table(data[key])]
            if data.get('performance'):
                title += ['', '## 源报告：性能范围', '', '```json',json.dumps(data['performance'],ensure_ascii=False,indent=2),'```']
            if data.get('protocol'):
                title += ['', '## 源报告：实验协议', '', '```json',json.dumps(data['protocol'],ensure_ascii=False,indent=2),'```']
            if data.get('warnings'):title += ['', '## 源警告', '',*['- '+w for w in data['warnings']]]
        title += ['', '## 历史候选（不同协议不直接排名）', '', '| 来源 | 版本 | PSNR | SSIM | LPIPS | IoU |', '|---|---|---|---|---|---|']
        for r in runs:
            if r['object'] != obj:continue
            for row in r['summary']:
                title.append('| ['+r['source']['original']+'](../'+r['source']['snapshot']+') | '+fmt(row['version'])+' | '+' | '.join(fmt(row.get(k)) for k in ('psnr','ssim','lpips','iou'))+' |')
        (OUT/'objects').mkdir(exist_ok=True);(OUT/'objects'/f'{obj}.md').write_text('\n'.join(title)+'\n')
    write(OUT/'all_runs.json',dict(numeric_encoding='exact original JSON numeric tokens stored as strings',runs=runs))
    write(OUT/'summary.json',dict(scope=config['scope'],selection_policy=config['selection_policy'],remote_status=config['remote_status'],
                                 numeric_encoding='metrics are exact original JSON numeric tokens stored as strings; source_record is unmodified structurally except numeric encoding',objects=selected))
    write(OUT/'source_manifest.json',list(sources.values()))
    lines=['# 全物品实验量化统计与推荐保留版本', '',
           '统计快照：2026-11-02。目录覆盖9个物品：Lego、Fan、Lamp、Metronome、Microphone、Arm、Robot、Gears、Rotation。', '',
           '**数值不修改**：源文件逐字节复制，SHA256登记在[source_manifest.json](source_manifest.json)。下表和分物品页保留源JSON数值精度；没有把缺失值补成0，也没有把各版最大值拼成一版。', '',
           '**“最佳”采用综合推荐口径**：每物品登记一个已有验收依据的版本，保留取舍和所有已发现的标准报告。Lamp/Gears等不存在所有指标同时最优的单一版本。没有任意加权总分、没有跨物品总平均或把不同测试集混算。', '',
           '## 推荐版本总表', '', '| 物品 | 版本 | 帧/样本数 | PSNR（dB） | SSIM | LPIPS | IoU |', '|---|---|---|---|---|---|---|']
    for item in selected:
        m=item['metrics_exact_decimal_strings'];lines.append(f'| [{item["object"]}](objects/{item["object"]}.md) | {item["recommended_version"]} | {item["sample_count"]} | '+' | '.join(fmt(m.get(k)) for k in ('psnr','ssim','lpips','iou'))+' |')
    lines += ['', '## 阅读范围与限制', '',
        '- 分物品页列出原报告的前景PSNR/SSIM、LPIPS、IoU、缺失率、部件缺失率、事件L1/余弦/极性/静默变化、延迟、点数，以及原报告已有的运动、性能字段。缺失项保留缺失。',
        '- 缺失率、部件缺失率、极性一致率在这些源汇总中已经是百分数，未再次乘100。SSIM、IoU、余弦为原尺度；运动单位与点采样对应关系按源协议解释。',
        '- latency_ms、FPS和显存来自不同采样/硬件/渲染负载，不能把单面离线渲染FPS当UI播放FPS。process-wide显存不是单版本显存。',
        '- V1/V2的含义以每份源报告的版本映射为准，不统一交换标签，不把不同物品的V1/V2当严格同构消融。',
        '- Arm/Robot/Gears/Rotation含同源RGB监督及选点；高分不是独立测试证明。Robot合成训练/验证/测试角度单独列示，当前24帧新V3不与旧12帧V1/V2拼接成同表。',
        '- Fan的本地完整报告包含真实事件校准/验证结果，按源协议保留；后续服务器缺数据不用于覆盖这些历史有效数值。其余缺事件的项目不补造指标。',
        '- 本次“保留最佳版”是保留推荐报告、模型身份与路径引用；不删除历史模型、不改UI或训练状态，也未复制大模型。模型文件的远端备份尚未完成。',
        '- '+config['remote_status'],
        '- 本表以本地可追溯标准实验为边界。Lego独立Hybrid FVV、clay/mesh和其他非同口径调试日志不混入RGB三版本统计；不声称已覆盖不可访问的服务器所有历史实验。', '',
        '## 文件与复核', '',
        '- [summary.json](summary.json)：每物品推荐版本、全部源指标、选择理由、模型来源。',
        '- [all_runs.json](all_runs.json)：发现的标准报告及完整summary/protocol/性能/运动，保留历史候选。',
        '- `best/<物品>.json`：每个物品的一份推荐版本登记，不包含拼接指标。',
        '- `sources/`：按SHA256命名的原始文件；包括可用逐帧数组、原始指标、协议、验证记录和模型身份清单。',
        '- [verification.json](verification.json)：独立快照哈希与逐指标原始数值token校验结果。', '',
        '只验证现有归档（不访问服务器）：', '', '```bash', 'python experiments/build_summary.py --verify', '```', '',
        '重新收集本地报告：`python experiments/build_summary.py`。选择配置为[selection.json](selection.json)。重建可能纳入新增报告；已归档原文件按内容哈希保留。', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    result=verify();write(OUT/'verification.json',result);print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.verify: print(json.dumps(verify()))
    else: build()
