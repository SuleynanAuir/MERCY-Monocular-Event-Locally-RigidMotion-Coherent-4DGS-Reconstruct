"""Build the experiment audit and forward protocol from verified frozen results.

Run from the repository root: python -m experiment.build_export_report --help
"""
import argparse
import csv
import gzip
import html
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
from experiment.export_report_analysis import analyze, sha, save, table, METRICS, PRIMARY

SOURCES = [
    ('SSIM: Wang et al., IEEE TIP 2004', 'https://ece.uwaterloo.ca/~z70wang/publications/ssim.html'),
    ('LPIPS: Zhang et al., CVPR 2018; official implementation', 'https://github.com/richzhang/PerceptualSimilarity'),
    ('Stationary bootstrap: Politis & Romano, JASA 1994', 'https://doi.org/10.1080/01621459.1994.10476870'),
]
LABELS = {
    'foreground_psnr':'前景 PSNR ↑ (dB)', 'psnr':'全图 PSNR ↑ (dB)',
    'ssim':'全图 SSIM ↑', 'foreground_ssim':'前景 SSIM ↑', 'lpips':'LPIPS-Alex ↓',
    'iou':'轮廓 IoU ↑', 'missing':'前景缺失率 ↓ (%)',
    'part_missing':'可见部件区域缺失率 ↓ (%)', 'event_l1':'事件相对 L1 ↓',
    'event_cosine':'事件余弦相似度 ↑', 'polarity':'极性一致率 ↑ (%)', 'silent':'无事件区 log 变化 ↓',
}


def content(output, root):
    a=json.loads((output/'data/analysis.json').read_text());s=a['summary'];p=a['protocol']
    lp=json.loads((output/'data/lpips.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    pages=[]
    def page(title,kicker,paragraphs=(),tables=(),figures=()):
        pages.append(dict(title=title,kicker=kicker,paragraphs=list(paragraphs),tables=list(tables),figures=list(figures)))
    def tab(caption,head,rows):return dict(caption=caption,head=head,rows=rows)
    fmt=lambda v:f'{v:.4f}' if abs(v)<1 else f'{v:.3f}'
    main=tab('表 1 · 全部 200 帧 / 180 个事件窗口的宏平均', ['指标','V1（原V2）','V2（原V1）','V3'],
             [[LABELS[k],*[fmt(r[k]) for r in s]] for k in METRICS])
    page('三版本对照试验评测','SC-GS / LEGO / FROZEN EXPORT AUDIT / 2026.10.27',[
        '本报告按要求重新编号：新 V1=原 V2（观测校正存档分支），新 V2=原 V1（未校正初始动画），V3 不变。四项主指标的均值按新编号递增改善；这是结果展示顺序调整，模型与测量值没有改变，不能据此证明严格逐层消融。',
        f'评测对象为同一 Lego 序列、一个真值相机、原始 640×640 分辨率，每 2 个源帧采样：200 个时刻、600 张预测图。全量评测，不筛选有利帧；按明确映射统一编号，保留原始包与来源。',
        f'V3 相对新 V2：前景 PSNR 增加 {s[2]["foreground_psnr"]-s[1]["foreground_psnr"]:.2f} dB，LPIPS 降低 {(1-s[2]["lpips"]/s[1]["lpips"])*100:.1f}%，IoU 增加 {(s[2]["iou"]-s[1]["iou"])*100:.2f} 个百分点，事件相对 L1 降低 {(1-s[2]["event_l1"]/s[1]["event_l1"])*100:.1f}%。平均渲染耗时为新 V2 的 {s[2]["latency_ms"]/s[1]["latency_ms"]:.2f} 倍。',
        f'改善并非覆盖所有指标：新 V1 的等权部件区域缺失率最低（{s[0]["part_missing"]:.2f}%，V3 为 {s[2]["part_missing"]:.2f}%）；新 V2/V3 的平均最大连续缺口相同，均占 GT 前景 {s[2]["largest_missing_fraction_percent"]:.4f}%。覆盖代理不能替代形状与外观质量。',
    ],[main,tab('表 2 · 四项主指标均值递进检查（描述性）',['比较','主指标均值','因果归因'],[
        ['V1 → V2','4 项均值改善','重新编号；非训练父子关系'],
        ['V2 → V3','4 项均值改善','未证实直接父子关系'],
        ['V1 → V3','4 项均值改善','存档与恢复分支的组合比较'],
    ])])
    page('数据、指标与公平比较协议','01 / MEASUREMENT CONTRACT',[
        '本报告是冻结产物的回顾性审计；主次指标组织方式与敏感性分析在现有结果之后制定，不宣称预注册。后续实验设计单独列为“待运行”，不会混入本次结果。',
        f'完整性：清单登记的 {len(manifest["files"]):,} 个文件均通过 SHA-256 与大小校验。RGB、固定前景和固定部件掩码各 200 帧；事件计数覆盖 199 个 (t0,t1] 窗口。相机、尺寸、背景和采样时间一致。',
        '主指标：前景 PSNR、标准全图 LPIPS、轮廓 IoU、事件相对 L1。背景会抬高全图质量，因此前景 PSNR 优先；全图 PSNR / SSIM 同时报告。所有均值为逐帧或逐窗宏平均，另存像素/事件量加权汇总。',
        '参考掩码由原始 Blender 几何的最近可见表面生成，对所有版本共用；它衡量几何覆盖而非透明度或光度真值。部件区域被任意预测表面覆盖也会计为覆盖，所以“部件缺失率”是可见区域覆盖代理，不能解释为语义部件识别率或遮挡背面的完整率。',
        '事件主评测仅用 180 个不与响应标定重叠的窗口。19 个重叠窗口保留原始分数但不纳入主表。response_validation 只是响应标定留出，不等于训练未见测试集；标定为固定经验 RGB–事件响应，不等于独立物理传感器标定。',
    ],[tab('表 3 · 明确、可复现的计算定义',['指标 / 设置','实现与解释'],[
        ['PSNR','RGB 归一化至 [0,1]；−10 log10(MSE)，先逐帧求分数再平均；前景使用公共掩码。'],
        ['SSIM','11×11 高斯核，σ=1.5，群体协方差，去除 5 像素边界后通道均值。前景 SSIM 按掩码中心取窗口，窗口可含邻域背景。'],
        ['LPIPS','官方 v0.1 / AlexNet，标准学习权重，RGB [-1,1]，原始分辨率、不裁剪不缩放；CPU float32 / eval。'],
        ['轮廓 / 缺失','预测 alpha≥0.5；IoU=交集/并集；缺失=(GT−预测覆盖)/GT。另测 alpha=0.25、0.75。'],
        ['事件 L1 / 极性','ΔlogI 与 C+N+−C−N− 比较；L1 除以观测绝对和；极性仅在非零净响应像素计算。'],
        ['部件 / 连通缺口','对每个可见 GT 部件求 alpha 缺失率，帧内部件等权，再帧间平均；最大缺口用 4 连通像素。'],
        ['不确定性','5000 次配对圆形 stationary bootstrap，平均块长 10；5、20 作敏感性分析。保留时间相关性；只是单序列描述性区间。'],
    ])])
    page('总体质量与时间分布','02 / FULL-SEQUENCE RESULTS',[
        '同一量纲使用一致坐标；雷达图按原始数值从零开始。当前 V1 是原观测校正存档分支，图像、轮廓与事件主指标均较差。新顺序不改变这个事实；部件覆盖和时延仍不满足全指标单调改善。CSV 同时保存展示编号与原始数据 ID。',
    ],figures=[('01_quality','图 1 · 雷达半径按原始数值线性增加，各轴上限与量纲分别标注；下方三组分面气泡图呈现逐指标改善和退步比例（非显著性），横轴在 ±25% 内线性、两侧对数，绿色 ↑ / + 表示越高越好，红色 ↑ / + 表示越高越差，没有综合总分。事件使用 180 个响应标定留出窗口；其余使用全部 200 帧。'),
               ('02_temporal','图 2 · 公共 GT 平移轨迹得到约 3.333 秒周期；交替底色为完整周期、点线为半周期。原始时间曲线、CDF 与差值保留，灰色事件窗口被排除。')])
    delta_rows=[]
    for key in PRIMARY:
        for pair in ('V1 -> V2','V2 -> V3','V1 -> V3'):
            r=next(r for r in a['comparisons'] if r['metric']==key and r['pair']==pair and r['expected_block_length']==10)
            delta_rows.append([LABELS[key].split(' ↑')[0].split(' ↓')[0],pair.replace('->','→'),
                               f'{r["oriented_delta"]:+.4f}',f'[{r["low"]:+.4f}, {r["high"]:+.4f}]',f'{100*r["win_fraction"]:.1f}%'])
    page('配对差异与结论边界','03 / PAIRED TEMPORAL ANALYSIS',[
        '统一差值方向为“正值=改善”：高者优的指标取后者减前者，低者优取前者减后者。三版本共享每次重采样索引，避免把不同时间段误当作独立比较。',
        '区间仅描述当前序列上的时间分布，不能替代独立场景或训练随机种子的置信区间。序列平稳性只是近似假设，因此同时提供块长 5 / 10 / 20 的结果，不给出总体显著性或跨场景泛化声明。',
    ],[tab('表 4 · 主指标配对差异；95% 描述性重采样区间',['指标','比较','改善差值','95% 区间','时刻胜率'],delta_rows)])
    page('固定时刻的原始渲染对照','04 / QUALITATIVE EVIDENCE',[
        '选择源帧 0、200、398，对应时间表的开始、中间、结束。选帧规则只依赖时间，不依赖任何方法的分数。图像仅为版面显示缩放；数值评测始终使用原始 640×640 图像。',
        '新 V1（原 V2）的轮廓、亮度和表面差异需要追溯输入资产、几何对齐、外观与模块开关。仅凭这组对比不能断言“观测校正模块本身无效”，因为该存档分支的父产物来源未被控制。',
    ],figures=[('04_qualitative','图 4 · RGB 真值与各版本无 UI 叠字渲染帧；同相机、同时间、同黑色背景。')])
    page('图像误差的空间分布','05 / SPATIAL ERROR',[
        '按全图 PSNR 两两绝对差之和降序选择最高的 4 个共同有效时刻，形成紧密 4×4 矩阵：前三列为各版误差，第四列为 error(V3)−error(V2)，负值绿色表示误差降低、正值红色表示误差增加。这些是主动选择的极端差异诊断时刻，不代表平均表现；完整序列统计不受影响。展示逐像素 RGB 绝对误差通道均值。所有面板共用 0–0.25 色标；超过上限的误差仅在图中饱和，指标计算不截断。',
        '全图误差热图同时包含前景和背景伪影，避免只展示局部优势区域。V3 在本视角的改善不能外推为不可见背面或任意新视角的完整性提升。',
    ],figures=[('05_error_maps','图 5 · PSNR 差异最大的四时刻、固定 GT 轮廓、共享误差色标、V3−V2 逐像素误差差值；右侧保留前景误差 CDF、p95 和对数尾部分布。图中缩放不影响原始误差计算。')])
    motion_rows=[]
    for key,title in [('points','轨迹采样点数'),('unassigned_points','未分组点数'),('joint_gap_max','最大关节间隙 / 场景单位'),
                      ('rigidity_error','组内刚体误差 / 场景单位'),('angular_jerk_rms','纯铰链角 jerk / rad·s⁻³'),
                      ('rotation_jerk_rms','SO(3) jerk / rad·s⁻³'),('position_jerk_rms','3D jerk（仅诊断）')]:
        motion_rows.append([title,*['不可用：非纯铰链' if r[key] is None else f'{r[key]:.6g}' for r in a['motion']]])
    page('公共部件覆盖与运动诊断','06 / STRUCTURE AND MOTION',[
        '新版热图纳入所有可见公共部件，按 GT 平均可见像素面积固定分组。每格等于组内缺失像素/全部共同可见 GT 像素×100，各行之和是总面积缺失，单位为百分点。三版统一 alpha=0.5、分组、分母与平方根色阶。面积缺失逐级降低，但部件等权缺失率不是逐级降低；两种口径均保留，不能混为同一指标。',
        'V1/V2 各有 896 个轨迹点，V3 有 640 个；未验证跨版本点对应。原始 3D jerk 只能作诊断，不能据此声称运动平滑度大幅提高。SO(3) 旋转 jerk 三者几乎相同。新 V1 存在明显离轴 swing，纯铰链角 jerk 留空，避免把投影角误写为真关节角。',
        '按组核验发现：V1/V2 中 3D jerk 最大的 group 9（约 276 / 280 场景单位·s⁻³）没有被 V3 的轨迹点采样覆盖。这是点群不一致的直接证据，不能把总 jerk 的下降解释为该组的运动已得到修复。新 V1 虽覆盖更多小部件区域，其整体 IoU 与前景缺失率仍退化。',
    ],[tab('表 5 · 运动量来自声明的源先验，无独立动作真值',['诊断项','V1','V2','V3'],motion_rows)],
       [('06_parts','图 6 · 逐采样时刻热图嵌入总面积缺失曲线、完整/半周期分界；下方展示部件内部缺失 CDF、超阈值部件占比与两种聚合比较。')])
    page('效率、资源与运动代价','07 / COST AND CONSTRAINTS',[
        f'原导出设备：{manifest["environment"]["gpu"]}，CUDA {manifest["environment"]["cuda"]}，PyTorch {manifest["environment"]["packages"]["torch"]}。预热 20 次，固定 200 帧重复 5 轮，获得 1000 组三版本串行测量。包含 GPU 同步与 CPU 像素回读，不含轨迹状态采集和图像编码。',
        f'V3 平均 {s[2]["latency_ms"]:.2f} ms，新 V2 为 {s[1]["latency_ms"]:.2f} ms；质量改进伴随 {s[2]["latency_ms"]/s[1]["latency_ms"]:.2f}× 耗时。三面板总平均 {a["performance"]["mean_ms"]:.2f} ms。串行固定顺序可能带来缓存与温度混杂，后续需独立进程、随机版本顺序复测。',
        f'显存仅有整个进程的 PyTorch 峰值 {a["performance"]["memory_peak"]/1024**2:.1f} MiB，包含全部已加载场景，不能分摊成各版本显存。加载耗时 {a["performance"]["load_seconds"]:.2f} s 也是进程级；不代表完整训练成本或独立冷启动统计。',
    ],[tab('表 6 · 原始性能测量；重复轮次不等于训练随机种子',['版本','点数','均值 ms','p95 ms','轮均值范围 ms'],
        [[r['version'],f'{r["points"]:,}',f'{r["mean_ms"]:.2f}',f'{r["p95_ms"]:.2f}',f'{r["repeat_mean_min_ms"]:.2f}–{r["repeat_mean_max_ms"]:.2f}'] for r in a['efficiency']])],
       [('07_efficiency','图 7 · 单一质量—时延坐标集成全部轮次×逐帧测量点、周期形状标记、50/90% 分布密度轮廓、四分位带、IoU 与高斯点数。下方时延密度轨道的高度不是 PSNR，轮廓不是置信区间。'),
        ('08_motion','图 8 · 关节间隙、离轴摆动、旋转 jerk 与轨迹 jerk/残差；同源动作周期统一标色。采样属性保留在数据文件中。')])
    page('评测参数敏感性','08 / ROBUSTNESS CHECKS',[
        '保持模型、渲染帧、GT 和相机不变，仅改变评测阈值 alpha=0.25/0.50/0.75；这检验结论是否过度依赖某个轮廓阈值，不是训练模块消融。',
        '轨迹按源帧间隔 2、4、8 做时间下采样，重新计算三阶差分与旋转 jerk。三阶导数对采样频率和量化噪声敏感，尤其不能跨不同点群直接解读。采样间隔 2 为主协议，4/8 只用于敏感性诊断。',
        '已检测到明确的采样风险：间隔 8 时，各版本均有 24 个组-时间区间的累计旋转超过 π（最大约 4.15 rad）；最短旋转增量可能发生混叠，SO(3) jerk 的跃升不能解释为真实动作突然变差。间隔 2 与 4 未检测到这一跨 π 风险。',
        '所有敏感性结果保留在 alpha_sensitivity.csv、motion_sampling_sensitivity.csv 与 paired_differences.csv；不会从中挑选最有利的参数替换主结果。',
    ],figures=[('09_sensitivity','图 9 · 无重新训练的敏感性分析。注意低阶运动变化与高阶导数量化噪声的区别。')])
    for old in ('tables/planned_experiments.csv','data/planned_protocol.json'):
        (output/old).unlink(missing_ok=True)
    page('复现说明与证据范围','10 / REPRODUCIBILITY AND LIMITS',[
        '输出包含 PDF/HTML 报告、8 组 PNG 图、逐帧与逐窗 CSV、运动/性能/部件明细、分析 JSON、紧凑 NPZ、压缩基础指标与源清单，以及构建脚本和文件哈希。无需再次导出或复制原始模型、视频和数百张渲染图。',
        '快速复现：在项目根目录安装 reproduce/requirements.txt，再执行 python -m experiment.build_export_report --reuse-analysis。此模式重绘已有分数；默认模式重新校验原导出并进行分析。完整基础指标重算命令及 LPIPS 重算命令见 reproduce/commands.txt。',
        f'原始包保持不变；原件中的旧 ID 通过 data/version_mapping.json 映射到当前编号。包名：{root.name}。源 manifest SHA-256：{p["manifest_sha256"]}。LPIPS 权重状态 SHA-256：{lp["weights_sha256"]}。代码和运行环境另见 data/analysis_environment.json 与 artifact_manifest.json。',
        '范围限制：单序列、单相机、每版本单个配置，无独立训练种子重复；不证明跨场景泛化、未见视角质量、物理传感器精确响应或独立关节重建精度。纯铰链角只在通过轴向验证的版本计算；各版本独立显存、完整训练耗时和编码基准尚未记录。',
        '本次证据能够支持“该固定视角下 V3 的图像与事件表现更好，并伴随更高时延”；不能支持“V1/V2/V3 已完成严格嵌套消融且所有指标单调改善”。本报告仅包含对照试验与评测参数敏感性，不包含消融试验。',
    ],[tab('表 9 · 关键方法来源',['方法','原始文献 / 官方实现'],SOURCES)])
    save(output/'data/report_content.json',pages)
    return pages


def pdf(output,pages,font_path=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from PIL import Image as PILImage
    fonts=([Path(font_path)] if font_path else [])+[Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')]
    font=next((f for f in fonts if f.exists()),None)
    if font is None:raise RuntimeError('Install Arial Unicode or set up a Chinese-capable TTF font')
    pdfmetrics.registerFont(TTFont('Report',str(font)))
    navy=colors.HexColor('#153D4A'); muted=colors.HexColor('#56717A');teal=colors.HexColor('#168574')
    styles={
        'body':ParagraphStyle('body',fontName='Report',fontSize=9,leading=14.2,spaceAfter=8,textColor=navy,wordWrap='CJK'),
        'title':ParagraphStyle('title',fontName='Report',fontSize=22,leading=30,spaceAfter=13,textColor=navy,wordWrap='CJK'),
        'kicker':ParagraphStyle('kicker',fontName='Report',fontSize=8,leading=12,spaceAfter=7,textColor=teal),
        'caption':ParagraphStyle('caption',fontName='Report',fontSize=8,leading=12,spaceAfter=8,textColor=muted,wordWrap='CJK'),
        'cell':ParagraphStyle('cell',fontName='Report',fontSize=7.6,leading=11,wordWrap='CJK',textColor=navy),
        'head':ParagraphStyle('head',fontName='Report',fontSize=7.8,leading=11,wordWrap='CJK',textColor=colors.white),
    }
    width=503
    doc=SimpleDocTemplate(str(output/'report.pdf'),pagesize=(595.28,841.89),rightMargin=46,leftMargin=46,
                         topMargin=43,bottomMargin=41,title='SC-GS 三版本对照试验评测',author='SC-GS experiment audit')
    story=[]
    for i,page in enumerate(pages):
        if i:story.append(PageBreak())
        story += [Paragraph(html.escape(page['kicker']),styles['kicker']),Paragraph(html.escape(page['title']),styles['title'])]
        for text in page['paragraphs']:story.append(Paragraph(html.escape(text),styles['body']))
        for t in page['tables']:
            rows=[[Paragraph(html.escape(str(v)),styles['head' if j==0 else 'cell']) for v in row]
                  for j,row in enumerate([t['head'],*t['rows']])]
            n=len(t['head'])
            if n==2:widths=[118,width-118]
            elif n==3:widths=[92,300,width-392] if '控制项' in t['head'][1] else [100,175,width-275]
            elif n==4:widths=[212,97,97,97] if len(t['rows'])>6 else [116,129,129,129]
            else:widths=[142,80,70,138,73] if '95%' in t['head'][3] else [64,105,80,80,174]
            table_obj=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table_obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F5'),colors.white]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#CCDADC'))]))
            story.extend([Paragraph(html.escape(t['caption']),styles['caption']),table_obj,Spacer(1,10)])
        for figure_index,(name,caption) in enumerate(page['figures']):
            if page['tables'] or figure_index:
                story.extend([PageBreak(),Paragraph(html.escape(page['kicker']),styles['kicker']),Paragraph(html.escape(page['title']+' · '+name[:2]),styles['title'])])
            path=output/f'figures/{name}.png'; w,h=PILImage.open(path).size
            targetw=width
            # Two-figure pages need compact layouts while individual image pages stay large.
            maxheight=540 if page['tables'] or figure_index else 430
            targeth=targetw*h/w
            if targeth>maxheight:targetw*=maxheight/targeth;targeth=maxheight
            story.append(KeepTogether([Image(str(path),targetw,targeth),Spacer(1,5),Paragraph(html.escape(caption),styles['caption'])]))
    def furniture(canvas,document):
        canvas.setStrokeColor(colors.HexColor('#CCDADC'));canvas.line(46,31,549,31)
        canvas.setFont('Report',7);canvas.setFillColor(muted)
        canvas.drawString(46,20,'SC-GS / LEGO  ·  FROZEN-ARTIFACT AUDIT  ·  2026-10-27')
        canvas.drawRightString(549,20,str(document.page))
    doc.build(story,onFirstPage=furniture,onLaterPages=furniture)


def webpage(output,pages):
    from experiment.comparison_explorer import HTML as EXPLORER, CSS as EXPLORER_CSS, JS as EXPLORER_JS
    def esc(s):return html.escape(str(s))
    blocks=[]
    for i,p in enumerate(pages):
        b=f'<section id="s{i}"><div class="kicker">{esc(p["kicker"])}</div><h2>{esc(p["title"])}</h2>'
        b+=''.join(f'<p>{esc(s)}</p>' for s in p['paragraphs'])
        for t in p['tables']:
            b+=f'<div class="tablewrap"><table><caption>{esc(t["caption"])}</caption><thead><tr>'+''.join(f'<th>{esc(s)}</th>' for s in t['head'])+'</tr></thead><tbody>'
            b+=''.join('<tr>'+''.join(f'<td>{esc(s)}</td>' for s in r)+'</tr>' for r in t['rows'])+'</tbody></table></div>'
        for name,caption in p['figures']:
            b+=f'<figure><a href="figures/{name}.png"><img loading="lazy" src="figures/{name}.png" alt="{esc(caption)}"></a><figcaption>{esc(caption)} <a href="figures/{name}.png">PNG</a></figcaption></figure>'
        if i==2:b+='<h3>多维指标 · 同量纲组合图谱</h3>'+EXPLORER
        blocks.append(b+'</section>')
    with np.load(output/'data/series.npz') as z:
        payload=dict(times=z['times'].tolist(),event_times=z['event_times'].tolist(),metrics={k:{'label':LABELS[k],
            'values':[[None if not np.isfinite(x) else float(x) for x in row] for row in z[k]]} for k in METRICS})
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f3;color:#183b49;font-family:Inter,-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;font-size:15px;line-height:1.8}header{background:#153d4a;color:#fff;padding:38px max(24px,calc((100vw - 1100px)/2));}h1{margin:6px 0;font-size:34px;line-height:1.4}header a{color:#a8e6d5}main{max-width:1160px;padding:24px;margin:auto}section{background:#fff;border:1px solid #d5e1e3;border-radius:12px;padding:34px;margin-bottom:24px;scroll-margin-top:20px}.kicker{letter-spacing:2px;font-size:12px;color:#168574}h2{font-size:27px;line-height:1.4;margin:8px 0 22px}p{max-width:94ch}nav{display:flex;gap:14px;flex-wrap:wrap;margin-top:20px}nav a{font-size:13px;text-decoration:none}a{color:#087563}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}caption{text-align:left;color:#637b83;padding:14px 0 8px}th,td{padding:10px 13px;text-align:left;border-bottom:1px solid #dbe5e7;vertical-align:top}th{background:#173e4a;color:#fff}tr:nth-child(even){background:#f2f7f7}.tablewrap{overflow:auto}figure{margin:24px 0}figure img{width:100%;display:block}figcaption,.note{font-size:13px;color:#5b737b}.explorer{border:1px solid #cbdcde;padding:20px;border-radius:10px;background:#f9fbfc}.choices{display:flex;gap:20px;margin:10px 0}select{padding:8px;border:1px solid #abc3c7;border-radius:5px;background:white;color:#173e4a}input[type=range]{width:55%;vertical-align:middle}output{display:block;font-variant-numeric:tabular-nums;margin-top:8px}svg#chart{width:100%;background:white}.links{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:700px){main{padding:12px}section{padding:20px}h1{font-size:27px}h2{font-size:23px}td,th{padding:8px;font-size:12px}}@media print{body{background:white}section{break-before:page;border:0}.explorer,nav{display:none}}'''
    css+=EXPLORER_CSS
    js=EXPLORER_JS
    payload['names']=['V1','V2','V3']
    nav=''.join(f'<a href="#s{i}">{i+1:02d} {esc(p["title"])}</a>' for i,p in enumerate(pages))
    text=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SC-GS · 三版本实验评测</title><style>{css}</style></head><body><header><div>REPRODUCIBLE EXPERIMENT / 2026.10.27</div><h1>三版本对照试验评测</h1><div>200 个时刻 · 600 张预测图 · 640×640 · 每 2 帧采样 · 一个固定真值相机</div><div class="links"><a href="report.pdf">下载 PDF 报告</a><a href="tables/main_results.csv">主结果 CSV</a><a href="reproduce/commands.txt">复现命令</a></div><nav>{nav}</nav></header><main>{''.join(blocks)}</main><script id="series" type="application/json">{json.dumps(payload,ensure_ascii=False,allow_nan=False)}</script><script>{js}</script></body></html>'''
    text=text.replace('</style>','p,td,output{overflow-wrap:anywhere}select{max-width:100%}</style>')
    (output/'report.html').write_text(text)


def reproduce(output,root):
    directory=output/'reproduce';directory.mkdir(exist_ok=True)
    source_files=['experiment/build_export_report.py','experiment/export_report_analysis.py',
        'experiment/export_report_figures.py','experiment/comparison_figures.py','experiment/comparison.py',
        'experiment/comparison_analysis.py','experiment/comparison_explorer.py','experiment/comparison_report.py','utils/comparison_io.py',
        'utils/viewer_comparisons.py','utils/comparison_outputs.py','tests/viewer/test_comparisons.py','requirements/comparison.txt','experiment/lpips_export.py','experiment/test_export_report.py','experiment/test_comparison_visuals.py',
        'scripts/evaluation/evaluate_experiment.py','utils/experiment_format.py','utils/mesh_motion.py','utils/version_presentation.py']
    for name in source_files:
        dest=directory/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    packages=['numpy','scipy','Pillow','matplotlib','torch','torchvision','lpips','reportlab']
    versions={name:importlib.metadata.version(name) for name in packages}
    save(output/'data/analysis_environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages=versions,source_code_sha256={name:sha(name) for name in source_files},references=SOURCES,
        base_metrics_sha256=sha(output/'data/base_metrics.json.gz'),
        note='Pinned versions describe the analysis machine; use compatible torch/torchvision wheels on other platforms.'))
    (directory/'requirements.txt').write_text('\n'.join(f'{k}=={v.split("+")[0]}' for k,v in versions.items())+'\n')
    (directory/'commands.txt').write_text(f'''运行目录：SC-GS 项目根目录。保留原始导出 {root.name}。
本文件以 python 表示安装了 requirements.txt 的 Python 3.11 环境。

1. 只从已经计算的分数重绘报告（最快；不加载原始模型）：
python -m experiment.build_export_report --reuse-analysis

2. 校验原导出并重新分析已有基础分数 + LPIPS（不重新计算逐部件指标）：
python -m experiment.build_export_report --cache experiment/output/data/base_metrics.json.gz

3. 若需全部从图像重算（约数分钟至十余分钟，依设备而定）：
python -m scripts.evaluation.evaluate_experiment experiment/exports/{root.name} --output /tmp/scgs-base-metrics.json
python -c "import gzip,pathlib; p=pathlib.Path('/tmp/scgs-base-metrics.json'); gzip.open('/tmp/scgs-base-metrics.json.gz','wb').write(p.read_bytes())"
python -m experiment.lpips_export --export experiment/exports/{root.name} --output experiment/output/data/lpips.json --weights /tmp/scgs-report-models
python -m experiment.build_export_report --cache /tmp/scgs-base-metrics.json.gz

4. 配对时间重采样测试：
python -m unittest experiment.test_export_report -v

图：figures/ 中仅 PNG；表：tables/；协议和中间量：data/。
复现依赖原 SC-GS 仓库；source/ 是本次分析源代码快照，可按相对路径还原。
若系统没有 Arial Unicode.ttf，请使用 --font 指向支持中文的 TrueType 字体。
无需复制视频/模型/全部帧至本输出目录；LPIPS 官方 AlexNet 权重保留在临时缓存。
''')
    # A compact publication table, without assuming a sequential causal chain.
    s=json.loads((output/'data/analysis.json').read_text())['summary']
    rows=['\\begin{tabular}{lrrrrrr}','\\hline','Version & FG PSNR $\\uparrow$ & LPIPS $\\downarrow$ & IoU $\\uparrow$ & Event L1 $\\downarrow$ & Latency (ms) & Points (M) \\\\', '\\hline']
    for r in s:rows.append(f'{r["version"]} & {r["foreground_psnr"]:.2f} & {r["lpips"]:.4f} & {r["iou"]:.4f} & {r["event_l1"]:.4f} & {r["latency_ms"]:.2f} & {r["point_count"]/1e6:.3f} \\\\')
    rows += ['\\hline','\\end{tabular}','% Renumbered: current V1 = original V2 (archived), V2 = original V1; not a verified nested ablation.']
    (output/'tables/main_results.tex').write_text('\n'.join(rows)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,default=Path('experiment/exports/lego_20261027T032447Z_4b1b2dfb'))
    parser.add_argument('--output',type=Path,default=Path('experiment/output'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/export-audit-4b1b2dfb/metrics.json.gz'))
    parser.add_argument('--reuse-analysis',action='store_true',help='Rebuild report from existing analysis, without recalculating metrics')
    parser.add_argument('--font',type=Path,help='Chinese-capable TrueType font for the PDF')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not args.reuse_analysis:analyze(args.export,args.cache,args.output)
    else:
        from scripts.evaluation.evaluate_experiment import verify
        manifest=json.loads((args.export/'manifest.json').read_text())
        saved_protocol=json.loads((args.output/'data/protocol.json').read_text())
        from utils.version_presentation import REVISION
        if saved_protocol.get('version_presentation',{}).get('revision')!=REVISION:
            raise ValueError('Recompute analysis for the current presentation mapping before reusing it')
        expected=saved_protocol['manifest_sha256']
        if expected!=sha(args.export/'manifest.json'):
            raise ValueError('Saved analysis does not belong to the requested export')
        verify(args.export,manifest)
    from experiment.export_report_figures import generate
    generate(args.export,args.output)
    pages=content(args.output,args.export);pdf(args.output,pages,args.font);webpage(args.output,pages);reproduce(args.output,args.export)
    files={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and p.relative_to(args.output).parts[0]!='comparisons'}
    save(args.output/'artifact_manifest.json',dict(source_manifest_sha256=sha(args.export/'manifest.json'),
        files=files,total_bytes=sum(f['bytes'] for f in files.values())))
    print(f'Report ready: {args.output}/report.pdf; {len(files)} files; {sum(f["bytes"] for f in files.values())/1024**2:.2f} MiB',flush=True)


if __name__=='__main__':main()
