"""Publication figures for the frozen export, with fixed method order and scales."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from experiment.export_report_analysis import METRICS, PRIMARY, NAMES

COLORS = ['#3465A4', '#D17A22', '#168574']
DASHES = ['-', '--', '-.']
RENUMBER_NOTE='Display renumbering: V1 = original V2; V2 = original V1; V3 unchanged. Not a nested ablation.'


def generate(root, output):
    root, output = Path(root), Path(output)
    report = json.loads((output/'data/analysis.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    source_ids=report['protocol']['source_version_ids']
    series = np.load(output/'data/series.npz', allow_pickle=False)
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    for path in figures.iterdir():
        if path.is_file() and path.suffix.lower() in ('.pdf','.svg'):path.unlink()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#CBD3DA', 'grid.color': '#E8ECEF', 'grid.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    files=[]
    def finish(fig, name, title, subtitle=None, legend=False):
        fig.suptitle(title, x=.055, ha='left', fontsize=16, fontweight='bold')
        if subtitle: fig.text(.055, .925, subtitle, fontsize=9, color='#52616B', va='top')
        if legend:
            fig.legend([Line2D([0],[0],color=c,lw=2,ls=d) for c,d in zip(COLORS,DASHES)],
                       NAMES, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.005))
        fig.tight_layout(rect=(.025,.04,.98,.88 if subtitle else .93), h_pad=2.2, w_pad=2.5)
        fig.text(.055,-.035,RENUMBER_NOTE,fontsize=8,color='#52616B')
        for ext in ('png',):
            fig.savefig(figures/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig); files.append(name)
    def bar(ax, key, scale=1, decimals=3):
        vals=[r[key]*scale for r in report['summary']]
        bars=ax.bar(NAMES,vals,color=COLORS,width=.6)
        ax.bar_label(bars,fmt=f'%.{decimals}f',padding=4,fontsize=10)
        ax.set_ylim(0,max(vals)*1.23 if max(vals)>0 else 1)
        title,unit,direction=METRICS[key]
        ax.set_title(title+(' ↑' if direction>0 else ' ↓'),loc='left')
        ax.set_ylabel('%' if scale==100 else unit)
        ax.grid(axis='y');ax.set_axisbelow(True)

    selected=[0,100,199]
    fig,axes=plt.subplots(3,4,figsize=(11.2,8.8))
    for row,i in enumerate(selected):
        paths=[root/manifest['reference']['samples'][i]['rgb']]+[root/f'versions/{v}/frames/{i:06d}.png' for v in source_ids]
        for col,path in enumerate(paths):
            ax=axes[row,col];ax.imshow(np.asarray(Image.open(path).convert('RGB')));ax.set_xticks([]);ax.set_yticks([])
            for sp in ax.spines.values():sp.set_visible(False)
            if row==0:ax.set_title(['RGB reference',*NAMES][col],color='#1E3442' if col==0 else COLORS[col-1])
            if col==0:ax.set_ylabel(f'Frame {i*2}\n{series["times"][i]:.3f} s',fontsize=10)
    finish(fig,'04_qualitative','Fixed-time visual comparison',
           'Beginning / middle / end selected by time only. Identical truth camera, native 640 × 640, black background.')

    fig,axes=plt.subplots(2,2,figsize=(11.5,7.2))
    for v in range(3):
        rows=[r for r in report['sensitivity'] if r['version']==NAMES[v]]
        for ax,key in zip(axes[0],('iou_macro','missing_macro_percent')):
            ax.plot([r['alpha_threshold'] for r in rows],[r[key] for r in rows],marker='o',ls=DASHES[v],color=COLORS[v])
        rows=[r for r in report['sampling'] if r['version']==NAMES[v]]
        for ax,key in zip(axes[1],('rotation_jerk_rms','position_jerk_rms')):
            ax.plot([r['source_frame_step'] for r in rows],[r[key] for r in rows],marker='o',ls=DASHES[v],color=COLORS[v])
    for ax,title,y in zip(axes.flat,('Silhouette threshold sensitivity','Missing-coverage sensitivity','Rotation jerk vs temporal stride','Raw 3D jerk vs temporal stride'),
                         ('IoU','Missing (%)','rad / s³','Scene units / s³')):
        ax.set_title(title,loc='left');ax.set_ylabel(y);ax.grid()
    for ax in axes[0]:ax.set_xlabel('Alpha threshold');ax.set_xticks([.25,.5,.75])
    for ax in axes[1]:ax.set_xlabel('Source-frame step');ax.set_xticks([2,4,8])
    axes[1,0].axvspan(7.7,8.2,color='#D17A22',alpha=.12)
    axes[1,0].annotate('Step 8: rotation alias risk',xy=(8,429.8),xytext=(3.3,355),fontsize=9,
                       arrowprops=dict(arrowstyle='->',color='#8F652F'),color='#8F652F')
    finish(fig,'09_sensitivity','Evaluation sensitivity, without retraining',
           'Default step = 2, alpha = 0.50. Raw 3D jerk uses different point populations: no cross-version smoothness claim.',True)
    series.close()
    from experiment.comparison_figures import generate as generate_atlas
    files.extend(generate_atlas(root,output))
    from experiment.comparison_figures import temporal
    temporal(output,report)
    files.append('02_temporal')
    return sorted(files)
