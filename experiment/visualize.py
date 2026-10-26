"""Export paper-style plots; unavailable measurements stay blank, never zero."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/scgs-experiment-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .protocol import ROOT, METRICS

COLORS=['#147D92','#D58046','#6865B0','#38957B','#C75B79']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
        'axes.labelcolor':'#314351','text.color':'#253645','axes.edgecolor':'#CBD5DF',
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,
        'svg.fonttype':'none','axes.grid':True,'grid.color':'#E8EDF2','grid.linewidth':.65})


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)


def inventory_plots(inventory,root=ROOT):
    style();rows=inventory['scenes'];names=[r['scene'].title() for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained',gridspec_kw={'width_ratios':[1.25,1,1]})
    x=np.arange(len(rows));w=.36
    for j,key in enumerate(('g0','corrected')):
        values=[r['gaussian_counts'][key]/1e6 for r in rows]
        axes[0].bar(x+(j-.5)*w,values,w,color=COLORS[j],label='G0' if j==0 else 'Supplied correction')
    axes[0].set(xticks=x,xticklabels=names,ylabel='Gaussians (millions)',title='Input model sizes')
    axes[0].legend(frameon=False,fontsize=8);axes[0].tick_params(axis='x',rotation=35)
    values=[r['events'].get('count',0)/1e6 if r['events']['valid'] else np.nan for r in rows]
    axes[1].bar(x,values,color=[COLORS[0] if np.isfinite(v) else '#DDE3E8' for v in values])
    for i,v in enumerate(values):
        if not np.isfinite(v):axes[1].text(i,.2,'Truncated',rotation=90,ha='center',color='#B04C49')
    axes[1].set(xticks=x,xticklabels=names,ylabel='Validated events (millions)',title='Raw event availability')
    axes[1].tick_params(axis='x',rotation=35)
    for i,row in enumerate(rows):
        for j,(label,ids) in enumerate(row['split']['frames'].items()):
            axes[2].barh(i,len(ids)/row['video']['fps'],left=ids[0]/row['video']['fps'],
                color=COLORS[j],label=label.title() if i==0 else None,height=.55)
    axes[2].set(yticks=x,yticklabels=names,xlabel='Time (s)',title='Frozen temporal split')
    axes[2].invert_yaxis();axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Five-scene input audit · dataset and protocol checks',fontsize=16,fontweight='bold')
    fig.supxlabel('Common input to C1 / C2 / C3: the supplied corrected G0',fontsize=10)
    save(fig,root/'01_main_comparison/visualization/input_audit')


def metric_panels(rows,path,title='Controlled method comparison'):
    """One panel per unit; do not normalize unlike units into a radar score."""
    style();fig,axes=plt.subplots(2,5,figsize=(18,7),layout='constrained')
    english=['RGB L1 ↓','PSNR ↑ (dB)','SSIM ↑','Held-out event L1 ↓','Passing intervals ↑',
             'Visible missing ↓','Rear missing ↓','Joint gap ↓','Angular jerk ↓','3D jerk ↓']
    for ax,(key,_,direction),label in zip(axes.flat,METRICS,english):
        finite=[r for r in rows if r.get('metrics',{}).get(key,{}).get('value') is not None]
        ax.set_title(label,loc='left',fontweight='bold',fontsize=11)
        if not finite:
            ax.text(.5,.5,'No eligible measurements',ha='center',va='center',transform=ax.transAxes,color='#7D8A98',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([]);continue
        labels=[r['scene']+' / '+r['method'] for r in finite]
        ax.bar(np.arange(len(finite)),[r['metrics'][key]['value'] for r in finite],color=[COLORS[i%3] for i in range(len(finite))])
        ax.set_xticks(np.arange(len(finite)),labels,rotation=70,ha='right',fontsize=7)
    fig.suptitle(title,fontsize=16,fontweight='bold')
    save(fig,path)
