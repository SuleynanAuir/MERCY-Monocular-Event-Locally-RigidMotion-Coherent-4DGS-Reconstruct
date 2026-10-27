"""Measured comparison atlas: common scales, explicit selection, PNG only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from experiment.export_report_analysis import METRICS, sha

COLORS = ['#507EAD', '#D77A38', '#309B8A', '#AD729B', '#879E53', '#CC6269']
INK = '#263440'
MUTED = '#667484'
RING = ['#91B8D4', '#DD9B69', '#80B8AF', '#B28EAF', '#93B783', '#D7797C']
SHORT = dict(foreground_psnr='FG PSNR', psnr='Full PSNR', ssim='Full SSIM',
             foreground_ssim='FG SSIM', lpips='LPIPS', iou='Silhouette IoU',
             missing='FG missing', part_missing='Part missing', event_l1='Event L1',
             event_cosine='Event cosine', polarity='Polarity', silent='Silent change')
PAIR_COLORS = ['#527EAA', '#AD729B', '#299A85']
GOOD = '#208573'
BAD = '#BC514E'
GAIN_CMAP = LinearSegmentedColormap.from_list('measured_gain', [BAD, '#F7F9FA', '#164F79'])
ERROR_DIFF_CMAP = LinearSegmentedColormap.from_list('error_difference', [GOOD, '#F7F9FA', BAD])


def finite(x):
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def nice_cap(value):
    if value <= 0 or not np.isfinite(value):
        return 1.
    step = 10. ** np.floor(np.log10(value))
    return float(np.ceil(value / step * 2) / 2 * step)


def radar_values(raw, keys):
    """Zero-origin linear spokes. Improvement direction never changes radius."""
    raw = np.asarray(raw, float)
    caps = np.array([1. if k in ('ssim', 'foreground_ssim', 'iou', 'event_cosine')
                     else 100. if k == 'polarity' else nice_cap(np.max(raw[:, j]))
                     for j, k in enumerate(keys)])
    return raw / caps, caps


def improvement(before, after, direction):
    """Arithmetic relative change of a reported metric, not a physical score."""
    if before is None or after is None or not np.isfinite([before, after]).all() or before == 0:
        return np.nan
    return 100 * direction * (after - before) / abs(before)


def select_gap_frames(psnr, count=4):
    """Rank all shared finite full-frame PSNRs; stable earlier-index tie break."""
    psnr = np.asarray(psnr, float)
    valid = np.isfinite(psnr).all(axis=0)
    score = sum(np.abs(psnr[i] - psnr[j]) for i in range(len(psnr)) for j in range(i+1, len(psnr)))
    eligible = np.flatnonzero(valid)
    selected = sorted(eligible, key=lambda i: (-score[i], i))[:count]
    return np.array(selected, dtype=int), score


def estimate_cycle(times, trajectory):
    """First strong recurrence after decorrelation; never infer cycles from method scores."""
    t=np.asarray(times,float);y=np.asarray(trajectory,float)
    if len(t)<16 or y.shape[0]!=len(t) or not np.isfinite(y).all():return None
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,np.median(dt),rtol=1e-3,atol=1e-8):return None
    y=y.reshape(len(t),-1);energy=np.mean(np.sum((y-y.mean(0))**2,axis=1))
    if energy<1e-12:return None
    limit=len(t)//2
    errors=np.array([np.mean(np.sum((y[lag:]-y[:-lag])**2,axis=1))/energy for lag in range(1,limit+1)])
    decorrelated=np.flatnonzero(errors>.5)
    if not len(decorrelated):return None
    first=int(decorrelated[0])+1
    for lag in range(max(3,first+1),limit+1):
        if errors[lag-1]<.1 and errors[lag-1]<=errors[lag-2] and (lag==limit or errors[lag-1]<=errors[lag]):
            return dict(period_samples=lag,period_seconds=float(lag*np.median(dt)),origin_seconds=float(t[0]),
                        recurrence_normalized_mse=float(errors[lag-1]),overlap_samples=len(t)-lag,
                        rule='first normalized recurrence MSE < 0.1 after MSE > 0.5; at least two sampled periods')
    return None


def source_cycle(root, manifest):
    path=manifest.get('reference',{}).get('source_motion')
    result=dict(status='unavailable',reason='No sufficiently recurrent common source-motion trajectory')
    if not path or not (Path(root)/path).is_file():return result
    with np.load(Path(root)/path,allow_pickle=False) as data:
        if not all(k in data for k in ('object_matrices','times_seconds','part_ids')):return result
        xyz=data['object_matrices'][:,:,:3,3].astype(float)
        variance=np.var(xyz,axis=0).sum(1)
        if not len(variance):return result
        part=int(np.argmax(variance));cycle=estimate_cycle(data['times_seconds'],xyz[:,part])
        if cycle:
            result=dict(cycle,status='available',source='common source object translation',source_file=path,
                        source_sha256=sha(Path(root)/path),part_id=int(data['part_ids'][part]),
                        part_selection='largest translation variance among source objects; independent of predicted methods',
                        boundary_convention='cycle origin at first exported source state; half-cycle dotted lines')
    return result


def shade_cycles(ax,cycle,start,end,labels=True):
    if cycle.get('status')!='available':return
    period=cycle['period_seconds'];origin=cycle['origin_seconds']
    first=int(np.floor((start-origin)/period));last=int(np.floor((end-origin)/period))
    for i in range(first,last+1):
        lo=origin+i*period;hi=lo+period
        ax.axvspan(max(lo,start),min(hi,end),color=['#DDEAF3','#F1E7D7'][i%2],alpha=.42,zorder=0,lw=0)
        if start<lo<end:ax.axvline(lo,color='#748C9A',ls='--',lw=1,alpha=.8)
        if start<lo+period/2<end:ax.axvline(lo+period/2,color='#AAB8C1',ls=':',lw=.75,alpha=.7)
        if labels and min(hi,end)>max(lo,start):
            ax.text((max(lo,start)+min(hi,end))/2,.965,f'Cycle {i+1}',transform=ax.get_xaxis_transform(),
                    ha='center',va='top',fontsize=9,color='#536675',weight='bold')


def area_contributions(support,detail):
    """Complete, common GT-area partition. Contributions sum to total pixel missing."""
    support=np.asarray(support,float);detail=np.asarray(detail,float)
    observed=support>0
    common=observed & np.isfinite(detail).all(0)
    weights=np.where(common,support,0.)
    missed=np.where(common[None,:,:],np.nan_to_num(detail),0.)*weights
    seen=observed.sum(0);mean_area=np.divide(support.sum(0),seen,out=np.zeros(len(seen)),where=seen>0)
    edges=np.array([0,16,32,64,128,256,512,1024,2048,4096,np.inf])
    bins=np.searchsorted(edges[1:],mean_area,side='right')
    groups=[np.flatnonzero((bins==j)&(seen>0)) for j in range(len(edges)-1)]
    labels=[f'{int(edges[j])}–{int(edges[j+1])} px' if np.isfinite(edges[j+1]) else f'≥{int(edges[j])} px' for j in range(len(groups))]
    keep=[j for j,g in enumerate(groups) if len(g)]
    groups=[groups[j] for j in keep];labels=[labels[j] for j in keep]
    denominator=weights.sum(1)
    numerator=np.stack([missed[:,:,group].sum(2) for group in groups],axis=2) if groups else np.empty((*detail.shape[:2],0))
    per_frame=np.divide(numerator,denominator[None,:,None],out=np.full(numerator.shape,np.nan),where=denominator[None,:,None]>0)*100
    return dict(groups=groups,labels=labels,numerator=numerator,denominator=denominator,per_frame=per_frame,
                mean_visible_pixels=mean_area,common=common,observed=observed,
                excluded_pixel_frames=float(support[observed & ~common].sum()))


def error_difference(error_before,error_after):
    """Signed per-pixel error delta: negative means improvement."""
    return np.asarray(error_after,float)-np.asarray(error_before,float)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.titleweight': 'bold', 'axes.labelcolor': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#CFD8E0',
        'grid.color': '#E4EAF0', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def canvas(title, subtitle, size=(16, 10)):
    f = plt.figure(figsize=size)
    f.text(.045, .97, title, fontsize=21, weight='bold', va='top')
    f.text(.045, .935, subtitle, fontsize=10, color=MUTED, va='top')
    return f


def empty(ax, message='Not available in this export'):
    ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes, color=MUTED)
    ax.set_axis_off()


def panel(ax, title):
    ax.set_title(title, loc='left', pad=12)
    ax.grid(alpha=.55)
    ax.set_axisbelow(True)


def method_legend(f, names, y=.89):
    f.legend([Line2D([0], [0], color=c, lw=2.5, marker='o', ms=4) for c in COLORS[:len(names)]],
             names, loc='center', bbox_to_anchor=(.5, y), ncol=len(names), frameon=False)


def save_figure(f, output, name, foot, note=''):
    f.text(.045, .065, foot, fontsize=9, color=MUTED, va='top', linespacing=1.5)
    f.text(.045, .014, note, fontsize=8, color=MUTED, va='bottom')
    f.savefig(Path(output) / 'figures' / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(f)


def quality(a, z, output, note, audit):
    names = [r['version'] for r in a['summary']]
    nv = len(names)
    f = canvas('01  /  Raw-magnitude profiles & measured gains',
               'Raw-magnitude spokes · green ↑ / +: higher is better · red ↑ / +: higher is worse', (16, 14))
    method_legend(f, names, .91)
    groups = [('Image reconstruction', ['psnr', 'foreground_psnr', 'ssim', 'foreground_ssim', 'lpips', 'iou']),
              ('Structure & event response', ['missing', 'part_missing', 'event_l1', 'event_cosine', 'polarity', 'silent'])]
    audit['radar'] = {}
    for j, (title, keys) in enumerate(groups):
        ax = f.add_axes([.14 + j*.34, .52, .36, .30], projection='polar')
        keys = [k for k in keys if all(r.get(k) is not None and np.isfinite(r[k]) and r[k] >= 0 for r in a['summary'])]
        if len(keys) < 3:
            empty(ax, title + '\nFewer than three comparable metrics'); continue
        raw = np.array([[r[k] for k in keys] for r in a['summary']])
        score, caps = radar_values(raw, keys)
        audit['radar'][title] = dict(keys=keys, cap=caps.tolist(), radius=score.tolist(), rule='raw / axis cap; no inversion')
        theta = np.arange(len(keys)) * 2*np.pi / len(keys)
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
        for v in range(nv):
            ax.plot(np.r_[theta, theta[0]], np.r_[score[v], score[v, 0]], color=COLORS[v], lw=2, marker='o', ms=3)
            ax.fill(np.r_[theta, theta[0]], np.r_[score[v], score[v, 0]], color=COLORS[v], alpha=.055)
        for k, t, cap, color in zip(keys, theta, caps, RING):
            ax.bar(t, .04, width=2*np.pi/len(keys)*.95, bottom=1.09, color=color, edgecolor='white')
            for fraction in (.5, 1.):
                ax.text(t + .07, fraction, f'{cap*fraction:.3g}', fontsize=7, color=MUTED,
                        bbox=dict(fc='white', ec='none', alpha=.8, pad=.6))
        labels = [SHORT[k] + ' ↑' + ('\n'+METRICS[k][1] if METRICS[k][1] else '') for k in keys]
        ax.set_xticks(theta, labels, fontsize=10); ax.tick_params(axis='x', pad=15)
        for tick,key in zip(ax.get_xticklabels(),keys):tick.set_color(GOOD if METRICS[key][2]>0 else BAD)
        ax.set_yticks([.25, .5, .75, 1.], ['']*4); ax.set_ylim(0, 1.14)
        ax.spines['polar'].set_visible(False)
        f.text(.32+j*.34,.875,title,ha='center',fontsize=13,weight='bold')
    keys=list(METRICS);pairs=[(i,j) for i,j in [(0,1),(1,2),(0,2)] if j<nv]
    gains=np.array([[improvement(a['summary'][b].get(k),a['summary'][e].get(k),METRICS[k][2]) for b,e in pairs] for k in keys])
    groups_by_key=dict(foreground_psnr=0,psnr=0,ssim=1,foreground_ssim=1,lpips=2,iou=2,
                       missing=3,part_missing=3,event_l1=4,event_cosine=4,polarity=5,silent=5)
    families=['PSNR','SSIM','Perception / shape','Missing coverage','Event response','Polarity / silence']
    family_colors=['#628DAE','#B28AA3','#D29A63','#6EA99D','#8B9B66','#A68CC4']
    valid=finite(gains);low=min(-100.,nice_cap(abs(valid.min()))*-1 if len(valid) and valid.min()<0 else -100.)
    high=max(350.,nice_cap(valid.max()) if len(valid) else 350.)
    audit['relative_gains']=[]
    f.text(.19,.466,'Relative improvement across metrics',fontsize=15,weight='bold')
    f.legend([Line2D([0],[0],marker='o',ls='',color=c,markersize=7) for c in family_colors],families,
             loc='center',bbox_to_anchor=(.56,.441),ncol=6,frameon=False,fontsize=8,columnspacing=1.1,handletextpad=.35)
    for col,(b,e) in enumerate(pairs):
        ax=f.add_axes([.19+col*.258,.13,.244,.278])
        ax.set_xscale('symlog',linthresh=25,linscale=.8)
        ax.set(xlim=(low*1.25,high*1.6),ylim=(len(keys)-.5,-.5))
        ax.axvspan(low*1.25,0,color='#F5E7E5',alpha=.58,lw=0)
        ax.axvspan(0,high*1.6,color='#E9F1F4',alpha=.65,lw=0)
        ax.axvline(0,color='#8A9DA9',ls='--',lw=1)
        for row,k in enumerate(keys):
            gain=gains[row,col];color=family_colors[groups_by_key[k]]
            ax.axhline(row,color='white',lw=.7,zorder=0)
            if np.isfinite(gain):
                ax.plot([0,gain],[row,row],color=color,alpha=.45,lw=1.3)
                ax.scatter(gain,row,s=38+90*np.sqrt(abs(gain)/max(abs(low),high)),color=color,
                           edgecolors='white',linewidths=.85,zorder=4)
                ax.annotate(f'{gain:+.1f}%',(gain,row),xytext=(-8 if gain>=100 else 8,0),textcoords='offset points',
                            ha='right' if gain>=100 else 'left',va='center',fontsize=8.5,color=INK,weight='bold')
            else:ax.text(.97,row,'N/A',transform=ax.get_yaxis_transform(),ha='right',va='center',color=MUTED,fontsize=8)
            audit['relative_gains'].append(dict(metric=k,pair=f'{names[b]} -> {names[e]}',percent=float(gain) if np.isfinite(gain) else None))
            if col==0:
                ax.text(-.055,row,SHORT[k],transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=9)
                ax.text(-.32,row,'+',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=13,weight='bold',color=GOOD if METRICS[k][2]>0 else BAD)
        for x,width,color in [(0,.47,COLORS[b]),(.53,.47,COLORS[e])]:
            ax.add_patch(FancyBboxPatch((x,1.025),width,.055,transform=ax.transAxes,clip_on=False,
                         boxstyle='round,pad=0.008,rounding_size=.012',facecolor=color,alpha=.18,edgecolor='none'))
        ax.text(.235,1.05,names[b],transform=ax.transAxes,ha='center',va='center',color=COLORS[b],weight='bold',fontsize=10)
        ax.text(.5,1.05,'→',transform=ax.transAxes,ha='center',va='center',color=MUTED,fontsize=13)
        ax.text(.765,1.05,names[e],transform=ax.transAxes,ha='center',va='center',color=COLORS[e],weight='bold',fontsize=10)
        ax.set_xticks([-100,0,25,100,300],['−100','0','25','100','300']);ax.tick_params(axis='x',labelsize=8,length=3)
        ax.set_yticks([]);ax.set_xlabel('Oriented change (%)',fontsize=9)
        ax.spines['left'].set_visible(False)
    audit['gain_display']=dict(kind='three paired dot facets',x_scale='symlog; linear within ±25%',
                              y='metric categories, not significance',size='absolute measured percentage',colors='six metric families')
    save_figure(f,output,'01_quality',
        'Radius: raw value / axis cap. Green ↑ / +: higher is better; red ↑ / +: higher is worse. Dots: measured gains, not significance.\nChange: 100 × direction × (after − before) / |before|. X: linear ±25%, logarithmic tails. PSNR % is arithmetic dB change.',note)


def temporal(output, a=None):
    output = Path(output); style()
    a = a or json.loads((output/'data/analysis.json').read_text())
    with np.load(output/'data/series.npz', allow_pickle=False) as zz: z = dict(zz)
    names = [r['version'] for r in a['summary']]; nv = len(names)
    cycle_path=output/'data/cycle_protocol.json';cycle=json.loads(cycle_path.read_text()) if cycle_path.exists() else {}
    cycle_label=f"GT cycle {cycle['period_seconds']:.3f} s · alternating cycle bands · dotted half-cycles" if cycle.get('status')=='available' else 'Common source cycle unavailable; no assumed period'
    f = canvas('02  /  Temporal dynamics, distributions & incremental change',
               cycle_label, (18, 13))
    method_legend(f, names, .9)
    outer = f.add_gridspec(2,3,left=.055,right=.965,bottom=.14,top=.835,wspace=.27,hspace=.36)
    for cell,k in zip(outer, ('foreground_psnr','lpips','iou','missing','event_l1','polarity')):
        grid = cell.subgridspec(2,2,width_ratios=[4,1],height_ratios=[3,1],hspace=.08,wspace=.08)
        ax = f.add_subplot(grid[0,0]); density = f.add_subplot(grid[0,1], sharey=ax); delta = f.add_subplot(grid[1,0],sharex=ax)
        values = z[k]; t = z['times'] if values.shape[1] == len(z['times']) else z['event_times']
        if len(t):
            shade_cycles(ax,cycle,float(z['times'][0]),float(z['times'][-1]))
            shade_cycles(delta,cycle,float(z['times'][0]),float(z['times'][-1]),labels=False)
        for v in range(nv):
            ax.plot(t,values[v],color=COLORS[v],lw=1.25)
            vals = np.sort(finite(values[v]))
            if vals.size:
                density.plot(np.arange(1,len(vals)+1)/len(vals)*100,vals,color=COLORS[v],lw=1.3)
                lo,med,hi=np.percentile(vals,[10,50,90])
                density.plot([12+v*30]*2,[lo,hi],color=COLORS[v],lw=3,alpha=.25)
                density.scatter(12+v*30,med,color=COLORS[v],s=12)
        for p in range(min(nv-1,2)):
            delta.plot(t, METRICS[k][2]*(values[p+1]-values[p]), color=PAIR_COLORS[p],lw=1.1,
                       label=f'{names[p]}→{names[p+1]}')
        if len(t) and len(t) != len(z['times']):
            for i in np.flatnonzero(~np.isfinite(values).all(0)):
                start = z['times'][i] if i < len(z['times']) else t[i]
                ax.axvspan(start,t[i],color='#BBC5CD',alpha=.35,lw=0)
                delta.axvspan(start,t[i],color='#BBC5CD',alpha=.35,lw=0)
        ax.set_title(METRICS[k][0],loc='left',pad=12);ax.set_ylabel(METRICS[k][1] or 'Native value')
        ax.grid(alpha=.5);ax.tick_params(labelbottom=False)
        density.set(xlim=(0,100),xlabel='CDF %');density.set_xticks([0,100]);density.tick_params(labelleft=False,labelsize=8);density.grid(alpha=.3)
        delta.axhline(0,color=MUTED,lw=.7);delta.set(xlabel='Time (s)',ylabel='Δ better ↑');delta.tick_params(labelsize=8);delta.grid(alpha=.4)
        if len(t):delta.set_xlim(t[0],t[-1] if t[-1]>t[0] else t[0]+1)
    f.legend([Line2D([0],[0],color=PAIR_COLORS[p],lw=1.8) for p in range(min(2,nv-1))],
             [f'{names[p]} → {names[p+1]} · oriented raw delta' for p in range(min(2,nv-1))],
             loc='center',bbox_to_anchor=(.5,.086),ncol=2,frameon=False,fontsize=9)
    save_figure(f, output, '02_temporal',
        'Cycle boundaries use common source-motion recurrence, not quality peaks. Curves are unsmoothed. Grey = excluded event windows.\nLower strips orient raw differences toward improvement; they retain each metric’s own unit. Distribution spread is not a confidence interval.',
        a['protocol'].get('figure_mapping_note','Version numbering follows the source mapping in data/version_mapping.json.'))


def spatial(root,a,z,manifest,output,note,audit):
    names=[r['version'] for r in a['summary']];nv=len(names)
    selected,scores=select_gap_frames(z['psnr'])
    audit['spatial_selection']=dict(metric='full-frame PSNR (dB)',rule='top 4 sum of all pairwise absolute PSNR differences; finite matched frames; earlier-index tie break',sample_indices=selected.tolist(),source_frames=[manifest['samples'][i]['frame'] for i in selected],seconds=[float(z['times'][i]) for i in selected],gap_sum_db=[float(scores[i]) for i in selected])
    has_delta=nv>=3;ncols=nv+int(has_delta)
    f=canvas('05  /  Spatial error & V3 − V2 pixel differences',
             'Error = mean-channel |RGB − GT| · Δ error = error(V3) − error(V2) · green: reduced error · red: increased error',(20,17.5))
    if not len(selected):
        empty(f.add_axes([.1,.2,.8,.6]),'Aligned finite PSNR unavailable')
        save_figure(f,output,'05_error_maps','No ranked frames available.',note);return
    refs=manifest.get('reference',{}).get('samples',[]);ids=a['protocol']['source_version_ids']
    grid=f.add_gridspec(len(selected),ncols,left=.075,right=.695,bottom=.16,top=.87,wspace=.012,hspace=.018)
    curves=f.add_gridspec(len(selected),1,left=.755,right=.965,bottom=.16,top=.87,hspace=.30)
    im=None;diffim=None;records=[];distributions=[]
    for row,i in enumerate(selected):
        ref=refs[i] if i<len(refs) else {};available=ref.get('rgb') and ref.get('aligned') and ref.get('background_matches',True)
        gt=np.asarray(Image.open(root/ref['rgb']).convert('RGB'),float)/255 if available else None
        mask=np.asarray(Image.open(root/ref['foreground_mask']))>0 if available and ref.get('foreground_mask') else None
        errors=[]
        curve_grid=curves[row,0].subgridspec(2,1,height_ratios=[3,1],hspace=.06)
        cdf=f.add_subplot(curve_grid[0]);tail=f.add_subplot(curve_grid[1],sharex=cdf)
        cdf.set_title('Foreground error · CDF & tail',loc='left',fontsize=11,pad=6)
        quantiles=[]
        for v in range(nv):
            ax=f.add_subplot(grid[row,v]);ax.set_xticks([]);ax.set_yticks([])
            if not available:empty(ax,'No aligned RGB');continue
            pred=np.asarray(Image.open(root/f'versions/{ids[v]}/frames/{i:06d}.png').convert('RGB'),float)/255
            err=np.abs(gt-pred).mean(-1);errors.append(err)
            im=ax.imshow(err,cmap='magma',vmin=0,vmax=.25,aspect='equal')
            if mask is not None and mask.any() and not mask.all():ax.contour(mask.astype(float),levels=[.5],colors=['#B9D9DB'],linewidths=.25,alpha=.5)
            ax.text(.04,.96,f'{names[v]} · {z["psnr"][v,i]:.2f} dB\nMAE {err.mean():.4f}',transform=ax.transAxes,color='white',fontsize=9,va='top')
            if row==0:ax.set_title(names[v],color=COLORS[v],fontsize=13,pad=9)
            if mask is not None and mask.any():
                vals=np.sort(err[mask]);ind=np.unique(np.linspace(0,len(vals)-1,min(1500,len(vals)),dtype=int));mass=(ind+1)/len(vals)*100
                q50,q95,q99=np.percentile(vals,[50,95,99]);quantiles.append(dict(version=names[v],p50=float(q50),p95=float(q95),p99=float(q99),foreground_pixels=len(vals)))
                cdf.step(vals[ind],mass,where='post',color=COLORS[v],lw=1.45,label=f'{names[v]}  p95 {q95:.3f}')
                tail.step(vals[ind],100-mass,where='post',color=COLORS[v],lw=1.1)
                cdf.scatter(q95,95,color=COLORS[v],s=18,edgecolors='white',linewidth=.4,zorder=4)
            if v==0:ax.set_ylabel(f'Rank {row+1} · frame {manifest["samples"][i]["frame"]}\n{z["times"][i]:.3f} s · Σ gap {scores[i]:.2f} dB',fontsize=9)
        if quantiles:
            cdf.legend(frameon=False,fontsize=8,loc='lower right',handlelength=1.4);cdf.axhline(95,ls=':',lw=.7,color=MUTED,alpha=.6)
        else:empty(cdf,'No common foreground mask');empty(tail)
        cdf.set(ylim=(0,102),xlim=(0,1),ylabel='CDF (%)');cdf.tick_params(labelbottom=False,labelsize=8);cdf.grid(alpha=.4)
        tail.set(yscale='log',ylim=(.1,100),xlabel='Foreground RGB error',ylabel='Tail %');tail.set_yticks([1,100],['1','100']);tail.tick_params(labelsize=8);tail.grid(alpha=.4)
        distributions.append(dict(sample_index=int(i),quantiles=quantiles))
        if has_delta:
            ax=f.add_subplot(grid[row,-1]);ax.set_xticks([]);ax.set_yticks([])
            if len(errors)<3:empty(ax,'No aligned error pair');continue
            difference=error_difference(errors[1],errors[2])
            diffim=ax.imshow(difference,cmap=ERROR_DIFF_CMAP,vmin=-.05,vmax=.05,aspect='equal')
            if mask is not None and mask.any() and not mask.all():ax.contour(mask.astype(float),levels=[.5],colors=['#AABBC4'],linewidths=.25,alpha=.5)
            record=dict(sample_index=int(i),frame=manifest['samples'][i]['frame'],time_seconds=float(z['times'][i]),mean_error_delta=float(difference.mean()))
            label=f'ΔMAE {difference.mean():+.4f}'
            if mask is not None and mask.any():
                vals=difference[mask];better=np.mean(vals < -1e-10)*100;worse=np.mean(vals > 1e-10)*100
                record.update(foreground_improved_percent=float(better),foreground_worse_percent=float(worse),foreground_tied_percent=float(100-better-worse))
                label+=f'\nFG − {better:.1f}% / + {worse:.1f}%'
            ax.text(.04,.96,label,transform=ax.transAxes,color=INK,fontsize=9,va='top')
            if row==0:ax.set_title(f'{names[2]} − {names[1]}',color=INK,fontsize=13,pad=9)
            records.append(record)
    if im is not None:f.colorbar(im,cax=f.add_axes([.075,.115,.44,.012]),orientation='horizontal',extend='max').set_label('RGB error · 0 to 0.25',fontsize=10)
    if diffim is not None:f.colorbar(diffim,cax=f.add_axes([.545,.115,.15,.012]),orientation='horizontal',extend='both',ticks=[-.05,0,.05]).set_label('Δ error · negative = improvement',fontsize=10)
    audit['spatial_distributions']=dict(scope='all foreground pixels per selected frame',curves='empirical CDF and exceedance tail',display_vertices_max=1500,rows=distributions)
    audit['spatial_delta']=dict(formula='mean(abs(V3-GT),RGB) - mean(abs(V2-GT),RGB)',display_range=[-.05,.05],tie_tolerance=1e-10,rows=records)
    if records:
        from experiment.export_report_analysis import table
        table(output/'tables/spatial_error_deltas.csv',records)
    save_figure(f,output,'05_error_maps',
        'Same four maximum-gap times; full frames, fixed GT, common color ranges. Display saturation only; Δ statistics are unclipped.\nRight: full-foreground CDF, p95 markers and logarithmic exceedance tails. Extreme diagnostic times; sequence-wide scores are unchanged.',note)


def parts(a,z,output,note,audit):
    names=[r['version'] for r in a['summary']];nv=len(names);n=len(z['times'])
    support=z.get('part_visibility',np.zeros((n,0)));detail=z.get('part_missing_detail',np.empty((nv,n,0)))
    f=canvas('06  /  Cycle-resolved missing area & part-loss distribution',
             'Common GT cycles · embedded total-loss traces · every observed part · fixed alpha 0.50',(18.5,13))
    method_legend(f,names,.895)
    if not support.shape[1] or not np.isfinite(detail).any():
        empty(f.add_axes([.1,.2,.8,.6]),'Common fixed part masks unavailable')
        save_figure(f,output,'06_parts','No comparable fixed part masks.',note);return
    c=area_contributions(support,detail);blocks=np.array_split(np.arange(n),min(24,n));denom=c['denominator']
    ng=len(c['groups']);heat=np.full((nv,ng,len(blocks)),np.nan)
    for j,block in enumerate(blocks):
        if denom[block].sum()>0:heat[:,:,j]=c['numerator'][:,block].sum(1)/denom[block].sum()*100
    total_denominator=denom.sum()
    pooled=c['numerator'].sum((1,2))/total_denominator*100 if total_denominator else np.full(nv,np.nan)
    group_mean=c['numerator'].sum(1)/total_denominator*100 if total_denominator else np.full((nv,ng),np.nan)
    frame_total=np.nansum(c['per_frame'],axis=2);frame_total[:,denom<=0]=np.nan
    full_heat=c['per_frame'].transpose(0,2,1)
    cap=nice_cap(np.nanmax(full_heat)) if np.isfinite(full_heat).any() else 1.
    trace_cap=nice_cap(np.nanmax(frame_total)) if np.isfinite(frame_total).any() else 1.
    cmap=LinearSegmentedColormap.from_list('visible_loss',['#F3F8FA','#BCD8E5','#619AAF','#245973']);cmap.set_bad('#DCE3E9')
    gs=f.add_gridspec(1,nv,left=.085,right=.92,bottom=.49,top=.825,wspace=.35);image=None
    labels=[f'{label} ({len(group)})' for label,group in zip(c['labels'],c['groups'])]
    times=np.asarray(z['times']);dt=float(np.median(np.diff(times))) if n>1 else 1.
    edges=np.r_[times[0]-dt/2,(times[:-1]+times[1:])/2,times[-1]+dt/2]
    cycle=audit.get('cycles',{});start=float(times[0]);end=float(edges[-1])
    for v in range(nv):
        ax=f.add_subplot(gs[0,v])
        image=ax.pcolormesh(edges,np.arange(ng+1)-.5,full_heat[v],cmap=cmap,norm=PowerNorm(gamma=.5,vmin=0,vmax=cap),shading='flat',rasterized=True)
        # One coordinate system: the trace occupies a reserved band above the heatmap rows.
        trace_y=lambda values:-.85-np.asarray(values)/trace_cap*2.2
        ax.fill_between(times,-.85,trace_y(frame_total[v]),color=COLORS[v],alpha=.10,lw=0)
        ax.plot(times,trace_y(frame_total[v]),color=COLORS[v],lw=1.7)
        ax.axhline(trace_y(pooled[v]),xmin=0,xmax=1,color=COLORS[v],ls=':',lw=.7,alpha=.6)
        for tick in [0,trace_cap/2,trace_cap]:
            ax.text(1.012,trace_y(tick),f'{tick:g}%',transform=ax.get_yaxis_transform(),fontsize=7,va='center',color=MUTED)
        ax.text(.02,-3.12,'Total missing',transform=ax.get_yaxis_transform(),fontsize=8,color=COLORS[v],va='bottom',weight='bold')
        for boundary in np.arange(ng+1)-.5:ax.axhline(boundary,color='white',lw=.35,alpha=.55)
        if cycle.get('status')=='available':
            period=cycle['period_seconds'];origin=cycle['origin_seconds']
            for cy in range(int(np.floor((start-origin)/period)),int(np.floor((end-origin)/period))+1):
                lo=max(start,origin+cy*period);hi=min(end,origin+(cy+1)*period)
                if hi<=lo:continue
                color=['#DDEAF3','#F1E7D7'][cy%2]
                ax.fill_between([lo,hi],-4.1,-3.45,color=color,lw=0)
                ax.text((lo+hi)/2,-3.78,f'Cycle {cy+1}',ha='center',va='center',fontsize=9,weight='bold',color='#536675')
                if start<lo<end:ax.axvline(lo,color='#536675',ls='--',lw=1.2,alpha=.9)
                half=origin+(cy+.5)*period
                if start<half<end:ax.axvline(half,color='#7B8F9A',ls=':',lw=1,alpha=.85)
            ticks=np.arange(origin,end+dt,period/2);ticks=ticks[(ticks>=start)&(ticks<=end)]
        else:
            ticks=np.linspace(start,float(times[-1]),5);ax.text((start+end)/2,-3.78,'Cycle unavailable',ha='center',color=MUTED,fontsize=8)
        ax.set_xticks(ticks,[f'{t:.2f}' for t in ticks],fontsize=8)
        ax.set_yticks(range(ng),labels,fontsize=8)
        ax.set(xlim=(start,end),ylim=(ng-.5,-4.25),xlabel='Time (s)',ylabel='GT area bin · part count' if v==0 else '')
        ax.set_title(f'{names[v]}  ·  pooled {pooled[v]:.3f}%',color=COLORS[v],pad=12)
    if image is not None:
        cb=f.colorbar(image,cax=f.add_axes([.955,.50,.009,.23]));cb.set_label('Missing-area contribution (pp) · √ scale',fontsize=9)
    cdf=f.add_axes([.075,.15,.53,.255]);agg=f.add_axes([.735,.15,.22,.255])
    part_weights=np.where(c['common'],support,0.);part_den=part_weights.sum(0)
    pooled_parts=np.divide(np.nansum(np.where(c['common'][None,:,:],detail,np.nan)*support,axis=1),part_den[None,:],out=np.full((nv,support.shape[1]),np.nan),where=part_den[None,:]>0)*100
    macro=[];tail_rows=[];thresholds=[5,25,50]
    tail=cdf.inset_axes([.29,.15,.67,.48]);tail.set_facecolor('#F8FAFB')
    for v in range(nv):
        valid=np.isfinite(pooled_parts[v])&(part_den>0);vals=pooled_parts[v,valid];weights=part_den[valid];order=np.argsort(vals)
        if len(vals):
            equal=100*np.arange(1,len(vals)+1)/len(vals);weighted=100*np.cumsum(weights[order])/weights.sum()
            cdf.step(vals[order],equal,where='post',color=COLORS[v],lw=1.7)
            cdf.step(vals[order],weighted,where='post',color=COLORS[v],lw=1.3,ls='--')
            cdf.fill_between(vals[order],equal,weighted,step='post',color=COLORS[v],alpha=.07)
            counts=np.array([(vals>=t).sum() for t in thresholds]);shares=counts/len(vals)*100
            positions=np.arange(len(thresholds))+(v-(nv-1)/2)*.22
            tail.bar(positions,shares,width=.19,color=COLORS[v],alpha=.80)
            for x,y,count in zip(positions,shares,counts):tail.text(x,y+.5,f'{y:.1f}',ha='center',va='bottom',fontsize=7,color=COLORS[v])
            tail_rows.append(dict(version=names[v],thresholds_percent=thresholds,part_counts=counts.tolist(),part_percent=shares.tolist(),total_parts=int(len(vals))))
        mean=a['summary'][v].get('part_missing');macro.append(mean)
        agg.scatter(v-.08,pooled[v],s=65,color=COLORS[v],marker='D',zorder=4)
        agg.annotate(f'{pooled[v]:.3f}%',(v-.08,pooled[v]),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=10,color=COLORS[v],weight='bold')
        if mean is not None:
            agg.plot([v-.08,v+.08],[pooled[v],mean],color=COLORS[v],alpha=.3,lw=2)
            agg.scatter(v+.08,mean,s=55,facecolors='white',edgecolors=COLORS[v],linewidth=1.7,zorder=4)
            agg.annotate(f'{mean:.3f}%',(v+.08,mean),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9,color=COLORS[v])
    cdf.set(xlim=(0,100),ylim=(0,102),xlabel='Within-part missing (%)',ylabel='Cumulative share (%)');panel(cdf,'Within-part loss · distribution & severe tails')
    cdf.legend([Line2D([0],[0],color=MUTED),Line2D([0],[0],color=MUTED,ls='--')],['Equal parts','GT area weighted'],fontsize=8,loc='lower left',frameon=False)
    tail.set(xticks=range(len(thresholds)),xticklabels=[f'≥ {t}%' for t in thresholds],ylabel='Parts (%)');tail.set_title('Parts exceeding loss thresholds',fontsize=9,loc='left',pad=5);tail.tick_params(labelsize=7);tail.margins(y=.3);tail.grid(axis='y',alpha=.3);tail.set_axisbelow(True)
    agg.plot(np.arange(nv)-.08,pooled,color=MUTED,lw=1,alpha=.6,zorder=1)
    agg.set(xticks=range(nv),xticklabels=names,ylabel='Missing (%)',xlim=(-.5,nv-.5));agg.margins(y=.3);panel(agg,'Two aggregation scales')
    agg.legend([Line2D([0],[0],marker='D',color=MUTED,ls=''),Line2D([0],[0],marker='o',markerfacecolor='white',color=MUTED,ls='')],['GT area pooled','Equal parts / frame'],fontsize=8,loc='upper left',frameon=False)
    f.text(.085,.442,'Full-resolution sample columns · dashed: full cycle · dotted: half-cycle · trace scale shared across versions',fontsize=9,color=MUTED)
    audit['parts']=dict(definition='missing pixel contribution by GT mean-area bins; percentage points of total common visible area',alpha_threshold=.5,
        size_groups=[dict(label=label,part_ids=z['part_ids'][g].tolist()) for label,g in zip(c['labels'],c['groups'])],
        total_parts=int((support.sum(0)>0).sum()),excluded_pixel_frames=c['excluded_pixel_frames'],
        heatmap_gamma=.5,heatmap_cap=cap,heatmap_samples=n,trace_scale_percent=[0,trace_cap],heatmap_cycle_source='data/cycle_protocol.json',
        gt_area_pooled_percent=[float(x) if np.isfinite(x) else None for x in pooled],within_part_tail_counts=tail_rows,
        original_equal_part_percent=macro,monotonic_gt_area_measured=bool(np.all(np.diff(pooled)<=0)),
        monotonic_equal_part_measured=bool(all(x is not None for x in macro) and np.all(np.diff(macro)<=0)))
    from experiment.export_report_analysis import table
    rows=[]
    for v in range(nv):
        for g,label in enumerate(c['labels']):
            for b,block in enumerate(blocks):rows.append(dict(version=names[v],area_group=label,part_count=len(c['groups'][g]),start_seconds=float(z['times'][block[0]]),end_seconds=float(z['times'][block[-1]]),missing_pixels=float(c['numerator'][v,block,g].sum()),common_gt_pixels=float(denom[block].sum()),contribution_pp=float(heat[v,g,b]) if np.isfinite(heat[v,g,b]) else None))
    table(output/'tables/part_area_contributions.csv',rows)
    save_figure(f,output,'06_parts',
        'Heatmap: area-loss contributions, not within-part missing rates. Embedded traces sum every GT-area bin at each sample.\nSame denominator, bins, masks and scales for all versions. CDF shading: aggregation gap. True regressions remain visible.',note)

def efficiency(a,z,output,note,audit=None):
    from scipy.ndimage import gaussian_filter
    names=[r['version'] for r in a['summary']];nv=len(names)
    f=canvas('07  /  Measured quality–latency distribution',
             'Individual measurements · cycle markers · 50 / 90% density contours · latency quantiles',(17,11.5))
    method_legend(f,names,.895)
    ax=f.add_axes([.075,.18,.89,.64]);lat=z.get('latencies',np.empty((0,0,nv)))
    qualities=finite(z['foreground_psnr']);all_latencies=finite(lat)
    if not len(qualities) or not len(all_latencies):
        empty(ax,'Aligned latency / quality measurements unavailable');save_figure(f,output,'07_efficiency','Performance data unavailable.',note);return
    qmin,qmax=float(qualities.min()),float(qualities.max());qspan=max(qmax-qmin,1.)
    lower=qmin-qspan*.30;upper=qmax+qspan*.15
    xmin=max(0,float(all_latencies.min())-1);xmax=float(all_latencies.max())+max(2.,float(np.ptp(all_latencies))*.08)
    cycle=(audit or {}).get('cycles',{});times=z['times']
    cycle_ids=np.floor((times-cycle['origin_seconds']+1e-8)/cycle['period_seconds']).astype(int) if cycle.get('status')=='available' else np.zeros(len(times),int)
    qrows=[]
    for v,r in enumerate(a['summary']):
        x,y=r.get('latency_ms'),r.get('foreground_psnr');values=finite(lat[...,v]);color=COLORS[v]
        if x is None or y is None or not np.isfinite([x,y]).all() or not len(values):continue
        if lat.shape[1]==z['foreground_psnr'].shape[1]:
            xx=lat[:,:,v].reshape(-1);yy=np.tile(z['foreground_psnr'][v],lat.shape[0]);cy=np.tile(cycle_ids,lat.shape[0])
            valid=np.isfinite(xx)&np.isfinite(yy)
            for j,marker in enumerate(['o','^']):
                pick=valid&(cy%2==j)
                ax.scatter(xx[pick],yy[pick],s=17 if j==0 else 21,color=color,alpha=.40,
                           marker=marker,edgecolors='white',linewidths=.25,zorder=3)
            if valid.sum()>10 and np.ptp(xx[valid])>0 and np.ptp(yy[valid])>0:
                hist,xe,ye=np.histogram2d(xx[valid],yy[valid],bins=(50,40));density=gaussian_filter(hist,1.2)
                ordered=np.sort(density.ravel())[::-1];mass=np.cumsum(ordered)/ordered.sum()
                levels=sorted(set(float(ordered[min(np.searchsorted(mass,p),len(ordered)-1)]) for p in [.5,.9]))
                if len(levels)==2 and levels[0]>0:
                    cx=(xe[:-1]+xe[1:])/2;cyy=(ye[:-1]+ye[1:])/2
                    ax.contourf(cx,cyy,density.T,levels=[levels[0],density.max()+1],colors=[color],alpha=.065,zorder=1)
                    ax.contour(cx,cyy,density.T,levels=levels,colors=[color],linewidths=[1,1.6],alpha=.65,zorder=4)
        lo,q25,med,q75,hi=np.percentile(values,[5,25,50,75,95]);qlo,qhi=np.percentile(finite(z['foreground_psnr'][v]),[5,95])
        ax.plot([lo,hi],[y,y],color=color,lw=1.2,zorder=5)
        ax.plot([q25,q75],[y,y],color=color,lw=5,alpha=.7,zorder=5,solid_capstyle='round')
        ax.plot([x,x],[qlo,qhi],color=color,lw=1.2,zorder=5)
        ax.scatter([lo,hi],[y,y],marker='|',color=color,s=70,zorder=5)
        points=r.get('point_count') or 0
        ax.scatter(x,y,s=max(65,points/5000),color=color,edgecolors='white',linewidths=1.8,zorder=6)
        # Marginal density lanes sit below every measured quality value and are explicitly labelled.
        baseline=lower+qspan*(.03+v*.06)
        hist,edges=np.histogram(values,bins=100,range=(xmin,xmax));centers=(edges[:-1]+edges[1:])/2
        density=gaussian_filter(hist.astype(float),.8);density=density/max(density.max(),1)*qspan*.04
        ax.fill_between(centers,baseline,baseline+density,color=color,alpha=.28,lw=0,zorder=1)
        ax.plot(centers,baseline+density,color=color,lw=1,zorder=2)
        ax.plot([med,med],[baseline,baseline+qspan*.037],color=color,lw=1.1)
        ax.plot([hi,hi],[baseline,baseline+qspan*.037],color=color,ls=':',lw=1.2)
        ax.text(xmax-.4,baseline+qspan*.006,f'{names[v]}  {med:.2f} / {hi:.2f}',ha='right',va='bottom',fontsize=9,color=color,weight='bold')
        qrows.append(dict(version=names[v],measurements=len(values),p05=float(lo),p25=float(q25),p50=float(med),p75=float(q75),p95=float(hi)))
        labelx=xmin+[.30,.29,.76][v]*(xmax-xmin) if nv==3 else x+.05*(xmax-xmin)
        labely=y+[.12,-.04,-.02][v]*qspan if nv==3 else y+.06*qspan
        iou=r.get('iou');label=f'{names[v]}  {y:.2f} dB\n{x:.2f} ms · {points/1e6:.3f} M'
        if iou is not None:label+=f'\n{iou*100:.2f}% IoU'
        ax.annotate(label,(x,y),xytext=(labelx,labely),textcoords='data',fontsize=11,color=color,weight='bold',
                    arrowprops=dict(arrowstyle='-',color=color,lw=.8,alpha=.65),va='center')
    ax.axhline(qmin-qspan*.044,color='#CCD7DF',lw=.8)
    ax.text(xmin+.25,qmin-qspan*.068,'Latency density',fontsize=9,color=MUTED,va='top')
    ax.text(xmax-.4,qmin-qspan*.068,'p50 / p95 · ms',fontsize=9,color=MUTED,va='top',ha='right')
    ax.set(xlim=(xmin,xmax),ylim=(lower,upper),xlabel='Latency (ms) ↓',ylabel='Foreground PSNR (dB) ↑')
    ax.set_yticks(np.arange(np.ceil(qmin/2)*2,upper,2));ax.grid(alpha=.42);ax.set_axisbelow(True)
    handles=[Line2D([0],[0],marker='o',color=MUTED,ls='',markersize=5),Line2D([0],[0],marker='^',color=MUTED,ls='',markersize=5),
             Line2D([0],[0],color=MUTED,lw=1),Line2D([0],[0],color=MUTED,lw=5)]
    cycle_labels=['Cycle 1','Cycle 2'] if cycle_ids.max()<2 else ['Odd cycles','Even cycles']
    labels=cycle_labels+['p5–p95','p25–p75'] if cycle.get('status')=='available' else ['Measurements','—','p5–p95','p25–p75']
    ax.legend(handles,labels,loc='upper left',frameon=False,ncol=4,fontsize=9,columnspacing=1.1)
    perf=a.get('performance',{});mem=perf.get('memory_peak');memory=f'{mem/1024**2:.1f} MiB' if mem else 'N/A'
    f.text(.075,.115,f'{lat.shape[0]} repeats × {lat.shape[1]} frames   ·   ● centroid area ∝ Gaussians   ·   process peak {memory}',fontsize=10,color=MUTED)
    if audit is not None:audit['efficiency_display']=dict(points='all finite repeat-by-frame latencies with matched frame PSNR; no jitter or trimming',cycle_markers='odd/even common GT cycle',
        density='50x40 equal-width 2D histogram; Gaussian sigma 1.2 bins; contours enclose 50/90% smoothed grid mass',latency_quantiles=qrows,latency_limits=[xmin,xmax])
    save_figure(f,output,'07_efficiency','Density lanes: offset, not PSNR. Contours: distribution mass, not confidence regions. Timing scope: export record.',note)

def motion(a,z,output,note,audit):
    names=[r['version'] for r in a['summary']];nv=len(names);rows=a.get('motion',[])
    f=canvas('08  /  Constraint signature & resolved motion differences',
             'Native constraint axes · source-aligned cycles · point-jerk trajectories and residuals',(17,10))
    method_legend(f,names,.895)
    ax=f.add_axes([.075,.27,.31,.45]);main=f.add_axes([.50,.45,.45,.36]);res=f.add_axes([.50,.17,.45,.17])
    keys=[('joint_gap_max','Joint gap\nscene units',True),('swing_max_rad','Off-axis swing\nrad',True),('rotation_jerk_rms','SO(3) jerk\nrad / s³',False)]
    scales=[]
    for j,(key,label,log) in enumerate(keys):
        vals=np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows],float)
        vv=finite(vals);lo,hi=(0.,1.) if not len(vv) else ((10.**np.floor(np.log10(max(vv.min(),1e-12))),10.**np.ceil(np.log10(max(vv.max(),1e-12)))) if log else (0.,nice_cap(vv.max())))
        if hi<=lo:hi=lo*10 if log else lo+1
        scaled=(np.log10(np.maximum(vals,1e-12))-np.log10(lo))/(np.log10(hi)-np.log10(lo)) if log else (vals-lo)/(hi-lo)
        scales.append(scaled)
        ax.plot([j,j],[0,1],color='#B3C4D0',lw=1.5)
        for tick in [0,.25,.5,.75,1]:
            val=10**(np.log10(lo)+tick*(np.log10(hi)-np.log10(lo))) if log else lo+tick*(hi-lo)
            ax.plot([j-.018,j+.018],[tick,tick],color=MUTED,lw=.6)
            ax.text(j-.045,tick,f'{val:.2g}',ha='right',va='center',fontsize=8,color=MUTED)
        ax.text(j,1.12,label+('\nlog scale' if log else '\nlinear scale'),ha='center',fontsize=10,weight='bold')
    if rows:
        scaled=np.array(scales).T
        for v,values in enumerate(scaled):
            ax.plot(range(3),values,'o-',color=COLORS[v],lw=2,ms=5,alpha=.8)
    ax.set(xlim=(-.35,2.3),ylim=(-.08,1.1));ax.set_axis_off()
    f.text(.075,.86,'Constraint profiles',fontsize=13,weight='bold')
    cycle=audit.get('cycles',{})
    if len(z['times']):
        shade_cycles(main,cycle,float(z['times'][0]),float(z['times'][-1]))
        shade_cycles(res,cycle,float(z['times'][0]),float(z['times'][-1]),labels=False)
    cycle_label=f"T = {cycle['period_seconds']:.3f} s · common GT" if cycle.get('status')=='available' else 'Cycle unavailable'
    f.text(.50,.855,cycle_label,fontsize=11,color=MUTED)
    curves=[]
    for v in range(nv):
        curve=z.get(f'position_{v}',np.array([]));curves.append(curve)
        if not len(curve):continue
        t=z['times'][3:3+len(curve)];main.plot(t,curve,color=COLORS[v],lw=1.5,label=names[v])
        peak=int(np.nanargmax(curve)) if np.isfinite(curve).any() else None
        if peak is not None:main.annotate(f'{names[v]} peak {curve[peak]:.2f}',(t[peak],curve[peak]),xytext=(12,12+v*18),textcoords='offset points',fontsize=9,color=COLORS[v],arrowprops=dict(arrowstyle='-',color=COLORS[v]))
    main.set(xlabel='Time (s)',ylabel='Point jerk · scene units / s³');main.margins(y=.35);main.set_ylim(bottom=0);panel(main,'Raw 3D jerk · full scale')
    if curves and len(curves[0]):
        base=curves[0]
        for v in range(1,nv):
            if len(curves[v]) != len(base):continue
            residual=curves[v]-base;res.plot(z['times'][3:3+len(base)],residual,color=COLORS[v],lw=1.5,label=f'{names[v]} − {names[0]}')
        res.set_yscale('symlog',linthresh=.1,linscale=1)
        res.axhline(0,color=MUTED,lw=.8)
        res.legend(frameon=False,fontsize=8,loc='lower right')
    res.set(xlabel='Time (s)',ylabel='Jerk difference · symlog');panel(res,'Residual · linear ±0.1 / logarithmic tails')
    audit['motion_display']=dict(constraints='parallel independent native axes; gap/swing log, SO3 jerk linear',residual='position_v - position_0',residual_scale='symlog linthresh=0.1',causal_smoothing_claim=False)
    save_figure(f,output,'08_motion',
        'Common GT cycles; dotted lines mark half-cycles. Native constraint axes retain independent units. Scene units are not metres.\nPoint populations differ across versions: lower raw point jerk cannot establish improved smoothing. No independent motion truth; no cross-version point matching.',note)


def generate(root, output, progress=None):
    root,output=Path(root),Path(output);(output/'figures').mkdir(exist_ok=True)
    # Only replace generated figure artifacts, never source render frames or numeric evidence.
    for path in (output/'figures').iterdir():
        if path.is_file() and (path.suffix.lower() in ('.svg','.pdf') or path.name=='03_paired_deltas.png'):path.unlink()
    a=json.loads((output/'data/analysis.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    with np.load(output/'data/series.npz',allow_pickle=False) as series:z=dict(series)
    style();note=a['protocol'].get('figure_mapping_note','Display renumbering: V1 = original V2; V2 = original V1; V3 unchanged. Source artifacts are unchanged.')
    audit={'revision':'cycle-integrated-atlas-20261027','numerical_metrics_recomputed':False};files=[]
    audit['cycles']=source_cycle(root,manifest)
    (output/'data/cycle_protocol.json').write_text(json.dumps(audit['cycles'],indent=2,allow_nan=False))
    for name,action in [
        ('01_quality',lambda:quality(a,z,output,note,audit)),
        ('05_error_maps',lambda:spatial(root,a,z,manifest,output,note,audit)),
        ('06_parts',lambda:parts(a,z,output,note,audit)),
        ('07_efficiency',lambda:efficiency(a,z,output,note,audit)),
        ('08_motion',lambda:motion(a,z,output,note,audit))]:
        action();files.append(name)
        if progress:progress(len(files),5,name)
    (output/'data/visualization_protocol.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False))
    return files
