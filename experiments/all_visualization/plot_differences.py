"""Descriptive object differences; no cross-protocol ranking or pooled scores."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def draw(objects,palette,markers,save):
    stats={}
    def series(o,key):
        return np.asarray(o['series'][o['recommended']][key],dtype=float)
    names=[o['id'].title() for o in objects]
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    for ax,key,label in zip(axes,['psnr','lpips'],['PSNR (dB) ↑','LPIPS ↓']):
        arrays=[series(o,key) for o in objects]
        for i,y in enumerate(arrays):
            y=y[np.isfinite(y)]
            violin=ax.violinplot(y,positions=[i],vert=False,widths=.75,showextrema=False)
            for body in violin['bodies']:body.set_facecolor(palette[i]);body.set_alpha(.25)
            q=np.quantile(y,[.05,.25,.5,.75,.95])
            ax.plot(q[[0,4]],[i,i],color=palette[i],lw=1.2)
            ax.plot(q[[1,3]],[i,i],color=palette[i],lw=5,solid_capstyle='round')
            ax.scatter(q[2],i,color='white',edgecolor=palette[i],s=30,zorder=4)
            ax.annotate(f'{q[2]:.3f}',(q[4],i),xytext=(6,0),textcoords='offset points',va='center',fontsize=9,color=palette[i])
            stats.setdefault(objects[i]['id'],{})[key]={'n':len(y),'p05':float(q[0]),'p25':float(q[1]),'median':float(q[2]),'p75':float(q[3]),'p95':float(q[4])}
        ax.set_yticks(range(9),names);ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis='x',alpha=.18);ax.margins(x=.2)
    fig.legend([Line2D([],[],color='#657784',lw=1),Line2D([],[],color='#657784',lw=5),Line2D([],[],marker='o',color='#657784',markerfacecolor='white',ls='')],['P05–P95','P25–P75','Median'],loc='upper center',ncol=3,frameon=False)
    fig.subplots_adjust(top=.91,wspace=.25,left=.09,right=.97)
    save(fig,'05_frame_distributions.png','Per-object frame distributions. Violin is descriptive KDE, whiskers P05-P95, box P25-P75, center median; no pooling.')

    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    for ax,key,label in zip(axes,['psnr','lpips'],['PSNR (dB)','LPIPS']):
        for i,o in enumerate(objects):
            y=series(o,key);y=np.sort(y[np.isfinite(y)])
            cumulative=np.arange(1,len(y)+1)/len(y)*100
            for low,high in zip(np.linspace(0,1,25)[:-1],np.linspace(0,1,25)[1:]):
                ax.fill_between(y,cumulative*low,cumulative*high,step='post',color=palette[i],alpha=.13*high**4,linewidth=0,zorder=1)
            ax.step(y,cumulative,where='post',color=palette[i],lw=1.8,label=names[i],zorder=3)
        ax.set_xlabel(label);ax.set_ylabel('Cumulative frames (%)');ax.set_ylim(0,101);ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False)
    fig.subplots_adjust(top=.82,wspace=.23)
    save(fig,'06_quality_ecdf.png','Unsmoothed empirical CDFs, each object weighted within its own sequence; different evaluation protocols.')

    fig,axes=plt.subplots(1,2,figsize=(15,7))
    coords=[];trajectories={}
    for i,o in enumerate(objects):
        y=series(o,'psnr');mu=float(np.nanmean(y));sd=float(np.nanstd(y))
        stats[o['id']]['psnr'].update(mean=mu,std=sd)
        coords.append([(mu,sd),(float(o['metrics']['lpips']),float(o['metrics']['iou']))])
        base='旧 V3' if o['special'] else 'V2'
        baseline=o['series'][base]
        old_psnr=np.asarray(baseline['psnr'],dtype=float)
        before=[(float(np.nanmean(old_psnr)),float(np.nanstd(old_psnr))),
                (float(np.nanmean(baseline['lpips'])),float(np.nanmean(baseline['iou'])))]
        trajectories[o['id']]={'baseline':base,'before':before,'selected':coords[-1]}
        # Measured frame spread, not uncertainty in the mean.
        q=np.nanquantile(y,[.25,.75])
        axes[0].plot(q,[sd,sd],color=palette[i],alpha=.3,lw=6,solid_capstyle='round',zorder=1)
        from matplotlib.patches import Rectangle
        lq=np.nanquantile(series(o,'lpips'),[.25,.75]);iq=np.nanquantile(series(o,'iou'),[.25,.75])
        axes[1].add_patch(Rectangle((lq[0],iq[0]),lq[1]-lq[0],iq[1]-iq[0],facecolor=palette[i],edgecolor=palette[i],alpha=.10,lw=1,zorder=1))
        for j,ax in enumerate(axes):
            bx,by=before[j];nx,ny=coords[-1][j]
            ax.scatter(bx,by,s=45,marker=markers[i],facecolor='white',edgecolor=palette[i],lw=1,zorder=2)
            ax.annotate('',xy=(nx,ny),xytext=(bx,by),arrowprops=dict(arrowstyle='->',color=palette[i],lw=1.2,alpha=.65,shrinkA=5,shrinkB=6),zorder=2)
            x,y0=coords[-1][j];ax.scatter(x,y0,s=90,color=palette[i],marker=markers[i],edgecolor='white',lw=.7,zorder=3)
            # Numeric IDs avoid colliding long names in dense metric clusters.
            ax.annotate(str(i+1),(x,y0),xytext=(6,7),textcoords='offset points',fontsize=9,color=palette[i],bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5))
    axes[0].set(xlabel='Mean PSNR (dB) ↑',ylabel='Frame PSNR standard deviation (dB) ↓')
    axes[1].set(xlabel='LPIPS ↓',ylabel='IoU ↑')
    for ax in axes:ax.grid(alpha=.18);ax.margins(.16)
    handles=[Line2D([],[],color=palette[i],marker=markers[i],ls='',label=f'{i+1}  {n}') for i,n in enumerate(names)]
    fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False)
    extra=[Line2D([],[],marker='o',ls='',color='#657784',markerfacecolor='white'),Line2D([],[],marker='o',ls='',color='#657784'),Line2D([],[],lw=6,alpha=.3,color='#657784')]
    fig.legend(extra,['V2 / Previous V3 *','Selected V3','Frame P25–P75'],loc='upper center',bbox_to_anchor=(.5,.89),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(top=.76,wspace=.23)
    save(fig,'07_quality_stability.png','Within-object baseline-to-selected arrows; measured interquartile spread, not confidence intervals. No cross-protocol ranking.')

    fig,ax=plt.subplots(figsize=(11,7))
    for i,o in enumerate(objects):
        full=float(o['metrics']['psnr']);fg=float(o['metrics']['foreground_psnr'])
        ax.plot([fg,full],[i,i],color=palette[i],lw=5,alpha=.22)
        ax.scatter(fg,i,s=60,marker='o',facecolor='white',edgecolor=palette[i],zorder=3)
        ax.scatter(full,i,s=60,marker='D',color=palette[i],zorder=3)
        ax.annotate(f'{full-fg:+.2f} dB',(max(fg,full),i),xytext=(8,0),textcoords='offset points',va='center',color=palette[i],fontsize=10)
        stats[o['id']]['foreground_gap_db']=full-fg
    ax.set_yticks(range(9),names);ax.invert_yaxis();ax.set_xlabel('PSNR (dB)');ax.margins(x=.2);ax.grid(axis='x',alpha=.18)
    ax.legend([Line2D([],[],marker='o',color='#657784',markerfacecolor='white',ls=''),Line2D([],[],marker='D',color='#657784',ls='')],['Foreground','Full frame'],loc='upper center',bbox_to_anchor=(.5,1.1),ncol=2,frameon=False)
    save(fig,'09_foreground_fullframe.png','Raw foreground/full-frame PSNR and their arithmetic gap; segmentation masks differ by protocol, gap is not a universal quality score.')
    return {'distribution_statistics':stats,'quality_stability_trajectories':trajectories}
