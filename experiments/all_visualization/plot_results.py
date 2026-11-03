"""Publication-style PNGs from verified archives; no invented measurements."""
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'scgs-mpl-cache'))
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from build_data import build
HERE=Path(__file__).resolve().parent
OUT=HERE/'png'
METRICS=[('psnr','PSNR',1),('ssim','SSIM',1),('lpips','LPIPS',-1),('iou','IoU',1),('foreground_psnr','Foreground PSNR',1),('missing','Missing',-1)]
PALETTE=['#2878b5','#e1812c','#3b923b','#c43c39','#9467bd','#8c564b','#d16b9c','#687782','#a39418']
MARKERS=['o','s','^','D','v','P','X','h','*']
OUTPUTS=[]
def number(row,key):
 v=row.get(key)
 return float(v) if v is not None and np.isfinite(float(v)) else np.nan

def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#64707a','axes.linewidth':.7,'xtick.color':'#39434c','ytick.color':'#39434c','savefig.dpi':300})

def save(fig,name,description):
 # Enforce English labels and no figure/axes titles or explanatory prose.
 assert not fig._suptitle
 for ax in fig.axes: assert not ax.get_title()
 fig.savefig(OUT/name,bbox_inches='tight',pad_inches=.18)
 plt.close(fig)
 OUTPUTS.append(dict(file='png/'+name,description=description,sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))

def legend(ax):
 ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=5,frameon=False,fontsize=10,columnspacing=1.5,handlelength=2.5)

def overview(objects):
 from matplotlib.lines import Line2D
 fig,axes=plt.subplots(2,3,figsize=(18,11))
 for ax,(key,label,direction) in zip(axes.flat,METRICS):
  for i,o in enumerate(objects):
   rows=o['versions']; vals=[number(r,key) for r in rows]; yy=len(objects)-1-i
   ax.axhspan(yy-.44,yy+.44,color=PALETTE[i],alpha=.035)
   ax.plot(vals,[yy]*len(vals),color=PALETTE[i],alpha=.4,lw=2)
   delta=direction*(vals[-1]-vals[0])
   extent=max(abs(number(q['versions'][-1],key)-number(q['versions'][0],key)) for q in objects)
   strength=abs(delta)/extent if extent else 0
   for t in range(40):
    x0=vals[0]+(vals[-1]-vals[0])*t/40
    x1=vals[0]+(vals[-1]-vals[0])*(t+1)/40
    ax.fill_between([x0,x1],yy-.22,yy+.22,color=PALETTE[i] if delta>=0 else '#b43c48',alpha=(.06+.28*strength)*(t+1)/40,linewidth=0,zorder=1)
   for j,v in enumerate(vals):
    selected=j==len(vals)-1
    marker='D' if selected else ('s' if j==1 else 'o')
    ax.scatter(v,yy,s=58 if selected else 34,marker=marker,facecolor=PALETTE[i] if selected else 'white',edgecolor=PALETTE[i],linewidth=1.2,zorder=3)
   ax.annotate(f'{vals[-1]:.3f}',(vals[-1],yy),xytext=(5,9),textcoords='offset points',fontsize=8,color=PALETTE[i])
  ax.set_yticks(range(len(objects)),[o['id'].title()+(' *' if o['special'] else '') for o in objects][::-1])
  ax.set_xlabel(label+(' (dB)' if 'psnr' in key else ' (source units)' if key=='missing' else '')+(' ↑' if direction==1 else ' ↓'))
  ax.set_ylim(-.65,len(objects)-.3);ax.margins(x=.18);ax.grid(axis='x',alpha=.16);ax.tick_params(axis='y',length=0)
 handles=[Line2D([],[],marker=m,color='#596778',linestyle='none',markerfacecolor='white' if m!='D' else '#596778',label=l) for m,l in [('o','V1 / Previous V3 *'),('s','V2'),('D','Selected V3')]]
 fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.995))
 fig.subplots_adjust(top=.94,bottom=.08,left=.085,right=.98,wspace=.42,hspace=.24)
 save(fig,'01_recommended_metrics.png','Six raw-unit metric panels; baseline and selected values with connectors and selected-value labels.')
 return {}

def improvements(objects,baseline):
 a=np.full((len(objects),len(METRICS)),np.nan)
 for i,o in enumerate(objects):
  old=next((v for v in o['versions'] if v['version']==('旧 V3' if o['special'] else baseline)),{})
  new=next(v for v in o['versions'] if v['version']==o['recommended'])
  for j,(key,_,direction) in enumerate(METRICS):
   b=number(old,key);n=number(new,key)
   if np.isfinite(b) and b!=0:a[i,j]=direction*(n-b)/abs(b)*100
 return a

def combined_heatmap(objects,a,b):
 # Adjacent baseline columns retain readable numbers without perspective occlusion.
 paired=np.stack([a,b],axis=2).reshape(len(objects),-1)
 cmap=colors.LinearSegmentedColormap.from_list('warm',['#100d23','#54204f','#a52c60','#e74648','#f58d69','#fae8d5'])
 norm=colors.Normalize(min(0,float(np.nanmin(paired))),float(np.nanmax(paired)))
 fig,ax=plt.subplots(figsize=(16,8))
 im=ax.imshow(paired,cmap=cmap,norm=norm,aspect='auto')
 # Beveled edges retain a flat, color-calibrated face and legible cell values.
 from matplotlib.patches import Polygon
 maxabs=np.nanmax(np.abs(paired))
 for r in range(paired.shape[0]):
  for c in range(paired.shape[1]):
   depth=.025+.10*abs(paired[r,c])/maxabs
   x0,x1,y0,y1=c-.5,c+.5,r-.5,r+.5
   ax.add_patch(Polygon([(x0,y1),(x1,y1),(x1,y0),(x1-depth,y0+depth),(x1-depth,y1-depth),(x0+depth,y1-depth)],facecolor='black',alpha=.20,edgecolor='none',zorder=2))
   ax.add_patch(Polygon([(x0,y1),(x0,y0),(x1,y0),(x1-depth,y0+depth),(x0+depth,y0+depth),(x0+depth,y1-depth)],facecolor='white',alpha=.23,edgecolor='none',zorder=2))
 for i in range(len(objects)):
  for j in range(paired.shape[1]):
   v=paired[i,j];label='N/A' if not np.isfinite(v) else f'{v:.1f}'
   if np.isfinite(v) and abs(v)<.05:label='0.0'
   ax.text(j,i,label,ha='center',va='center',fontsize=11,color='white' if np.isfinite(v) and norm(v)<.62 else '#29272b')
 ax.set_yticks(range(len(objects)),[o['id'].title()+(' *' if o['special'] else '') for o in objects])
 ax.set_xticks(range(12),['vs V1','vs V2']*6);ax.set_ylabel('Object');ax.set_xlabel('Baseline')
 top=ax.secondary_xaxis('top');top.set_xticks(np.arange(6)*2+.5,['PSNR','SSIM','LPIPS','IoU','FG-PSNR','Missing']);top.tick_params(length=0,pad=12)
 ax.set_xticks(np.arange(-.5,12,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True);ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='both',length=0,pad=9)
 for x in np.arange(1.5,11,2):ax.axvline(x,color='white',lw=4)
 for spine in ax.spines.values():spine.set_visible(False)
 cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.025);cb.set_label('Relative improvement (%)');cb.outline.set_visible(False)
 save(fig,'02_improvement_combined.png','Paired V1/V2 columns per metric, shared color scale. Robot both columns use previous V3.')

def endpoint_labels(ax,entries,xtext,gap):
 # Ordered endpoint routing: preserve vertical order, reserve a text-only gutter.
 ordered=sorted(entries,key=lambda e:e[1])
 positions=[]
 for _,y,_,_ in ordered:positions.append(max(y,positions[-1]+gap) if positions else y)
 overflow=max(0,positions[-1]-(ax.get_ylim()[1]-.8))
 positions=np.array(positions)-overflow
 texts=[]
 for (x,y,text,color),placed in zip(ordered,positions):
  texts.append(ax.annotate(text,xy=(x,y),xytext=(xtext,placed),ha='left',va='center',fontsize=9,color=color,
   bbox=dict(facecolor='white',edgecolor='none',pad=2,alpha=.97),
   arrowprops=dict(arrowstyle='-',color=color,lw=.8,connectionstyle='arc3,rad=0'),zorder=20,annotation_clip=False))
 ax.figure.canvas.draw()
 boxes=[t.get_bbox_patch().get_window_extent() for t in texts]
 assert all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]), 'Overlapping endpoint labels'

def gradient_fill(ax,x,y,floor,color):
 y=np.asarray(y)
 for low,high in zip(np.linspace(0,1,25)[:-1],np.linspace(0,1,25)[1:]):
  ax.fill_between(x,floor+(y-floor)*low,floor+(y-floor)*high,color=color,alpha=.10*high**4,linewidth=0,zorder=1)

def versions(objects):
 fig,ax=plt.subplots(figsize=(14,7));entries=[]
 floor=15
 for i,o in enumerate(objects):
  x=[2,3] if o['special'] else [0,1,3]
  y=[number(v,'psnr') for v in o['versions']]
  gradient_fill(ax,x,y,floor,PALETTE[i])
  ax.plot(x,y,color=PALETTE[i],marker=MARKERS[i],ms=7,lw=2,label=o['id'].title(),ls='--' if o['special'] else '-')
  entries.append((3,y[-1],f'{y[-1]:.2f}',PALETTE[i]))
 ax.set_xticks([0,1,2,3],['V1','V2','Previous V3\n(Robot)','Selected V3']);ax.set_ylabel('PSNR (dB)');ax.set_xlabel('Version');ax.grid(axis='y',alpha=.18);ax.set_ylim(floor,46);ax.set_xlim(-.15,3.35);legend(ax)
 fig.subplots_adjust(top=.85)
 endpoint_labels(ax,entries,3.10,1.3)
 save(fig,'03_version_psnr.png','Version curves with translucent fill to 15 dB; separated endpoint labels show selected PSNR.')

def temporal(objects):
 fig,ax=plt.subplots(figsize=(16,7));missing=[];entries=[]
 for i,o in enumerate(objects):
  raw=o['series'].get(o['recommended'],{}).get('psnr',[])
  if not raw:missing.append(o['id']);continue
  y=np.array([np.nan if v is None else v for v in raw]);frames=np.array(o['times']);assert len(frames)==len(y)
  x=(frames-frames[0])/(frames[-1]-frames[0])*100
  gradient_fill(ax,x,y,18,PALETTE[i])
  ax.plot(x,y,color=PALETTE[i],lw=1.6,marker=MARKERS[i],ms=3,markevery=max(1,len(x)//12),label=o['id'].title())
  entries.append((100,y[-1],f"{o['id'].title()}  μ={np.nanmean(y):.2f}  σ={np.nanstd(y):.2f}  n={len(y)}",PALETTE[i]))
 ax.set_xlabel('Normalized sequence progress (%)');ax.set_ylabel('PSNR (dB)');ax.set_xlim(0,142);ax.set_xticks(np.arange(0,101,20));ax.set_ylim(18,46);ax.grid(alpha=.18);legend(ax)
 fig.subplots_adjust(top=.85)
 endpoint_labels(ax,entries,104,1.35)
 save(fig,'04_temporal_psnr.png','All nine original curves; x is per-sequence normalized source-frame progress, not common time. No smoothing or resampling. Fill to 18 dB; population standard deviation ddof=0.')
 return missing

def main():
 build();setup();OUT.mkdir(exist_ok=True)
 # Remove only the six known outputs of the previous plotting implementation.
 for name in ['01_recommended_metrics.png','02_improvement_vs_v1.png','02_improvement_vs_v2.png','03_version_psnr.png','04_temporal_psnr.png','05_metric_coverage.png']:
  (OUT/name).unlink(missing_ok=True)
 data=json.loads((HERE/'data.json').read_text());objects=data['objects'];stats=overview(objects)
 a=improvements(objects,'V1');b=improvements(objects,'V2')
 stats['improvement_vs_V1']=a.tolist();stats['improvement_vs_V2']=b.tolist()
 combined_heatmap(objects,a,b)
 versions(objects);stats['missing_temporal_objects']=temporal(objects)
 from plot_differences import draw
 stats.update(draw(objects,PALETTE,MARKERS,save))
 (HERE/'plot_statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
 manifest=dict(source_summary_sha256=data['provenance']['summary_sha256'],verified=data['integrity']['passed'],objects=[dict(object=o['id'],source=o['snapshot'],sha256=o['sha256'],pointer=o['pointer'],series_source=o['seriesSource']) for o in objects],figures=OUTPUTS)
 (HERE/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(f'Generated {len(OUTPUTS)} PNG figures; missing temporal data: {stats["missing_temporal_objects"]}')
if __name__=='__main__':main()
