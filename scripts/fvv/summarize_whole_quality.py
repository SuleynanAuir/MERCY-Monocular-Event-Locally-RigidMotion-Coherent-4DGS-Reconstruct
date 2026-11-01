"""Summarize whole-scene evaluation, including regressions outside protected parts."""
import argparse,json
from pathlib import Path


def summarize(report):
    protected=set(range(144,190))|set(range(203,217))
    result={}
    for tag,model in report['models'].items():
        values=model['visible_parts'];groups={}
        for name,keep in [('all',lambda i:True),('protected_bucket_arm',lambda i:i in protected),('remaining_scene',lambda i:i not in protected)]:
            rows=[v for i,v in values.items() if keep(int(i))];pixels=sum(v['pixels'] for v in rows)
            groups[name]=dict(parts=len(rows),pixels=pixels,means={k:sum(v['means'][k]*v['pixels'] for v in rows)/pixels for k in rows[0]['means']} if pixels else {})
        result[tag]=groups
    changes=[]
    previous=report['models']['previous']['visible_parts'];candidate=report['models']['candidate']['visible_parts']
    for ident,before in previous.items():
        after=candidate.get(ident)
        if not after or min(before['pixels'],after['pixels'])<100:continue
        change=dict(id=int(ident),name=before['name'],pixels=before['pixels'],before=before['means'],after=after['means'],dark_fraction_change=after['means']['dark_fraction']-before['means']['dark_fraction'])
        if 'rgb_mse' in before['means']:
            change['rgb_mse_change']=after['means']['rgb_mse']-before['means']['rgb_mse']
            change['edge_residual_change']=after['means']['edge_residual']-before['means']['edge_residual']
        changes.append(change)
    result['largest_dark_regressions']=sorted(changes,key=lambda r:r['dark_fraction_change'],reverse=True)[:10]
    result['largest_dark_improvements']=sorted(changes,key=lambda r:r['dark_fraction_change'])[:10]
    detailed=[r for r in changes if 'rgb_mse_change' in r]
    if detailed:
        result['part_detail_audit']=dict(eligible_parts=len(detailed),
            rgb_improved=sum(r['rgb_mse_change']<0 for r in detailed),
            edges_improved=sum(r['edge_residual_change']<0 for r in detailed),
            largest_rgb_regressions=sorted(detailed,key=lambda r:r['rgb_mse_change'],reverse=True)[:10],
            largest_rgb_improvements=sorted(detailed,key=lambda r:r['rgb_mse_change'])[:10])
    result['scope']='Pixel-weighted supplied-source part diagnostics. Protected parts are bucket 144..167 and six moving arm groups 168..189,203..216; remaining includes body, tracks, fixed supports and base. Rankings require >=100 visible pixels and include regressions, not only improvements.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=summarize(json.loads(a.input.read_text()));a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k in ['previous','candidate']},indent=2))


if __name__=='__main__':main()
