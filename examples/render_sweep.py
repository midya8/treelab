"""Render the complete selected one-axis partitions, not a response sample grid."""
import html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def render(r):
    ink='#251f21';muted='#585254';teal='#73a89a';rule='#eae9ea';out=[]
    def text(x,y,s,size=21,color=ink,weight=400):out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(str(s))}</text>')
    row_id=r['source_rows'][0];selected=[a for a in r['sweeps'] if a['row_id']==row_id];curves=[a for a in selected if a['feature_name']=='alcohol']
    metadata={'protocol':r['protocol'],'source_rows':r['source_rows'],'limits':r['limits'],'displayed_sweeps':selected}
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1600" viewBox="0 0 1400 1600" role="img" aria-labelledby="title desc"><title id="title">TreeLab one-feature response geometry</title><desc id="desc">Complete alcohol slice and first-row all-feature attainable model-output ranges. Not causal effects or prediction intervals.</desc><metadata>'+html.escape(json.dumps(metadata,sort_keys=True,separators=(',',':')))+ '</metadata><rect width="1400" height="1600" fill="white"/><g font-family="Arial, sans-serif">')
    text(40,60,'TreeLab / exact slices, not causal effects',34,weight=600)
    text(40,101,'Hold every other feature fixed. Partition the stored trees, not a sampled response grid.',23,muted)
    text(40,151,'66 saved slices / 1,028 segments / first three test rows x 11 features x two frozen models',22,weight=600)
    text(40,189,'Domains use fitting-only min/max. Rows 1 and 8 are identical copies: their curves agree.',21,muted)
    text(40,241,'1. Alcohol slice for the first test row, original CSV row 1 (zero based)',25,weight=600)
    text(110,278,'Model prediction / quality-score units',18,muted)
    left=110;right=1320;top=292;bottom=570;xlo,xhi=curves[0]['domain'];ylo=4.8;yhi=6.8
    px=lambda x:left+(x-xlo)/(xhi-xlo)*(right-left)
    py=lambda y:bottom-(y-ylo)/(yhi-ylo)*(bottom-top)
    for y in [5.,5.5,6.,6.5]:
        out.append(f'<line x1="{left}" y1="{py(y)}" x2="{right}" y2="{py(y)}" stroke="{rule}"/>');text(47,py(y)+7,f'{y:.1f}',18,muted)
    for x in [8.,10.,12.,14.]:text(px(x)-15,bottom+32,f'{x:.0f}',18,muted)
    for a in curves:
        color=teal if a['model']=='bagging_32' else ink;points=[]
        for segment in a['segments']:
            points.extend([(px(segment['lower']),py(segment['prediction'])),(px(segment['upper']),py(segment['prediction']))])
        out.append(f'<polyline data-model="{a["model"]}" data-segments="{len(a["segments"])}" points="'+ ' '.join(f'{x:.6f},{y:.6f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="3"/>')
    text(585,628,'Alcohol / % by volume (stored feature)',20,muted)
    text(110,670,'Selected CART: 3 leaf segments',21,ink,600);text(660,670,'32-tree mean: 50 common segments',21,teal,600)
    text(40,718,'Step joins are visual guides; exact <= boundary inclusion and nextafter witnesses are saved.',20,muted)
    text(40,771,'2. Attainable outputs on each one-feature slice, first test row',25,weight=600)
    cols=[40,470,800,1150]
    for x,label in zip(cols,['Varied feature','Selected CART min / max','32-tree mean min / max','Segments*']):text(x,817,label,20,weight=600)
    for i,name in enumerate(r['features']):
        a=next(s for s in selected if s['feature_name']==name and s['model']=='selected_cart');b=next(s for s in selected if s['feature_name']==name and s['model']=='bagging_32');y=861+38*i
        out.append(f'<line x1="40" y1="{y+11}" x2="1360" y2="{y+11}" stroke="{rule}"/>')
        for x,value in zip(cols,[name,f"{a['minimum']['prediction']:.5f} / {a['maximum']['prediction']:.5f}",f"{b['minimum']['prediction']:.5f} / {b['maximum']['prediction']:.5f}",f"{len(a['segments'])} / {len(b['segments'])}"]):text(x,y,value,20)
    text(40,1314,'* Selected CART / ensemble segments. Output units are ordinal quality scores treated as numeric.',20,muted)
    text(40,1362,'Common-axis extrema, with concrete witness inputs',24,weight=600)
    text(40,1400,'Opposite stumps can each attain 10 separately, yet their mean is always 5. Separate maxima mislead.',20,muted)
    text(40,1440,'These ranges are not uncertainty bands, causal effects, plausible-wine guarantees or joint-feature bounds.',20,muted)
    text(40,1477,'Correlated features may create impossible combinations. No model refit and no held-out accuracy claim.',20,muted)
    text(40,1550,'Established ICE-style response analysis / UCI Wine Quality, Cortez et al., CC BY 4.0 / MIT code',19,muted)
    out.append('</g></svg>');return ''.join(out)+'\n'

if __name__=='__main__':
    r=json.loads((ROOT/'treelab/artifacts/sweep-report.json').read_text());(ROOT/'assets/sweep-audit.svg').write_text(render(r))
