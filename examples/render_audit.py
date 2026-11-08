"""Render only the frozen report, retaining its complete data as SVG metadata."""
import html,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def render(r):
    ink='#251f21';muted='#585254';teal='#73a89a';rule='#eae9ea';out=[]
    def text(x,y,s,size=20,color=ink,weight=400):out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(str(s))}</text>')
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1500" viewBox="0 0 1400 1500" role="img" aria-labelledby="title desc">');out.append('<title id="title">TreeLab: the linear control still wins</title><desc id="desc">Full real-data pruning path, held-out metrics, and one complete selected-tree prediction route. One fixed split, not a deployment claim.</desc>');out.append('<metadata>'+html.escape(json.dumps(r,sort_keys=True,separators=(',',':')))+ '</metadata>');out.append('<rect width="1400" height="1500" fill="white"/>');out.append('<g font-family="Arial, sans-serif">')
    text(40,60,'TreeLab / pruning reduces complexity, not every error',32,weight=600)
    text(40,98,'From-scratch CART on UCI white wine. Same-fit least squares remains the stronger held-out control.',22,muted)
    text(40,141,'1. Group-disjoint source split, selected without held-out targets',24,weight=600)
    counts=r['split_counts'];text(40,180,f"4,898 rows / 3,961 exact feature groups. Fit {counts['fit']}; validation {counts['validation']}; test {counts['test']}.")
    text(40,214,'Exact copies stay together. Near copies, batch overlap and future-distribution shift are not ruled out.',20,muted)
    columns=[40,430,780,1040];labels=['Model / same fitting rows','Complexity','Test RMSE / score units','Test MAE / score units']
    for x,label in zip(columns,labels):text(x,264,label,19,weight=600)
    rows=[('fit_mean','Fitting mean','1 constant'),('fit_median','Fitting median','1 constant'),('grown','Grown CART',str(r['grown_leaves'])+' leaves'),('selected','Validation-selected CART',str(r['selected_leaves'])+' leaves'),('least_squares','Ordinary least squares','12 coefficients')]
    for i,(key,label,cost) in enumerate(rows):
        y=307+44*i;m=r['test_metrics'][key];color=teal if key=='least_squares' else ink;weight=600 if key=='least_squares' else 400
        out.append(f'<g data-model="{key}" data-rmse="{m["rmse"]}" data-mae="{m["mae"]}">')
        for x,s in zip(columns,[label,cost,f"{m['rmse']:.5f}",f"{m['mae']:.5f}"]):text(x,y,s,21,color,weight)
        out.append('</g>')
    text(40,542,'2. Every weakest-link state is retained; validation chooses 19 leaves',24,weight=600)
    text(40,576,f"{len(r['path'])} states: normalized fitting SSE + alpha x leaf count. No test curve is used for selection.",20,muted)
    x0,y0,w,h=120,630,1130,360;maxleaves=r['grown_leaves'];ymax=math.ceil(max(max(p['fit_mse'],p['validation_mse']) for p in r['path'])*10)/10
    def xy(p,k):return x0+w*math.log2(p['leaves'])/math.log2(maxleaves),y0+h*(1-p[k]/ymax)
    for i in range(5):
        v=ymax*i/4;yy=y0+h*(1-v/ymax);out.append(f'<line x1="{x0}" y1="{yy}" x2="{x0+w}" y2="{yy}" stroke="{rule}"/>');text(48,yy+7,f'{v:.2f}',18,muted)
    for n in [1,2,4,8,16,32,64,128,256,425]:
        xx=x0+w*math.log2(n)/math.log2(maxleaves);text(xx-12,y0+h+30,n,17,muted)
    for key,color,label in [('fit_mse','#999496','Fit MSE'),('validation_mse',teal,'Validation MSE')]:
        points=' '.join(f'{x:.6f},{y:.6f}' for x,y in [xy(p,key) for p in reversed(r['path'])]);out.append(f'<polyline data-series="{key}" fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
    selected=r['path'][r['selected_index']];xx,yy=xy(selected,'validation_mse');out.append(f'<circle data-selected-index="{r["selected_index"]}" cx="{xx:.6f}" cy="{yy:.6f}" r="7" fill="{teal}"/>');text(xx+12,yy-13,f"Selected: 19 leaves / validation MSE {selected['validation_mse']:.5f}",19,weight=600)
    text(470,1060,'Leaf count / logarithmic horizontal scale',20,muted);text(120,612,'Mean squared score error',18,muted)
    text(930,612,'Fit: gray / validation: teal',18,muted)
    text(40,1110,'3. One complete route, not feature causality',24,weight=600)
    example=r['selected_examples'][0]
    for i,step in enumerate(example['steps']):
        op='<=' if step['direction']=='left' else '>'
        text(40,1150+i*35,f"Node {step['node']}: {r['features'][step['feature']]} = {step['input']:g} {op} {step['threshold']:g}  /  {step['direction']}",20)
    text(40,1350,f"First test row {example['row']}: leaf {example['leaf']}, {example['leaf_fit_rows']} fitting rows, prediction {example['prediction']:.5f}; recorded score {example['target']:g}.",20,weight=600)
    text(40,1396,'One target-blind hash-group split. Ordinal scores treated as numeric. No confidence, safety or SOTA claim.',19,muted)
    text(40,1430,'Data: Cortez et al. (2009), UCI DOI 10.24432/C56S3T, CC BY 4.0. Derived metrics and chart; not endorsed.',19,muted)
    text(40,1464,'Established CART / weakest-link pruning. Binary64 arithmetic and finite oracles, not exact all-input proofs.',19,muted)
    out.append('</g></svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':
    r=json.loads((ROOT/'treelab/artifacts/report.json').read_text());(ROOT/'assets/tree-audit.svg').write_text(render(r))
