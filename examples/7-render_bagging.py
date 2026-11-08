"""Render frozen results only. Complete summary data remains in SVG metadata."""
import html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def render(r):
    ink='#251f21';muted='#585254';teal='#73a89a';rule='#eae9ea';out=[]
    def text(x,y,s,size=21,color=ink,weight=400):out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(str(s))}</text>')
    summary={k:v for k,v in r.items() if k not in ['oob','test_member_predictions','validation_member_predictions','test_rows','indices']}
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1380" viewBox="0 0 1400 1380" role="img" aria-labelledby="title desc"><title id="title">TreeLab: group-bootstrap bagging with honest OOB support</title><desc id="desc">Fixed validation prefix grid, OOB coverage and held-out metrics with preserved same-fit controls. Historical test reused, not a new blind benchmark.</desc><metadata>'+html.escape(json.dumps(summary,sort_keys=True,separators=(',',':')))+ '</metadata><rect width="1400" height="1380" fill="white"/><g font-family="Arial, sans-serif">')
    text(40,60,'TreeLab / bagging with inspectable exclusions',34,weight=600)
    text(40,103,'32 from-scratch trees. Exact copies travel together. Every draw, tree and prediction is saved.',23,muted)
    text(40,153,'1. Fixed validation grid, not OOB or test tuning',25,weight=600)
    text(40,193,'2,376 fit groups / 2,921 fit rows. G group draws per tree; expand copies with multiplicity.')
    text(40,228,f"Seed 20261108 / min leaf 5 / depth 20. Bootstrap row counts vary: {min(r['bootstrap_rows']):,} to {max(r['bootstrap_rows']):,}.",21,muted)
    cols=[40,310,580,900,1150]
    for x,s in zip(cols,['Trees','Validation MSE','OOB supported rows','OOB MSE*','Chosen']):text(x,280,s,20,weight=600)
    for i,p in enumerate(r['prefixes']):
        y=326+i*44;out.append(f'<line x1="40" y1="{y+12}" x2="1360" y2="{y+12}" stroke="{rule}"/>')
        for x,s in zip(cols,[p['trees'],f"{p['validation_mse']:.6f}",f"{p['oob_supported_rows']:,} / 2,921",f"{p['oob_mse_supported']:.6f}",'selected' if p['trees']==r['selected_trees'] else '-']):text(x,y,s,21,teal if p['trees']==r['selected_trees'] else ink,600 if p['trees']==r['selected_trees'] else 400)
    text(40,510,'* OOB MSE is row-weighted on each prefix\'s supported subset, not a fixed evaluation set.',20,muted)
    text(40,545,'Unsupported rows: 1,832 / 482 / 1 / 0. Zero votes mean null, never a fabricated zero prediction.',20,muted)
    text(40,605,'2. All controls use the same fitting rows',25,weight=600)
    text(40,642,'969 held-out rows. Historical test reused across versions: descriptive, not a new blind benchmark.',21,muted)
    columns=[40,640,950,1190]
    for x,s in zip(columns,['Model','RMSE / score units','MAE / score units','Test rows']):text(x,692,s,20,weight=600)
    for i,(key,label) in enumerate([('fit_mean','Fitting mean'),('fit_median','Fitting median'),('grown','Single grown CART'),('selected','Validation-selected CART'),('least_squares','Ordinary least squares'),('bagged_selected','Validation-selected bagging / 32 trees')]):
        m=r['test_metrics'][key];y=736+i*44;out.append(f'<line x1="40" y1="{y+12}" x2="1360" y2="{y+12}" stroke="{rule}"/>')
        for x,s in zip(columns,[label,f"{m['rmse']:.6f}",f"{m['mae']:.6f}",m['rows']]):text(x,y,s,21,teal if key=='bagged_selected' else ink,600 if key=='bagged_selected' else 400)
    text(40,1050,'3. Row absence is not group absence',25,weight=600)
    text(40,1090,'Witness: features [0, 0, 1, 1]; row draws [1, 1, 2, 2]. Row 0 is absent, its exact copy is in-bag.')
    text(40,1127,'Naive row-OOB squared error is 0. Group-OOB correctly excludes that prediction.',21,muted)
    text(40,1187,'What this does not establish',25,weight=600)
    text(40,1225,'No feature subsampling: this is bagging, not a random forest. No calibrated uncertainty interval.',21,muted)
    text(40,1260,'Exact-copy groups do not recover producer/batch identity. One seed and one split; no causal isolation.',21,muted)
    text(40,1322,'UCI Wine Quality / Cortez et al. / CC BY 4.0. Established CART and bootstrap averaging, implemented here.',19,muted)
    out.append('</g></svg>');return ''.join(out)

if __name__=='__main__':
    r=json.loads((ROOT/'treelab/artifacts/bagging-report.json').read_text());(ROOT/'assets/bagging-audit.svg').write_text(render(r))
