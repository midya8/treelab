"""Exact binary64 one-axis tree-response partitions, not causal effects."""
import json,math
from pathlib import Path
import numpy as np
from . import tree,bagging


def members(model,prefix=None):
    if isinstance(model,dict) and model.get('format')=='treelab-group-bagging-v1':
        bagging.validate(model);n=len(model['trees'])
        if prefix is None:prefix=n
        if type(prefix) is not int or not 1<=prefix<=n:raise ValueError('invalid ensemble prefix')
        return [t['model'] for t in model['trees'][:prefix]]
    if prefix is not None:raise ValueError('prefix applies only to an ensemble')
    return [tree.validate(model)]


def response(model,row,feature,low,high,prefix=None):
    models=members(model,prefix);row=tree.array(row,1,'row')
    if len(row)!=models[0]['features']:raise ValueError('wrong feature width')
    if type(feature) is not int or not 0<=feature<len(row):raise ValueError('invalid feature index')
    def number(x):return type(x) in (int,float) and math.isfinite(x) and abs(x)<=1e100
    if not number(low) or not number(high) or low>high:raise ValueError('domain must be finite ordered numbers within +/-1e100')
    low=float(low);high=float(high)
    # Enumerate leaf intervals, routing non-axis splits with the fixed row.
    # (lower, upper] domains follow the learner's <= left rule.
    leaves=[]
    def visit(n,lower,upper,out):
        if lower>=upper:return
        if 'feature' not in n:
            out.append((lower,upper,n['mean'],n['id']));return
        f=n['feature'];threshold=n['threshold']
        if f==feature:
            visit(n['left'],lower,min(upper,threshold),out);visit(n['right'],max(lower,threshold),upper,out)
        else:visit(n['left'] if row[f]<=threshold else n['right'],lower,upper,out)
    for m in models:
        out=[];visit(m['root'],-math.inf,math.inf,out);leaves.append(out)
    breaks=sorted({upper for out in leaves for lower,upper,_,_ in out if low<=upper<high})
    segments=[];start=low;closed=True
    for stop in breaks+[high]:
        # Closed first lower endpoint, then open cut thresholds. An interval
        # with no representable binary64 value is omitted, never sampled.
        witness=start if closed else math.nextafter(start,math.inf)
        if witness<=stop:
            values=[];ids=[]
            for out in leaves:
                matches=[(mean,leaf_id) for lower,upper,mean,leaf_id in out if lower<witness<=upper]
                if len(matches)!=1:raise ArithmeticError('leaf partition is incomplete or overlapping')
                mean,leaf_id=matches[0];values.append(float(mean));ids.append(leaf_id)
            prediction=float(np.mean(values))
            segments.append({'lower':start,'lower_inclusive':closed,'upper':stop,'upper_inclusive':True,'witness':witness,'prediction':prediction,'member_predictions':values,'member_leaf_ids':ids})
        start=stop;closed=False
    if not segments:raise ArithmeticError('empty finite domain partition')
    minimum=min(range(len(segments)),key=lambda i:(segments[i]['prediction'],i));maximum=max(range(len(segments)),key=lambda i:(segments[i]['prediction'],-i))
    def extremum(i):
        segment=segments[i];x=row.copy();x[feature]=segment['witness']
        return {'segment':i,'prediction':segment['prediction'],'axis_value':segment['witness'],'input':x.tolist()}
    current=float(np.mean([tree.predict(m,row[None,:])[0] for m in models]))
    return {'format':'treelab-axis-sweep-v1','row':row.tolist(),'feature':feature,'domain':[low,high],'members':len(models),'base_prediction':current,'base_axis_in_domain':bool(low<=row[feature]<=high),'segments':segments,'minimum':extremum(minimum),'maximum':extremum(maximum),'meaning':'Binary64 model-response slice holding all other features fixed. Attainable extrema only on this one-axis domain, not causal effects, plausible-input guarantees, prediction intervals or multifeature robustness.'}


def run(folder):
    from . import data
    from .__main__ import strict_load
    x,y,names=data.load();s=data.split(x);root=Path(__file__).resolve().parent
    models={'selected_cart':strict_load(root/'artifacts/selected-model.json'),'bagging_32':strict_load(root/'artifacts/bagging-model.json.gz')};lo=x[s['fit']].min(axis=0);hi=x[s['fit']].max(axis=0);sweeps=[]
    for row_id in s['test'][:3]:
        for name,model in models.items():
            for feature in range(len(names)):
                r=response(model,x[row_id],feature,float(lo[feature]),float(hi[feature]));r.update(row_id=int(row_id),model=name,feature_name=names[feature]);sweeps.append(r)
    report={'format':'treelab-axis-report-v1','protocol':{'rows':'First three original test rows, no response-driven row choice','domains':'Feature-wise min/max of fitting rows only','models':'Unchanged selected CART and unchanged 32-member group-bagging ensemble','units':'Ordinal wine-quality scores treated as numeric model outputs','precision':'Binary64 threshold/routing semantics, no input casting to float32','no_refit':True},'source_rows':s['test'][:3].tolist(),'features':names,'fit_ranges':[[float(a),float(b)] for a,b in zip(lo,hi)],'sweeps':sweeps,'limits':['One-feature slices may combine correlated features unrealistically','Exact here means the stored binary64 tree partition on one finite axis, not mathematical exact arithmetic or a universal numerical certificate','No causal inference, calibrated predictive interval, plausible-wine guarantee or joint multifeature extremum','Stored models, existing artifacts and historical test metrics remain unchanged']}
    out=Path(folder);out.mkdir(parents=True,exist_ok=True);(out/'sweep-report.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');return report
