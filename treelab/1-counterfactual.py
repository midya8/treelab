"""Constrained closest-leaf model counterfactuals, not causal interventions.

Exhaustive leaf-box projection minimizes weighted L1 over finite binary64
inputs. Exact rational costs avoid rounding a genuinely closer leaf into a tie.
"""
import json
import math
from fractions import Fraction
from pathlib import Path
from . import tree


def search(model, row, target, bounds, weights=None, immutable=None):
    """Find a globally minimum-cost witness for ONE CART under a closed band.

    All costs use exact rational values of the supplied binary64 numbers.
    Bounds are closed, independent per-feature intervals. Immutable feature
    indices are intersected with their source values, even outside the bounds.
    Equal exact costs break by leaf ID. No causal/plausibility claim is made.
    """
    tree.validate(model)
    row=tree.array(row,1,'row').tolist();d=model['features']
    if len(row)!=d:raise ValueError('wrong feature width')
    target=tree.array(target,1,'target').tolist()
    if len(target)!=2 or target[0]>target[1]:raise ValueError('target must be an ordered closed prediction band')
    bounds=tree.array(bounds,2,'bounds').tolist()
    if len(bounds)!=d or any(len(b)!=2 or b[0]>b[1] for b in bounds):raise ValueError('bounds must contain one ordered closed pair per feature')
    weights=[1.]*d if weights is None else tree.array(weights,1,'weights').tolist()
    if len(weights)!=d or any(w<=0 for w in weights):raise ValueError('weights must be strictly positive and match feature width')
    immutable=[] if immutable is None else immutable
    if not isinstance(immutable,(list,tuple)) or any(type(i) is not int or not 0<=i<d for i in immutable) or len(set(immutable))!=len(immutable):raise ValueError('immutable must contain unique feature indices')
    immutable=sorted(immutable);lower=[b[0] for b in bounds];upper=[b[1] for b in bounds]
    for i in immutable:lower[i]=max(lower[i],row[i]);upper[i]=min(upper[i],row[i])
    fr=[Fraction(v) for v in row];fw=[Fraction(v) for v in weights]
    candidates=[];leaf_count=0;target_leaves=0
    def visit(node,lo,hi):
        nonlocal leaf_count,target_leaves
        if 'feature' in node:
            i=node['feature'];t=float(node['threshold'])
            left_hi=hi.copy();left_hi[i]=min(left_hi[i],t)
            right_lo=lo.copy();right_lo[i]=max(right_lo[i],math.nextafter(t,math.inf))
            visit(node['left'],lo,left_hi);visit(node['right'],right_lo,hi)
            return
        leaf_count+=1
        if not target[0]<=node['mean']<=target[1]:return
        target_leaves+=1
        if any(a>b for a,b in zip(lo,hi)):return
        witness=[min(max(v,a),b) for v,a,b in zip(row,lo,hi)]
        cost=sum((w*abs(Fraction(v)-x) for w,v,x in zip(fw,witness,fr)),Fraction())
        candidates.append((cost,node['id'],node,witness,lo.copy(),hi.copy()))
    visit(model['root'],lower,upper)
    candidates.sort(key=lambda c:(c[0],c[1]))
    def rational(cost):return {'numerator':str(cost.numerator),'denominator':str(cost.denominator)}
    def serialize(c):
        cost,leaf_id,node,witness,lo,hi=c
        pred,trace=tree.predict(model,[witness],paths=True)
        if trace[0]['leaf']!=leaf_id or float(pred[0])!=node['mean']:raise ArithmeticError('counterfactual witness routing failed')
        return {'leaf':leaf_id,'prediction':node['mean'],'input':witness,'cost':float(cost),'cost_exact':rational(cost),'changed_features':[i for i,(a,b) in enumerate(zip(row,witness)) if a!=b],'box':[[a,b] for a,b in zip(lo,hi)],'path':trace[0]['steps']}
    original=float(tree.predict(model,[row])[0])
    return {'format':'treelab-counterfactual-v1','row':row,'base_prediction':original,'target':target,'bounds':bounds,'weights':weights,'immutable':immutable,'status':'optimal' if candidates else 'infeasible','solution':serialize(candidates[0]) if candidates else None,'candidates':[serialize(c) for c in candidates],'leaf_count':leaf_count,'target_leaves':target_leaves,'feasible_target_leaves':len(candidates),'meaning':'Global weighted-L1 optimum over finite binary64 inputs of this single stored CART, within independent closed bounds and immutable features. Rational costs compare the exact supplied binary64 values; displayed decimal cost is rounded. Equal exact costs break by leaf ID. Model counterfactual only, not causal recourse, a plausible wine, a prediction guarantee for reality, or an ensemble optimum.'}


def run(folder):
    from . import data
    from .__main__ import strict_load
    x,y,names=data.load();split=data.split(x);root=Path(__file__).resolve().parent
    model=strict_load(root/'artifacts/selected-model.json')
    low=x[split['fit']].min(axis=0);high=x[split['fit']].max(axis=0)
    bounds=[[float(a),float(b)] for a,b in zip(low,high)]
    weights=[float(1/(b-a)) if b>a else 1. for a,b in bounds]
    # Fixed, documented requests, not selected for an attractive result.
    policies=[('all_features',[]),('alcohol_only',[i for i in range(len(names)) if names[i]!='alcohol']),('fixed_alcohol',[names.index('alcohol')]),('all_fixed',list(range(len(names))))]
    cases=[]
    for row_id in split['test'][:3]:
        for target in [[6.,7.],[7.,8.],[9.,9.]]:
            for policy,immutable in policies:
                result=search(model,x[row_id],target,bounds,weights,immutable)
                result.update(row_id=int(row_id),policy=policy);cases.append(result)
    report={'format':'treelab-counterfactual-report-v1','features':names,'source_rows':split['test'][:3].tolist(),'protocol':{'rows':'First three original test rows, including their duplicate-feature pair','targets':'Fixed closed bands [6,7], [7,8], [9,9] in model-output units','bounds':'Fitting-only feature min/max; independent boxes do not guarantee plausible inputs','weights':'Stored binary64 reciprocal fitting ranges; exact rational arithmetic on those stored weights, not exact reciprocals','model':'Unchanged validation-selected 19-leaf CART only; no ensemble guarantee','no_refit':True,'cost':'Weighted L1 over finite binary64 inputs; exact rational cost for ordering; deterministic leaf-ID ties'},'cases':cases,'limits':['No causal knowledge, feasibility-of-intervention model or distribution support check','Quality scores are ordinal and treated numerically by this model; not a winemaking recommendation','Independent feature bounds and immutability are constraints, not proof of actionability','No new accuracy result; historical models and reports remain unchanged']}
    out=Path(folder);out.mkdir(parents=True,exist_ok=True);(out/'counterfactual-report.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');return report
