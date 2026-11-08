"""Budgeted best-first constrained counterfactual search for frozen tree means.

Global guarantees apply only when the frontier proves completion. A budget
limit returns honest lower/upper objective bounds, never invented infeasibility.
"""
import heapq,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from . import bagging,tree,sweep


def search(model,row,target,bounds,weights=None,immutable=None,max_states=10000,prefix=None,seed_inputs=None):
    models=sweep.members(model,prefix);d=models[0]['features']
    row=tree.array(row,1,'row').tolist();target=tree.array(target,1,'target').tolist();bounds=tree.array(bounds,2,'bounds').tolist()
    if len(row)!=d:raise ValueError('wrong feature width')
    if len(target)!=2 or target[0]>target[1]:raise ValueError('target must be an ordered closed band')
    if len(bounds)!=d or any(len(b)!=2 or b[0]>b[1] for b in bounds):raise ValueError('bounds must contain one ordered closed pair per feature')
    weights=[1.]*d if weights is None else tree.array(weights,1,'weights').tolist()
    if len(weights)!=d or any(w<=0 for w in weights):raise ValueError('weights must be strictly positive and match width')
    immutable=[] if immutable is None else immutable
    if not isinstance(immutable,(list,tuple)) or any(type(i) is not int or not 0<=i<d for i in immutable) or len(set(immutable))!=len(immutable):raise ValueError('immutable must contain unique feature indices')
    if type(max_states) is not int or not 0<=max_states<=100000:raise ValueError('max_states must be an integer in 0..100000')
    seed_inputs=[] if seed_inputs is None else seed_inputs
    if not isinstance(seed_inputs,(list,tuple)) or len(seed_inputs)>128:raise ValueError('seed_inputs must contain at most 128 input vectors')
    seeds_extra=[]
    for seed in seed_inputs:
        z=tree.array(seed,1,'seed input').tolist()
        if len(z)!=d or any(not bounds[i][0]<=v<=bounds[i][1] for i,v in enumerate(z)) or any(z[i]!=row[i] for i in immutable):raise ValueError('seed input violates width, bounds or immutability')
        seeds_extra.append(tuple(z))
    immutable=sorted(immutable);lo=tuple(b[0] for b in bounds);hi=tuple(b[1] for b in bounds)
    lo=tuple(max(v,row[i]) if i in immutable else v for i,v in enumerate(lo));hi=tuple(min(v,row[i]) if i in immutable else v for i,v in enumerate(hi))
    ranges={}
    def prepare(n,k):
        for field in ['mean']+(['threshold'] if 'feature' in n else []):
            if float(n[field])!=n[field]:raise ValueError('model '+field+' must be exactly representable as binary64')
        if 'feature' not in n:a=b=float(n['mean'])
        else:
            a1,b1=prepare(n['left'],k);a2,b2=prepare(n['right'],k);a=min(a1,a2);b=max(b1,b2)
        ranges[k,n['id']]=(a,b);return a,b
    for k,m in enumerate(models):prepare(m['root'],k)
    fr=tuple(Fraction(x) for x in row);fw=tuple(Fraction(x) for x in weights)
    def projection(a,b):
        x=tuple(min(max(v,l),h) for v,l,h in zip(row,a,b));cost=sum((w*abs(Fraction(v)-r) for w,v,r in zip(fw,x,fr)),Fraction());return x,cost
    def predict(x):
        values=[];ids=[]
        for m in models:
            n=m['root']
            while 'feature' in n:n=n['left'] if x[n['feature']]<=n['threshold'] else n['right']
            values.append(float(n['mean']));ids.append(n['id'])
        return float(np.mean(values)),values,ids
    def output_bounds(nodes):return (float(np.mean([ranges[k,n['id']][0] for k,n in enumerate(nodes)])),float(np.mean([ranges[k,n['id']][1] for k,n in enumerate(nodes)])))
    def rational(x):return None if x is None else {'numerator':str(x.numerator),'denominator':str(x.denominator)}
    initial=tuple(m['root'] for m in models);queue=[];serial=0;best=None;visited=0;generated=0;pruned_domain=0;pruned_output=0;pruned_cost=0;peak=0;seed_evaluations=0;seed_feasible=0
    def consider(x,cost):
        nonlocal best
        prediction,values,ids=predict(x)
        if target[0]<=prediction<=target[1] and (best is None or cost<best[0]):best=(cost,x,prediction,values,ids)
    def push(nodes,a,b):
        nonlocal serial,generated,pruned_domain,pruned_output,pruned_cost,peak
        generated+=1
        if any(l>h for l,h in zip(a,b)):pruned_domain+=1;return
        # If the whole current box lies on one side, that split is forced.
        # Propagate across every member before output bounds or branching.
        forced=[]
        for n in nodes:
            while 'feature' in n:
                f=n['feature'];t=n['threshold']
                if b[f]<=t:n=n['left']
                elif a[f]>t:n=n['right']
                else:break
            forced.append(n)
        nodes=tuple(forced)
        l,u=output_bounds(nodes)
        if u<target[0] or l>target[1]:pruned_output+=1;return
        x,cost=projection(a,b)
        if best is not None and cost>=best[0]:pruned_cost+=1;return
        serial+=1;heapq.heappush(queue,(cost,serial,nodes,a,b,x));peak=max(peak,len(queue))
    if not any(l>h for l,h in zip(lo,hi)):
        projected,_=projection(lo,hi)
        # Fixed, disclosed warm starts. They improve only feasible upper
        # bounds, never prune on the basis of an unverified candidate.
        seeds=[projected,lo,hi,tuple(min(max(a/2+b/2,a),b) for a,b in zip(lo,hi))]
        for x in dict.fromkeys(seeds+seeds_extra):
            seed_evaluations+=1
            c=sum((w*abs(Fraction(v)-r) for w,v,r in zip(fw,x,fr)),Fraction())
            prediction=predict(x)[0]
            if target[0]<=prediction<=target[1]:seed_feasible+=1
            consider(x,c)
    push(initial,lo,hi)
    while queue and visited<max_states:
        lower,_,nodes,a,b,x=heapq.heappop(queue)
        if best is not None and lower>=best[0]:queue.clear();break
        visited+=1;consider(x,lower)
        if best is not None and lower>=best[0]:continue
        unresolved=next((k for k,n in enumerate(nodes) if 'feature' in n),None)
        if unresolved is None:continue
        n=nodes[unresolved];f=n['feature'];threshold=float(n['threshold'])
        for side in ['left','right']:
            nn=list(nodes);nn[unresolved]=n[side];aa=list(a);bb=list(b)
            if side=='left':bb[f]=min(bb[f],threshold)
            else:aa[f]=max(aa[f],math.nextafter(threshold,math.inf))
            push(tuple(nn),tuple(aa),tuple(bb))
    # Frontier nodes at or above a feasible cost cannot improve it.
    while queue and best is not None and queue[0][0]>=best[0]:heapq.heappop(queue);pruned_cost+=1
    complete=not queue;status=('optimal' if best is not None else 'infeasible') if complete else 'budget_exhausted'
    lower=best[0] if complete and best else (queue[0][0] if queue else None)
    upper=best[0] if best else None
    solution=None
    if best:
        cost,x,prediction,values,ids=best
        # Independent public inference route, beyond the search's direct walk.
        actual=[float(tree.predict(m,[x])[0]) for m in models]
        native=float(bagging.predict(model,[x],prefix)[0]) if isinstance(model,dict) and model.get('format')=='treelab-group-bagging-v1' else float(tree.predict(model,[x])[0])
        if actual!=values or float(np.mean(actual))!=prediction or native!=prediction:raise ArithmeticError('ensemble witness verification failed')
        solution={'input':list(x),'prediction':prediction,'member_predictions':values,'member_leaf_ids':ids,'cost':float(cost),'cost_exact':rational(cost),'changed_features':[i for i,(u,v) in enumerate(zip(row,x)) if u!=v]}
    return {'format':'treelab-forest-search-v1','row':row,'base_prediction':predict(row)[0],'target':target,'bounds':bounds,'weights':weights,'immutable':immutable,'members':len(models),'seed_inputs':[list(z) for z in seeds_extra],'max_states':max_states,'seed_evaluations':seed_evaluations,'feasible_seeds':seed_feasible,'status':status,'solution':solution,'objective_lower_exact':rational(lower),'objective_upper_exact':rational(upper),'absolute_gap_exact':rational(upper-lower) if upper is not None and lower is not None else None,'complete':complete,'frontier_states':len(queue),'expanded_states':visited,'generated_states':generated,'peak_frontier_states':peak,'pruned':{'domain':pruned_domain,'output':pruned_output,'cost':pruned_cost},'meaning':'Weighted-L1 search over finite binary64 inputs of the supplied mean ensemble. Exact rational objective bounds, rounded NumPy binary64 ensemble output as in inference. Global optimum/infeasibility only when complete; budget exhaustion is not infeasibility. One minimizer, no canonical tie or all-optima enumeration. Independent feature constraints are not causal recourse or proof of plausible inputs.'}


def run(folder):
    from . import data
    from .__main__ import strict_load
    x,y,names=data.load();split=data.split(x);root=Path(__file__).resolve().parent
    model=strict_load(root/'artifacts/bagging-model.json.gz');low=x[split['fit']].min(axis=0);high=x[split['fit']].max(axis=0);bounds=[[float(a),float(b)] for a,b in zip(low,high)];weights=[float(1/(b-a)) if b>a else 1. for a,b in bounds];cases=[]
    slices=strict_load(root/'artifacts/sweep-report.json')['sweeps']
    def axis_seeds(row_id,target,policy):
        seeds=[]
        for r in slices:
            if r['row_id']!=row_id or r['model']!='bagging_32' or (policy=='alcohol_only' and r['feature_name']!='alcohol'):continue
            options=[];j=r['feature']
            for segment in r['segments']:
                if target[0]<=segment['prediction']<=target[1]:
                    a=segment['lower'] if segment['lower_inclusive'] else math.nextafter(segment['lower'],math.inf);b=segment['upper']
                    z=r['row'].copy();z[j]=min(max(z[j],a),b)
                    options.append((sum((Fraction(w)*abs(Fraction(v)-Fraction(u)) for w,v,u in zip(weights,z,r['row'])),Fraction()),tuple(z)))
            if options:seeds.append(min(options)[1])
        return [list(z) for z in dict.fromkeys(seeds)]
    for row_id in split['test'][:3]:
        for target in [[6.,7.],[7.,8.]]:
            for policy,immutable in [('all_features',[]),('alcohol_only',[i for i in range(len(names)) if names[i]!='alcohol'])]:
                for budget in [0,128,4096]:
                    r=search(model,x[row_id],target,bounds,weights,immutable,budget,seed_inputs=axis_seeds(int(row_id),target,policy));r.update(row_id=int(row_id),policy=policy);cases.append(r)
    report={'format':'treelab-forest-search-report-v1','features':names,'source_rows':split['test'][:3].tolist(),'protocol':{'model':'Unchanged frozen 32-tree group-bagging ensemble; original member order','rows':'First three original historical test rows, including exact-copy pair','targets':'Fixed closed model-output bands [6,7] and [7,8]','policies':'All features or alcohol only','budgets':'0, 128, 4096 expanded frontier states; not wall time or total compute','constraints':'Fitting-only min/max and stored binary64 reciprocal ranges; no refit or test-target use','warm_starts':'Projected source, lower/upper/midpoint domain seeds plus closest feasible target point from each previously frozen one-axis ensemble slice (only alcohol for alcohol-only policy). All seed vectors are retained and verified on the full ensemble before becoming an incumbent. Seeds improve upper bounds, never establish optimality.','search':'Exact rational weighted-L1 priority, first unresolved member split, forced-branch propagation across every tree, subtree-mean output bounds using the same NumPy reduction as inference','completion':'Optimal and infeasible only when the search frontier certifies completion; budget_exhausted with or without a witness is not infeasibility'},'cases':cases,'limits':['Exponential worst-case region search; no scalable-solver or speed benchmark claim','Exact objective bounds on supplied binary64 numbers, not exact rational model outputs or training arithmetic','Lower/upper costs concern the same full-ensemble input, never separately attainable member optima','Independent feature boxes do not prove causal actionability, feasible chemistry or distribution support','No fresh accuracy claim or change to previous models, metrics, reports or figures']}
    out=Path(folder);out.mkdir(parents=True,exist_ok=True);(out/'forest-search-report.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');return report
