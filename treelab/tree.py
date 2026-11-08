"""Greedy squared-error CART and weakest-link pruning, implemented in NumPy.

Established algorithms, not a globally optimal tree learner or exact certificate.
"""
import copy
import math
import numpy as np


def array(value, ndim, name):
    a=np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.ndim!=ndim or a.size==0:
        raise ValueError(name+' must be a nonempty numeric array')
    a=a.astype(np.float64)
    if not np.isfinite(a).all() or np.any(np.abs(a)>1e100):
        raise ValueError(name+' must be finite and within +/-1e100')
    return a


def stats(y):
    origin=float(y[0]);center=origin+math.fsum(float(v)-origin for v in y)/len(y)
    return center,math.fsum((float(v)-center)**2 for v in y)


def best_split(x,y,min_leaf=1):
    """Enumerate all distinct adjacent-value splits with prefix SSE statistics."""
    center,_=stats(y);best=None
    for feature in range(x.shape[1]):
        order=np.argsort(x[:,feature],kind='stable');v=x[order,feature];d=y[order]-center
        sums=np.cumsum(d);squares=np.cumsum(d*d);left=np.arange(1,len(y))
        valid=(left>=min_leaf)&(len(y)-left>=min_leaf)&(v[:-1]<v[1:])
        positions=np.flatnonzero(valid)
        if not len(positions):continue
        n=left[positions];s=sums[positions];q=squares[positions]
        losses=np.maximum(0,q-s*s/n)+np.maximum(0,squares[-1]-q-(sums[-1]-s)**2/(len(y)-n))
        j=int(np.argmin(losses));i=int(positions[j]);loss=float(losses[j])
        threshold=float(v[i]+(v[i+1]-v[i])/2)
        if threshold>=v[i+1]:threshold=float(v[i])
        candidate=(loss,feature,threshold)
        if best is None or candidate<best:best=candidate
    return best


def fit(x,y,min_leaf=5,max_depth=20):
    x=array(x,2,'features');y=array(y,1,'targets')
    if len(x)!=len(y):raise ValueError('row counts differ')
    if type(min_leaf) is not int or min_leaf<1:raise ValueError('min_leaf must be a positive integer')
    if type(max_depth) is not int or not 0<=max_depth<=30:raise ValueError('max_depth must be an integer in 0..30')
    next_id=0
    def grow(rows,depth):
        nonlocal next_id
        mean,sse=stats(y[rows]);node={'id':next_id,'n':len(rows),'mean':mean,'sse':sse};next_id+=1
        if depth>=max_depth or len(rows)<2*min_leaf or sse==0:return node
        split=best_split(x[rows],y[rows],min_leaf)
        if split is None or split[0]>=sse:return node
        _,feature,threshold=split;mask=x[rows,feature]<=threshold
        if min(mask.sum(),(~mask).sum())<min_leaf:raise ArithmeticError('split partition failed')
        node.update(feature=feature,threshold=threshold,left=grow(rows[mask],depth+1),right=grow(rows[~mask],depth+1))
        return node
    return {'format':'treelab-cart-v1','features':x.shape[1],'fit_rows':len(y),'min_leaf':min_leaf,'max_depth':max_depth,'root':grow(np.arange(len(y)),0)}


def validate(model):
    if not isinstance(model,dict) or set(model)!={'format','features','fit_rows','min_leaf','max_depth','root'} or model['format']!='treelab-cart-v1':raise ValueError('invalid model header')
    for k in ['features','fit_rows','min_leaf']:
        if type(model[k]) is not int or model[k]<1:raise ValueError('invalid '+k)
    if type(model['max_depth']) is not int or not 0<=model['max_depth']<=30:raise ValueError('invalid max_depth')
    ids=set()
    def walk(n,depth):
        if not isinstance(n,dict) or set(n) not in ({'id','n','mean','sse'},{'id','n','mean','sse','feature','threshold','left','right'}):raise ValueError('invalid node fields')
        for k in ['id','n']:
            if type(n[k]) is not int or n[k]<(0 if k=='id' else 1):raise ValueError('invalid node '+k)
        if n['id'] in ids:raise ValueError('duplicate node id')
        ids.add(n['id'])
        for k in ['mean','sse']:
            if type(n[k]) not in (float,int) or not math.isfinite(n[k]) or abs(n[k])>(1e100 if k=='mean' else 1e250):raise ValueError('invalid node '+k)
        if n['sse']<0 or depth>model['max_depth']:raise ValueError('invalid SSE/depth')
        if 'left' in n:
            if type(n['feature']) is not int or not 0<=n['feature']<model['features']:raise ValueError('invalid split feature')
            if type(n['threshold']) not in (float,int) or not math.isfinite(n['threshold']) or abs(n['threshold'])>1e100:raise ValueError('invalid threshold')
            walk(n['left'],depth+1);walk(n['right'],depth+1)
            if n['n']!=n['left']['n']+n['right']['n']:raise ValueError('child counts disagree')
            child_mean=(n['left']['n']/n['n'])*n['left']['mean']+(n['right']['n']/n['n'])*n['right']['mean']
            if not math.isclose(n['mean'],child_mean,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('child means disagree')
            risk=n['left']['sse']+n['right']['sse']+n['left']['n']*(n['left']['mean']-n['mean'])**2+n['right']['n']*(n['right']['mean']-n['mean'])**2
            if not math.isclose(risk,n['sse'],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('SSE decomposition disagrees')
    walk(model['root'],0)
    if model['root']['n']!=model['fit_rows']:raise ValueError('fit count disagrees')
    return model


def predict(model,x,paths=False):
    validate(model);x=array(x,2,'features')
    if x.shape[1]!=model['features']:raise ValueError('wrong feature width')
    values=[];traces=[]
    for row in x:
        n=model['root'];trace=[]
        while 'left' in n:
            left=bool(row[n['feature']]<=n['threshold']);trace.append({'node':n['id'],'feature':n['feature'],'threshold':n['threshold'],'input':float(row[n['feature']]),'direction':'left' if left else 'right'});n=n['left' if left else 'right']
        values.append(n['mean']);traces.append({'steps':trace,'leaf':n['id'],'leaf_fit_rows':n['n'],'prediction':n['mean']})
    return (np.array(values),traces) if paths else np.array(values)


def subtree(n):
    if 'left' not in n:return n['sse'],1
    a,la=subtree(n['left']);b,lb=subtree(n['right']);return a+b,la+lb


def effective_alphas(model):
    """SSE divided by ROOT fitting count, so alpha is MSE per leaf."""
    validate(model);result=[]
    def visit(n):
        if 'left' in n:
            risk,leaves=subtree(n);alpha=max(0.,(n['sse']-risk)/model['fit_rows']/(leaves-1))
            result.append({'node':n['id'],'alpha':alpha,'subtree_leaves':leaves,'leaf_risk':n['sse']/model['fit_rows'],'subtree_risk':risk/model['fit_rows']});visit(n['left']);visit(n['right'])
    visit(model['root']);return result


def pruning_path(model):
    validate(model);current=copy.deepcopy(model);risk,leaves=subtree(current['root']);path=[{'alpha':0.,'risk':risk/current['fit_rows'],'leaves':leaves,'pruned':[],'model':copy.deepcopy(current)}]
    while leaves>1:
        candidates=effective_alphas(current);alpha=min(c['alpha'] for c in candidates);chosen={c['node'] for c in candidates if c['alpha']==alpha};pruned=[]
        def cut(n):
            if n['id'] in chosen:
                pruned.append(n['id'])
                for k in ['feature','threshold','left','right']:del n[k]
            elif 'left' in n:cut(n['left']);cut(n['right'])
        cut(current['root']);risk,newleaves=subtree(current['root'])
        if not newleaves<leaves:raise ArithmeticError('pruning did not shrink tree')
        leaves=newleaves
        # Tiny inversions from binary64 sums do not define a new lower breakpoint.
        alpha=max(alpha,path[-1]['alpha'])
        path.append({'alpha':alpha,'risk':risk/current['fit_rows'],'leaves':leaves,'pruned':pruned,'model':copy.deepcopy(current)})
    return path


def select(path,x,y):
    """Validation-only MSE selection; exact ties prefer the smaller subtree."""
    y=array(y,1,'validation targets');scores=[]
    for p in path:
        v=predict(p['model'],x)
        if len(v)!=len(y):raise ValueError('validation row counts differ')
        scores.append(float(np.mean((v-y)**2)))
    best=min(range(len(path)),key=lambda i:(scores[i],path[i]['leaves'],i));return best,scores


def prune(model,alpha):
    if type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<0:raise ValueError('alpha must be finite and nonnegative')
    path=pruning_path(model);eligible=[p for p in path if p['alpha']<=alpha]
    return copy.deepcopy(eligible[-1]['model'])
