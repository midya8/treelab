"""Group-bootstrap tree averaging with explicit group-excluded OOB support.

Established bagging. No feature subsampling or calibrated uncertainty interval.
"""
import hashlib,json
import numpy as np
from . import data,tree


def fingerprint(a):
    return hashlib.sha256(json.dumps(a.tolist(),separators=(',',':'),allow_nan=False).encode()).hexdigest()


def groups(x):
    by_key={}
    for i,row in enumerate(x):by_key.setdefault(data.group_key(row),[]).append(i)
    keys=sorted(by_key);return keys,[by_key[k] for k in keys]


def fit(x,y,n_trees=32,seed=20261108,min_leaf=5,max_depth=20):
    x=tree.array(x,2,'features');y=tree.array(y,1,'targets')
    if len(x)!=len(y):raise ValueError('row counts differ')
    if type(n_trees) is not int or not 1<=n_trees<=256:raise ValueError('n_trees must be an integer in 1..256')
    if type(seed) is not int or not 0<=seed<2**64:raise ValueError('seed must be an integer in 0..2**64-1')
    keys,g=groups(x);rng=np.random.default_rng(seed);members=[]
    for _ in range(n_trees):
        draws=rng.integers(0,len(g),len(g)).tolist();rows=np.concatenate([g[j] for j in draws]);m=tree.fit(x[rows],y[rows],min_leaf,max_depth);members.append({'draws':draws,'model':m})
    return {'format':'treelab-group-bagging-v1','features':x.shape[1],'fit_rows':len(y),'seed':seed,'min_leaf':min_leaf,'max_depth':max_depth,'feature_sha256':fingerprint(x),'target_sha256':fingerprint(y),'group_keys':keys,'groups':g,'trees':members}


def validate(e):
    fields={'format','features','fit_rows','seed','min_leaf','max_depth','feature_sha256','target_sha256','group_keys','groups','trees'}
    if not isinstance(e,dict) or set(e)!=fields or e['format']!='treelab-group-bagging-v1':raise ValueError('invalid ensemble header')
    for k in ['features','fit_rows','min_leaf']:
        if type(e[k]) is not int or e[k]<1:raise ValueError('invalid '+k)
    if type(e['max_depth']) is not int or not 0<=e['max_depth']<=30:raise ValueError('invalid max_depth')
    if type(e['seed']) is not int or not 0<=e['seed']<2**64:raise ValueError('invalid seed')
    def digest(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
    if not digest(e['feature_sha256']) or not digest(e['target_sha256']):raise ValueError('invalid training digest')
    g=e['groups'];keys=e['group_keys']
    if not isinstance(g,list) or not g or not isinstance(keys,list) or len(g)!=len(keys) or any(not digest(k) for k in keys) or keys!=sorted(set(keys)):raise ValueError('invalid groups or keys')
    flat=[]
    for rows in g:
        if not isinstance(rows,list) or not rows or any(type(i) is not int for i in rows) or rows!=sorted(set(rows)):raise ValueError('invalid group rows')
        flat+=rows
    if sorted(flat)!=list(range(e['fit_rows'])):raise ValueError('group partition invalid')
    if not isinstance(e['trees'],list) or not 1<=len(e['trees'])<=256:raise ValueError('invalid member count')
    for member in e['trees']:
        if not isinstance(member,dict) or set(member)!={'draws','model'}:raise ValueError('invalid member fields')
        draws=member['draws']
        if not isinstance(draws,list) or len(draws)!=len(g) or any(type(j) is not int or not 0<=j<len(g) for j in draws):raise ValueError('invalid group draws')
        m=tree.validate(member['model'])
        if m['features']!=e['features'] or m['min_leaf']!=e['min_leaf'] or m['max_depth']!=e['max_depth'] or m['fit_rows']!=sum(len(g[j]) for j in draws):raise ValueError('bootstrap model metadata disagrees')
    return e


def prediction_matrix(e,x):
    validate(e);x=tree.array(x,2,'features')
    if x.shape[1]!=e['features']:raise ValueError('wrong feature width')
    return np.array([tree.predict(t['model'],x) for t in e['trees']])


def predict(e,x,n_trees=None):
    matrix=prediction_matrix(e,x)
    if n_trees is None:n_trees=len(matrix)
    if type(n_trees) is not int or not 1<=n_trees<=len(matrix):raise ValueError('invalid prefix size')
    return matrix[:n_trees].mean(axis=0)


def oob_mask(e):
    validate(e);mask=np.zeros((len(e['trees']),e['fit_rows']),dtype=bool)
    for t,member in enumerate(e['trees']):
        seen=set(member['draws'])
        for j,rows in enumerate(e['groups']):
            if j not in seen:mask[t,rows]=True
    return mask


def oob(e,x,y,prefixes):
    x=tree.array(x,2,'OOB features');y=tree.array(y,1,'OOB targets');validate(e)
    if fingerprint(x)!=e['feature_sha256'] or fingerprint(y)!=e['target_sha256']:raise ValueError('OOB requires original fitting data, in original row order')
    keys,g=groups(x)
    if keys!=e['group_keys'] or g!=e['groups']:raise ValueError('OOB group bindings disagree')
    if not isinstance(prefixes,list) or not prefixes or any(type(b) is not int or not 1<=b<=len(e['trees']) for b in prefixes) or prefixes!=sorted(set(prefixes)):raise ValueError('invalid OOB prefixes')
    matrix=prediction_matrix(e,x);mask=oob_mask(e);reports=[]
    for b in prefixes:
        counts=mask[:b].sum(axis=0);sums=np.where(mask[:b],matrix[:b],0).sum(axis=0);supported=counts>0;pred=np.divide(sums,counts,out=np.zeros_like(sums),where=supported)
        errors=pred[supported]-y[supported];mse=float(np.mean(errors**2)) if np.any(supported) else None
        reports.append({'trees':b,'supported_rows':int(supported.sum()),'unsupported_rows':int((~supported).sum()),'min_votes':int(counts.min()),'max_votes':int(counts.max()),'row_weighted_mse_on_supported':mse,'rows':[{'fit_row':i,'votes':int(n),'prediction':float(pred[i]) if n else None,'target':float(y[i])} for i,n in enumerate(counts)]})
    return {'meaning':'Average only trees that excluded the entire exact feature group; unsupported rows have null predictions. Descriptive, not calibrated uncertainty or guaranteed unbiased error.','prefixes':reports,'member_predictions':matrix.tolist()}


def duplicate_witness():
    x=np.array([[0.],[0.],[1.],[1.]]);y=np.array([0.,0.,10.,10.]);draws=[1,1,2,2];m=tree.fit(x[draws],y[draws],1);pred=float(tree.predict(m,x[[0]])[0])
    return {'features':x.tolist(),'targets':y.tolist(),'row_bootstrap_draws':draws,'withheld_row':0,'row_absent':0 not in draws,'identical_feature_copy_in_bag':1 in draws,'naive_row_oob_prediction':pred,'naive_row_oob_squared_error':float((pred-y[0])**2),'group_oob_eligible':False,'meaning':'Row absence alone does not exclude an exact feature copy; a single witness, not a causal estimate of real-data optimism.'}


def select(matrix,y,prefixes):
    scores=[float(np.mean((matrix[:b].mean(axis=0)-y)**2)) for b in prefixes]
    chosen=min(range(len(prefixes)),key=lambda i:(scores[i],prefixes[i]))
    return prefixes[chosen],scores


def run(folder):
    from pathlib import Path
    from . import experiment
    x,y,names=data.load();s=data.split(x);f=s['fit'];v=s['validation'];t=s['test'];prefixes=[1,4,16,32]
    e=fit(x[f],y[f],32,20261108,5,20);vm=prediction_matrix(e,x[v]);b,validation=select(vm,y[v],prefixes)
    # OOB and held-out test predictions do not choose the prefix.
    o=oob(e,x[f],y[f],prefixes);tm=prediction_matrix(e,x[t]);pred=tm[:b].mean(axis=0)
    prior=json.loads((Path(__file__).resolve().parent/'artifacts/report.json').read_text())
    if prior['indices']!={k:a.tolist() for k,a in s.items()}:raise ValueError('v0.1 split changed')
    report={'format':'treelab-bagging-report-v1','dataset_sha256':data.HASH,'features':names,'fit_feature_sha256':e['feature_sha256'],'fit_target_sha256':e['target_sha256'],'protocol':{'seed':20261108,'trees_fitted':32,'prefix_grid':prefixes,'min_leaf':5,'max_depth':20,'sampling':'Draw G fitting-feature groups G times with replacement per tree; include every original row of every drawn group, repeated for multiplicity','selection':'Minimum validation MSE, exact ties prefer smaller prefix; no OOB or test selection','prediction':'Unweighted average of tree predictions; no feature subsampling, no pruning, not a random forest','OOB':'Exclude the entire exact feature group. Supported-subset row-weighted MSE; zero votes -> null, not a zero prediction.'},'split_counts':prior['split_counts'],'indices':prior['indices'],'fit_groups':len(e['groups']),'bootstrap_rows':[m['model']['fit_rows'] for m in e['trees']],'bootstrap_leaves':[tree.subtree(m['model']['root'])[1] for m in e['trees']],'prefixes':[{'trees':bb,'validation_mse':score,'oob_supported_rows':oo['supported_rows'],'oob_unsupported_rows':oo['unsupported_rows'],'oob_mse_supported':oo['row_weighted_mse_on_supported']} for bb,score,oo in zip(prefixes,validation,o['prefixes'])],'selected_trees':b,'test_metrics':{'bagged_selected':experiment.metrics(y[t],pred),**prior['test_metrics']},'validation_member_predictions':vm.tolist(),'test_member_predictions':tm.tolist(),'test_rows':[{'row':int(row),'target':float(y[row]),'prediction':float(pred[j])} for j,row in enumerate(t)],'oob':o,'duplicate_witness':duplicate_witness(),'limits':['Feature groups are exact-copy exclusions, not known producer/batch/time-independent experimental units','Group sampling changes the resampling unit; row counts vary with group sizes, and validation/test metrics remain row weighted','OOB uses fewer, row-dependent trees and supported subsets; do not compare different-prefix OOB MSE as the same set or treat spread as calibrated uncertainty','One seed and one split; prefix selection uses validation, so the reused historical test is descriptive rather than a new blind benchmark','Established bootstrap averaging, not feature-subsampled random forests, causal component isolation, confidence intervals or proof of future performance']}
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('bagging-report',report),('bagging-model',e)]:
        (out/(name+'.json')).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    return report
