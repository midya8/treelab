"""Fixed protocol. Test rows are evaluated only after validation selection."""
import hashlib,json
from pathlib import Path
import numpy as np
from . import data,tree


def metrics(y,p):
    return {'rows':len(y),'mse':float(np.mean((y-p)**2)),'rmse':float(np.sqrt(np.mean((y-p)**2))),'mae':float(np.mean(np.abs(y-p)))}


def run(folder):
    x,y,names=data.load();indices=data.split(x);fit=indices['fit'];val=indices['validation'];test=indices['test']
    model=tree.fit(x[fit],y[fit],min_leaf=5,max_depth=20);path=tree.pruning_path(model);chosen,validation=tree.select(path,x[val],y[val]);selected=path[chosen]['model']
    # No test prediction or metric participates in fitting/path/selection.
    predictions={name:tree.predict(m,x[test]) for name,m in [('grown',model),('selected',selected)]};mean=float(y[fit].mean());median=float(np.median(y[fit]));predictions.update(fit_mean=np.full(len(test),mean),fit_median=np.full(len(test),median))
    # Standardized ridge-free least squares is a separate strong linear control.
    center=x[fit].mean(axis=0);scale=x[fit].std(axis=0);scale=np.where(scale==0,1,scale)
    design=np.c_[np.ones(len(fit)),(x[fit]-center)/scale];coef=np.linalg.lstsq(design,y[fit],rcond=None)[0];predictions['least_squares']=np.c_[np.ones(len(test)),(x[test]-center)/scale]@coef
    _,traces=tree.predict(selected,x[test[:6]],paths=True)
    report={'format':'treelab-report-v1','dataset_sha256':data.HASH,'features':names,'protocol':{'min_leaf':5,'max_depth':20,'split':'Sort SHA-256 of canonical parsed feature values; first floor(60%) unique groups fit, next through floor(80%) validation, rest test','selection':'Minimum validation MSE over full weakest-link path; exact ties prefer fewer leaves; no refit','alpha_units':'SSE / number of fitting rows + alpha * leaves; alpha in target-score-squared units per leaf','source_target':'Ordinal expert quality score treated as numeric; not a price, safety or causal target'},'indices':{k:v.tolist() for k,v in indices.items()},'split_counts':{k:len(v) for k,v in indices.items()},'unique_feature_groups':len({data.group_key(row) for row in x}),'conflicting_target_groups':0,'selected_index':chosen,'selected_alpha':path[chosen]['alpha'],'selected_leaves':path[chosen]['leaves'],'grown_leaves':path[0]['leaves'],'path':[{'alpha':p['alpha'],'fit_mse':p['risk'],'leaves':p['leaves'],'pruned':p['pruned'],'validation_mse':validation[i]} for i,p in enumerate(path)],'test_metrics':{k:metrics(y[test],p) for k,p in predictions.items()},'test_rows':[{'row':int(row),'target':float(y[row]),**{k:float(p[j]) for k,p in predictions.items()}} for j,row in enumerate(test)],'selected_examples':[{'row':int(row),'target':float(y[row]),**trace} for row,trace in zip(test[:6],traces)],'linear_control':{'fit_center':center.tolist(),'fit_scale':scale.tolist(),'intercept_and_coefficients':coef.tolist()},'limits':['One deterministic group-disjoint split, not confidence intervals, writer/batch independence, time-forward testing or a deployment benchmark','Greedy CART and floating weakest-link pruning are established methods, not globally optimal tree learning or exact certificates','Feature duplicates stay together, including target conflicts; near duplicates and missing batch identifiers remain unresolved','Squared error treats ordinal scores as numeric and penalizes large errors; other losses answer different questions']}
    groups={}
    for row,yi in zip(x,y):groups.setdefault(data.group_key(row),set()).add(float(yi))
    report['conflicting_target_groups']=sum(len(v)>1 for v in groups.values())
    output=Path(folder);output.mkdir(parents=True,exist_ok=True)
    for name,obj in [('report',report),('grown-model',model),('selected-model',selected)]:
        (output/(name+'.json')).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    return report
