import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from treelab import data,tree,experiment
ROOT=Path(__file__).resolve().parents[1]


def equal(test,a,b):
    test.assertEqual(type(a),type(b))
    if isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,list):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    elif isinstance(a,float):test.assertTrue(np.isfinite(a) and np.isfinite(b));test.assertAlmostEqual(a,b,delta=2e-9)
    else:test.assertEqual(a,b)

class DataAndExperiment(unittest.TestCase):
    def test_original_csv_and_names_hashes_and_shape(self):
        x,y,names=data.load();self.assertEqual(x.shape,(4898,11));self.assertEqual(len(names),11);self.assertTrue(np.isfinite(x).all());self.assertEqual(set(y),{3,4,5,6,7,8,9});self.assertEqual(hashlib.sha256((ROOT/'treelab/data/winequality.names').read_bytes()).hexdigest(),'15d215e73b39105952380fd487e4ce1bf5a6b83a425abb285c2ec55193d3ca92')
    def test_group_disjoint_complete_partition_and_counts(self):
        x,_,_=data.load();s=data.split(x);self.assertEqual({k:len(v) for k,v in s.items()},{'fit':2921,'validation':1008,'test':969});self.assertEqual(sorted(np.concatenate(list(s.values())).tolist()),list(range(4898)));groups={k:{tuple(row) for row in x[v]} for k,v in s.items()}
        for a,b in [('fit','validation'),('fit','test'),('validation','test')]:self.assertFalse(groups[a]&groups[b])
        self.assertEqual(len(set(map(tuple,x))),3961)
    def test_duplicate_features_with_conflicting_targets_stay_together(self):
        x=np.array([[i] for i in range(10)]+[[3],[3]],float);s=data.split(x);sets=[set(v) for v in s.values()];self.assertTrue(any({3,10,11}<=v for v in sets))
    def test_split_is_target_blind_and_permutation_equivariant(self):
        x,_,_=data.load();p=np.random.default_rng(32).permutation(len(x));a=data.split(x);b=data.split(x[p]);self.assertEqual({k:set(v) for k,v in a.items()},{k:set(p[v]) for k,v in b.items()})
    def test_full_refit_and_report_reproduction(self):
        with tempfile.TemporaryDirectory() as out:
            r=experiment.run(out);saved=json.loads((ROOT/'treelab/artifacts/report.json').read_text());equal(self,saved,r)
            for name in ['grown-model','selected-model']:equal(self,json.loads((ROOT/'treelab/artifacts'/f'{name}.json').read_text()),json.loads((Path(out)/f'{name}.json').read_text()))
    def test_changed_test_targets_cannot_change_fitting_or_selection(self):
        x,y,names=data.load();y=y.copy();s=data.split(x);y[s['test']]=10-y[s['test']]
        with tempfile.TemporaryDirectory() as out,mock.patch('treelab.data.load',return_value=(x,y,names)):
            r=experiment.run(out);saved=json.loads((ROOT/'treelab/artifacts/report.json').read_text());equal(self,r['path'],saved['path']);self.assertEqual(r['selected_index'],saved['selected_index']);equal(self,json.loads((Path(out)/'selected-model.json').read_text()),json.loads((ROOT/'treelab/artifacts/selected-model.json').read_text()))
    def test_test_predictions_start_only_after_validation_selection(self):
        x,_,_=data.load();s=data.split(x);selected=False;original_select=tree.select;original_predict=tree.predict
        def choose(*args):
            nonlocal selected
            r=original_select(*args);selected=True;return r
        def prediction(model,rows,*args,**kwargs):
            if len(rows)==len(s['test']):self.assertTrue(selected)
            return original_predict(model,rows,*args,**kwargs)
        with tempfile.TemporaryDirectory() as out,mock.patch('treelab.tree.select',side_effect=choose),mock.patch('treelab.tree.predict',side_effect=prediction):experiment.run(out)
    def test_every_saved_test_prediction_independent_traversal(self):
        x,y,_=data.load();r=json.loads((ROOT/'treelab/artifacts/report.json').read_text());m=json.loads((ROOT/'treelab/artifacts/selected-model.json').read_text());errors=[]
        for row in r['test_rows']:
            n=m['root']
            while 'feature' in n:n=n['left'] if x[row['row'],n['feature']]<=n['threshold'] else n['right']
            self.assertEqual(n['mean'],row['selected']);self.assertEqual(row['target'],y[row['row']]);errors.append((n['mean']-row['target'])**2)
        self.assertAlmostEqual(sum(errors)/len(errors),r['test_metrics']['selected']['mse'],places=12)
    def test_same_fit_linear_control_and_normal_equation(self):
        x,y,_=data.load();r=json.loads((ROOT/'treelab/artifacts/report.json').read_text());s=data.split(x);c=r['linear_control'];np.testing.assert_array_equal(c['fit_center'],x[s['fit']].mean(axis=0));np.testing.assert_array_equal(c['fit_scale'],x[s['fit']].std(axis=0));a=np.c_[np.ones(len(s['fit'])),(x[s['fit']]-c['fit_center'])/c['fit_scale']];coef=np.array(c['intercept_and_coefficients']);self.assertLess(float(np.max(np.abs(a.T@(a@coef-y[s['fit']])))),1e-8);self.assertLess(r['test_metrics']['least_squares']['mse'],r['test_metrics']['selected']['mse'])
    def test_saved_metrics_and_selected_index_are_not_cherry_picked(self):
        r=json.loads((ROOT/'treelab/artifacts/report.json').read_text());idx=min(range(len(r['path'])),key=lambda i:(r['path'][i]['validation_mse'],r['path'][i]['leaves'],i));self.assertEqual(idx,r['selected_index']);self.assertEqual(r['selected_leaves'],19);self.assertEqual(r['grown_leaves'],425);self.assertEqual(len(r['path']),290)
        for name,m in r['test_metrics'].items():
            error=np.array([row[name]-row['target'] for row in r['test_rows']]);self.assertAlmostEqual(float(np.mean(error**2)),m['mse']);self.assertAlmostEqual(float(np.mean(np.abs(error))),m['mae'])
    def test_report_guard_rejects_hidden_float_discrete_and_missing(self):
        r=json.loads((ROOT/'treelab/artifacts/report.json').read_text())
        for kind in range(4):
            q=copy.deepcopy(r)
            if kind==0:q['test_rows'][-1]['selected']+=.01
            if kind==1:q['selected_leaves']+=1
            if kind==2:q['test_rows'][0]['grown']=float('nan')
            if kind==3:del q['dataset_sha256']
            with self.assertRaises(AssertionError):equal(self,r,q)
