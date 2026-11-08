import copy,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from treelab import bagging,data,tree
from test_experiment import equal
ROOT=Path(__file__).resolve().parents[1]

class BaggingTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0.],[0.],[1.],[2.],[3.],[3.]])
        self.y=np.array([0.,2.,4.,6.,8.,10.])
        self.e=bagging.fit(self.x,self.y,8,7,1,5)
    def test_determinism_prefix_stability_and_exact_group_multiplicity(self):
        self.assertEqual(self.e,bagging.fit(self.x,self.y,8,7,1,5));self.assertEqual(self.e['trees'][:4],bagging.fit(self.x,self.y,4,7,1,5)['trees'])
        g=self.e['groups'];rng=np.random.default_rng(7)
        for t in self.e['trees']:
            draws=rng.integers(0,len(g),len(g)).tolist();self.assertEqual(t['draws'],draws);rows=[i for j in draws for i in g[j]]
            self.assertEqual(t['model'],tree.fit(self.x[rows],self.y[rows],1,5));self.assertEqual(t['model']['fit_rows'],len(rows))
    def test_oob_support_and_means_independent_membership_oracle(self):
        mask=bagging.oob_mask(self.e);r=bagging.oob(self.e,self.x,self.y,[1,4,8]);g=self.e['groups']
        for p in r['prefixes']:
            errors=[]
            for i,row in enumerate(p['rows']):
                group=next(j for j,rows in enumerate(g) if i in rows);values=[]
                for k,t in enumerate(self.e['trees'][:p['trees']]):
                    eligible=group not in t['draws'];self.assertEqual(eligible,bool(mask[k,i]))
                    if eligible:values.append(float(tree.predict(t['model'],self.x[[i]])[0]))
                self.assertEqual(row['votes'],len(values))
                if not values:self.assertIsNone(row['prediction'])
                else:self.assertAlmostEqual(row['prediction'],sum(values)/len(values));errors.append((row['prediction']-self.y[i])**2)
            self.assertEqual(p['supported_rows'],len(errors));self.assertAlmostEqual(p['row_weighted_mse_on_supported'],sum(errors)/len(errors))
    def test_duplicate_targets_do_not_split_feature_group(self):
        for group in self.e['groups']:
            for a in group:
                self.assertTrue(all(np.array_equal(self.x[a],self.x[b]) for b in group))
        mask=bagging.oob_mask(self.e);np.testing.assert_array_equal(mask[:,0],mask[:,1]);np.testing.assert_array_equal(mask[:,4],mask[:,5])
    def test_single_group_has_no_supported_oob_rows(self):
        x=np.ones((3,2));y=np.arange(3.);e=bagging.fit(x,y,4,0,1);r=bagging.oob(e,x,y,[1,4])
        for p in r['prefixes']:
            self.assertEqual(p['supported_rows'],0);self.assertIsNone(p['row_weighted_mse_on_supported']);self.assertTrue(all(a['prediction'] is None and a['votes']==0 for a in p['rows']))
    def test_predict_is_independent_tree_average_and_valid_prefix(self):
        for b in [1,4,8]:np.testing.assert_array_equal(bagging.predict(self.e,self.x,b),np.array([tree.predict(t['model'],self.x) for t in self.e['trees'][:b]]).mean(axis=0))
        for b in [True,0,9,1.0]:
            with self.assertRaises(ValueError):bagging.predict(self.e,self.x,b)
    def test_duplicate_row_absence_witness_is_not_group_exclusion(self):
        r=bagging.duplicate_witness();self.assertTrue(r['row_absent']);self.assertTrue(r['identical_feature_copy_in_bag']);self.assertFalse(r['group_oob_eligible']);self.assertEqual(r['naive_row_oob_squared_error'],0.)
    def test_invalid_training_and_prefixes(self):
        for k,v in [('n_trees',True),('n_trees',0),('n_trees',257),('seed',True),('seed',-1),('seed',2**64)]:
            with self.assertRaises(ValueError):bagging.fit(self.x,self.y,**{k:v})
        for p in [[],[0],[9],[True],[1.,4],[4,1],[1,1]]:
            with self.assertRaises(ValueError):bagging.oob(self.e,self.x,self.y,p)
        with self.assertRaises(ValueError):bagging.fit(self.x,self.y[:-1])
        with self.assertRaises(ValueError):bagging.predict(self.e,[[1,2]])
    def test_oob_requires_original_order_features_targets_and_group_bindings(self):
        for x,y in [(self.x[::-1],self.y[::-1]),(self.x,self.y+1),(self.x+1,self.y)]:
            with self.assertRaises(ValueError):bagging.oob(self.e,x,y,[1])
        e=copy.deepcopy(self.e);e['group_keys'][0]='0'*64
        with self.assertRaises(ValueError):bagging.oob(e,self.x,self.y,[1])
    def test_strict_ensemble_serialization_guards(self):
        mutations=[lambda e:e.update(extra=1),lambda e:e.update(features=True),lambda e:e.update(seed=False),lambda e:e.update(max_depth=31),lambda e:e.update(feature_sha256='x'),lambda e:e['group_keys'].reverse(),lambda e:e['groups'][0].append(99),lambda e:e['trees'][0]['draws'].__setitem__(0,True),lambda e:e['trees'][0]['draws'].append(0),lambda e:e['trees'][0]['model'].update(features=2),lambda e:e['trees'][0]['model'].update(fit_rows=999),lambda e:e['trees'][0]['model']['root'].update(mean=float('nan'))]
        for f in mutations:
            e=copy.deepcopy(self.e);f(e)
            with self.assertRaises((ValueError,TypeError)):bagging.validate(e)
    def test_full_refit_reproduces_both_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            bagging.run(d)
            for n in ['bagging-model','bagging-report']:equal(self,json.load(open(Path(d)/(n+'.json'))),json.load(open(ROOT/'treelab/artifacts'/(n+'.json'))))
    def test_real_every_group_oob_membership_and_saved_tree_prediction(self):
        e=json.load(open(ROOT/'treelab/artifacts/bagging-model.json'));r=json.load(open(ROOT/'treelab/artifacts/bagging-report.json'));x,y,_=data.load();s=data.split(x);f=s['fit'];v=s['validation'];t=s['test'];mask=bagging.oob_mask(e)
        for k,m in enumerate(e['trees']):
            included={i for j in m['draws'] for i in e['groups'][j]};np.testing.assert_array_equal(mask[k],[i not in included for i in range(len(f))])
            for rows,saved in [(f,r['oob']['member_predictions'][k]),(v,r['validation_member_predictions'][k]),(t,r['test_member_predictions'][k])]:
                values=[]
                for i in rows:
                    n=m['model']['root']
                    while 'feature' in n:n=n['left'] if x[i,n['feature']]<=n['threshold'] else n['right']
                    values.append(n['mean'])
                np.testing.assert_array_equal(values,saved)
        o=bagging.oob(e,x[f],y[f],[1,4,16,32]);equal(self,o,r['oob']);self.assertEqual([p['unsupported_rows'] for p in o['prefixes']],[1832,482,1,0])
        b,scores=bagging.select(np.array(r['validation_member_predictions']),y[v],[1,4,16,32]);self.assertEqual(b,32);np.testing.assert_array_equal(scores,[p['validation_mse'] for p in r['prefixes']]);pred=np.array(r['test_member_predictions'])[:b].mean(axis=0);np.testing.assert_array_equal(pred,[a['prediction'] for a in r['test_rows']]);self.assertAlmostEqual(float(np.mean((pred-y[t])**2)),r['test_metrics']['bagged_selected']['mse'])
    def test_changed_test_targets_do_not_change_ensemble_or_prefix_selection(self):
        x,y,names=data.load();y=y.copy();s=data.split(x);y[s['test']]=10-y[s['test']]
        with tempfile.TemporaryDirectory() as d,mock.patch('treelab.data.load',return_value=(x,y,names)):
            r=bagging.run(d);saved=json.load(open(ROOT/'treelab/artifacts/bagging-report.json'));self.assertEqual(r['selected_trees'],saved['selected_trees']);self.assertEqual(r['prefixes'],saved['prefixes']);self.assertEqual(r['test_member_predictions'],saved['test_member_predictions']);equal(self,json.load(open(Path(d)/'bagging-model.json')),json.load(open(ROOT/'treelab/artifacts/bagging-model.json')))
    def test_test_predictions_are_only_computed_after_selection(self):
        x,_,_=data.load();s=data.split(x);selected=False;original=bagging.select;predict=bagging.prediction_matrix
        def choose(*a):
            nonlocal selected
            ans=original(*a);selected=True;return ans
        def check(e,rows):
            if len(rows)==len(s['test']):self.assertTrue(selected)
            return predict(e,rows)
        with tempfile.TemporaryDirectory() as d,mock.patch('treelab.bagging.select',side_effect=choose),mock.patch('treelab.bagging.prediction_matrix',side_effect=check):bagging.run(d)
    def test_exact_selection_tie_prefers_fewer_trees(self):self.assertEqual(bagging.select(np.zeros((4,2)),np.zeros(2),[1,4]),(1,[0.,0.]))
