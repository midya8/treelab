import copy,json,math,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from treelab import bagging,data,sweep,tree
from treelab.__main__ import strict_load
from test_experiment import equal
ROOT=Path(__file__).resolve().parents[1]

class Sweeps(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[i,j] for i in range(4) for j in range(3)],float);self.y=self.x[:,0]**2-self.x[:,1];self.m=tree.fit(self.x,self.y,1,5)
    def route(self,m,row):
        n=m['root']
        while 'feature' in n:n=n['left'] if row[n['feature']]<=n['threshold'] else n['right']
        return n
    def direct(self,m,row):return self.route(m,row)['mean']
    def check(self,m,row,f,lo,hi,prefix=None):
        r=sweep.response(m,row,f,lo,hi,prefix);models=sweep.members(m,prefix)
        def direct(axis):
            x=list(row);x[f]=axis;return float(np.mean([self.direct(a,x) for a in models]))
        self.assertTrue(r['segments']);self.assertEqual(r['segments'][0]['lower'],lo);self.assertTrue(r['segments'][0]['lower_inclusive']);self.assertEqual(r['segments'][-1]['upper'],hi)
        for i,s in enumerate(r['segments']):
            self.assertTrue(s['upper_inclusive']);self.assertEqual(s['prediction'],float(np.mean(s['member_predictions'])))
            if i:self.assertEqual(s['lower'],r['segments'][i-1]['upper']);self.assertFalse(s['lower_inclusive'])
            probes={s['witness'],s['upper']};left=s['lower'] if s['lower_inclusive'] else math.nextafter(s['lower'],math.inf);probes.add(left);probes.add(left+(s['upper']-left)/2)
            if left<s['upper']:probes.add(math.nextafter(s['upper'],-math.inf))
            for axis in probes:self.assertEqual(s['prediction'],direct(axis))
            for j,model in enumerate(models):
                x=list(row);x[f]=s['witness'];leaf=self.route(model,x);self.assertEqual(leaf['id'],s['member_leaf_ids'][j]);self.assertEqual(leaf['mean'],s['member_predictions'][j])
        for name,fun in [('minimum',min),('maximum',max)]:
            e=r[name];self.assertEqual(e['prediction'],fun(a['prediction'] for a in r['segments']));self.assertEqual(e['prediction'],direct(e['axis_value']));self.assertEqual(e['input'][f],e['axis_value']);self.assertEqual([v for i,v in enumerate(e['input']) if i!=f],[v for i,v in enumerate(row) if i!=f])
        return r
    def test_each_random_tree_and_ensemble_interval_matches_independent_direct_probes(self):
        for seed in range(8):
            rng=np.random.default_rng(seed);x=rng.integers(-3,4,(18,3));y=rng.normal(size=18);m=tree.fit(x,y,1,4);e=bagging.fit(x,y,4,seed,1,4)
            for f in range(3):
                row=rng.normal(size=3).tolist();self.check(m,row,f,-4.,4.);self.check(e,row,f,-4.,4.);self.check(e,row,f,-4.,4.,2)
    def test_global_leaf_box_oracle_keeps_only_fixed_complement_compatible_regions(self):
        row=[1.,.5];f=0;r=self.check(self.m,row,f,-1.,5.);boxes=[]
        def enumerate_all(n,low,high):
            if 'feature' not in n:boxes.append((low,high,n));return
            j=n['feature'];left=list(high);left[j]=min(left[j],n['threshold']);right=list(low);right[j]=max(right[j],n['threshold']);enumerate_all(n['left'],low,left);enumerate_all(n['right'],right,high)
        enumerate_all(self.m['root'],[-math.inf]*2,[math.inf]*2)
        reachable=[(lo,hi,n) for lo,hi,n in boxes if all(lo[j]<row[j]<=hi[j] for j in range(2) if j!=f)]
        for s in r['segments']:
            matches=[n for lo,hi,n in reachable if lo[f]<s['witness']<=hi[f]];self.assertEqual(len(matches),1);self.assertEqual(s['prediction'],matches[0]['mean'])
        self.assertLess(len(reachable),len(boxes))
    def test_singleton_and_threshold_at_left_endpoint(self):
        m=tree.fit([[0],[1]],[2,8],1);t=m['root']['threshold'];r=self.check(m,[0],0,t,1.)
        self.assertEqual(len(r['segments']),2);self.assertEqual(r['segments'][0]['lower'],t);self.assertEqual(r['segments'][0]['upper'],t);self.assertEqual(r['segments'][0]['prediction'],2.);self.assertEqual(r['segments'][1]['witness'],math.nextafter(t,math.inf))
        r=self.check(m,[0],0,t,t);self.assertEqual(len(r['segments']),1);self.assertEqual(r['minimum']['prediction'],2.)
    def test_adjacent_float_domain_has_no_epsilon_gap(self):
        a=1.;b=math.nextafter(a,math.inf);m=tree.fit([[a],[b]],[0.,1.],1);self.assertEqual(m['root']['threshold'],a);r=self.check(m,[a],0,a,b);self.assertEqual([s['witness'] for s in r['segments']],[a,b]);self.assertEqual([s['prediction'] for s in r['segments']],[0.,1.])
    def test_constant_tree_and_axis_unused(self):
        self.assertEqual(len(self.check(tree.fit([[0],[1]],[2,2],1),[0],0,-5.,5.)['segments']),1)
        m=tree.fit([[0,1],[1,1]],[0,1],1);self.assertEqual(len(self.check(m,[0,1],1,0.,2.)['segments']),1)
    def test_ensemble_extrema_need_common_axis_not_sum_of_separate_extrema(self):
        # Two opposite stumps have separate maxima 10, but their average is 5 everywhere.
        e=bagging.fit([[0],[1]],[0,10],2,0,1);a=tree.fit([[0],[1]],[0,10],1);b=tree.fit([[0],[1]],[10,0],1)
        # Build valid deterministic membership metadata for these two fitted members.
        e['trees']=[{'draws':[0,1],'model':a},{'draws':[0,1],'model':b}];r=self.check(e,[0],0,0.,1.)
        self.assertEqual(r['minimum']['prediction'],5.);self.assertEqual(r['maximum']['prediction'],5.)
        self.assertEqual(np.mean([sweep.response(t,[0],0,0.,1.)['maximum']['prediction'] for t in [a,b]]),10.)
    def test_nested_repeated_axis_and_signed_zero(self):
        self.check(self.m,[-0.,1.],0,-0.,3.);r=self.check(self.m,[0.,1.],0,-0.,0.);self.assertEqual(len(r['segments']),1)
    def test_invalid_inputs_and_prefix_rules(self):
        for f in [True,-1,2,1.]:
            with self.assertRaises(ValueError):sweep.response(self.m,[1,1],f,0.,3.)
        for lo,hi in [(True,1),(0,float('nan')),(0,float('inf')),(1,0),(-1e101,1),('0',1)]:
            with self.assertRaises(ValueError):sweep.response(self.m,[1,1],0,lo,hi)
        for row in [[1],[1,1,1],[True,False],[float('nan'),0]]:
            with self.assertRaises(ValueError):sweep.response(self.m,row,0,0.,3.)
        with self.assertRaises(ValueError):sweep.response(self.m,[1,1],0,0.,3.,1)
        e=bagging.fit(self.x,self.y,2,0,1)
        for prefix in [True,0,3,1.]:
            with self.assertRaises(ValueError):sweep.response(e,[1,1],0,0.,3.,prefix)
        m=copy.deepcopy(self.m);m['root']['threshold']=float('inf')
        with self.assertRaises(ValueError):sweep.response(m,[1,1],0,0.,3.)
    def test_real_all_66_sweeps_and_1028_segments_reproduce_and_match_direct_boundaries(self):
        report=json.load(open(ROOT/'treelab/artifacts/sweep-report.json'));self.assertEqual(len(report['sweeps']),66);self.assertEqual(sum(len(r['segments']) for r in report['sweeps']),1028)
        models={'selected_cart':strict_load(ROOT/'treelab/artifacts/selected-model.json'),'bagging_32':strict_load(ROOT/'treelab/artifacts/bagging-model.json.gz')}
        for r in report['sweeps']:
            actual=self.check(models[r['model']],r['row'],r['feature'],*r['domain']);self.assertEqual(actual,{k:v for k,v in r.items() if k not in ['row_id','model','feature_name']})
    def test_full_report_and_fitting_only_domains_no_refit(self):
        x,y,names=data.load();s=data.split(x);original=sweep.response;seen=[]
        def record(*a):seen.append(np.array(a[1]));return original(*a)
        with tempfile.TemporaryDirectory() as d,mock.patch('treelab.tree.fit',side_effect=AssertionError('unexpected fit')),mock.patch('treelab.sweep.response',side_effect=record):
            actual=sweep.run(d);equal(self,actual,json.load(open(ROOT/'treelab/artifacts/sweep-report.json')))
        self.assertEqual(actual['source_rows'],[1,8,11]);np.testing.assert_array_equal(actual['fit_ranges'],np.c_[x[s['fit']].min(axis=0),x[s['fit']].max(axis=0)]);self.assertEqual(len(seen),66)
