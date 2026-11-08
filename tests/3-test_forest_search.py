import copy,itertools,math,unittest
from fractions import Fraction
import numpy as np
from treelab import bagging,forest_search as fs,tree
from test_counterfactual import route,cost,rational


def ensemble(models):
    # Valid bootstrap metadata for two unique input groups; fitted trees
    # preserve two rows and weighted target summaries independently.
    e=bagging.fit([[0],[1]],[0,1],len(models),0,1)
    e['trees']=[{'draws':[0,1],'model':m} for m in models];return e

class ForestSearch(unittest.TestCase):
    def oracle(self,models,row,target,bounds,weights,immutable=()):
        axes=[{a,b} for a,b in bounds]
        for i in range(len(row)):
            if bounds[i][0]<=row[i]<=bounds[i][1]:axes[i].add(row[i])
        def add(n):
            if 'feature' in n:
                i=n['feature'];t=n['threshold']
                for v in [t,math.nextafter(t,math.inf)]:
                    if bounds[i][0]<=v<=bounds[i][1]:axes[i].add(v)
                add(n['left']);add(n['right'])
        for m in models:add(m['root'])
        result=[]
        for z in itertools.product(*axes):
            if any(z[i]!=row[i] for i in immutable):continue
            pred=float(np.mean([float(route(m,z)['mean']) for m in models]))
            if target[0]<=pred<=target[1]:result.append((cost(row,z,weights),z))
        return min(result) if result else None
    def check(self,r,models,optimum):
        if r['solution']:
            s=r['solution'];x=s['input'];self.assertEqual(s['member_leaf_ids'],[route(m,x)['id'] for m in models]);self.assertEqual(s['member_predictions'],[route(m,x)['mean'] for m in models]);self.assertEqual(s['prediction'],float(np.mean(s['member_predictions'])));self.assertTrue(r['target'][0]<=s['prediction']<=r['target'][1]);self.assertEqual(rational(s['cost_exact']),cost(r['row'],x,r['weights']));self.assertEqual(s['cost_exact'],r['objective_upper_exact'])
            for i,v in enumerate(x):
                self.assertTrue(r['bounds'][i][0]<=v<=r['bounds'][i][1])
                if i in r['immutable']:self.assertEqual(v,r['row'][i])
        if optimum:
            if r['objective_lower_exact']:self.assertLessEqual(rational(r['objective_lower_exact']),optimum[0])
            if r['objective_upper_exact']:self.assertGreaterEqual(rational(r['objective_upper_exact']),optimum[0])
            if r['status']=='optimal':self.assertEqual(rational(r['solution']['cost_exact']),optimum[0]);self.assertTrue(r['complete']);self.assertEqual(r['absolute_gap_exact'],{'numerator':'0','denominator':'1'})
            self.assertNotEqual(r['status'],'infeasible')
        else:self.assertIsNone(r['solution']);self.assertNotEqual(r['status'],'optimal')
        if r['status']=='budget_exhausted':self.assertFalse(r['complete']);self.assertGreater(r['frontier_states'],0)
        if r['status']=='infeasible':self.assertTrue(r['complete']);self.assertIsNone(r['objective_lower_exact'])
        self.assertLessEqual(r['expanded_states'],r['max_states']);self.assertLessEqual(r['peak_frontier_states'],r['expanded_states']+1)
    def test_random_forests_against_exhaustive_independent_grid_and_all_budget_bounds(self):
        for seed in range(8):
            rng=np.random.default_rng(seed);x=rng.integers(-2,3,(10,2));y=rng.integers(0,4,10);e=bagging.fit(x,y,3,seed,1,3);models=[t['model'] for t in e['trees']];row=rng.uniform(-2,2,2).tolist();bounds=[[-2,2],[-1.5,1.5]];weights=[.3,1.7]
            for target in [[1.,2.],[4.,5.]]:
                o=self.oracle(models,row,target,bounds,weights)
                for budget in [0,1,2,4,16,10000]:
                    r=fs.search(e,row,target,bounds,weights,max_states=budget);self.check(r,models,o)
                    if budget==10000:self.assertTrue(r['complete'])
    def test_separate_member_maxima_are_not_jointly_feasible(self):
        a=tree.fit([[0],[1]],[0,10],1);b=tree.fit([[0],[1]],[10,0],1);e=ensemble([a,b])
        r=fs.search(e,[0],[9,10],[[0,1]],max_states=100);self.check(r,[a,b],None);self.assertEqual(r['status'],'infeasible');self.assertGreater(r['pruned']['output'],0)
        r=fs.search(e,[0],[5,5],[[0,1]],max_states=0);self.assertEqual(r['status'],'optimal');self.assertEqual(r['solution']['cost'],0)
    def test_budget_exhaustion_no_witness_is_not_infeasibility(self):
        m=tree.fit([[0],[.25],[.5],[.75],[1]],[0,1,0,0,0],1);r=fs.search(m,[0],[1,1],[[0,1]],max_states=0);self.assertEqual(r['status'],'budget_exhausted');self.assertIsNone(r['solution']);self.assertEqual(rational(r['objective_lower_exact']),0)
    def test_budget_with_incumbent_retains_nonzero_gap(self):
        m=tree.fit([[0],[1]],[0,1],1);r=fs.search(m,[0],[1,1],[[0,1]],max_states=0)
        self.assertEqual(r['status'],'budget_exhausted');self.assertEqual(r['solution']['input'],[1]);self.assertEqual(rational(r['absolute_gap_exact']),1)
        done=fs.search(m,[0],[1,1],[[0,1]],max_states=100);self.assertLess(rational(done['objective_upper_exact']),rational(r['objective_upper_exact']))
    def test_midpoint_seed_underflow_never_escapes_singleton_bounds(self):
        z=math.nextafter(0.,math.inf);m=tree.fit([[0],[z]],[0,1],1)
        r=fs.search(m,[0],[0,1],[[z,z]],max_states=0)
        self.assertEqual(r['status'],'optimal');self.assertEqual(r['solution']['input'],[z])
    def test_single_tree_agrees_with_complete_leaf_enumeration(self):
        from treelab import counterfactual
        for seed in range(4):
            rng=np.random.default_rng(seed);x=rng.integers(-3,4,(20,3));y=rng.integers(0,5,20);m=tree.fit(x,y,1,4);row=rng.normal(size=3).tolist();bounds=[[-2,2]]*3
            a=counterfactual.search(m,row,[2,4],bounds);r=fs.search(m,row,[2,4],bounds);self.assertEqual(a['status'],r['status']);self.assertEqual(a['solution']['cost_exact'] if a['solution'] else None,r['objective_upper_exact'])
    def test_adjacent_float_and_subnormal_objective(self):
        z=math.nextafter(0.,math.inf);m=tree.fit([[0],[z]],[0,1],1);r=fs.search(m,[0],[1,1],[[0,z]],weights=[z]);self.assertEqual(r['status'],'optimal');self.assertEqual(r['solution']['input'],[z]);self.assertEqual(r['solution']['cost'],0);self.assertGreater(rational(r['objective_upper_exact']),0)
    def test_constraints_source_bounds_immutability_direction_and_prefix(self):
        a=tree.fit([[0],[1]],[0,10],1);b=tree.fit([[0],[1]],[10,0],1);e=ensemble([a,b])
        self.assertEqual(fs.search(e,[0],[9,10],[[0,1]],prefix=1)['status'],'optimal')
        self.assertEqual(fs.search(a,[0],[9,10],[[0,1]],immutable=[0])['status'],'infeasible')
        self.assertEqual(fs.search(a,[-1],[0,10],[[0,1]],immutable=[0])['status'],'infeasible')
        self.assertEqual(fs.search(a,[-1],[0,10],[[0,1]])['solution']['cost'],1)
        self.assertEqual(fs.search(a,[1],[0,0],[[1,2]])['status'],'infeasible')
    def test_mean_rounding_uses_same_numpy_inference_semantics(self):
        models=[tree.fit([[0],[1]],[v,v],1) for v in [1e16,1.,-1e16]];e=ensemble(models)
        value=float(np.mean([m['root']['mean'] for m in models]));r=fs.search(e,[0],[value,value],[[0,1]],max_states=0);self.assertEqual(r['solution']['prediction'],value);self.assertEqual(r['status'],'optimal')
    def test_determinism_no_mutation_and_validation(self):
        m=tree.fit([[0],[1]],[0,1],1);before=copy.deepcopy(m);base=dict(model=m,row=[0],target=[1,1],bounds=[[0,1]])
        a=fs.search(**base);self.assertEqual(a,fs.search(**base));self.assertEqual(m,before)
        for k,vals in {'row':[[1,2],[True],[float('nan')]],'target':[[2,1],[1],[False,True]],'bounds':[[[1,0]],[[0,1,2]]],'weights':[[0],[-1],[1,2]],'immutable':[[True],[1],[0,0]],'max_states':[-1,True,1.,100001],'prefix':[0,1]}.items():
            for v in vals:
                with self.subTest(k=k,v=v),self.assertRaises((ValueError,TypeError)):fs.search(**(base|{k:v}))
        bad=copy.deepcopy(m);bad['root']['threshold']=2**53+1
        with self.assertRaises(ValueError):fs.search(**(base|{'model':bad}))
    def test_optional_seed_inputs_are_verified_not_trusted(self):
        m=tree.fit([[0],[1]],[0,1],1)
        r=fs.search(m,[0],[1,1],[[0,1]],max_states=0,seed_inputs=[[.75]])
        self.assertEqual(r['status'],'budget_exhausted');self.assertEqual(r['solution']['input'],[.75]);self.assertEqual(r['seed_inputs'],[[.75]])
        for seeds in [[[2]],[[True]],[[0,1]],[[.7]]*129]:
            with self.assertRaises(ValueError):fs.search(m,[0],[1,1],[[0,1]],seed_inputs=seeds)
        with self.assertRaises(ValueError):fs.search(m,[0],[1,1],[[0,1]],immutable=[0],seed_inputs=[[1]])
    def test_frozen_real_report_reproduction_and_independent_one_axis_target_oracle(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from treelab import data
        from treelab.__main__ import strict_load
        root=Path(__file__).resolve().parents[1];saved=strict_load(root/'treelab/artifacts/forest-search-report.json')
        with tempfile.TemporaryDirectory() as out,mock.patch('treelab.tree.fit',side_effect=AssertionError('no fit')):self.assertEqual(fs.run(out),saved)
        sweep=strict_load(root/'treelab/artifacts/sweep-report.json');x,y,names=data.load();split=data.split(x);bounds=np.c_[x[split['fit']].min(0),x[split['fit']].max(0)].tolist();model=strict_load(root/'treelab/artifacts/bagging-model.json.gz');models=[t['model'] for t in model['trees']]
        self.assertEqual(len(saved['cases']),36)
        for r in saved['cases']:
            self.assertEqual(r['bounds'],bounds);self.assertEqual(r['row'],x[r['row_id']].tolist())
            axis=next(s for s in sweep['sweeps'] if s['row_id']==r['row_id'] and s['feature_name']=='alcohol' and s['model']=='bagging_32');candidates=[]
            for s in axis['segments']:
                if r['target'][0]<=s['prediction']<=r['target'][1]:
                    a=s['lower'] if s['lower_inclusive'] else math.nextafter(s['lower'],math.inf);z=r['row'].copy();z[10]=min(max(z[10],a),s['upper']);candidates.append((cost(z,r['row'],r['weights']),z))
            oracle=min(candidates) if candidates else None
            if r['policy']=='alcohol_only':self.check(r,models,oracle)
            else:
                if oracle:
                    self.assertIsNotNone(r['solution']);self.assertLessEqual(rational(r['objective_upper_exact']),oracle[0]);self.assertLessEqual(rational(r['objective_lower_exact']),oracle[0])
                if r['solution']:self.assertEqual(r['solution']['prediction'],float(np.mean([route(m,r['solution']['input'])['mean'] for m in models])))
