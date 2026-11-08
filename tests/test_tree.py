import copy,itertools,json,unittest
from fractions import Fraction as F
import numpy as np
from treelab import tree


def oracle_split(x,y,min_leaf):
    candidates=[]
    for j in range(x.shape[1]):
        values=sorted(set(x[:,j]))
        for a,b in zip(values,values[1:]):
            t=float(a+(b-a)/2)
            if t>=b:t=float(a)
            left=x[:,j]<=t
            if min(left.sum(),(~left).sum())<min_leaf:continue
            risk=sum(float(np.sum((v-v.mean())**2)) for v in [y[left],y[~left]])
            candidates.append((risk,j,t))
    return min(candidates) if candidates else None


def exact_subtrees(node,x,y,rows):
    values=[F(float(v)) for v in y[rows]];mean=sum(values)/len(values);risk=sum((v-mean)**2 for v in values)
    result=[(risk,1)]
    if 'left' in node:
        mask=x[rows,node['feature']]<=node['threshold']
        result += [(a+b,la+lb) for a,la in exact_subtrees(node['left'],x,y,rows[mask]) for b,lb in exact_subtrees(node['right'],x,y,rows[~mask])]
    return result

class Trees(unittest.TestCase):
    def test_40_split_references_direct_sse_and_all_thresholds(self):
        for seed in range(40):
            rng=np.random.default_rng(seed);x=rng.integers(0,7,(17,3)).astype(float);y=rng.normal(size=17);m=1+seed%4
            actual=tree.best_split(x,y,m);expected=oracle_split(x,y,m);self.assertAlmostEqual(actual[0],expected[0],places=11);self.assertEqual(actual[1:],expected[1:])
    def test_split_is_stable_under_large_target_translation(self):
        x=np.arange(10.)[:,None];y=np.array([1,1,1,2,2,6,6,7,7,7.]);a=tree.best_split(x,y,2);b=tree.best_split(x,y+2**40,2);self.assertEqual(a,b)
    def test_equal_split_losses_choose_feature_then_threshold(self):
        x=np.c_[np.arange(4.),np.arange(4.)];y=np.array([0,1,0,1.]);s=tree.best_split(x,y,1);self.assertEqual(s[1:],(0,.5))
    def test_adjacent_binary64_values_do_not_empty_right_child(self):
        a=1.;b=np.nextafter(a,2.);x=np.array([[a],[b]]);s=tree.best_split(x,np.array([0.,1.]),1);self.assertEqual(s[2],a);self.assertEqual(tree.predict(tree.fit(x,[0,1],1),x).tolist(),[0,1])
    def test_constant_feature_or_target_stays_leaf(self):
        for x,y in [(np.ones((6,2)),np.arange(6.)),(np.arange(6.)[:,None],np.ones(6))]:self.assertNotIn('left',tree.fit(x,y,1)['root'])
    def test_min_leaf_and_depth_are_enforced(self):
        x=np.arange(30.)[:,None];y=np.arange(30.)%7;m=tree.fit(x,y,4,2)
        def walk(n,d):
            self.assertLessEqual(d,2);self.assertGreaterEqual(n['n'],4)
            if 'left' in n:walk(n['left'],d+1);walk(n['right'],d+1)
        walk(m['root'],0)
    def test_fit_input_is_not_mutated(self):
        x=np.arange(12.).reshape(6,2);y=np.arange(6.);a=x.copy();b=y.copy();tree.fit(x,y,1);np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
    def test_midpoint_left_inclusive_and_paths_reproduce_prediction(self):
        m=tree.fit([[0],[1],[2],[3]],[0,0,2,2],1);pred,traces=tree.predict(m,[[1.5],[1.500001]],True);self.assertEqual(pred.tolist(),[0,2]);self.assertEqual(traces[0]['steps'][0]['direction'],'left');self.assertEqual(traces[1]['steps'][0]['direction'],'right')
    def test_saved_node_sse_matches_independent_training_routing(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(60,4));y=rng.integers(0,8,60);m=tree.fit(x,y,3)
        def check(n,rows):
            self.assertEqual(n['n'],len(rows));self.assertAlmostEqual(n['mean'],float(y[rows].mean()));self.assertAlmostEqual(n['sse'],float(np.sum((y[rows]-y[rows].mean())**2)),places=10)
            if 'left' in n:
                mask=x[rows,n['feature']]<=n['threshold'];check(n['left'],rows[mask]);check(n['right'],rows[~mask])
        check(m['root'],np.arange(len(y)))
    def test_effective_alpha_definition_and_no_mutation(self):
        m=tree.fit(np.arange(8.)[:,None],[0,0,2,2,1,1,3,3],1);before=json.dumps(m);path=tree.pruning_path(m);self.assertEqual(json.dumps(m),before)
        for state in path:
            for c in tree.effective_alphas(state['model']):self.assertAlmostEqual(c['alpha'],(c['leaf_risk']-c['subtree_risk'])/(c['subtree_leaves']-1),places=14)
    def test_pruning_risk_increases_leaves_decrease_to_root(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(100,3));y=rng.normal(size=100);path=tree.pruning_path(tree.fit(x,y,2))
        self.assertEqual(path[-1]['leaves'],1)
        for a,b in zip(path,path[1:]):self.assertLess(b['leaves'],a['leaves']);self.assertGreaterEqual(b['alpha']+1e-14,a['alpha']);self.assertGreaterEqual(b['risk']+1e-14,a['risk'])
    def test_simultaneous_disjoint_weakest_links(self):
        x=np.arange(8.)[:,None];y=np.array([0,0,1,1,10,10,11,11.]);path=tree.pruning_path(tree.fit(x,y,1));self.assertEqual([s['leaves'] for s in path],[4,2,1]);self.assertEqual(len(path[1]['pruned']),2)
    def test_all_small_tree_subtrees_exact_fraction_objective_oracle(self):
        checked=0
        for seed in range(12):
            rng=np.random.default_rng(seed);x=rng.integers(0,10,(10,2)).astype(float);y=rng.integers(-3,6,10).astype(float);model=tree.fit(x,y,1,3);path=tree.pruning_path(model);choices=exact_subtrees(model['root'],x,y,np.arange(len(y)))
            alphas=sorted(set([0.]+[p['alpha'] for p in path]));probes=alphas+[a+(b-a)/2 for a,b in zip(alphas,alphas[1:])]+[alphas[-1]+1]
            for alpha in probes:
                exact=min(r/F(len(y))+F(alpha)*leaves for r,leaves in choices)
                actual=min(p['risk']+alpha*p['leaves'] for p in path);self.assertAlmostEqual(actual,float(exact),places=11);pruned=tree.prune(model,alpha);risk,leaves=tree.subtree(pruned['root']);self.assertAlmostEqual(risk/len(y)+alpha*leaves,float(exact),places=11);checked+=1
        self.assertGreater(checked,60)
    def test_pruned_node_retains_fit_mean_and_count(self):
        m=tree.fit([[0],[1],[2],[3]],[1,2,7,8],1);root=tree.pruning_path(m)[-1]['model']['root'];self.assertEqual(root['mean'],4.5);self.assertEqual(root['n'],4);self.assertEqual(set(root),{'id','n','mean','sse'})
    def test_selection_uses_validation_mse_and_tie_prefers_small_tree(self):
        m=tree.fit([[0],[1],[2],[3]],[0,0,2,2],1);path=tree.pruning_path(m);idx,scores=tree.select(path,[[0],[3]],[1,1]);self.assertEqual(path[idx]['leaves'],1);self.assertEqual(scores,[1,0])
        idx,_=tree.select(path,[[0],[3]],[.5,1.5]);self.assertEqual(path[idx]['leaves'],1)
    def test_json_round_trip_and_predictions(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(20,2));m=tree.fit(x,rng.normal(size=20),2);np.testing.assert_array_equal(tree.predict(m,x),tree.predict(json.loads(json.dumps(m)),x))
    def test_invalid_training_values_and_shapes(self):
        for x,y in [([[True]],[1]),([[np.nan]],[1]),([[np.inf]],[1]),([[1e101]],[1]),([1,2],[1,2]),([[1]],[[1]]),([[1],[2]],[1]),([[]],[1]),([],[])]:
            with self.assertRaises(ValueError):tree.fit(x,y)
        for kw in [{'min_leaf':True},{'min_leaf':0},{'max_depth':31},{'max_depth':-1},{'max_depth':1.5}]:
            with self.assertRaises(ValueError):tree.fit([[0],[1]],[0,1],**kw)
    def test_invalid_prediction_width_or_nonfinite(self):
        m=tree.fit([[0],[1]],[0,1],1)
        for x in [[[0,1]],[[np.nan]],[[True]],[],[1,2]]:
            with self.assertRaises(ValueError):tree.predict(m,x)
    def test_model_rejects_corruption_fields_ids_counts_and_finiteness(self):
        m=tree.fit([[0],[1],[2],[3]],[0,0,2,2],1)
        for kind in range(9):
            q=copy.deepcopy(m)
            if kind==0:q['root']['sse']=-1
            if kind==1:q['root']['mean']=float('nan')
            if kind==2:q['root']['left']['id']=q['root']['id']
            if kind==3:q['root']['n']=8
            if kind==4:q['root']['feature']=1
            if kind==5:q['root']['threshold']=float('inf')
            if kind==6:q['root']['extra']=1
            if kind==7:q['features']=True
            if kind==8:q['root']['n']=True
            with self.assertRaises(ValueError):tree.validate(q)
    def test_prune_at_breakpoints_prefers_smaller_tied_tree(self):
        m=tree.fit(np.arange(8.)[:,None],[0,0,1,1,10,10,11,11],1);path=tree.pruning_path(m)
        for state in path:
            self.assertEqual(tree.subtree(tree.prune(m,state['alpha'])['root'])[1],state['leaves'])
        self.assertEqual(tree.subtree(tree.prune(m,1e10)['root'])[1],1)
        for alpha in [-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):tree.prune(m,alpha)
    def test_target_translation_and_scale_change_means_and_alpha_units(self):
        x=np.arange(8.)[:,None];y=np.array([0,0,1,1,10,10,11,11.]);a=tree.pruning_path(tree.fit(x,y,1));b=tree.pruning_path(tree.fit(x,2*y+7,1));self.assertEqual([p['leaves'] for p in a],[p['leaves'] for p in b])
        for p,q in zip(a,b):self.assertAlmostEqual(q['alpha'],4*p['alpha']);np.testing.assert_allclose(tree.predict(q['model'],x),2*tree.predict(p['model'],x)+7,atol=1e-12,rtol=0)
    def test_model_rejects_impossible_mean_or_impurity(self):
        m=tree.fit([[0],[1],[2],[3]],[0,0,2,2],1)
        for k,v in [('mean',10),('sse',0)]:
            q=copy.deepcopy(m);q['root'][k]=v
            with self.assertRaises(ValueError):tree.validate(q)
