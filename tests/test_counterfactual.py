import copy,itertools,json,math,tempfile,unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock
import numpy as np
from treelab import counterfactual as cf,data,tree
from treelab.__main__ import strict_load
ROOT=Path(__file__).resolve().parents[1]

def route(m,x):
    n=m['root']
    while 'feature' in n:n=n['left'] if x[n['feature']]<=n['threshold'] else n['right']
    return n

def cost(x,z,w):return sum((Fraction(a)*abs(Fraction(b)-Fraction(c)) for a,b,c in zip(w,x,z)),Fraction())
def rational(c):return Fraction(int(c['numerator']),int(c['denominator']))

class Counterfactual(unittest.TestCase):
    def setUp(self):self.m=tree.fit([[0,0],[0,1],[1,0],[1,1]],[0,1,2,3],1)
    def check(self,m,row,target,bounds,weights=None,immutable=None):
        r=cf.search(m,row,target,bounds,weights,immutable);w=r['weights']
        costs=[]
        for c in r['candidates']:
            z=c['input'];self.assertEqual(route(m,z)['id'],c['leaf']);self.assertEqual(route(m,z)['mean'],c['prediction']);self.assertTrue(target[0]<=c['prediction']<=target[1])
            for i,v in enumerate(z):
                self.assertTrue(bounds[i][0]<=v<=bounds[i][1]);self.assertTrue(c['box'][i][0]<=v<=c['box'][i][1])
                if i in (immutable or []):self.assertEqual(v,row[i])
            self.assertEqual(rational(c['cost_exact']),cost(row,z,w));self.assertEqual(c['cost'],float(cost(row,z,w)));costs.append((rational(c['cost_exact']),c['leaf']))
        self.assertEqual(costs,sorted(costs));self.assertEqual(r['solution'],r['candidates'][0] if costs else None);return r
    def test_multifeature_and_immutability(self):
        r=self.check(self.m,[0,0],[3,3],[[0,1],[0,1]],[1,2]);self.assertEqual(r['solution']['changed_features'],[0,1])
        self.assertEqual(self.check(self.m,[0,0],[3,3],[[0,1],[0,1]],immutable=[0])['status'],'infeasible')
    def test_exact_threshold_and_adjacent_float(self):
        a=1.;b=math.nextafter(a,math.inf);m=tree.fit([[a],[b]],[0,1],1)
        r=self.check(m,[a],[1,1],[[a,b]]);self.assertEqual(r['solution']['input'],[b]);self.assertEqual(rational(r['solution']['cost_exact']),Fraction(b)-Fraction(a))
        self.assertEqual(self.check(m,[a],[1,1],[[a,a]])['status'],'infeasible')
        self.assertEqual(self.check(m,[b],[0,0],[[a,b]])['solution']['input'],[a])
    def test_source_feasible_zero_and_source_outside_bounds(self):
        self.assertEqual(self.check(self.m,[0,0],[0,3],[[0,1],[0,1]])['solution']['cost'],0.)
        self.assertGreater(self.check(self.m,[-2,0],[0,3],[[0,1],[0,1]])['solution']['cost'],0.)
        self.assertEqual(self.check(self.m,[-2,0],[0,3],[[0,1],[0,1]],immutable=[0])['status'],'infeasible')
    def test_directional_bounds(self):
        m=tree.fit([[0],[1]],[0,1],1)
        self.assertEqual(self.check(m,[1],[0,0],[[1,2]])['status'],'infeasible')
        self.assertEqual(self.check(m,[1],[0,0],[[0,1]])['solution']['input'],[.5])
    def test_no_target_leaf_and_singleton_domains(self):
        r=self.check(self.m,[0,0],[4,5],[[0,1],[0,1]]);self.assertEqual(r['target_leaves'],0);self.assertIsNone(r['solution'])
        self.assertEqual(self.check(self.m,[0,0],[3,3],[[1,1],[1,1]])['solution']['input'],[1,1])
    def test_constant_tree_and_equal_cost_id_tie(self):
        m=tree.fit([[0],[1]],[2,2],1);r=self.check(m,[0],[2,2],[[-1,1]]);self.assertEqual(r['leaf_count'],1);self.assertEqual(r['solution']['cost'],0)
        # All-target source on an exact split: the source's zero-cost leaf wins.
        r=self.check(self.m,[.5,.5],[0,3],[[0,1],[0,1]]);self.assertEqual(r['solution']['cost_exact'],{'numerator':'0','denominator':'1'})
    def test_random_small_grid_independent_exhaustive_oracle(self):
        for seed in range(12):
            rng=np.random.default_rng(seed);x=rng.integers(-2,3,(12,2));y=rng.integers(0,4,12);m=tree.fit(x,y,1,4);row=rng.uniform(-2,2,2).tolist();weights=[.3,1.7];bounds=[[-2.,2.],[-1.5,1.5]];target=[1.,2.]
            # Every 1D projection can only be a bound, source coordinate,
            # threshold, or its immediate representable successor. Build this
            # grid without using the production leaf-box enumeration.
            axes=[{a,b} for a,b in bounds]
            for i in range(2):
                if bounds[i][0]<=row[i]<=bounds[i][1]:axes[i].add(row[i])
            def add(n):
                if 'feature' in n:
                    i=n['feature'];t=n['threshold']
                    for v in [t,math.nextafter(t,math.inf)]:
                        if bounds[i][0]<=v<=bounds[i][1]:axes[i].add(v)
                    add(n['left']);add(n['right'])
            add(m['root']);oracle=[]
            for z in itertools.product(*axes):
                n=route(m,z)
                if target[0]<=n['mean']<=target[1]:oracle.append((cost(row,z,weights),n['id']))
            r=self.check(m,row,target,bounds,weights)
            self.assertEqual(r['status'],'optimal' if oracle else 'infeasible')
            if oracle:self.assertEqual((rational(r['solution']['cost_exact']),r['solution']['leaf']),min(oracle))
    def test_exact_cost_tie_breaks_by_leaf_id(self):
        m=tree.fit([[-1],[0],[1]],[1,0,1],1)
        r=self.check(m,[2**-54],[1,1],[[-1,1]])
        a,b=r['candidates'];self.assertEqual(a['cost_exact'],b['cost_exact']);self.assertLess(a['leaf'],b['leaf'])
    def test_rational_cost_keeps_subnormal_change(self):
        b=math.nextafter(0.,math.inf);m=tree.fit([[0],[b]],[0,1],1)
        r=self.check(m,[0],[1,1],[[0,b]],[b]);self.assertEqual(r['solution']['cost'],0.)
        self.assertGreater(rational(r['solution']['cost_exact']),0)
    def test_rational_cost_rejects_rounded_false_tie(self):
        # Exact costs differ below a unit in the last place of their sum.
        # Synthetic valid tree: nearest target leaves sit behind y<0 and
        # y>small, both needing x >=1 via <= split semantics.
        tiny=2**-55
        x=[[0,-tiny],[1,-tiny],[0,0],[1,0],[0,tiny],[1,tiny]]
        m=tree.fit(x,[0,1,0,0,0,1],1,5)
        r=self.check(m,[0,tiny/4],[1,1],[[0,1],[-tiny,tiny]])
        cs=r['candidates'];self.assertGreaterEqual(len(cs),2)
        self.assertEqual(cs[0]['cost'],cs[1]['cost']);self.assertLess(rational(cs[0]['cost_exact']),rational(cs[1]['cost_exact']))
    def test_nonrepresentable_integer_model_scalars_rejected(self):
        m=tree.fit([[0],[1]],[0,0],1);m['root']['mean']=2**53+1
        with self.assertRaisesRegex(ValueError,'exactly representable'):cf.search(m,[0],[2**53,2**53],[[0,1]])
        m=tree.fit([[0],[1]],[0,1],1);m['root']['threshold']=2**53+1
        with self.assertRaisesRegex(ValueError,'exactly representable'):cf.search(m,[0],[0,1],[[0,1]])
    def test_validation_and_model_not_mutated(self):
        before=copy.deepcopy(self.m);cf.search(self.m,[0,0],[0,3],[[0,1],[0,1]]);self.assertEqual(before,self.m)
        base=dict(model=self.m,row=[0,0],target=[0,3],bounds=[[0,1],[0,1]])
        bad={'row':[[0],[True,False],[0,float('nan')]],'target':[[3,0],[0],[0,float('inf')],[True,False]],'bounds':[[[0,1]],[[1,0],[0,1]],[[0,1,2],[0,1,2]]],'weights':[[0,1],[-1,1],[1],[True,True]],'immutable':[[True],[2],[-1],[0,0],{0}]}
        for k,values in bad.items():
            for v in values:
                with self.subTest(k=k,v=v),self.assertRaises((ValueError,TypeError)):cf.search(**(base|{k:v}))
        with self.assertRaises(ValueError):cf.search({'format':'treelab-group-bagging-v1'},[0,0],[0,1],[[0,1],[0,1]])
    def test_frozen_real_report_reproduces_without_fit(self):
        with tempfile.TemporaryDirectory() as out,mock.patch('treelab.tree.fit',side_effect=AssertionError('no fit')):
            r=cf.run(out);self.assertEqual(r,strict_load(ROOT/'treelab/artifacts/counterfactual-report.json'))
        self.assertEqual(len(r['cases']),36);self.assertEqual(r['source_rows'],[1,8,11]);x,y,names=data.load();s=data.split(x);bounds=np.c_[x[s['fit']].min(axis=0),x[s['fit']].max(axis=0)].tolist();m=strict_load(ROOT/'treelab/artifacts/selected-model.json')
        for c in r['cases']:
            self.assertEqual(c['bounds'],bounds);self.assertEqual(c['row'],x[c['row_id']].tolist());self.check(m,c['row'],c['target'],bounds,c['weights'],c['immutable'])
