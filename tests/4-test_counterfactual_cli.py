import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class CounterfactualCLI(unittest.TestCase):
    def cli(self,*args):return subprocess.run([sys.executable,'-m','treelab',*map(str,args)],cwd=ROOT,capture_output=True,text=True)
    def test_three_modes_and_rejected_extra_fields(self):
        r=self.cli('counterfactual-demo');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),json.load(open(ROOT/'treelab/artifacts/counterfactual-report.json')))
        with tempfile.TemporaryDirectory() as d:
            r=self.cli('counterfactual-audit','--output',d);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(Path(d,'counterfactual-report.json').read_text(),r.stdout)
            inp=Path(d,'in.json');inp.write_text(json.dumps({'row':[0,0],'target':[3,3],'bounds':[[0,1],[0,1]]}))
            from treelab import tree
            model=Path(d,'m.json');model.write_text(json.dumps(tree.fit([[0,0],[1,1]],[0,3],1)))
            r=self.cli('counterfactual','--model',model,'--input',inp);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['status'],'optimal')
            inp.write_text('{"row":[0,0],"target":[3,3],"bounds":[[0,1],[0,1]],"typo":1}')
            r=self.cli('counterfactual','--model',model,'--input',inp);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
    def test_complete_browser_export_and_provenance(self):
        import hashlib,importlib.util
        spec=importlib.util.spec_from_file_location('export_cf',ROOT/'examples/export_counterfactual.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=(ROOT/'docs/counterfactual-data.js').read_text();self.assertEqual(s,m.export());w=json.loads(s.removeprefix('globalThis.TREELAB_COUNTERFACTUAL=').removesuffix(';\n'))
        for k,p in [('source','counterfactual-report.json'),('model','selected-model.json')]:self.assertEqual(w[k+'_sha256'],hashlib.sha256((ROOT/'treelab/artifacts'/p).read_bytes()).hexdigest())
        self.assertEqual(w['report'],json.load(open(ROOT/'treelab/artifacts/counterfactual-report.json')))
        self.assertNotIn('fetch(', (ROOT/'docs/counterfactual.js').read_text());self.assertIn('not advice', (ROOT/'docs/counterfactual.html').read_text())
