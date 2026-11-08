import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class SweepCLI(unittest.TestCase):
    def runcli(self,*args,ok=True):
        r=subprocess.run([sys.executable,'-m','treelab',*map(str,args)],capture_output=True,text=True,cwd=ROOT);self.assertEqual(r.returncode,0 if ok else 2,r.stderr);return json.loads(r.stdout) if ok else r.stderr
    def test_three_modes_tree_gzip_ensemble_and_prefix(self):
        self.assertEqual(len(self.runcli('sweep-demo')['sweeps']),66)
        with tempfile.TemporaryDirectory() as d:
            r=self.runcli('sweep-audit','--output',d);self.assertEqual(len(r['sweeps']),66);self.assertEqual(r,json.loads((Path(d)/'sweep-report.json').read_text()))
        self.assertEqual(self.runcli('sweep','--model','treelab/artifacts/selected-model.json','--input','examples/sweep.json')['members'],1)
        self.assertEqual(self.runcli('sweep','--model','treelab/artifacts/bagging-model.json.gz','--input','examples/sweep.json')['members'],32)
        self.assertEqual(self.runcli('sweep','--model','treelab/artifacts/bagging-model.json.gz','--input','examples/sweep.json','--trees',4)['members'],4)
    def test_strict_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json'
            for body in ['{"row":[1],"row":[2],"feature":0,"low":0,"high":1}','{"row":[NaN],"feature":0,"low":0,"high":1}',json.dumps({'row':[1]*11,'feature':True,'low':0,'high':1}),json.dumps({'row':[1]*11,'feature':0,'low':2,'high':1}),json.dumps({'row':[1]*11,'feature':0,'low':0,'high':1,'extra':2})]:
                path.write_text(body);self.runcli('sweep','--model','treelab/artifacts/selected-model.json','--input',path,ok=False)
        self.runcli('sweep','--model','treelab/artifacts/selected-model.json','--input','examples/sweep.json','--trees',1,ok=False)
