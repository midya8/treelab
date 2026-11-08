import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from treelab.__main__ import strict_load
ROOT=Path(__file__).resolve().parents[1]

class CLI(unittest.TestCase):
    def call(self,*args):return subprocess.run([sys.executable,'-m','treelab',*map(str,args)],cwd=ROOT,capture_output=True,text=True)
    def test_demo_exact_saved_report(self):
        r=self.call('demo');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,(ROOT/'treelab/artifacts/report.json').read_text())
    def test_predict_saved_model_full_paths(self):
        r=self.call('predict','--model',ROOT/'treelab/artifacts/selected-model.json','--input',ROOT/'examples/input.json');self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertEqual(len(v['predictions']),2);self.assertTrue(v['paths'][0]['steps']);self.assertEqual(v['predictions'][0],v['paths'][0]['prediction'])
    def test_fit_prune_predict_round_trip_outside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);f=tmp/'train.json';f.write_text(json.dumps({'features':[[0],[1],[2],[3]],'targets':[0,0,2,2]}));m=tmp/'m.json';q=tmp/'q.json'
            a=self.call('fit','--input',f,'--output',m,'--min-leaf',1);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout)['leaves'],2)
            a=self.call('prune','--model',m,'--alpha',1,'--output',q);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout)['leaves'],1)
            f.write_text('{"features":[[0],[3]]}');a=self.call('predict','--model',q,'--input',f);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout)['predictions'],[1,1])
    def test_reject_nonfinite_duplicate_and_extra_json_fields(self):
        for text in ['{"features":[[NaN]]}','{"features":[[0]],"features":[[1]]}','{"features":[[0]],"extra":1}']:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'input.json';p.write_text(text);r=self.call('predict','--model',ROOT/'treelab/artifacts/selected-model.json','--input',p);self.assertEqual(r.returncode,2);self.assertFalse(r.stdout)
    def test_invalid_model_and_missing_file_no_traceback(self):
        for path in ['missing.json',ROOT/'examples/input.json']:
            r=self.call('predict','--model',path,'--input',ROOT/'examples/input.json');self.assertEqual(r.returncode,2);self.assertNotIn('Traceback',r.stderr)
    def test_fit_rejects_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.json';p.write_text('{"features":[[1]],"targets":[1],"extra":1}');r=self.call('fit','--input',p,'--output',Path(tmp)/'model.json');self.assertEqual(r.returncode,2);self.assertFalse((Path(tmp)/'model.json').exists())
