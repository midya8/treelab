import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class BaggingCLI(unittest.TestCase):
    def cmd(self,*args,ok=True):
        r=subprocess.run([sys.executable,'-m','treelab',*map(str,args)],capture_output=True,text=True,cwd=ROOT)
        self.assertEqual(r.returncode,0 if ok else 2,r.stderr)
        return json.loads(r.stdout) if ok else r.stderr
    def test_five_bagging_modes_roundtrip(self):
        self.assertEqual(self.cmd('bag-demo')['selected_trees'],32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=p/'model.json';self.assertEqual(self.cmd('bag-fit','--input','examples/train.json','--output',m,'--min-leaf',1,'--trees',4)['trees'],4)
            raw=json.load(open(ROOT/'examples/train.json'));(p/'input.json').write_text(json.dumps({'features':raw['features']}));self.assertEqual(len(self.cmd('bag-predict','--model',m,'--input',p/'input.json')['predictions']),len(raw['features']))
            r=self.cmd('bag-oob','--model',m,'--input','examples/train.json');self.assertEqual(r['prefixes'][0]['trees'],4)
            r=self.cmd('bag-audit','--output',p/'audit');self.assertEqual(r['selected_trees'],32);self.assertTrue((p/'audit/bagging-model.json').is_file())
    def test_invalid_cli_data_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=p/'model.json';self.cmd('bag-fit','--input','examples/train.json','--output',m,'--trees',2)
            for s in ['{"features":[[0]],"features":[[1]]}','{"features":[[NaN]]}','{"features":[[true]]}','{"features":[[1]],"extra":1}']:
                (p/'input.json').write_text(s);self.cmd('bag-predict','--model',m,'--input',p/'input.json',ok=False)
            self.cmd('bag-predict','--model',m,'--input','examples/input.json','--trees',3,ok=False)
            self.cmd('bag-fit','--input','examples/train.json','--output',m,'--trees',0,ok=False)
            raw=json.load(open(ROOT/'examples/train.json'));raw['targets'][0]+=1;(p/'input.json').write_text(json.dumps(raw));self.cmd('bag-oob','--model',m,'--input',p/'input.json',ok=False)
