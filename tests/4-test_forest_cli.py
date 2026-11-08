import hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ForestCLI(unittest.TestCase):
    def cli(self,*a):return subprocess.run([sys.executable,'-m','treelab',*map(str,a)],cwd=ROOT,capture_output=True,text=True)
    def test_all_three_cli_modes_strict_requests_and_optional_prefix(self):
        r=self.cli('forest-demo');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,(ROOT/'treelab/artifacts/forest-search-report.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            r=self.cli('forest-audit','--output',d);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(Path(d,'forest-search-report.json').read_text(),r.stdout)
            r=self.cli('forest-search','--model',ROOT/'treelab/artifacts/bagging-model.json.gz','--input',ROOT/'examples/forest-search.json');self.assertEqual(r.returncode,0,r.stderr);obj=json.loads(r.stdout);self.assertEqual(obj['members'],32);self.assertEqual(obj['status'],'budget_exhausted');self.assertIsNotNone(obj['solution'])
            r=self.cli('forest-search','--model',ROOT/'treelab/artifacts/bagging-model.json.gz','--input',ROOT/'examples/forest-search.json','--trees',1);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['members'],1)
            p=Path(d,'bad.json');raw=json.loads((ROOT/'examples/forest-search.json').read_text());raw['typo']=1;p.write_text(json.dumps(raw));r=self.cli('forest-search','--model',ROOT/'treelab/artifacts/bagging-model.json.gz','--input',p);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
    def test_full_browser_data_and_three_provenance_hashes(self):
        spec=importlib.util.spec_from_file_location('export_forest',ROOT/'examples/export_forest.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=(ROOT/'docs/forest-data.js').read_text();self.assertEqual(s,m.export());w=json.loads(s.removeprefix('globalThis.TREELAB_FOREST=').removesuffix(';\n'));self.assertEqual(w['report'],json.loads((ROOT/'treelab/artifacts/forest-search-report.json').read_text()))
        for k in ['report','model','warm_start_report']:self.assertEqual(w[k+'_sha256'],hashlib.sha256((ROOT/w[k+'_path']).read_bytes()).hexdigest())
        self.assertNotIn('fetch(', (ROOT/'docs/forest.js').read_text());self.assertIn('Budget exhaustion is not infeasibility.',(ROOT/'docs/forest.html').read_text())
