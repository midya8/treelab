import hashlib,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ExplorerExport(unittest.TestCase):
    def test_generated_browser_report_is_exact_complete_frozen_source(self):
        spec=importlib.util.spec_from_file_location('export_explorer',ROOT/'examples/export_explorer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=(ROOT/'docs/data.js').read_text();self.assertEqual(s,m.export());wrapper=json.loads(s.removeprefix('globalThis.TREELAB_DATA=').removesuffix(';\n'));raw=(ROOT/'treelab/artifacts/sweep-report.json').read_bytes();self.assertEqual(wrapper['report'],json.loads(raw));self.assertEqual(wrapper['source_sha256'],hashlib.sha256(raw).hexdigest());self.assertEqual(wrapper['source_path'],'treelab/artifacts/sweep-report.json')
    def test_static_ui_source_boundaries(self):
        html=(ROOT/'docs/index.html').read_text();self.assertIn('not uncertainty intervals',html);self.assertIn('no refit or new inference',html);self.assertIn('CC BY 4.0',html);self.assertIn('No accounts, analytics, APIs or data uploads',html)
        for p in ['app.js','geometry.js']:self.assertNotIn('fetch(', (ROOT/'docs'/p).read_text())
