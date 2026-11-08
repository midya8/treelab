import importlib.util,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class SweepFigure(unittest.TestCase):
    def test_all_displayed_series_intervals_and_summary_values(self):
        spec=importlib.util.spec_from_file_location('render_sweep',ROOT/'examples/render_sweep.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);r=json.loads((ROOT/'treelab/artifacts/sweep-report.json').read_text());s=(ROOT/'assets/sweep-audit.svg').read_text();self.assertEqual(s,m.render(r));root=ET.fromstring(s);metadata=json.loads(root.find('{http://www.w3.org/2000/svg}metadata').text);selected=[a for a in r['sweeps'] if a['row_id']==r['source_rows'][0]];self.assertEqual(metadata['displayed_sweeps'],selected);self.assertEqual(metadata['protocol'],r['protocol']);ns='{http://www.w3.org/2000/svg}';lines=root.findall(ns+'g/'+ns+'polyline');self.assertEqual([int(a.attrib['data-segments']) for a in lines],[3,50]);self.assertEqual([len(a.attrib['points'].split()) for a in lines],[6,100]);text=' '.join(a.text or '' for a in root.iter(ns+'text'))
        for a in selected:self.assertIn(f"{a['minimum']['prediction']:.5f} / {a['maximum']['prediction']:.5f}",text)
        self.assertIn('not uncertainty bands',text);self.assertIn('no held-out accuracy claim',text)
