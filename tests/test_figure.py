import importlib.util,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class Figure(unittest.TestCase):
    def test_full_metadata_regeneration_metrics_and_every_path_point(self):
        spec=importlib.util.spec_from_file_location('render',ROOT/'examples/render_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);r=json.loads((ROOT/'treelab/artifacts/report.json').read_text());svg=(ROOT/'assets/tree-audit.svg').read_text();self.assertEqual(m.render(r),svg);root=ET.fromstring(svg);ns={'s':'http://www.w3.org/2000/svg'};self.assertEqual(json.loads(root.find('s:metadata',ns).text),r)
        for g in root.findall('.//s:g[@data-model]',ns):
            metrics=r['test_metrics'][g.attrib['data-model']];self.assertEqual(float(g.attrib['data-rmse']),metrics['rmse']);self.assertEqual(float(g.attrib['data-mae']),metrics['mae']);texts=[v.text for v in g.findall('s:text',ns)];self.assertIn(f"{metrics['rmse']:.5f}",texts);self.assertIn(f"{metrics['mae']:.5f}",texts)
        import math
        ymax=math.ceil(max(max(p['fit_mse'],p['validation_mse']) for p in r['path'])*10)/10
        for line in root.findall('.//s:polyline',ns):
            key=line.attrib['data-series'];points=[[float(v) for v in p.split(',')] for p in line.attrib['points'].split()];self.assertEqual(len(points),290)
            for (x,y),p in zip(points,reversed(r['path'])):self.assertAlmostEqual(x,120+1130*math.log2(p['leaves'])/math.log2(r['grown_leaves']),places=5);self.assertAlmostEqual(y,630+360*(1-p[key]/ymax),places=5)
        text=' '.join(t.text or '' for t in root.findall('.//s:text',ns))
        for step in r['selected_examples'][0]['steps']:self.assertIn('Node '+str(step['node'])+':',text)
        self.assertIn('12 coefficients',text);self.assertIn('CC BY 4.0',text)
