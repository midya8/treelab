"""Bundle the complete frozen report, with independently checkable provenance."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def export():
    report=ROOT/'treelab/artifacts/counterfactual-report.json';model=ROOT/'treelab/artifacts/selected-model.json'
    wrapper={'report':json.loads(report.read_bytes()),'source_path':'treelab/artifacts/counterfactual-report.json','source_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'model_path':'treelab/artifacts/selected-model.json','model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
    return 'globalThis.TREELAB_COUNTERFACTUAL='+json.dumps(wrapper,sort_keys=True,separators=(',',':'),allow_nan=False)+';\n'
if __name__=='__main__':(ROOT/'docs/counterfactual-data.js').write_text(export())
