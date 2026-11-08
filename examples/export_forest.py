"""Bundle a complete frozen search report without running a browser optimizer."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def export():
    paths={'report':'forest-search-report.json','model':'bagging-model.json.gz','warm_start_report':'sweep-report.json'};out={}
    for key,name in paths.items():
        raw=(ROOT/'treelab/artifacts'/name).read_bytes();out[key+'_path']='treelab/artifacts/'+name;out[key+'_sha256']=hashlib.sha256(raw).hexdigest()
        if key=='report':out['report']=json.loads(raw)
    return 'globalThis.TREELAB_FOREST='+json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False)+';\n'
if __name__=='__main__':(ROOT/'docs/forest-data.js').write_text(export())
