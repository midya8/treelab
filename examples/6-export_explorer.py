"""Export the exact frozen response report for a dependency-free static explorer."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def export():
    raw=(ROOT/'treelab/artifacts/sweep-report.json').read_bytes();report=json.loads(raw)
    wrapper={'source_path':'treelab/artifacts/sweep-report.json','source_sha256':hashlib.sha256(raw).hexdigest(),'report':report}
    return 'globalThis.TREELAB_DATA='+json.dumps(wrapper,sort_keys=True,separators=(',',':'),allow_nan=False)+';\n'

if __name__=='__main__':(ROOT/'docs/data.js').write_text(export())
