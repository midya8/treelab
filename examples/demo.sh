#!/usr/bin/env bash
set -euo pipefail
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python -m treelab demo > "$work/saved.json"
cmp "$work/saved.json" treelab/artifacts/report.json
python -m treelab audit --output "$work/refit" > "$work/actual.json"
python - "$work" <<'PY'
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,'tests')
from test_experiment import equal
root=Path(sys.argv[1])
for name in ['report','grown-model','selected-model']:
    equal(unittest.TestCase(),json.load(open('treelab/artifacts/'+name+'.json')),json.load(open(root/'refit'/(name+'.json'))))
PY
python -m treelab predict --model treelab/artifacts/selected-model.json --input examples/input.json > "$work/predictions.json"
python -m treelab fit --input examples/train.json --output "$work/tree.json" --min-leaf 1 > "$work/fit.json"
python -m treelab prune --model "$work/tree.json" --alpha 1 --output "$work/pruned.json" > "$work/pruned-summary.json"
python examples/render_audit.py
git diff --exit-code assets/tree-audit.svg 2>/dev/null || test ! -d .git
printf '%s\n' 'Full real-data refit, validation-only pruning, five CLI modes and report reproduction: OK'

python -m treelab bag-demo > "$work/bagging-saved.json"
cmp "$work/bagging-saved.json" treelab/artifacts/bagging-report.json
python -m treelab bag-audit --output "$work/bagging-refit" > "$work/bagging-report.json"
python - "$work" <<'PY2'
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,'tests')
from test_experiment import equal
from test_bagging import load
for name in ['bagging-model.json.gz','bagging-report.json']:
    equal(unittest.TestCase(),load(Path('treelab/artifacts')/name),load(Path(sys.argv[1])/'bagging-refit'/name))
PY2
python examples/render_bagging.py
git diff --exit-code assets/bagging-audit.svg 2>/dev/null || test ! -d .git
printf '%s\n' 'Group-bootstrap full refit, OOB support and bagging artifacts: OK'
