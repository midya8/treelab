"""Original UCI data plus target-blind exact-feature-group splitting."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent/'data'
HASH='76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836'


def load():
    p=ROOT/'winequality-white.csv'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=HASH:raise ValueError('source hash mismatch')
    with p.open() as f:
        rows=list(csv.reader(f,delimiter=';'))
    names=rows[0][:-1];a=np.array(rows[1:],dtype=float)
    if a.shape!=(4898,12) or not np.isfinite(a).all():raise ValueError('invalid source shape/values')
    return a[:,:-1],a[:,-1],names


def group_key(row):
    # Canonical numeric representations identify equal parsed feature vectors.
    return hashlib.sha256(json.dumps([float(v).hex() for v in row],separators=(',',':')).encode()).hexdigest()


def split(x):
    keys=[group_key(row) for row in x];unique=sorted(set(keys));n=len(unique);mapping={k:('fit' if i<int(.6*n) else 'validation' if i<int(.8*n) else 'test') for i,k in enumerate(unique)}
    return {name:np.array([i for i,k in enumerate(keys) if mapping[k]==name],dtype=np.int64) for name in ['fit','validation','test']}
