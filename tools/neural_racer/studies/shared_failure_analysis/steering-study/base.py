"""Workspace-only imports. Run scripts from the pinned workspace root."""
from pathlib import Path
import sys, os
R=Path.cwd()
D=Path(__file__).resolve().parent
for p in (R/'.sim/python',R/'tools/neural_racer',R/'src/racerbot_car'):
    sys.path.insert(0,str(p))
from neural_racer_train import referee, tracks
from neural_racer import observation, model_store
import numpy as np
import json, hashlib
A=np.deg2rad(np.linspace(-135.,135.,109))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def encode(r):
    return np.log1p(np.clip(np.where((r>.05)&(r<25)&np.isfinite(r),r,10.),.05,10.)).astype('f')
def predict(w, scan):
    x=encode(np.asarray(scan)[::10])
    assert x.size==109
    def apply(x):
        for i in range(2):
            x=np.maximum(0,w[f'w{i}']@x+w[f'b{i}'])
        return float((w['w2']@x+w['b2']).item())
    # An exactly mirror-equivariant output from the same fitted weights.
    return float(np.clip(.5*(apply(x)-apply(x[::-1])),-.33,.33))
