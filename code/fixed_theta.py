"""No calibration: theta is fixed, so sparsity comes from the weights, not from a chosen percentile.

The designer's distinction, which the earlier code got wrong:

  "do not use top-k; guarantee sparse activation through the weight adjustment"

  - the weight rule                each cell fires when its score exceeds theta times ITS OWN weight
                                   budget (|bias| + sum |w|).  Cells differ in that budget, so under
                                   one fixed theta some fire easily and others do not.  That is
                                   sparsity coming from the weights.
  - what this file used to do       bisect theta until exactly 20 % of the cells fire.  Picking a
                                   fraction and then finding the cutoff for it IS a top-k in
                                   disguise, whatever the implementation looks like.

So here theta is FIXED and never tuned.  The activation ratio is whatever the weights produce, and it
is reported for every image rather than forced to a target.

Usage: python fixed_theta.py [train_n] [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC, MZ = 28, 10, 16

from torchvision.datasets import MNIST

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def prep(ds, n, seed):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, S * S), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        X[i] = (np.asarray(img, np.float32) / 255.0).reshape(-1)
        Y[i] = lab
    return X, Y


Xtr, ytr = prep(tr, N_TRAIN, 101)
Xte, yte = prep(te, N_TEST, 108)
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)

print('theta is FIXED -- no calibration, no target fraction', flush=True)
print('', flush=True)
print('theta | visual area active%% (mean / min / max over images)', flush=True)

for theta in (0.6, 0.9, 1.0, 1.2):
    vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
    vis.theta = [theta]                        # fixed, never calibrated
    with T.no_grad():
        A = vis.forward(Xt)[-1]
    per = A.mean(1)
    print('  %.1f | %.1f%% / %.1f%% / %.1f%%' % (
        theta, 100 * float(per.mean()), 100 * float(per.min()), 100 * float(per.max())), flush=True)

# with the activation ratio left free, how well can the patterns still be told apart?
print('', flush=True)
print('theta | held-out, nearest-neighbour on the patterns', flush=True)
for theta in (0.6, 0.9, 1.0, 1.2):
    vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
    vis.theta = [theta]
    with T.no_grad():
        Atr = vis.forward(Xt)[-1]
        Ate = vis.forward(Xe)[-1]
    An = Atr / Atr.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = Ate / Ate.norm(dim=1, keepdim=True).clamp(min=1e-9)
    ii = (Bn @ An.T).topk(5, dim=1).indices.cpu().numpy()
    pred = np.array([np.bincount(ytr[r], minlength=NC).argmax() for r in ii])
    print('  %.1f | %.3f' % (theta, float((pred == yte).mean())), flush=True)
