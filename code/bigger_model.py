"""Does a bigger model help once there is enough data?

The earlier z sweep was run at 4000 training images.  If data was the limit there, a bigger model
could look useless when it is not.  This re-tests z at 10000 and 30000 images, with a larger
calibration set (1000 instead of 400) so that the threshold itself is not the weak link, and with
the k of the read-out scaled as well.

Patterns are stored as bool and the neighbour search is done in batches, so the bigger volumes fit
in 8 GB.

Usage: python bigger_model.py [train_n] [test_n]
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
S, NC = 28, 10
ACT = 0.20

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
Xt = T.as_tensor(Xtr, device=DEV)
Xe = T.as_tensor(Xte, device=DEV)


def patterns(st, X, bs=2000):
    out = []
    with T.no_grad():
        for i in range(0, len(X), bs):
            out.append(st.forward(X[i:i + bs])[-1] > 0.5)
    return T.cat(out, 0)


def knn_bool(A, B, k, bs=200):
    """nearest exemplars over boolean patterns, in batches"""
    Af = A.float()
    An = Af / Af.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs].float()
        Bn = Bf / Bf.norm(dim=1, keepdim=True).clamp(min=1e-9)
        idx = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(idx)):
            pred.append(int(np.bincount(ytr[idx[r]], minlength=NC).argmax()))
    return float((np.array(pred) == yte).mean())


print('train %d | activation %.2f | calibration on 1000 images' % (N_TRAIN, ACT), flush=True)
print('z  | cells  | k=3   k=5   k=9', flush=True)
for z in (8, 16, 32, 64):
    st = Local3DStack([(S, S, 1), (S, S, z)], fan_in=4, seed=1, dev=DEV)
    st.calibrate(Xt[:1000], target=ACT)
    A = patterns(st, Xt)
    B = patterns(st, Xe)
    row = ['%.3f' % knn_bool(A, B, k) for k in (3, 5, 9)]
    print('%-2d | %5d | %s' % (z, A.shape[1], '  '.join(row)), flush=True)
    del A, B
    T.cuda.empty_cache()
