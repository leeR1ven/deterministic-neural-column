"""Push the same pipeline as far as it goes: all 60000 training images, best read-out settings.

Architecture unchanged: the fixed local 3D layer, then a nearest-exemplar read-out.  Only the amount
of training data and the read-out parameter k change -- nothing about the network.

Reported against a k-NN on the raw pixels with exactly the same training set, so the reader can see
whether the layer adds anything at the extremes.

Usage: python push_100.py [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20

from torchvision.datasets import MNIST

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def load(ds, idx):
    X = np.zeros((len(idx), S * S), np.float32)
    Y = np.zeros(len(idx), int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        X[i] = (np.asarray(img, np.float32) / 255.0).reshape(-1)
        Y[i] = lab
    return X, Y


rng = np.random.default_rng(0)
all_tr = np.arange(len(tr))
te_idx = rng.choice(len(te), N_TEST, replace=False)
Xtr, ytr = load(tr, all_tr)
Xte, yte = load(te, te_idx)
print('training on ALL %d images | testing on %d held-out' % (len(Xtr), len(Xte)), flush=True)

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)


def patterns(X, bs=2000):
    out = []
    with T.no_grad():
        for i in range(0, len(X), bs):
            out.append(vis.forward(T.as_tensor(X[i:i + bs], device=DEV))[-1] > 0.5)
    return T.cat(out, 0)


def knn(A, B, k, bs=100):
    Af = A.float()
    An = Af / Af.norm(dim=1, keepdim=True).clamp(min=1e-9)
    correct = 0
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs].float()
        Bn = Bf / Bf.norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred = int(np.bincount(ytr[ii[r]], minlength=NC).argmax())
            correct += int(pred == yte[i + r])
    return correct / float(len(B))


Ptr = patterns(Xtr)
Pte = patterns(Xte)
Xtr_t = T.as_tensor((Xtr > 0.2).astype(np.float32), device=DEV)
Xte_t = T.as_tensor((Xte > 0.2).astype(np.float32), device=DEV)
print('', flush=True)
print('k  | layer-1 patterns | raw pixels', flush=True)
for k in (1, 3, 5, 9):
    print(' %d | %.3f            | %.3f' % (k, knn(Ptr, Pte, k), knn(Xtr_t, Xte_t, k)), flush=True)
