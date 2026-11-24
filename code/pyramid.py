"""A spatial pyramid: pool between layers so the receptive field grows.

All previous experiments used one local layer over a 28x28 volume.  In such a layer a cell sees
three near neighbours, so its receptive field is a few pixels -- it cannot look at the shape of a
digit, only at local texture.  Pooling between layers is the standard way to grow the field without
adding cells, and it is a structural change rather than a parameter.

    L1  28x28xZ local layer   -> 2x2 mean pool -> 14x14xZ
    L2  14x14xZ local layer   -> 2x2 mean pool ->  7x7xZ
    L3   7x7xZ  local layer   -> read-out (k nearest exemplars)

The read-out is also compared between the last layer and the concatenation of all layers, since
earlier layers keep finer detail and later ones have the larger field.

Usage: python pyramid.py [train_n] [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import coords

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
NC = 10
Z = 8
ACT = 0.20
K = 5

from torchvision.datasets import MNIST

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def prep(ds, n, seed, S=28):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, S, S), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        X[i] = (np.asarray(img, np.float32) / 255.0)
        Y[i] = lab
    return X, Y


Xtr, ytr = prep(tr, N_TRAIN, 101)
Xte, yte = prep(te, N_TEST, 108)


def build_layer(S, seed, fan=4):
    rng = np.random.default_rng(seed)
    dim = (S, S, Z)
    yy, xx, zz = np.meshgrid(np.arange(S), np.arange(S), np.arange(Z), indexing='ij')
    oc = np.stack([yy.ravel(), xx.ravel(), zz.ravel()], -1)
    n = len(oc)
    idx = np.zeros((n, fan), np.int64)
    w = np.zeros((n, fan), np.float32)
    idx[:, 0] = np.arange(n)
    w[:, 0] = 1.0
    for j in range(1, fan):
        nb = oc.copy()
        nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, S - 1)
        nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, S - 1)
        nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, Z - 1)
        idx[:, j] = np.ravel_multi_index(nb.T, dim)
        w[:, j] = (rng.random(n) * 0.5 + 0.5) * 0.01
    b = (rng.random(n) * 1.0).astype(np.float32)
    return idx, w, b


def run_layer(idx, w, b, Xvol, theta, bs=1000):
    IDX = T.as_tensor(idx, device=DEV, dtype=T.long)
    W = T.as_tensor(w, device=DEV)
    B = T.as_tensor(b, device=DEV)
    out = []
    with T.no_grad():
        for i in range(0, len(Xvol), bs):
            xb = Xvol[i:i + bs]
            s = (xb[:, IDX] * W.unsqueeze(0)).sum(-1) + B.unsqueeze(0)
            out.append((s > theta * (B.abs() + W.abs().sum(1)).unsqueeze(0)).float())
    return T.cat(out, 0)


def calibrate(idx, w, b, Xvol, target=ACT):
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if float(run_layer(idx, w, b, Xvol[:400], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pool2(h, S):
    """2x2 mean pooling of a (n, S*S*Z) binary pattern"""
    n = h.shape[0]
    v = h.reshape(n, S, S, Z).permute(0, 3, 1, 2)
    v = v.reshape(n, Z, S // 2, 2, S // 2, 2).mean(dim=(3, 5))
    return v.permute(0, 2, 3, 1).reshape(n, (S // 2) * (S // 2) * Z)


def knn(A, B, k=K, bs=200):
    Af = A.float(); An = Af / Af.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs].float(); Bn = Bf / Bf.norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(ytr[ii[r]], minlength=NC).argmax()))
    return float((np.array(pred) == yte).mean())


layers = [build_layer(28, 1), build_layer(14, 2), build_layer(7, 3)]
print('pyramid 28 -> 14 -> 7, Z=%d, activation %.2f, train %d' % (Z, ACT, N_TRAIN), flush=True)

outs_tr, outs_te = [], []
# the first layer needs the full volume: replicate the image across its z slices
Ain = T.as_tensor(np.repeat(Xtr.reshape(len(Xtr), -1)[:, None, :], Z, 1).reshape(len(Xtr), -1), device=DEV)
Bin = T.as_tensor(np.repeat(Xte.reshape(len(Xte), -1)[:, None, :], Z, 1).reshape(len(Xte), -1), device=DEV)
S_ = 28
for li, (idx, w, b) in enumerate(layers):
    th = calibrate(idx, w, b, Ain)
    A = run_layer(idx, w, b, Ain, th)
    B = run_layer(idx, w, b, Bin, th)
    outs_tr.append(A); outs_te.append(B)
    print('  L%d (%dx%d) | acc %.3f' % (li + 1, S_, S_, knn(A, B)), flush=True)
    if S_ > 7:
        Ain = pool2(A, S_); Bin = pool2(B, S_)
        S_ //= 2

cat_tr = T.cat(outs_tr, 1); cat_te = T.cat(outs_te, 1)
print('  all layers concatenated | acc %.3f' % knn(cat_tr, cat_te), flush=True)
