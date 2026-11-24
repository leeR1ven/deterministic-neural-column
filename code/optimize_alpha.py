"""Optimisation of the conforming model: make the layer actually compute.

Independent reproduction found that the local 3D layer is decorative: the self weight is 1.0 while the
three neighbours together total 0.0225, so removing them changes only 0.17 % of the activations.  The
layer is effectively a per-pixel threshold.

The library exposes the neighbour weight as `alpha` (default 0.01).  This sweeps it, together with the
number of prototypes per action, and reports every result against two baselines measured on the same
split: a nearest-neighbour read-out on the raw pixels, and one on the layer's own patterns.

No architecture change: same layer, same motor cells, same cosine read-out.  Only the construction
parameter of the layer and the number of prototypes vary.

Usage: python optimize_alpha.py [train_n] [test_n] [epochs]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, ETA = 0.20, 0.05

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


def knn(Xa, Xb, k):
    A = Xa / np.maximum(np.linalg.norm(Xa, axis=1, keepdims=True), 1e-9)
    B = Xb / np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-9)
    ii = (B @ A.T).argsort(1)[:, -k:]
    pred = np.array([np.bincount(ytr[r], minlength=NC).argmax() for r in ii])
    return float((pred == yte).mean())


print('baselines on this split: raw pixels 5-NN %.3f | 1-NN %.3f' % (knn(Xtr, Xte, 5), knn(Xtr, Xte, 1)), flush=True)


def run(alpha, M, seed=1):
    vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, alpha=alpha, bias_scale=1.0, seed=1, dev=DEV)
    vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)
    with T.no_grad():
        A = vis.forward(T.as_tensor(Xtr, device=DEV))[-1]
        B = vis.forward(T.as_tensor(Xte, device=DEV))[-1]
    Wm = T.zeros(NC * M, A.shape[1], device=DEV)

    def sims(X):
        Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)
        Wn = Wm / Wm.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return Xn @ Wn.T

    def answer(X):
        return sims(X).reshape(len(X), NC, M).max(-1).values.argmax(1).cpu().numpy()

    g = T.Generator(device=DEV).manual_seed(5 + seed)
    for ep in range(EPOCHS):
        order = T.randperm(len(A), device=DEV, generator=g)
        for st in range(0, len(A), 256):
            sel = order[st:st + 256]
            a, y = A[sel], ytr[sel.cpu().numpy()]
            s = sims(a)
            cls = (T.as_tensor(y, device=DEV) * M).unsqueeze(1) + T.arange(M, device=DEV).unsqueeze(0)
            own = Wm.reshape(NC, M, -1)[T.as_tensor(y, device=DEV)]
            free = (own.sum(-1) <= 1e-6)
            who = T.where(free.any(1), free.float().argmax(1), s.gather(1, cls).argmax(1))
            rows = T.as_tensor(y, device=DEV) * M + who
            empty = Wm[rows].sum(1) <= 1e-6
            if bool(empty.any()):
                Wm[rows[empty]] = a[empty]
            nz = ~empty
            if bool(nz.any()):
                Wm.index_add_(0, rows[nz], ETA * (a[nz] - Wm[rows[nz]]))
            Wm.clamp_(min=0.0)
    # how much does the layer compute?  fraction of activations that change when neighbours are removed
    return float((answer(B) == yte).mean()), A, B


print('', flush=True)
print('', flush=True)
print('seed | held-out (150/action) | layer 5-NN | raw 5-NN', flush=True)
res = []
for seed in (1, 2, 3):
    acc, Atr_, Ate_ = run(0.01, 150, seed=seed)
    lay = knn(Atr_.cpu().numpy(), Ate_.cpu().numpy(), 5)
    raw = knn(Xtr, Xte, 5)
    res.append((acc, lay, raw))
    print('  %d  | %.3f                 | %.3f     | %.3f' % (seed, acc, lay, raw), flush=True)
r = np.array(res)
print('mean | %.3f                 | %.3f     | %.3f' % tuple(r.mean(0)), flush=True)
