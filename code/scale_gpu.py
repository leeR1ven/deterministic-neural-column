"""GPU version of the scale/sparsity sweep (the numpy one was too slow above 4096 cells).

Same mechanisms, same measurement: nearest class prototype on the representation, with the
raw-pixel ceiling reported alongside.  Everything runs on the GPU so that 8192 x 4096
configurations are practical.

Usage: python scale_gpu.py [train_n] [test_n] [resolution] [seed]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
S = int(sys.argv[3]) if len(sys.argv) > 3 else 28
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
DEV = 'cuda' if T.cuda.is_available() else 'cpu'

from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def prep(ds, n, seed):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, S * S), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        X[i] = (np.asarray(img.resize((S, S), Image.BOX), np.float32) / 255.0).reshape(-1)
        Y[i] = lab
    return X, Y


Xtr, ytr = prep(tr, N_TRAIN, SEED + 1)
Xte, yte = prep(te, N_TEST, SEED + 2)
NC = 10
DIM = 2 * S * S


def complement_t(X):
    Xt = T.as_tensor(X, device=DEV)
    return T.cat([2 * Xt - 1, 1 - 2 * Xt], dim=1)


Vtr, Vte = complement_t(Xtr), complement_t(Xte)


def acc_proto(A, y, B, y2):
    """nearest class prototype: this is a measuring instrument, not a model component"""
    P = T.stack([A[T.as_tensor(y == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = (Bn @ P.T).argmax(1).cpu().numpy()
    return float((pred == y2).mean())


ceiling = acc_proto(T.as_tensor(Xtr, device=DEV), ytr, T.as_tensor(Xte, device=DEV), yte)
print('MNIST %dx%d | train %d | test %d | seed %d' % (S, S, N_TRAIN, N_TEST, SEED), flush=True)
print('raw-pixel ceiling: %.3f' % ceiling, flush=True)


def first_layer(N1, k=40, gain=2.0, seed=0):
    g = T.Generator(device=DEV).manual_seed(seed + 1)
    W = T.randn(N1, DIM, device=DEV, generator=g)
    mask = T.zeros_like(W, dtype=T.bool)
    for j in range(N1):
        idx = T.randperm(DIM, device=DEV, generator=g)[:k]
        mask[j, idx] = True
    W = W * mask
    W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
    bud = W.abs().sum(1) + 1e-9
    return ((gain * Vtr @ W.T > 0.5 * bud).float(), (gain * Vte @ W.T > 0.5 * bud).float())


def concepts(Ctr, Cte, N2, qlev, epochs=3, seed=0):
    g = T.Generator(device=DEV).manual_seed(seed + 2)
    n1 = Ctr.shape[1]
    W = T.randn(N2, n1, device=DEV, generator=g) * 0.1
    mask = T.zeros_like(W, dtype=T.bool)
    for j in range(N2):
        idx = T.randperm(n1, device=DEV, generator=g)[:100]
        mask[j, idx] = True
    W = W * mask
    W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
    for ep in range(epochs):
        order = T.randperm(len(Ctr), device=DEV, generator=g)
        for i in order.tolist():
            x = Ctr[i]
            s = W @ x
            w = T.topk(s, 3).indices
            W[w] += 0.15 * (x.unsqueeze(0) - W[w] * x.sum() / max(float(x.sum()), 1.0))
            W *= 0.998
            W.clamp_(min=0.0)
            W[w] /= W[w].abs().sum(1, keepdim=True).clamp(min=1e-9)
    th = T.quantile((W @ Ctr.T), qlev, dim=1, keepdim=True)
    return ((W @ Ctr.T > th).T.float(), (W @ Cte.T > th).T.float())


print('', flush=True)
print('first | concept | active%% | first-layer | concept-layer', flush=True)
for N1 in (8192,):
    Atr, Ate = first_layer(N1, seed=SEED)
    a1 = acc_proto(Atr, ytr, Ate, yte)
    for N2, q in ((4096, 0.90),):
        if N2 > N1:
            continue
        Btr, Bte = concepts(Atr, Ate, N2, q, epochs=3, seed=SEED)
        a2 = acc_proto(Btr, ytr, Bte, yte)
        print('  %5d | %5d  | %4.1f    | %.3f       | %.3f' % (
            N1, N2, 100.0 * (1 - q), a1, a2), flush=True)
