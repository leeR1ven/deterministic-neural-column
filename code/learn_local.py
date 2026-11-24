"""Let the local layer's weights be shaped by the data instead of drawn at random.

Every image result so far used the library's Local3DStack, whose weights are FIXED random values --
it never learns.  The word model and the muscle binding in this repository do learn, so this simply
connects the image path to the same idea.

The connectivity stays local and fixed (each cell receives itself plus three near neighbours); only
the four strengths change, by a local rule with no labels, no top-k and no global error:

    for each image, for each cell that fires (threshold rule):
        w_ij <- w_ij + eta * y_i * (x_j - y_i * w_ij)      (Oja: Hebbian growth, then normalise)

then the threshold is recalibrated so the activation ratio stays at the target.

Compared against the fixed-weight layer on held-out images, three seeds.

Usage: python learn_local.py [train_n] [test_n] [epochs]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
Z, ACT, K = 8, 0.20, 5
FAN = 4

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


def build(seed):
    rng = np.random.default_rng(seed)
    dim = (S, S, Z)
    yy, xx, zz = np.meshgrid(np.arange(S), np.arange(S), np.arange(Z), indexing='ij')
    oc = np.stack([yy.ravel(), xx.ravel(), zz.ravel()], -1)
    n = len(oc)
    idx = np.zeros((n, FAN), np.int64)
    w = np.zeros((n, FAN), np.float32)
    idx[:, 0] = np.arange(n)
    w[:, 0] = 1.0
    for j in range(1, FAN):
        nb = oc.copy()
        nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, S - 1)
        nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, S - 1)
        nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, Z - 1)
        idx[:, j] = np.ravel_multi_index(nb.T, dim)
        w[:, j] = (rng.random(n) * 0.5 + 0.5) * 0.01
    b = (rng.random(n) * 1.0).astype(np.float32)
    return idx, w, b


def volume(X):
    """replicate the image across the z slices so every cell has an input"""
    return np.repeat(X[:, None, :], Z, 1).reshape(len(X), -1)


def respond(W, B, IDX, Xvol, theta, bs=2000):
    out = []
    with T.no_grad():
        for i in range(0, len(Xvol), bs):
            xb = T.as_tensor(Xvol[i:i + bs], device=DEV)
            s = (xb[:, IDX] * W.unsqueeze(0)).sum(-1) + B.unsqueeze(0)
            out.append((s > theta * (B.abs() + W.abs().sum(1)).unsqueeze(0)).float())
    return T.cat(out, 0)


def calibrate(W, B, IDX, Xvol, target=ACT):
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if float(respond(W, B, IDX, Xvol[:500], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def knn(A, Bm, ytr, yte, k=K, bs=200):
    Af = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(Bm), bs):
        Bf = Bm[i:i + bs] / Bm[i:i + bs].norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bf @ Af.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(ytr[ii[r]], minlength=NC).argmax()))
    return float((np.array(pred) == yte).mean())


print('local layer, %d cells, fixed connectivity, %d epochs of local Oja learning' % (
    S * S * Z, EPOCHS), flush=True)
print('seed | fixed weights | learned weights', flush=True)

for seed in (1, 2, 3):
    Xtr, ytr = prep(tr, N_TRAIN, seed * 101)
    Xte, yte = prep(te, N_TEST, seed * 101 + 7)
    Vtr, Vte = volume(Xtr), volume(Xte)

    idx, w, b = build(seed)
    IDX = T.as_tensor(idx, device=DEV, dtype=T.long)
    Wt = T.as_tensor(w, device=DEV)
    Bt = T.as_tensor(b, device=DEV)

    th0 = calibrate(Wt, Bt, IDX, Vtr)
    acc_fixed = knn(respond(Wt, Bt, IDX, Vtr, th0), respond(Wt, Bt, IDX, Vte, th0), ytr, yte)

    # ---- local Oja learning: only the cells that fire update, no labels, no top-k ----
    ETA = 0.05
    g = T.Generator(device=DEV).manual_seed(seed)
    for ep in range(EPOCHS):
        order = T.randperm(len(Vtr), device=DEV, generator=g)
        for start in range(0, len(Vtr), 200):
            sel = order[start:start + 200].cpu().numpy()
            xb = T.as_tensor(Vtr[sel], device=DEV)
            s = (xb[:, IDX] * Wt.unsqueeze(0)).sum(-1) + Bt.unsqueeze(0)
            y = (s > th0 * (Bt.abs() + Wt.abs().sum(1)).unsqueeze(0)).float()   # (n, cells)
            # gather the inputs each cell sees, then Oja on the active ones
            xj = xb[:, IDX]                                                     # (n, cells, FAN)
            yv = y.unsqueeze(-1)
            hebb = (yv * xj).sum(0)                                             # (cells, FAN)
            norm = (yv * yv).sum(0)                                             # (cells, 1)
            Wt += ETA * (hebb - norm * Wt)
            Wt.clamp_(min=0.0)

    th1 = calibrate(Wt, Bt, IDX, Vtr)
    acc_learned = knn(respond(Wt, Bt, IDX, Vtr, th1), respond(Wt, Bt, IDX, Vte, th1), ytr, yte)
    print('  %d  | %.3f         | %.3f' % (seed, acc_fixed, acc_learned), flush=True)
