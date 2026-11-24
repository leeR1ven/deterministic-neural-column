"""How long should it look?  A local layer run recurrently, and how the settling time matters.

Everything so far was a single feed-forward pass, so the model had no "duration" at all.  Here the
first layer is run as a local recurrent network:

    h_{t+1} = fire( W_local @ (h_t + g * x) )

where W_local is the library's local 3D connectivity (self + three near neighbours, fixed weights),
x is the input image, g is how strongly the input keeps driving the state, and fire() is the
threshold rule.  t = 0 is the plain feed-forward pass; later steps let the pattern settle and the
input keep acting.

Questions:
  * does looking longer help, and where does it stop helping?
  * how strong should the ongoing input drive be (g)?

Usage: python settle_duration.py [train_n] [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import coords

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
Z = 16
ACT = 0.20
K = 5

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


def knn(A, B, k=K):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    idx = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
    pred = np.array([np.bincount(ytr[idx[i]], minlength=NC).argmax() for i in range(len(B))])
    return float((pred == yte).mean())


# --- local recurrent layer over the 28x28xZ volume ---
rng = np.random.default_rng(7)
dim = (S, S, Z)
oc = coords(dim)
n = len(oc)
FAN = 4
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
IDX = T.as_tensor(idx, device=DEV, dtype=T.long)
W = T.as_tensor(w, device=DEV)
BUD = (W.abs().sum(1)).unsqueeze(0) + 1e-9

# the input enters the first z-slice and spreads through the local connectivity
x_full = T.zeros(Xt.shape[0], n, device=DEV)
x_full_te = T.zeros(Xe.shape[0], n, device=DEV)
plane = S * S
x_full[:, :plane] = Xt
x_full_te[:, :plane] = Xe


def fire(s, theta):
    return (s > theta * BUD).float()


def settle(x_full_batch, steps, g, theta):
    h = fire(g * x_full_batch, theta)
    for t in range(steps):
        s = (h[:, IDX] * W.unsqueeze(0)).sum(-1) + g * x_full_batch
        h = fire(s, theta)
    return h


def calibrate(xb, g, target=ACT, iters=16):
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        act = float(settle(xb, 0, g, mid).mean())
        if act > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print('local recurrent layer, %d cells, activation target %.2f, k=%d' % (n, ACT, K), flush=True)
print('input drive g | steps: 0      1      2      3      5      8', flush=True)
for g in (1.0, 2.0, 4.0):
    theta = calibrate(x_full[:400], g)
    row = []
    with T.no_grad():
        for steps in (0, 1, 2, 3, 5, 8):
            h1 = settle(x_full, steps, g, theta)
            h2 = settle(x_full_te, steps, g, theta)
            row.append('%.3f' % knn(h1, h2))
    print('   %.1f        | %s' % (g, '  '.join(row)), flush=True)
