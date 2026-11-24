"""The z axis is wasted: every slice receives the same picture.

In the current construction the input goes into one z-slice and all the others start empty, so the
extra slices add cells that see the same neighbourhood -- copies, which is why doubling the volume
never changed anything.

This makes the z axis carry DIFFERENT views of the same image instead:
    slice 0  raw grey levels
    slice 1  blurred (coarse scale)
    slice 2  horizontal edge response
    slice 3  vertical edge response
    remaining slices  copies of the raw slice (so the volume keeps its size)

Nothing is labelled and no cell is assigned to anything: these are just different inputs to the same
local layer, and the read-out is still k nearest exemplars.

Usage: python multichannel.py [train_n] [test_n]
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
S, NC = 28, 10
Z = 8
ACT = 0.20
K = 5

from torchvision.datasets import MNIST

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def prep(ds, n, seed):
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


def blur(a):
    b = a.reshape(len(a), S // 4, 4, S // 4, 4).mean(axis=(2, 4))
    return np.repeat(np.repeat(b, 4, axis=1), 4, axis=2)[:, :S, :S]


def dh(a):
    out = np.zeros_like(a)
    out[:, :, 1:-1] = a[:, :, 2:] - a[:, :, :-2]
    return np.abs(out)


def dv(a):
    out = np.zeros_like(a)
    out[:, 1:-1, :] = a[:, 2:, :] - a[:, :-2, :]
    return np.abs(out)


def stack_channels(X, mode):
    n = len(X)
    out = np.zeros((n, Z, S, S), np.float32)
    if mode == 'same':
        out[:] = X[:, None]
    else:
        out[:, 0] = X
        out[:, 1] = blur(X)
        out[:, 2] = dh(X)
        out[:, 3] = dv(X)
        out[:, 4:] = X[:, None]
    return out.reshape(n, -1)


def build(seed=1, fan=4):
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


def forward(idx, w, b, X, theta, bs=1000):
    n = idx.shape[0]
    IDX = T.as_tensor(idx, device=DEV, dtype=T.long)
    W = T.as_tensor(w, device=DEV)
    B = T.as_tensor(b, device=DEV)
    out = []
    with T.no_grad():
        for i in range(0, len(X), bs):
            xb = T.as_tensor(X[i:i + bs], device=DEV)
            s = (xb[:, IDX] * W.unsqueeze(0)).sum(-1) + B.unsqueeze(0)
            out.append((s > theta * (B.abs() + W.abs().sum(1)).unsqueeze(0)).float())
    return T.cat(out, 0)


def calibrate(idx, w, b, X, target=ACT):
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if float(forward(idx, w, b, X[:500], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def knn(A, B, k=K, bs=200):
    Af = A.float(); An = Af / Af.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs].float(); Bn = Bf / Bf.norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(ytr[ii[r]], minlength=NC).argmax()))
    return float((np.array(pred) == yte).mean())


print('multi-channel input vs copies, train %d, activation %.2f, k=%d' % (N_TRAIN, ACT, K), flush=True)
idx, w, b = build()
for mode in ('same', 'multi'):
    A_in = stack_channels(Xtr, mode)
    B_in = stack_channels(Xte, mode)
    th = calibrate(idx, w, b, A_in)
    A = forward(idx, w, b, A_in, th)
    B = forward(idx, w, b, B_in, th)
    print('  z slices = %-5s | acc %.3f' % (mode, knn(A, B)), flush=True)
