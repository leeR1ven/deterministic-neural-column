"""A fully connected concept layer, trained by Hebb -- can training beat the fixed layer?

The previous trained layer sampled only 128 of 12544 inputs per cell, so it lost information before
learning even started.  Here each concept cell sees the WHOLE layer-1 pattern, and learning is
Hebbian with a threshold rule (no top-k, no labels, no global error):

    s = A @ W.T                    (concept scores)
    y = s > theta                  (threshold; theta keeps the activation ratio fixed)
    W += eta * (y.T @ A) / n       (co-active inputs grow)
    W <- clamp(W, 0) and renormalise each row

Read-out is k nearest exemplars on the concept pattern, as everywhere else.

Usage: python dense_concept.py [train_n] [test_n] [epochs] [cells]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N2 = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, K = 0.20, 5
A2 = 0.10          # activation ratio on the concept layer

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

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Atr = vis.forward(Xt)[-1]
    Ate = vis.forward(Xe)[-1]
print('layer 1: %d cells, %.1f%% active' % (Atr.shape[1], 100 * float(Atr.mean())), flush=True)

n1 = Atr.shape[1]
rng = np.random.default_rng(2)
W = T.as_tensor((rng.random((N2, n1)).astype(np.float32) * 0.1), device=DEV)
W = W / W.sum(1, keepdim=True)


def respond(A, theta, bs=512):
    out = []
    with T.no_grad():
        for i in range(0, len(A), bs):
            out.append(((A[i:i + bs] @ W.T) > theta).float())
    return T.cat(out, 0)


def calibrate(A, target=A2, iters=14):
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(respond(A[:1000], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def knn(Aa, Bb, k=K, bs=200):
    Af = Aa / Aa.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(Bb), bs):
        Bf = Bb[i:i + bs] / Bb[i:i + bs].norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bf @ Af.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(ytr[ii[r]], minlength=NC).argmax()))
    return float((np.array(pred) == yte).mean())


print('dense concept layer: %d cells x %d inputs, target %.0f%% active, %d epochs' % (
    N2, n1, 100 * A2, EPOCHS), flush=True)
print('  layer-1 read-out (reference): %.3f' % knn(Atr, Ate), flush=True)
th = calibrate(Atr)
print('  concept before training: %.3f (active %.1f%%)' % (
    knn(respond(Atr, th), respond(Ate, th)), 100 * float(respond(Atr, th).mean())), flush=True)

ETA = 0.02
g = T.Generator(device=DEV).manual_seed(3)
for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for start in range(0, len(Atr), 256):
        sel = order[start:start + 256]
        a = Atr[sel]
        y = ((a @ W.T) > th).float()
        W += ETA * (y.T @ a) / max(float(y.sum()), 1.0)
        W -= ETA * 0.01 * W
        W.clamp_(min=0.0)
        W /= W.sum(1, keepdim=True).clamp(min=1e-9)
    th = calibrate(Atr)
    print('  epoch %2d: %.3f (active %.1f%%)' % (
        ep + 1, knn(respond(Atr, th), respond(Ate, th)), 100 * float(respond(Atr, th).mean())), flush=True)
