"""Actually train it: a concept layer with lateral inhibition, learned by Hebb, many epochs.

Previous attempts never really trained the image path.  The library stack has fixed weights, and the
one "learning" run applied three quick Oja passes and was called a failure.

This builds the layer the main line calls for:

  * inputs: the local 3D pattern of layer 1 (fixed, as before)
  * concept cells: each receives a random subset of those inputs
  * DIFFERENTIATION BY LATERAL INHIBITION, not top-k: a cell's score is reduced by the activity of
    its neighbours in the concept sheet, so cells compete locally without any global sorting
  * learning: Hebbian on the cells that fire, then each cell's weights are renormalised
  * many epochs, and the threshold is recalibrated so the activation ratio stays at the target

The read-out is k nearest exemplars, as established earlier.

Usage: python train_concept.py [train_n] [test_n] [epochs] [cells]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack, coords

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N2 = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, K = 0.20, 5

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

# ---------------- layer 1: the fixed local stack ----------------
vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Atr = vis.forward(Xt)[-1]
    Ate = vis.forward(Xe)[-1]
print('layer 1: %d cells, %.1f%% active' % (Atr.shape[1], 100 * float(Atr.mean())), flush=True)

# ---------------- layer 2: concept sheet with lateral inhibition ----------------
n1 = Atr.shape[1]
rng = np.random.default_rng(2)
SIDE = int(np.sqrt(N2))
FAN2 = int(os.environ.get("FAN2", "64"))
idx2 = np.stack([rng.choice(n1, FAN2, replace=False) for _ in range(N2)]).astype(np.int64)
W2 = rng.random((N2, FAN2)).astype(np.float32) * 0.1
W2 = W2 / W2.sum(1, keepdims=True)

# lateral neighbourhood on a SIDE x SIDE sheet (wrap-around): each cell takes the mean of its neighbours
yy, xx = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing='ij')
flat = (yy * SIDE + xx).ravel()
NB = []
for i in range(N2):
    y, x = divmod(i, SIDE)
    nb = [flat[((y + dy) % SIDE) * SIDE + ((x + dx) % SIDE)]
          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    NB.append(nb)
NB = T.as_tensor(np.array(NB), device=DEV)          # (N2, 8)

IDX2 = T.as_tensor(idx2, device=DEV)
W = T.as_tensor(W2, device=DEV)
BETA = 0.9        # strength of lateral inhibition


def scores(A):
    """A: (n, n1) layer-1 pattern -> concept-layer score before threshold"""
    return (A[:, IDX2] * W.unsqueeze(0)).sum(-1)


def respond_batch(a, theta, beta=BETA, steps=3):
    s = scores(a)
    for _ in range(steps):
        inh = s[:, NB].mean(-1) * beta
        y = (s - inh > theta).float()
        s = scores(a) - inh * y
    return (s > theta).float()


def respond(A, theta, beta=BETA, bs=256):
    out = []
    for i in range(0, len(A), bs):
        out.append(respond_batch(A[i:i + bs], theta, beta))
    return T.cat(out, 0)


def calibrate(A, target=0.10, iters=14):
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(respond(A[:500], mid).mean()) > target:
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


print('concept layer: %d cells, %d inputs each, lateral inhibition %.1f, %d epochs' % (
    N2, FAN2, BETA, EPOCHS), flush=True)
th = calibrate(Atr)
print('  before training: %.3f (active %.1f%%)' % (
    knn(respond(Atr, th), respond(Ate, th)), 100 * float(respond(Atr, th).mean())), flush=True)

ETA = 0.02
g = T.Generator(device=DEV).manual_seed(3)
for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for start in range(0, len(Atr), 256):
        sel = order[start:start + 256]
        a = Atr[sel]
        y = respond(a, th)
        xj = a[:, IDX2]                                  # (n, cells, FAN2)
        hebb = (y.unsqueeze(-1) * xj).sum(0)
        W += ETA * (hebb / max(float(y.sum()), 1.0) - W * y.mean())
        W.clamp_(min=0.0)
        W /= W.sum(1, keepdim=True).clamp(min=1e-9)
    th = calibrate(Atr)
    acc = knn(respond(Atr, th), respond(Ate, th))
    print('  epoch %2d: %.3f (active %.1f%%)' % (
        ep + 1, acc, 100 * float(respond(Atr, th).mean())), flush=True)
