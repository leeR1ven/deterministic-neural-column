"""Several concepts per digit: classify with many templates instead of one average.

The designer's point: if one digit activates very different cell groups for different
writings, that is fine -- several concepts can all mean the same digit.  A single average
prototype blurs exactly those groups together, so this keeps K templates per digit.

Measured on held-out images, for K = 1 (the old way), 2, 4, 8, 16 templates per digit.
The templates are found by k-means on the concept responses of the training images of that
digit -- unsupervised, no labels used inside the cluster.

Usage: python multi_concept.py [train_n] [test_n]
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = np.random.default_rng(SEED)

from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def prep(ds, n, seed):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, 64), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        X[i] = np.asarray(img.resize((8, 8), Image.BOX), np.float32).reshape(-1) / 255.0
        Y[i] = lab
    return X, Y


Xtr, ytr = prep(tr, N_TRAIN, SEED + 1)
Xte, yte = prep(te, N_TEST, SEED + 2)
NC = 10

# first layer
r = np.random.default_rng(SEED + 1)
N1, K1, GAIN = 1024, 40, 2.0
W1 = r.normal(0, 1, (N1, 128))
mask = np.zeros_like(W1, bool)
for j in range(N1):
    mask[j, r.choice(128, K1, replace=False)] = True
W1 = W1 * mask
W1 = W1 / np.abs(W1).sum(1, keepdims=True)
BUD1 = np.abs(W1).sum(1) + 1e-9


def first_layer(X):
    V = np.concatenate([2 * X - 1, 1 - 2 * X], axis=1)
    return (W1 @ (GAIN * V).T > 0.5 * BUD1[:, None]).T.astype(np.float32)


Ctr, Cte = first_layer(Xtr), first_layer(Xte)

# concept layer
N2, K2 = 1024, 100
r2 = np.random.default_rng(SEED + 2)
W2 = r2.normal(0, 1, (N2, N1)) * 0.1
m2 = np.zeros_like(W2, bool)
for j in range(N2):
    m2[j, r2.choice(N1, K2, replace=False)] = True
W2 = W2 * m2
W2 = W2 / np.abs(W2).sum(1, keepdims=True)
for ep in range(10):
    order = r2.permutation(len(Ctr))
    for i in order:
        x = Ctr[i]
        s = W2 @ x
        w = np.argpartition(s, -3)[-3:]
        W2[w] += 0.15 * (x[None, :] - W2[w] * x.sum() / max(x.sum(), 1))
        W2 *= 0.998
        np.maximum(W2, 0.0, out=W2)
        W2[w] /= np.maximum(np.abs(W2[w]).sum(1, keepdims=True), 1e-9)
TH = np.quantile(W2 @ Ctr.T, 0.90, axis=1, keepdims=True)
Atr = (W2 @ Ctr.T > TH).T.astype(np.float32)
Ate = (W2 @ Cte.T > TH).T.astype(np.float32)
print('train %d | test %d | concept cells %.0f active' % (
    N_TRAIN, N_TEST, Atr.sum(1).mean()), flush=True)


def kmeans(X, k, iters=25, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = np.argmin(D, axis=1)
        for c in range(k):
            if (a == c).any():
                C[c] = X[a == c].mean(0)
    return C


def classify(templates, tlabels, A):
    """nearest template, normalised; ties broken by summing similarity per digit"""
    Tn = templates / np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9)
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    S = An @ Tn.T
    out = np.zeros((len(A), NC), np.float32)
    for j, c in enumerate(tlabels):
        out[:, c] += S[:, j]
    return np.argmax(out, axis=1)


print('', flush=True)
print('templates per digit | accuracy', flush=True)
for K in (1, 2, 4, 8, 16):
    Tm, Tl = [], []
    for c in range(NC):
        Xc = Atr[ytr == c]
        if K == 1:
            C = Xc.mean(0, keepdims=True)
        else:
            C = kmeans(Xc.astype(np.float32), min(K, len(Xc)), seed=SEED + c)
        Tm.append(C)
        Tl.extend([c] * len(C))
    pred = classify(np.concatenate(Tm), np.array(Tl), Ate)
    print('  %2d                | %.3f' % (K, float((pred == yte).mean())), flush=True)
