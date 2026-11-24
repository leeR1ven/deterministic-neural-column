"""What do the pictures it gets wrong actually look like?

Trains the best model (150 stored patterns per action, five accumulated glances) and prints the
misclassified test images as characters, with the true digit and what the model answered.

Usage: python look_at_errors.py [train_n] [test_n] [epochs] [n_show]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NSHOW = int(sys.argv[4]) if len(sys.argv) > 4 else 8
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC, MZ = 28, 10, 16
ACT, M, ETA = 0.20, 150, 0.05

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
    return X, Y, idx


Xtr, ytr, _ = prep(tr, N_TRAIN, 101)
Xte, yte, te_idx = prep(te, N_TEST, 108)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)


def patterns(X):
    with T.no_grad():
        return vis.forward(T.as_tensor(X, device=DEV))[-1]


Atr = patterns(Xtr)
n1 = Atr.shape[1]
Wm = T.zeros(NC * M, n1, device=DEV)


def sims(A):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Wn = Wm / Wm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return An @ Wn.T


g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for st in range(0, len(Atr), 256):
        sel = order[st:st + 256]
        A = Atr[sel]
        y = ytr[sel.cpu().numpy()]
        s = sims(A)
        cls = (T.as_tensor(y, device=DEV) * M).unsqueeze(1) + T.arange(M, device=DEV).unsqueeze(0)
        own = Wm.reshape(NC, M, -1)[T.as_tensor(y, device=DEV)]
        free = (own.sum(-1) <= 1e-6)
        who = T.where(free.any(1), free.float().argmax(1), s.gather(1, cls).argmax(1))
        rows = T.as_tensor(y, device=DEV) * M + who
        empty = Wm[rows].sum(1) <= 1e-6
        if bool(empty.any()):
            Wm[rows[empty]] = A[empty]
        nz = ~empty
        if bool(nz.any()):
            Wm.index_add_(0, rows[nz], ETA * (A[nz] - Wm[rows[nz]]))
        Wm.clamp_(min=0.0)


def score(X):
    s = sims(patterns(X))
    return s.reshape(len(X), NC, M).max(-1).values


def shifted(X, dy, dx):
    a = X.reshape(-1, S, S)
    b = np.zeros_like(a)
    y0, y1 = max(dy, 0), S + min(dy, 0)
    x0, x1 = max(dx, 0), S + min(dx, 0)
    b[:, y0:y1, x0:x1] = a[:, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return b.reshape(len(a), -1)


SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
acc = None
for dy, dx in SHIFTS:
    sc = score(Xte if (dy == 0 and dx == 0) else shifted(Xte, dy, dx))
    acc = sc if acc is None else acc + sc
pred = acc.argmax(1).cpu().numpy()
wrong = np.where(pred != yte)[0]
print('held-out %d | wrong %d (%.1f%%)' % (len(yte), len(wrong), 100.0 * len(wrong) / len(yte)), flush=True)
from collections import Counter
pairs = Counter((int(yte[w]), int(pred[w])) for w in wrong)
print('confusions (true -> answered), most common first:', flush=True)
for (t, q), c in pairs.most_common(10):
    print('   %d -> %d : %2d times' % (t, q, c), flush=True)
same_shape = sum(c for (t, q), c in pairs.items() if {t, q} in ({1,7},{3,8},{4,9},{5,6},{5,8},{0,6},{0,8},{2,7}))
print('   of which are classic look-alike pairs (1/7, 3/8, 4/9, 5/6, 5/8, 0/6, 0/8, 2/7): %d of %d (%.0f%%)' % (
    same_shape, len(wrong), 100.0*same_shape/max(len(wrong),1)), flush=True)
print('', flush=True)

for j, w in enumerate(wrong[:NSHOW]):
    a = Xte[w].reshape(S, S)
    print('  #%d  true=%d  model answered %d' % (j + 1, yte[w], pred[w]), flush=True)
    for row in a[::2]:
        print('     ' + ''.join('#' if v > 0.6 else ('+' if v > 0.25 else ('.' if v > 0.05 else ' ')) for v in row), flush=True)
    print('', flush=True)
