"""Take the 0.95 model and let it look several times.

The model that reached 0.95 (code/motor_multi.py): the visual area's pattern, then each action owns 150
stored patterns and is scored by its closest one; the highest action wins.  This keeps that model exactly
as it is and adds the designer's idea on top: look several times (small eye movements) and ACCUMULATE the
per-action scores before deciding.

  per glance   score_d = max cosine between the pattern and action d's own patterns
  collapse     the scores of all glances are summed, then argmax
  control      decide from a single glance, no accumulation

Usage: python glances_095.py [train_n] [test_n] [epochs] [glances]
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
NG = int(sys.argv[4]) if len(sys.argv) > 4 else 9
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
    return X, Y


SEED = int(os.environ.get("SEED", "1"))
N_TEST = 2000 if SEED > 1 else N_TEST
Xtr, ytr = prep(tr, N_TRAIN, SEED * 101)
Xte, yte = prep(te, N_TEST, SEED * 101 + 7)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)     # as in the 0.95 model


def patterns(X):
    with T.no_grad():
        return vis.forward(T.as_tensor(X, device=DEV))[-1]


Atr, Ate = patterns(Xtr), patterns(Xte)
n1 = Atr.shape[1]
Wm = T.zeros(NC * M, n1, device=DEV)


def sims(A):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Wn = Wm / Wm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return An @ Wn.T


g = T.Generator(device=DEV).manual_seed(5 + SEED)
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
    """per-action score: the closest of that action's own patterns"""
    s = sims(patterns(X))
    return s.reshape(len(X), NC, M).max(-1).values


print('the 0.95 model, %d patterns per action, calibrated visual theta (as it was)' % M, flush=True)
print('', flush=True)
print('glances | accumulated score -> held-out accuracy', flush=True)
SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (2, 0), (-2, 0)]
acc = None
for k in range(NG):
    dy, dx = SHIFTS[k % len(SHIFTS)]
    Xk = Xte if (dy == 0 and dx == 0) else Xte.reshape(-1, S, S)
    if dy or dx:
        a = Xte.reshape(-1, S, S)
        b = np.zeros_like(a)
        y0, y1 = max(dy, 0), S + min(dy, 0)
        x0, x1 = max(dx, 0), S + min(dx, 0)
        b[:, y0:y1, x0:x1] = a[:, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        Xk = b.reshape(len(a), -1)
    sc = score(Xk)
    acc = sc if acc is None else acc + sc
    print('  %d      | %.3f' % (k + 1, float((acc.argmax(1).cpu().numpy() == yte).mean())), flush=True)

one = score(Xte).argmax(1).cpu().numpy()
print('', flush=True)
print('  single glance, no accumulation : %.3f' % float((one == yte).mean()), flush=True)
