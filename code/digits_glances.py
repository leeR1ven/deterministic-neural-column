"""Look several times, and accumulate the evidence INSIDE the loop.

Earlier "multiple looks" experiments averaged several independently-computed patterns after the
fact, which is not what an animal does.  Here the loop keeps a trace: each glance updates a state
that already carries the previous ones,

    m_t = m_{t-1} + (V_t - m_{t-1}) / TAU

so the decision is made from the accumulated state, not from any single frame.  The glances are
small eye movements (shifts of one or two pixels), so each one sees the same digit from a
slightly different position.

Everything else stays: local 3D stacks with fixed weights, no top-k, threshold sparsity; the
prototype read-out is only a measuring instrument.

Usage: python digits_glances.py [train_n] [test_n] [act]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28
NC = 10
TAU = 3.0            # the trace's time constant, in glances

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


Xtr, ytr = prep(tr, N_TRAIN, 1)
Xte, yte = prep(te, N_TEST, 2)


def glance(img, t):
    """eye movement: a small shift, cycling through a fixed set so every look is reproducible"""
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    dy, dx = shifts[t % len(shifts)]
    a = img.reshape(S, S)
    b = np.zeros_like(a)
    y0, y1 = max(dy, 0), S + min(dy, 0)
    x0, x1 = max(dx, 0), S + min(dx, 0)
    b[y0:y1, x0:x1] = a[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return b.reshape(-1)


vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:400], device=DEV), target=ACT)


def state(X, glances):
    """run the loop for `glances` steps and return the accumulated trace"""
    m = None
    with T.no_grad():
        for t in range(glances):
            V = vis.forward(T.as_tensor(np.stack([glance(x, t) for x in X]), device=DEV))[-1]
            m = V if m is None else m + (V - m) / TAU
    return m


def acc(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


print('MNIST, local 3D stack, activation %.2f, trace TAU %.1f' % (ACT, TAU), flush=True)
print('glances | accumulated state acc', flush=True)
for g in (1, 2, 3, 5, 7, 10):
    A = state(Xtr, g)
    B = state(Xte, g)
    print('  %2d    | %.3f' % (g, acc(A, B)), flush=True)
