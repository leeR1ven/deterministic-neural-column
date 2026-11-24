"""Looking continuously changes the prefrontal CONTEXT, which then colours the reading.

The designer's point: continuous looking is like continuously hearing sound -- it slowly moves
the prefrontal context rather than stacking up a static picture.  So the loop keeps a slow state

    h_t = h_{t-1} + (V_t - h_{t-1}) / TAU          the context

and the interpretation at time t is read from the CURRENT glance together with the context:

    score_c = cos(V_t, class_c) + beta * cos(h_t, class_c)

With beta = 0 this is the old single-glance reading; with beta > 0 the context (which the earlier
glances built) biases the interpretation, so the reading should sharpen as glances accumulate.

The eye movement still fills in the field: each fixation writes its sharp patch into what the
next glance sees (write-where-you-look), since averaging or accumulating thresholded patterns was
shown not to work.

Usage: python digits_context.py [seed] [train_n] [test_n] [glances]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
T_G = int(sys.argv[4]) if len(sys.argv) > 4 else 6
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.35
TAU = 2.0

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


Xtr, ytr = prep(tr, N_TRAIN, SEED * 101)
Xte, yte = prep(te, N_TEST, SEED * 101 + 7)

yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
FIX = [(14, 14), (9, 9), (19, 9), (9, 19), (19, 19), (14, 7)]
SIG = 5.0
MASKS = [(np.exp(-(((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * SIG ** 2))).reshape(-1) > 0.3).astype(np.float32)
         for fy, fx in FIX]

# prototypes from the full image (the measuring instrument, as before)
vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=SEED, dev=DEV)
first = np.stack([Xtr * MASKS[0] + 0.0 for _ in range(1)])[0]
vis.calibrate(T.as_tensor(Xtr[:400], device=DEV), target=ACT)
with T.no_grad():
    FULL_tr = vis.forward(T.as_tensor(Xtr, device=DEV))[-1]
    FULL_te = vis.forward(T.as_tensor(Xte, device=DEV))[-1]
P = T.stack([FULL_tr[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)


def sequence(X):
    """run the glances; return the context after each glance and the current pattern"""
    buf = np.zeros_like(X)
    h = None
    out = []
    for t in range(T_G):
        m = MASKS[t % len(MASKS)]
        buf = buf * (1 - m) + X * m                  # write-where-you-look
        with T.no_grad():
            V = vis.forward(T.as_tensor(buf, device=DEV))[-1]
        h = V if h is None else h + (V - h) / TAU    # the context
        out.append((V, h))
    return out


seq_tr = sequence(Xtr)
seq_te = sequence(Xte)


def readout(V, h, beta):
    Vn = V / V.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Hn = h / h.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return (Vn @ P.T + beta * (Hn @ P.T)).argmax(1).cpu().numpy()


print('seed %d | glances up to %d | activation %.2f | TAU %.1f' % (SEED, T_G, ACT, TAU), flush=True)
print('beta | t=1   t=2   t=3   t=4   t=5   t=6', flush=True)
for beta in (0.0, 0.5, 1.0, 2.0):
    row = []
    for t in range(T_G):
        V, h = seq_te[t]
        row.append('%.3f' % float((readout(V, h, beta) == yte).mean()))
    print('  %.1f | %s' % (beta, '  '.join(row)), flush=True)
