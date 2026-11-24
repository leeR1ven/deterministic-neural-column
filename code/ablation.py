"""Which of my changes cost the accuracy?  One ablation, four rows.

The version that reached 0.95 and the version that reaches 0.71 differ in three things, and two of them
were my own additions rather than anything the designer asked for:

  (1) INPUT TO EACH MOTOR CELL   full 12544-dim pattern   vs  a random 800-input subset
  (2) LEARNING RULE              only the best cell of the action learns each pattern (winner takes all)
                                 vs  every cell of the action moves toward every pattern (a mean)
  (3) SURFACE THRESHOLD          the read-out uses the continuous similarity
                                 vs  the surface fires above a threshold and only 0/1 is passed on

Starting from (full inputs, winner-takes-all, no threshold) -- which is the 0.95 version -- each of the
three is switched on one at a time, so the cost of each is visible separately.

The motor area's own local stack is left out of this table on purpose: it is the part the designer asked
for and a previous measurement showed it recovers accuracy rather than costing it.

Usage: python ablation.py [train_n] [test_n] [epochs] [cells_per_action]
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
PER = int(sys.argv[4]) if len(sys.argv) > 4 else 150
FAN = int(os.environ.get('FAN', '800'))
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC, MZ = 28, 10, 16
ACT, ETA = 0.20, 0.05

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
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Vtr, Vte = vis.forward(Xt)[-1], vis.forward(Xe)[-1]
n1 = Vtr.shape[1]


def run(full_input, wta, threshold):
    """full_input: each cell sees the whole pattern | wta: only the best cell learns
       threshold: the surface fires (0/1) before the read-out"""
    N = PER * NC
    rng = np.random.default_rng(3)
    if full_input:
        MASK = T.ones(N, n1, device=DEV)
    else:
        m = np.zeros((N, n1), bool)
        for j in range(N):
            m[j, rng.choice(n1, FAN, replace=False)] = True
        MASK = T.as_tensor(m, device=DEV).float()
    W = T.zeros(N, n1, device=DEV)
    REG = T.as_tensor(np.array([j // PER for j in range(N)]), device=DEV)

    def sim(V):
        s = V @ W.T
        return s / ((V.norm(dim=1, keepdim=True) * W.norm(dim=1).unsqueeze(0)) + 1e-9)

    def score(s, binary=False):
        # on a CONTINUOUS similarity the closest cell decides; on a BINARY firing pattern the max is a
        # tie, so the region that fires most cells decides (a count, i.e. a rate code)
        if binary:
            return T.stack([s[:, REG == d].sum(1) for d in range(NC)], 1)
        return T.stack([s[:, REG == d].max(1).values for d in range(NC)], 1)

    g = T.Generator(device=DEV).manual_seed(5)
    for ep in range(EPOCHS):
        order = T.randperm(len(Vtr), device=DEV, generator=g)
        for st in range(0, len(Vtr), 256):
            sel = order[st:st + 256]
            V = Vtr[sel]
            yb = ytr[sel.cpu().numpy()]
            if wta:
                s = sim(V)
                for i, d in enumerate(yb):
                    idxs = T.nonzero(REG == d).squeeze(1)
                    empty = idxs[W[idxs].sum(1) <= 1e-6]
                    if len(empty) > 0:
                        W[empty[0]] = V[i]               # fill an empty cell first
                    else:
                        j = idxs[s[i, idxs].argmax()]
                        W[j] += ETA * (V[i] - W[j])
            else:
                prof = T.zeros(len(sel), N, device=DEV)
                for i, d in enumerate(yb):
                    prof[i, d * PER:(d + 1) * PER] = 1.0
                W += ETA * ((prof.T @ V) / max(float(prof.sum()), 1.0) * len(sel)) * MASK
            W.clamp_(min=0.0)
            W *= MASK
            W /= W.norm(dim=1, keepdim=True).clamp(min=1e-9)

    s = sim(Vte)
    if threshold:
        str_ = sim(Vtr)
        lo, hi = 0.0, 1.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if float((str_[:1000] > mid).float().mean()) > ACT:
                lo = mid
            else:
                hi = mid
        th = 0.5 * (lo + hi)
        s = (s > th).float()
    return float((score(s, binary=threshold).argmax(1).cpu().numpy() == yte).mean())


print('cells per action %d | epochs %d' % (PER, EPOCHS), flush=True)
print('', flush=True)
print('  full inputs | winner-takes-all | surface fires | held-out', flush=True)
for full, wta, thr, label in ((True, True, False, 'the 0.95 version'),
                              (True, True, True, '+ surface fires'),
                              (False, True, False, '+ sparse inputs (mine)'),
                              (True, False, False, '+ every cell learns (mine)'),
                              (False, False, True, 'all three (0.71 version)')):
    acc = run(full, wta, thr)
    print('  %-11s | %-15s | %-13s | %.3f   %s' % (
        'yes' if full else 'no', 'yes' if wta else 'no', 'yes' if thr else 'no', acc, label), flush=True)
