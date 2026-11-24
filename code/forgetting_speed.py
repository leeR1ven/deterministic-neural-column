"""Are the weights changing so fast that it forgets what it just learned?

The designer's suspicion: the weights move too quickly, so no sooner learned than forgotten.

This runs online learning and interleaves a retention test:

    for each new sample:  W[label] += eta * (pattern - W[label])     (local, no labels used elsewhere)
    after every block:    measure accuracy on (a) everything seen so far, (b) a held-out set

The learning rate eta is the speed at which the weights move.  If fast weights cause forgetting, the
seen-set accuracy should sag as eta rises, especially at the beginning of a block.

Usage: python forgetting_speed.py [train_n] [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 600
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
BLOCK = 50

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
vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)


def patterns(X):
    with T.no_grad():
        return vis.forward(T.as_tensor(X, device=DEV))[-1]


Ptr = patterns(Xtr)
Pte = patterns(Xte)
print('train %d | test %d | block %d | read-out: one weight vector per class' % (
    N_TRAIN, N_TEST, BLOCK), flush=True)
print('', flush=True)
print('eta   | after 300 seen | after 1000 seen | after 3000 seen | held-out (final)', flush=True)

for eta in (0.01, 0.05, 0.2, 0.5, 1.0):
    W = T.zeros(NC, Ptr.shape[1], device=DEV)
    counts = T.zeros(NC, device=DEV)
    order = np.random.default_rng(7).permutation(len(Ptr))
    seen_acc = {}
    for step, i in enumerate(order):
        c = int(ytr[i])
        p = Ptr[i]
        W[c] += eta * (p - W[c])                     # local update toward this pattern
        counts[c] += 1
        if (step + 1) % BLOCK == 0 and (step + 1) in (300, 1000, 3000):
            Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)
            pred = (Ptr[order[:step + 1]] @ Wn.T).argmax(1).cpu().numpy()
            seen_acc[step + 1] = float((pred == ytr[order[:step + 1]]).mean())
    Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = (Pte @ Wn.T).argmax(1).cpu().numpy()
    held = float((pred == yte).mean())
    print('  %.2f | %.3f          | %.3f           | %.3f           | %.3f' % (
        eta, seen_acc.get(300, float('nan')), seen_acc.get(1000, float('nan')),
        seen_acc.get(3000, float('nan')), held), flush=True)
