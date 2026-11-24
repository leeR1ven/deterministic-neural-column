"""Teach it slowly and continuously: small steps should learn, but not forget.

The designer's plan: make the weights hard to change -- a small learning rate -- then keep teaching.
Learning will be slow, but so will forgetting, and continuous teaching should accumulate rather than
overwrite.

    every sample:  W[label] += eta * (pattern - W[label])      with a very small eta
    every 1000:    measure accuracy on a FIXED sample of what it has already seen, and on a
                   held-out set it has never seen

A gentle learning rate on its own cannot stop interference from a single vector per class, so this
also reports a variant that keeps the ordinary memory while blocking the update from pulling a class
vector too far in one step (a cap on how much one sample may move it).

Usage: python teach_slowly.py [train_n] [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 600
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
REPORT = 1000

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
# a FIXED probe of the first 200 training images, so "how much of what it has seen does it still know"
# can be tracked without the probe itself changing
probe, yprobe = Xtr[:200], ytr[:200]

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)


def patterns(X):
    with T.no_grad():
        return vis.forward(T.as_tensor(X, device=DEV))[-1]


Ptr = patterns(Xtr)
Pte = patterns(Xte)
Pprobe = patterns(probe)
print('continuous teaching | train %d | probe %d | held-out %d' % (
    N_TRAIN, len(probe), N_TEST), flush=True)
print('', flush=True)

for eta in (0.02, 0.005, 0.001):
    W = T.zeros(NC, Ptr.shape[1], device=DEV)
    order = np.random.default_rng(7).permutation(len(Ptr))
    line = []
    for step, i in enumerate(order):
        c = int(ytr[i])
        W[c] += eta * (Ptr[i] - W[c])
        if (step + 1) % REPORT == 0:
            Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)
            pr = (Pprobe @ Wn.T).argmax(1).cpu().numpy()
            ho = (Pte @ Wn.T).argmax(1).cpu().numpy()
            line.append('%d:%.2f/%.2f' % (step + 1, float((pr == yprobe).mean()),
                                          float((ho == yte).mean())))
    print('eta %.3f  (probe/held-out)  %s' % (eta, '  '.join(line)), flush=True)
