"""Different activation ratios per layer: dense near the input, sparse near the front.

The designer's point: the layer closest to the visual input should carry more information
(high activation), and the ratio should fall as you move towards the prefrontal end, where
sparsity is what separates things.

The library calibrates ONE theta per layer but with the same target everywhere.  This lets
each layer have its own target and measures the accuracy at every layer.

Usage: python digits_gradient.py [train_n] [test_n] [targets like 0.35,0.20,0.10]
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
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28
NC = 10

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
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)


def acc(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


def calibrate_per_layer(stack, code, targets):
    """one theta per layer, each with its own target activation"""
    with T.no_grad():
        for l in range(len(stack.layers)):
            lo, hi = 0.0, 2.0
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                stack.theta[l] = mid
                act = float(stack.forward(code)[l + 1].mean())
                if act > targets[l]:
                    lo = mid
                else:
                    hi = mid
            stack.theta[l] = 0.5 * (lo + hi)


DIMS = [(S, S, 1), (S, S, 2), (S, S, 4), (S, S, 8)]
print('raw pixels: %.3f' % acc(Xt, Xe), flush=True)
print('', flush=True)
print('targets            | active per layer | accuracy per layer', flush=True)
for targets in ([0.35, 0.20, 0.10],
                [0.35, 0.15, 0.05],
                [0.50, 0.20, 0.05],
                [0.10, 0.10, 0.10]):
    st = Local3DStack(DIMS, fan_in=4, seed=1, dev=DEV)
    calibrate_per_layer(st, Xt[:600], targets)
    with T.no_grad():
        htr = st.forward(Xt)
        hte = st.forward(Xe)
    acts = ['%.0f%%' % (100 * float(htr[l + 1].mean())) for l in range(3)]
    accs = ['%.3f' % acc(htr[l + 1], hte[l + 1]) for l in range(3)]
    print('  %-16s | %-16s | %s' % ('/'.join('%.2f' % t for t in targets),
                                    ' / '.join(acts), ' / '.join(accs)), flush=True)
