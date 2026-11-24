"""Saccades with a fovea: each glance brings NEW information, not a shifted copy.

The previous attempt shifted the whole image, which adds no information -- it only blurs.  A real
eye movement brings the high-resolution fovea to a different place, so each glance resolves a
different part of the shape.  That is what the accumulation can exploit.

Each glance t:
    mask_t   = a Gaussian around the t-th fixation point
    V_t      = sharp inside the mask, blurred outside   (a fovea)
    m_t      = m_{t-1} + (V_t - m_{t-1}) / TAU          (the trace, inside the loop)

Fixations cycle over centre, corners and edges of the 28x28 field.

Usage: python digits_saccade.py [train_n] [test_n] [act] [sigma]
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
SIG = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28
NC = 10
TAU = 3.0

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

# a coarse, low-resolution version of the whole field (what the periphery sees)
def blur(img):
    a = img.reshape(S, S)
    b = a.reshape(S // 4, 4, S // 4, 4).mean(axis=(1, 3))
    return np.repeat(np.repeat(b, 4, axis=0), 4, axis=1)[:S, :S].reshape(-1)


FIX = [(14, 14), (8, 8), (20, 8), (8, 20), (20, 20), (14, 6), (6, 14)]
yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')


def glance(img, t):
    fy, fx = FIX[t % len(FIX)]
    mask = np.exp(-(((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * SIG ** 2))).reshape(-1).astype(np.float32)
    return img * mask + blur(img) * (1 - mask)


vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
center = np.stack([glance(x, 0) for x in Xtr[:400]])
vis.calibrate(T.as_tensor(center, device=DEV), target=ACT)


def state(X, glances):
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


print('MNIST, fovea sigma %.1f, activation %.2f, trace TAU %.1f' % (SIG, ACT, TAU), flush=True)
print('glances | accumulated state acc', flush=True)
for g in (1, 2, 3, 5, 7):
    print('  %2d    | %.3f' % (g, acc(state(Xtr, g), state(Xte, g))), flush=True)
