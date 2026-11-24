"""How dense should the centre of the visual field be?

The designer's question: when the network grows, does the CENTRE of the visual field get more
dense and the periphery sparser, and what degree of that is best?

Foveation here is a radial warp of the 28x28 image: for an output pixel at radius r (normalised
to 1 at the corner), the source radius is r_src = r ** gamma.  gamma = 1 is the flat, uniform
sampling; gamma > 1 means the centre is magnified (sampled finely) and the periphery compressed
(sampled coarsely) -- more "optic nerve" per degree in the middle.

Each gamma is fed through the same local 3D stack, so only the input density changes.

Usage: python digits_fovea.py [train_n] [test_n] [act]
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

from torchvision.datasets import MNIST

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def foveate(img, gamma):
    """radial warp: source radius = (output radius) ** gamma, bilinear sampled"""
    if abs(gamma - 1.0) < 1e-6:
        return img
    c = (S - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
    dx, dy = (xx - c) / c, (yy - c) / c
    r = np.sqrt(dx * dx + dy * dy)
    f = np.where(r > 1e-9, (r ** gamma) / np.maximum(r, 1e-9), 0.0)
    sx = np.clip(c + dx * f * c, 0, S - 1)
    sy = np.clip(c + dy * f * c, 0, S - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, S - 1), np.minimum(y0 + 1, S - 1)
    fx, fy = sx - x0, sy - y0
    return ((img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
             + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy).astype(np.float32))


def prep(ds, n, seed, gamma):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, S * S), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        a = np.asarray(img, np.float32) / 255.0
        X[i] = foveate(a, gamma).reshape(-1)
        Y[i] = lab
    return X, Y


def run(gamma, Xtr, ytr, Xte, yte):
    Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)

    def acc(A, B):
        P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
        P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
        Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())

    st = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
    st.calibrate(Xt[:400], target=ACT)
    with T.no_grad():
        h1 = st.forward(Xt)[-1]
        h2 = st.forward(Xe)[-1]
    return acc(Xt, Xe), acc(h1, h2)


print('MNIST, local 3D stack, activation %.2f' % ACT, flush=True)
print('gamma | centre sampling | raw pixels | stack', flush=True)
for gamma in (1.0, 1.3, 1.7, 2.2, 3.0):
    Xtr, ytr = prep(tr, N_TRAIN, 1, gamma)
    Xte, yte = prep(te, N_TEST, 2, gamma)
    c, a = run(gamma, Xtr, ytr, Xte, yte)
    print('  %.1f | %-15s | %.3f      | %.3f' % (
        gamma, '%.1fx' % (gamma ** 2), c, a), flush=True)
