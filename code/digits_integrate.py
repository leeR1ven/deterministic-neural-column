"""Accumulate BEFORE the sparsification, not after.

The previous two attempts accumulated the binary patterns, which cannot work: the threshold has
already discarded what each glance saw, so adding more patterns only averages the loss away.

Here the integration happens in the graded signal, which is how a membrane potential works:

    acc = acc + (glance_t - acc) / TAU          (graded, no threshold)
    then the local stack sees the accumulated image once

Two kinds of glance:
  (a) a fovea without magnification -- sharp in one place, blurred elsewhere.  If the mechanism
      is right, accumulating several of them should RECOVER the single-view accuracy, because
      between them they cover the whole field sharply.
  (b) a MAGNIFIED patch at the fixation, squeezed back into the 28x28 frame.  A magnified patch
      carries more resolution than the same place has in the original frame, so accumulating
      several of these can in principle give MORE than a single view.

Usage: python digits_integrate.py [train_n] [test_n] [act] [mode(a|b)]
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
MODE = sys.argv[4] if len(sys.argv) > 4 else 'a'
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28
NC = 10
TAU = 2.0
SIG = 6.0
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
yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
FIX = [(14, 14), (9, 9), (19, 9), (9, 19), (19, 19)]


def blur(img):
    a = img.reshape(S, S)
    b = a.reshape(S // 4, 4, S // 4, 4).mean(axis=(1, 3))
    return np.repeat(np.repeat(b, 4, axis=0), 4, axis=1)[:S, :S].reshape(-1)


def glance(img, t):
    fy, fx = FIX[t % len(FIX)]
    mask = np.exp(-(((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * SIG ** 2))).reshape(-1).astype(np.float32)
    if MODE == 'a':
        return img * mask + blur(img) * (1 - mask)
    # (b) magnified patch around the fixation, placed back into the frame
    R = 7
    y0, y1 = max(fy - R, 0), min(fy + R, S)
    x0, x1 = max(fx - R, 0), min(fx + R, S)
    patch = img.reshape(S, S)[y0:y1, x0:x1]
    big = np.asarray(T.as_tensor(patch).repeat_interleave(2, 0).repeat_interleave(2, 1).cpu())
    h, w = min(big.shape[0], S), min(big.shape[1], S)
    out = np.zeros((S, S), np.float32)
    oy, ox = (S - h) // 2, (S - w) // 2
    out[oy:oy + h, ox:ox + w] = big[:h, :w]
    return out.reshape(-1)


vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(np.stack([glance(x, 0) for x in Xtr[:400]]), device=DEV), target=ACT)


def integrated(X, glances):
    """write-where-you-look: each fixation writes its sharp patch into the buffer.
    Averaging blurred views cannot help (blur is irreversible), but writing sharp regions
    into the places they belong can reconstruct the whole field."""
    acc = np.zeros((len(X), S * S), np.float32)
    cov = np.zeros((1, S * S), np.float32)
    for t in range(glances):
        fy, fx = FIX[t % len(FIX)]
        mask = np.exp(-(((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * SIG ** 2))).reshape(-1).astype(np.float32)
        m = (mask > 0.3).astype(np.float32)
        g = np.stack([x for x in X])
        acc = acc * (1 - m) + g * m
        cov = cov * (1 - m) + m
    return acc


def acc_of(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


print('MNIST, mode (%s), activation %.2f, TAU %.1f' % (MODE, ACT, TAU), flush=True)
print('glances | accuracy on the integrated image', flush=True)
for g in (1, 2, 3, 5):
    with T.no_grad():
        A = vis.forward(T.as_tensor(integrated(Xtr, g), device=DEV))[-1]
        B = vis.forward(T.as_tensor(integrated(Xte, g), device=DEV))[-1]
    print('  %2d    | %.3f' % (g, acc_of(A, B)), flush=True)
