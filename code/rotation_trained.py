"""Train on a RANGE of rotations, then test on angles it never saw exactly.

The previous test failed a model that had never seen a rotated digit, which says nothing about
abstraction -- a CNN trained on upright digits also fails at 45 degrees.  The fair question is
whether a model that HAS seen a spread of appearances fills in BETWEEN them (interpolation) and
whether it stretches BEYOND them (extrapolation), compared with a nearest-exemplar read-out on the
raw pixels given exactly the same training set.

Training angles: -20, -10, 0, +10, +20 degrees.
Held-out interpolation: -15, -5, +5, +15 degrees   (between the trained angles)
Extrapolation: 30, 45 degrees                      (outside the trained range)

Both read-outs use k nearest exemplars; the only difference is the representation.

Usage: python rotation_trained.py [train_n] [test_n]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 400
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, Z, K = 0.20, 16, 5

from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def render(ds, idx, angle):
    r = np.zeros((len(idx), S * S), np.float32)
    for i, k in enumerate(idx):
        img, _ = ds[int(k)]
        if angle:
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        r[i] = (np.asarray(img, np.float32) / 255.0).reshape(-1)
    return r


def labels(ds, idx):
    return np.array([ds[int(k)][1] for k in idx])


rng = np.random.default_rng(0)
train_idx = rng.choice(len(tr), N_TRAIN, replace=False)
test_idx = rng.choice(len(te), N_TEST, replace=False)
ytr = labels(tr, train_idx)
yte = labels(te, test_idx)

TRAIN_ANGLES = (-20, -10, 0, 10, 20)
Xtr, ytr_aug = [], []
for a in TRAIN_ANGLES:
    Xtr.append(render(tr, train_idx, a))
    ytr_aug.extend(ytr)
Xtr = np.concatenate(Xtr, 0)
ytr_aug = np.array(ytr_aug)

vis = Local3DStack([(S, S, 1), (S, S, Z)], fan_in=4, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)


def patterns(X, bs=2000):
    out = []
    with T.no_grad():
        for i in range(0, len(X), bs):
            out.append(vis.forward(T.as_tensor(X[i:i + bs], device=DEV))[-1])
    return T.cat(out, 0)


def knn_batch(A, B, labels_ref, k=K, bs=200):
    Af = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs] / B[i:i + bs].norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bf @ Af.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(labels_ref[ii[r]], minlength=NC).argmax()))
    return np.array(pred)


print('trained on angles %s, %d images each (%d total)' % (TRAIN_ANGLES, N_TRAIN, len(Xtr)), flush=True)
Vtr = patterns(Xtr)
Xtr_t = T.as_tensor(Xtr, device=DEV)

print('', flush=True)
print('angle | raw pixels | local 3D stack   (trained: -20..+20)', flush=True)
for ang, kind in ((-15, 'interp'), (-5, 'interp'), (5, 'interp'), (15, 'interp'),
                  (30, 'extrap'), (45, 'extrap')):
    Xte = render(te, test_idx, ang)
    pr = knn_batch(Xtr_t, T.as_tensor(Xte, device=DEV), ytr_aug)
    pm = knn_batch(Vtr, patterns(Xte), ytr_aug)
    print(' %2d° %-6s | %.3f      | %.3f' % (
        ang, kind, float((pr == yte).mean()), float((pm == yte).mean())), flush=True)
