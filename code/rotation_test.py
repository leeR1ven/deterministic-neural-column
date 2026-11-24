"""Does it generalise beyond appearance?  Train on upright digits, test on rotated ones.

Every accuracy reported so far used a test set drawn from exactly the same distribution as training,
so a nearest-exemplar method scores well by definition -- it only has to find something that looks
alike.  That is generalisation to new INSTANCES, not to a new appearance.

This keeps the training set upright and rotates the test digits by an increasing angle.  Two
read-outs are compared on the SAME rotated input:

    raw pixels      nearest exemplar on the pixels
    local 3D stack  nearest exemplar on the fixed local layer's pattern

If the layer builds any abstraction, its accuracy should hold up better than the pixel one as the
angle grows.  If both fall together, the layer contributes nothing beyond "looks alike".

Also included: a control where the training set is rotated the same way as the test set, which
shows how much of any drop is simply the unseen appearance.

Usage: python rotation_test.py [train_n] [test_n]
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
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, Z, K = 0.20, 16, 5

from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def prep(ds, n, seed, angle=0.0):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, S * S), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        if angle:
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        X[i] = (np.asarray(img, np.float32) / 255.0).reshape(-1)
        Y[i] = lab
    return X, Y


def knn(A, B, ytr, k=K, bs=200):
    Af = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs] / B[i:i + bs].norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bf @ Af.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(ytr[ii[r]], minlength=NC).argmax()))
    return float((np.array(pred) == yte).mean())


Xtr, ytr = prep(tr, N_TRAIN, 101)
vis = Local3DStack([(S, S, 1), (S, S, Z)], fan_in=4, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)

with T.no_grad():
    Vtr = vis.forward(T.as_tensor(Xtr, device=DEV))[-1]

print('train: upright only (%d images) | test: rotated | activation %.2f | k=%d' % (N_TRAIN, ACT, K), flush=True)
print('angle | raw pixels | local 3D stack', flush=True)
for ang in (0, 10, 20, 30, 45):
    Xte, yte = prep(te, N_TEST, 108, angle=ang)
    with T.no_grad():
        Vte = vis.forward(T.as_tensor(Xte, device=DEV))[-1]
    raw = knn(T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV), ytr)
    stk = knn(Vtr, Vte, ytr)
    print(' %2d°  | %.3f      | %.3f' % (ang, raw, stk), flush=True)

print('', flush=True)
print('control: training set rotated the same way (so the appearance is not new)', flush=True)
print('angle | raw pixels | local 3D stack', flush=True)
for ang in (20, 45):
    Xtr2, _ = prep(tr, N_TRAIN, 101, angle=ang)
    Xte2, yte2 = prep(te, N_TEST, 108, angle=ang)
    yte = yte2
    with T.no_grad():
        V2 = vis.forward(T.as_tensor(Xtr2, device=DEV))[-1]
        V2e = vis.forward(T.as_tensor(Xte2, device=DEV))[-1]
    print(' %2d°  | %.3f      | %.3f' % (
        ang, knn(T.as_tensor(Xtr2, device=DEV), T.as_tensor(Xte2, device=DEV), ytr), knn(V2, V2e, ytr)), flush=True)
