"""Does the 0.951 design generalise to an appearance it has never seen?

0.951 on 1000 held-out pictures is generalisation to new INSTANCES of the same task.  That is real and
it is the standard meaning, but it is weak: the read-out keeps patterns, so it is similarity matching.

The stronger question: train on a spread of appearances and test on angles that were never trained.
Here the training set contains -20, -10, 0, +10, +20 degrees only, and the test set contains -15, -5,
+5, +15 (in between, never seen) plus 30 and 45 (outside the trained range).

Same design as the 0.951 result: local 3D layer (fixed) + motor cells (60 per action), no external
store of pictures.  A nearest-exemplar read-out on the raw pixels gets exactly the same training set,
as the thing to beat.

Usage: python motor_rotation.py [per_angle] [test_n] [epochs]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

PER_ANGLE = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 400
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, M, ETA = 0.20, 60, 0.05
TRAIN_ANGLES = (-20, -10, 0, 10, 20)

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
tr_idx = rng.choice(len(tr), PER_ANGLE, replace=False)
te_idx = rng.choice(len(te), N_TEST, replace=False)
ytr_one = labels(tr, tr_idx)

Xtr, ytr = [], []
for a in TRAIN_ANGLES:
    Xtr.append(render(tr, tr_idx, a))
    ytr.extend(ytr_one)
Xtr = np.concatenate(Xtr, 0)
ytr = np.array(ytr)

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:1000], device=DEV), target=ACT)


def pats(X, bs=2000):
    out = []
    with T.no_grad():
        for i in range(0, len(X), bs):
            out.append(vis.forward(T.as_tensor(X[i:i + bs], device=DEV))[-1])
    return T.cat(out, 0)


Atr = pats(Xtr)
n1 = Atr.shape[1]
print('training on angles %s, %d images each (%d total)' % (TRAIN_ANGLES, PER_ANGLE, len(Xtr)), flush=True)

Wm = T.zeros(NC * M, n1, device=DEV)


def sims(A):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Wn = Wm / Wm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return An @ Wn.T


def answer(A):
    return sims(A).reshape(len(A), NC, M).max(-1).values.argmax(1).cpu().numpy()


g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for st in range(0, len(Atr), 256):
        sel = order[st:st + 256].cpu().numpy()
        A = Atr[sel]
        y = ytr[sel]
        s = sims(A)
        cls = (T.as_tensor(y, device=DEV) * M).unsqueeze(1) + T.arange(M, device=DEV).unsqueeze(0)
        own = Wm.reshape(NC, M, -1)[T.as_tensor(y, device=DEV)]
        free = (own.sum(-1) <= 1e-6)
        who = T.where(free.any(1), free.float().argmax(1), s.gather(1, cls).argmax(1))
        rows = T.as_tensor(y, device=DEV) * M + who
        empty = Wm[rows].sum(1) <= 1e-6
        if bool(empty.any()):
            Wm[rows[empty]] = A[empty]
        nz = ~empty
        if bool(nz.any()):
            Wm.index_add_(0, rows[nz], ETA * (A[nz] - Wm[rows[nz]]))
    print('  epoch %d done' % (ep + 1), flush=True)

print('', flush=True)
print('angle | kind   | motor answer | raw pixels', flush=True)
for ang, kind in ((-15, 'unseen'), (-5, 'unseen'), (5, 'unseen'), (15, 'unseen'), (30, 'outside'), (45, 'outside')):
    Xte = render(te, te_idx, ang)
    yte = labels(te, te_idx)
    Ate = pats(Xte)
    motor = float((answer(Ate) == yte).mean())
    # the raw-pixel reference gets exactly the same training images
    Xn = T.as_tensor(Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9), device=DEV)
    Yn = T.as_tensor(Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-9), device=DEV)
    ii = (Yn @ Xn.T).topk(5, dim=1).indices.cpu().numpy()
    raw = float((np.array([np.bincount(ytr[r], minlength=NC).argmax() for r in ii]) == yte).mean())
    print(' %3d  | %-6s | %.3f        | %.3f' % (ang, kind, motor, raw), flush=True)
