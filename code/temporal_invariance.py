"""Learn invariance from TIME: consecutive views of the same digit should give the same state.

Hebbian learning on single images only finds what co-occurs inside one picture, which is why every
layer above the local stack so far has been a lossy compression of it.  The one signal that carries
information about what should be IGNORED is time: as the eye moves, the digit's position and angle
change, but it stays the same digit.  A representation that is stable across consecutive frames is
therefore invariant to those changes.

The rule is local and needs no labels:

    s_t   = W · a_t                         concept score for frame t
    y_t   = s_t > theta                     sparse firing
    W    += eta * ( y_t^T a_t  -  LAMBDA * (y_t - y_{t-1})^2 gradient )   ... simplified below

In practice the temporal term is applied as a pull of the current state towards the previous one, so
cells that fire for BOTH frames are strengthened and cells that flip are weakened:

    co = y_t * y_{t-1}                      (cells stable across the pair)
    W += eta * ( co^T a_t / n  -  flip^T a_t / n )     flip = |y_t - y_{t-1}|

Everything stays local, no top-k, no labels.

Evaluated on the rotation benchmark from section 21 (train on -20..+20 degrees, test on unseen
angles), where invariance actually shows up.

Usage: python temporal_invariance.py [train_n] [test_n] [epochs]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, K = 0.20, 5
N2 = 2048
FAN2 = 128
A2 = 0.10
ETA = 0.02
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


rng = np.random.default_rng(0)
train_idx = rng.choice(len(tr), N_TRAIN, replace=False)
test_idx = rng.choice(len(te), N_TEST, replace=False)
ytr = np.array([tr[int(k)][1] for k in train_idx])
yte = np.array([te[int(k)][1] for k in test_idx])

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(render(tr, train_idx[:1000], 0), device=DEV), target=ACT)


def l1(X):
    out = []
    with T.no_grad():
        for i in range(0, len(X), 1000):
            out.append(vis.forward(T.as_tensor(X[i:i + 1000], device=DEV))[-1])
    return T.cat(out, 0)


# layer-1 patterns for every training angle, kept as CONSECUTIVE FRAMES of one sequence
F = [l1(render(tr, train_idx, a)) for a in TRAIN_ANGLES]
print('layer 1 ready: %d cells, %d frames' % (F[0].shape[1], len(F)), flush=True)

n1 = F[0].shape[1]
r2 = np.random.default_rng(2)
IDX2 = T.as_tensor(np.stack([r2.choice(n1, FAN2, replace=False) for _ in range(N2)]), device=DEV)
W = T.as_tensor((r2.random((N2, FAN2)).astype(np.float32) * 0.1), device=DEV)
W = W / W.sum(1, keepdim=True)


def score(a):
    return (a[:, IDX2] * W.unsqueeze(0)).sum(-1)


def fire(a, th, bs=256):
    out = []
    with T.no_grad():
        for i in range(0, len(a), bs):
            out.append((score(a[i:i + bs]) > th).float())
    return T.cat(out, 0)


def calibrate(a, target=A2, iters=14):
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(fire(a[:1000], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def knn(Aa, Bb, labels, k=K, bs=200):
    Af = Aa / Aa.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(Bb), bs):
        Bf = Bb[i:i + bs] / Bb[i:i + bs].norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bf @ Af.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(labels[ii[r]], minlength=NC).argmax()))
    return np.array(pred)


th = calibrate(F[0])


def evaluate(tag):
    line = []
    for ang in (-15, -5, 5, 15, 30, 45):
        Xte = render(te, test_idx, ang)
        Ate = l1(Xte)
        p = knn(fire(F[0], th), fire(Ate, th), ytr)
        line.append('%d:%.3f' % (ang, float((p == yte).mean())))
    print('  %-18s %s' % (tag, '  '.join(line)), flush=True)


print('', flush=True)
print('test angles: ', flush=True)
evaluate('before training')

g = T.Generator(device=DEV).manual_seed(3)
for ep in range(EPOCHS):
    for t in range(1, len(F)):
        order = T.randperm(len(F[t]), device=DEV, generator=g)
        for st in range(0, len(F[t]), 256):
            sel = order[st:st + 256]
            a = F[t][sel]
            a_prev = F[t - 1][sel]
            y = fire(a, th)
            y_prev = fire(a_prev, th)
            co = y * y_prev                                  # stable across the pair
            flip = (y - y_prev).abs()                        # cells that switched
            xj = a[:, IDX2]
            hebb = (co.unsqueeze(-1) * xj).sum(0) / max(float(co.sum()), 1.0)
            anti = (flip.unsqueeze(-1) * xj).sum(0) / max(float(flip.sum()), 1.0)
            W += ETA * (hebb - 0.5 * anti)
            W.clamp_(min=0.0)
            W /= W.sum(1, keepdim=True).clamp(min=1e-9)
    th = calibrate(F[0])
    evaluate('epoch %d' % (ep + 1))
