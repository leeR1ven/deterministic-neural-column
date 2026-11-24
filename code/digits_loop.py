"""Closed loop: look several times, then say what you saw with an action.

The full chain, with no top-k, no hand-drawn regions and no chosen cells:

  1. the eye moves: each fixation writes its sharp patch into a buffer (write-where-you-look;
     a previous round showed that averaging or accumulating binary patterns does not work)
  2. the buffer goes through a local 3D stack -> the visual population V
  3. Hebbian binding: W += outer(motor cells of the action, visual cells active)
  4. an outsider scores each action's REGION by how much drive arrives there

The diagnostic prints three read-outs so that a failure can be attributed:
  prototype      nearest class prototype on V              (what the internal code knows)
  W-cosine       cosine between the visual pattern and each W row, region summed
  S-cosine       the same but with the exact class sums instead of the learned W
                 (if this is high and W-cosine is low, the binding is at fault; if both are
                  low, the read-out itself is at fault)

Usage: python digits_loop.py [seed] [train_n] [test_n] [glances]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
G = int(sys.argv[4]) if len(sys.argv) > 4 else 5
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.35
MACT = 0.10

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


Xtr, ytr = prep(tr, N_TRAIN, SEED * 101)
Xte, yte = prep(te, N_TEST, SEED * 101 + 7)

yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
FIX = [(14, 14), (9, 9), (19, 9), (9, 19), (19, 19), (14, 7), (7, 14)]
SIG = 5.0
MASKS = []
for t in range(len(FIX)):
    fy, fx = FIX[t]
    MASKS.append((np.exp(-(((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * SIG ** 2))).reshape(-1) > 0.3).astype(np.float32))


def bufferise(X, glances):
    acc = np.zeros((len(X), S * S), np.float32)
    for t in range(glances):
        m = MASKS[t % len(MASKS)]
        acc = acc * (1 - m) + X * m
    return acc


Btr = bufferise(Xtr, G)
Bte = bufferise(Xte, G)

vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=SEED, dev=DEV)
vis.calibrate(T.as_tensor(Btr[:400], device=DEV), target=ACT)
with T.no_grad():
    Vtr = vis.forward(T.as_tensor(Btr, device=DEV))[-1]
    Vte = vis.forward(T.as_tensor(Bte, device=DEV))[-1]

ACT_SIG = np.stack([np.clip(np.random.default_rng(100 + c).normal(0.5, 0.25, 64), 0, 1) for c in range(NC)]).astype(np.float32)
mot = Local3DStack([(8, 8, 1), (8, 8, 8)], fan_in=4, seed=SEED + 50, dev=DEV)
mot.calibrate(T.as_tensor(ACT_SIG, device=DEV), target=MACT)
with T.no_grad():
    M = mot.forward(T.as_tensor(ACT_SIG, device=DEV))[-1]      # (10, motor cells)

# Hebbian binding
W = T.zeros(M.shape[1], Vtr.shape[1], device=DEV)
g = T.Generator(device=DEV).manual_seed(SEED + 9)
for ep in range(6):
    for i in T.randperm(len(Vtr), device=DEV, generator=g).tolist():
        onv = T.nonzero(Vtr[i] > 0).squeeze(1)
        onm = T.nonzero(M[ytr[i]] > 0).squeeze(1)
        if onv.numel() == 0 or onm.numel() == 0:
            continue
        W[onm.unsqueeze(1), onv.unsqueeze(0)] += 1.0

S_c = T.stack([Vtr[T.as_tensor(ytr == c, device=DEV)].sum(0) for c in range(NC)])


def region_score(rows, V):
    """an action is a region: score the drive arriving there, cosine-normalised"""
    Rn = rows / rows.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Vn = V / V.norm(dim=1, keepdim=True).clamp(min=1e-9)
    d = Vn @ Rn.T
    return d @ M.T / (M.sum(1).unsqueeze(0) + 1e-9)


def proto(V):
    P = T.stack([Vtr[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Vn = V / V.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return (Vn @ P.T).argmax(1).cpu().numpy()


def a(pred):
    return float((pred == yte).mean())


print('seed %d | train %d | glances %d | motor %d cells at %.0f%%' % (
    SEED, N_TRAIN, G, M.shape[1], 100 * MACT), flush=True)
print('  prototype (internal)   : %.3f' % a(proto(Vte)), flush=True)
print('  W rows  (learned)      : %.3f' % a(region_score(W, Vte).argmax(1).cpu().numpy()), flush=True)
Sn = S_c / S_c.norm(dim=1, keepdim=True).clamp(min=1e-9)
Vn_ = Vte / Vte.norm(dim=1, keepdim=True).clamp(min=1e-9)
print('  S sums  (exact)        : %.3f' % a((Vn_ @ Sn.T).argmax(1).cpu().numpy()), flush=True)
