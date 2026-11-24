"""The full designed chain: image -> local layer -> concept cells -> motor action -> answer.

This is the architecture as the designer describes it, with nothing extra:

  layer 1   local 3D volume, fixed weights (the library's construction)
  concept   cells that see part of layer 1, fire by the threshold rule (no top-k), synapses LEARNED
            by Hebb from what they see (no labels)
  motor     one action per digit; during teaching the movement fires that action's cells
  binding   Hebbian between the concept cells active at that moment and the motor cells of the
            performed action
  answer    at test no movement is given: the concept activity drives the motor cells and whichever
            action's region receives the most drive is the answer

Nothing stores pictures.  There is no kNN and no label layer -- the answer travels through the motor
output, as the design says.

Usage: python motor_concept.py [train_n] [test_n] [epochs] [concept_cells]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
N2 = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, A2 = 0.20, 0.10
FAN2 = 128
MOTOR = 1000
PER = MOTOR // NC

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
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Atr = vis.forward(Xt)[-1]
    Ate = vis.forward(Xe)[-1]
n1 = Atr.shape[1]
print('layer 1: %d cells, %.1f%% active (fixed)' % (n1, 100 * float(Atr.mean())), flush=True)

rng = np.random.default_rng(2)
IDX2 = T.as_tensor(np.stack([rng.choice(n1, FAN2, replace=False) for _ in range(N2)]), device=DEV)
W2 = T.as_tensor((rng.random((N2, FAN2)).astype(np.float32) * 0.1), device=DEV)
W2 = W2 / W2.sum(1, keepdim=True)


def concept(a, th, bs=256):
    out = []
    with T.no_grad():
        for i in range(0, len(a), bs):
            out.append(((a[i:i + bs][:, IDX2] * W2.unsqueeze(0)).sum(-1) > th).float())
    return T.cat(out, 0)


def calibrate(a, target=A2, iters=14):
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(concept(a[:1000], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


ACT_CELLS = [T.arange(d * PER, (d + 1) * PER, device=DEV) for d in range(NC)]
Wm = T.zeros(MOTOR, N2, device=DEV)


def answer(Y2):
    d = Y2 @ Wm.T
    dn = d / (Wm.norm(dim=1) + 1e-9)
    scores = T.stack([dn[:, ACT_CELLS[c]].sum(1) for c in range(NC)], 1)
    return scores.argmax(1).cpu().numpy()


th = calibrate(Atr)
ETA2, ETAM = 0.02, 0.05
g = T.Generator(device=DEV).manual_seed(3)
print('concept: %d cells | motor: %d cells (%d per action)' % (N2, MOTOR, PER), flush=True)

for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for st in range(0, len(Atr), 256):
        sel = order[st:st + 256]
        a = Atr[sel]
        y2 = concept(a, th)
        # concept synapses learn from what they see (unsupervised Hebb)
        hebb = (y2.unsqueeze(-1) * a[:, IDX2]).sum(0) / max(float(y2.sum()), 1.0)
        W2 += ETA2 * (hebb - W2 * y2.mean())
        W2.clamp_(min=0.0)
        W2 /= W2.sum(1, keepdim=True).clamp(min=1e-9)
        # binding to the action being performed (slow, so nothing is overwritten)
        Wm *= (1.0 - 0.01)
        for j, i in enumerate(sel.cpu().numpy()):
            on2 = T.nonzero(y2[j] > 0).squeeze(1)
            if on2.numel() == 0:
                continue
            tgt = ACT_CELLS[ytr[i]]
            Wm[tgt.unsqueeze(1), on2.unsqueeze(0)] += ETAM
    th = calibrate(Atr)
    Ytr2, Yte2 = concept(Atr, th), concept(Ate, th)
    print('  epoch %2d: train %.3f | held-out %.3f' % (
        ep + 1, float((answer(Ytr2) == ytr).mean()), float((answer(Yte2) == yte).mean())), flush=True)
