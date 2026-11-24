"""Where did the 24 points go?  Measure the motor-area chain stage by stage.

The visual area is untouched; only the motor side changed.  Going from the "each motor cell holds a
whole pattern" version (0.95) to the "motor area built like the visual area" version (0.71) lost about
0.24.  This measures the accuracy after every stage of the chain, so the loss can be attributed:

  stage 1  the prefrontal pattern -> motor surface synapses, CONTINUOUS similarity, best cell per region
  stage 2  the same after thresholding the surface to 0/1  (the surface is a layer of neurons, it fires)
  stage 3  after the motor area's own local 3D stack
  stage 4  after summing activity over each region (that is the action decision)

If stage 1 is already low, the prefrontal->surface synapses are the problem.  If stage 1 is high and
stage 2 low, thresholding is the problem.  If stage 2 is high and stage 3 low, the motor area's stack
is destroying it, and if stage 3 is high and stage 4 low, the region read-out is.

Usage: python where_lost.py [train_n] [test_n] [epochs] [fan_in]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
FANM = int(sys.argv[4]) if len(sys.argv) > 4 else 800
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
MS, MZ = 10, 16
ETA = 0.05

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

vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Vtr = vis.forward(Xt)[-1]
    Vte = vis.forward(Xe)[-1]
n1 = Vtr.shape[1]

rng = np.random.default_rng(3)
mask = np.zeros((MS * MS, n1), bool)
for j in range(MS * MS):
    mask[j, rng.choice(n1, FANM, replace=False)] = True
MASK = T.as_tensor(mask, device=DEV)
W = T.zeros(MS * MS, n1, device=DEV)


def raw_sim(V):
    s = V @ W.T
    denom = (V.norm(dim=1, keepdim=True) * W.norm(dim=1).unsqueeze(0)) + 1e-9
    return s / denom


REG = np.array([j // MS for j in range(MS * MS)])     # row d = action d
REGt = T.as_tensor(REG, device=DEV)


def region_score(sim):
    return T.stack([sim[:, REGt == d].max(1).values for d in range(NC)], 1)


g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Vtr), device=DEV, generator=g)
    for st in range(0, len(Vtr), 256):
        sel = order[st:st + 256]
        V = Vtr[sel]
        yb = ytr[sel.cpu().numpy()]
        prof = T.zeros(len(sel), MS * MS, device=DEV)
        for i, d in enumerate(yb):
            prof[i, d * MS:(d + 1) * MS] = 1.0
        W += ETA * ((prof.T @ V) / max(float(prof.sum()), 1.0) * len(sel)) * MASK.float()
        W.clamp_(min=0.0)
        W /= W.norm(dim=1, keepdim=True).clamp(min=1e-9)

# threshold for the surface, so that it fires like a layer of neurons
sim_tr = raw_sim(Vtr)
lo, hi = 0.0, 1.0
for _ in range(16):
    mid = 0.5 * (lo + hi)
    if float((sim_tr[:1000] > mid).float().mean()) > ACT:
        lo = mid
    else:
        hi = mid
TH = 0.5 * (lo + hi)

mot = Local3DStack([(MS, MS, 1), (MS, MS, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
mot.calibrate((sim_tr[:1000] > TH).float(), target=ACT)


def stages(V):
    sim = raw_sim(V)
    a1 = float((region_score(sim).argmax(1).cpu().numpy() == yte).mean())
    surf = (sim > TH).float()
    a2 = float((region_score(surf).argmax(1).cpu().numpy() == yte).mean())
    with T.no_grad():
        out = mot.forward(surf)[-1].reshape(len(V), MS, MS, MZ)
    # stage 3: best region by the stack's own activity, before collapsing it
    per_row = out.sum(-1).sum(-1)                     # (n, MS)
    a3 = float((per_row.argmax(1).cpu().numpy() == yte).mean())
    # stage 4: the same collapse (identical here) -- reported to make the chain explicit
    a4 = a3
    return a1, a2, a3, a4


a1, a2, a3, a4 = stages(Vte)
print('visual area: %d cells | surface %dx%d, %d inputs per cell | epochs %d' % (
    n1, MS, MS, FANM, EPOCHS), flush=True)
print('', flush=True)
print('  stage 1  continuous similarity, best cell per region : %.3f' % a1, flush=True)
print('  stage 2  after the surface fires (threshold)          : %.3f' % a2, flush=True)
print('  stage 3  after the motor area\'s local 3D stack        : %.3f' % a3, flush=True)
print('', flush=True)
print('  surface activity %.1f%% | motor output activity %.1f%%' % (
    100 * float((raw_sim(Vte) > TH).float().mean()),
    100 * float(mot.forward((raw_sim(Vte) > TH).float())[-1].mean())), flush=True)
