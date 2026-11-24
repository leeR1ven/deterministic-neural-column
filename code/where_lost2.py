"""Where is the accuracy lost in the motor chain, and does more cells per region help?

Measured with the first version: the prefrontal -> motor-surface synapses only support 0.707, the
surface threshold drops that to 0.359, and the motor area's own local stack recovers it to 0.704.  So
the motor area's structure is fine; the loss is in the first step.

The suspicion: each surface cell sees only a random subset of the visual pattern, so its similarity is
a noisy estimate.  If that is the reason, giving each action MORE surface cells -- each seeing its own
subset -- should raise step 1, because the region is scored by its best cell.

Usage: python where_lost2.py [train_n] [test_n] [epochs] [cells_per_region] [fan_in]
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
PER = int(sys.argv[4]) if len(sys.argv) > 4 else 10
FANM = int(sys.argv[5]) if len(sys.argv) > 5 else 800
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC, MZ = 28, 10, 16
ACT, ETA = 0.20, 0.05
NCELLS = PER * NC

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
mask = np.zeros((NCELLS, n1), bool)
for j in range(NCELLS):
    mask[j, rng.choice(n1, FANM, replace=False)] = True
MASK = T.as_tensor(mask, device=DEV)
W = T.zeros(NCELLS, n1, device=DEV)
# the surface is laid out as (PER, NC); region d is COLUMN d, so it is contiguous in x
REG = np.array([min(j % NC, NC - 1) for j in range(NCELLS)])
REGt = T.as_tensor(REG, device=DEV)


def sim_of(V):
    s = V @ W.T
    return s / ((V.norm(dim=1, keepdim=True) * W.norm(dim=1).unsqueeze(0)) + 1e-9)


def region_score(sim, binary=False):
    # continuous similarity -> the closest cell decides; binary firing -> the region that fires most
    # cells decides (a count), because on 0/1 the maximum is a tie
    if binary:
        return T.stack([sim[:, REGt == d].sum(1) for d in range(NC)], 1)
    return T.stack([sim[:, REGt == d].max(1).values for d in range(NC)], 1)


g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Vtr), device=DEV, generator=g)
    for st in range(0, len(Vtr), 256):
        sel = order[st:st + 256]
        V = Vtr[sel]
        yb = ytr[sel.cpu().numpy()]
        prof = T.zeros(len(sel), NCELLS, device=DEV)
        for i, d in enumerate(yb):
            prof[i, d::NC] = 1.0                      # column d of the (PER, NC) surface
        W += ETA * ((prof.T @ V) / max(float(prof.sum()), 1.0) * len(sel)) * MASK.float()
        W.clamp_(min=0.0)
        W /= W.norm(dim=1, keepdim=True).clamp(min=1e-9)

sim_tr = sim_of(Vtr)
lo, hi = 0.0, 1.0
for _ in range(16):
    mid = 0.5 * (lo + hi)
    if float((sim_tr[:1000] > mid).float().mean()) > ACT:
        lo = mid
    else:
        hi = mid
TH = 0.5 * (lo + hi)

mot = Local3DStack([(PER, NC, 1), (PER, NC, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
mot.calibrate((sim_tr[:1000] > TH).float(), target=ACT)

sim = sim_of(Vte)
a1 = float((region_score(sim).argmax(1).cpu().numpy() == yte).mean())
surf = (sim > TH).float()
a2 = float((region_score(surf, binary=True).argmax(1).cpu().numpy() == yte).mean())
with T.no_grad():
    out = mot.forward(surf)[-1].reshape(len(surf), PER, NC, MZ)
per_col = out.sum(-1).sum(1)   # sum over z and over y: region d is COLUMN x=d
a3 = float((per_col.argmax(1).cpu().numpy() == yte).mean())

print('cells/region %3d | fan-in %5d | stage1 %.3f | stage2 %.3f | stage3 %.3f' % (
    PER, FANM, a1, a2, a3), flush=True)
