"""The motor area built the same way as the visual area, as the designer described.

The design:

  visual area   Local3DStack over 28x28x16  (local 3D volume, fixed weights, threshold)
                  -> its output is what the prefrontal loop carries
  motor area    the SAME kind of stack, over a 10x10x16 volume
                  -> its SURFACE (the 10x10 input layer) is structurally the same as the visual
                     input layer, and it is what the prefrontal signal arrives at
  binding       the prefrontal pattern connects to the motor surface by Hebb; during teaching the
                movement drives the region of that action, so the co-active surface cells learn it
  action        each action is a REGION of the motor surface (row d for digit d); after the motor
                area's own local stack has processed it, the region with the most activity is the
                action, hence the digit

Nothing is stored as a list of prototypes: the memory is the synaptic weight between the prefrontal
pattern and the motor surface, plus the fixed local wiring inside the motor area.

Usage: python motor_area.py [train_n] [test_n] [epochs]
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
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
MS = 10                      # motor surface is 10x10
MZ = 16                      # motor volume depth, same as the visual one
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

# ---------------- visual area (unchanged construction) ----------------
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Vtr = vis.forward(Xt)[-1]
    Vte = vis.forward(Xe)[-1]
print('visual area: %d cells, %.1f%% active' % (Vtr.shape[1], 100 * float(Vtr.mean())), flush=True)

# ---------------- motor area: the same kind of stack, smaller volume ----------------
mot = Local3DStack([(MS, MS, 1), (MS, MS, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
# the motor surface receives the prefrontal signal, so calibrate it on such signals
FANM = int(os.environ.get('FANM', '200'))   # inputs each motor-surface cell receives
rng = np.random.default_rng(3)
_mask = np.zeros((MS * MS, Vtr.shape[1]), bool)
for j in range(MS * MS):
    _mask[j, rng.choice(Vtr.shape[1], FANM, replace=False)] = True
MASK = T.as_tensor(_mask, device=DEV)
W = T.zeros(MS * MS, Vtr.shape[1], device=DEV)          # prefrontal -> motor surface
print('motor area: surface %dx%d, volume %dx%dx%d' % (MS, MS, MS, MS, MZ), flush=True)


def motor_surface(V, th):
    """the prefrontal signal drives the motor surface cells through the learned synapses"""
    # cosine: normalise by BOTH the prefrontal pattern's length and each synapse row's length,
    # otherwise the value scales with how active the visual area happens to be
    s = V @ W.T
    denom = (V.norm(dim=1, keepdim=True) * W.norm(dim=1).unsqueeze(0)) + 1e-9
    return (s / denom > th).float()


def calibrate_motor(V, target=ACT, iters=14):
    lo, hi = 0.0, 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(motor_surface(V[:1000], mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# teach: the movement drives the region of that action on the surface (row d for digit d)
print('', flush=True)
print('teaching: the movement fires row d of the motor surface for digit d', flush=True)
g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Vtr), device=DEV, generator=g)
    for st in range(0, len(Vtr), 256):
        sel = order[st:st + 256]
        V = Vtr[sel]
        yb = ytr[sel.cpu().numpy()]
        # the performed movement: a soft profile over the surface, brightest on the action's row
        prof = T.zeros(len(sel), MS * MS, device=DEV)
        for i, d in enumerate(yb):
            prof[i, d * MS:(d + 1) * MS] = 1.0
        # Hebbian: co-active surface cells grow toward the prefrontal pattern
        W += ETA * ((prof.T @ V) / max(float(prof.sum()), 1.0) * len(sel)) * MASK.float()
        W.clamp_(min=0.0)
        W /= W.norm(dim=1, keepdim=True).clamp(min=1e-9)

th = calibrate_motor(Vtr)
# the motor area's own stack must be calibrated like any other area (its default theta is 0.9)
with T.no_grad():
    surf0 = motor_surface(Vtr[:1000], th)
mot.calibrate(surf0, target=ACT)
with T.no_grad():
    o0 = mot.forward(surf0)[-1]
print('DIAG surface active %.1f%% | motor output active %.1f%%' % (
    100 * float(surf0.mean()), 100 * float(o0.mean())), flush=True)


def answer(V):
    surf = motor_surface(V, th)                       # (n, MS*MS) -- the motor surface
    # the motor area's own local stack processes it, exactly like the visual area processes an image
    with T.no_grad():
        out = mot.forward(surf)[-1]                   # (n, MS*MS*MZ)
    # an action is a REGION: the cells whose surface position is in row d
    sc = out.reshape(len(V), MS, MS, MZ).sum(-1).sum(-1)   # (n, MS) activity per row
    return sc.argmax(1).cpu().numpy()


print('', flush=True)
print('  train: %.3f | held-out: %.3f' % (
    float((answer(Vtr) == ytr).mean()), float((answer(Vte) == yte).mean())), flush=True)
