"""One action, several ways of performing it.

Measured facts from the previous section: the layer-1 patterns of one digit overlap each other only
0.40-0.55, so a single weight vector per action (a class mean) cannot cover them.  The same patterns
read by a nearest exemplar give 0.928, while the one-vector-per-action motor read-out gives 0.764 --
so the loss is in the motor side, not in the representation.

The designer approved the fix: give each action SEVERAL motor cells, each free to hold a different way
of performing it.  That is the motor-side counterpart of "a concept can be several".

  image -> local 3D layer (fixed, unchanged)
        -> motor cells of action d; during teaching the movement fires them, and the one that already
           matches best moves slowly toward the current pattern (an empty cell takes it directly)
  answer  for each action take the drive of its BEST matching motor cell; the action with the highest
          drive is what the model says it saw.  No movement is shown at test time.

No pictures are stored; the memory is the synaptic weights of the motor cells.

Usage: python motor_multi.py [train_n] [test_n] [epochs] [cells_per_action]
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
M = int(sys.argv[4]) if len(sys.argv) > 4 else 30
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
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


SEED = int(os.environ.get("SEED", "1"))
Xtr, ytr = prep(tr, N_TRAIN, SEED * 101)
Xte, yte = prep(te, N_TEST, SEED * 101 + 7)
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Atr = vis.forward(Xt)[-1]
    Ate = vis.forward(Xe)[-1]
n1 = Atr.shape[1]
print('layer 1: %d cells, %.1f%% active' % (n1, 100 * float(Atr.mean())), flush=True)

# motor: NC actions x M cells, each cell a weight vector over layer-1 cells
Wm = T.zeros(NC * M, n1, device=DEV)


def sims(A):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Wn = Wm / Wm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return An @ Wn.T                                   # (n, NC*M)


def answer(A):
    """each action is scored by its single best motor cell"""
    s = sims(A).reshape(len(A), NC, M).max(-1).values
    return s.argmax(1).cpu().numpy()


def teach(A, y):
    s = sims(A)
    cls = (T.as_tensor(y, device=DEV) * M).unsqueeze(1) + T.arange(M, device=DEV).unsqueeze(0)
    cls_sim = s.gather(1, cls)
    own = Wm.reshape(NC, M, -1)[T.as_tensor(y, device=DEV)]
    free = (own.sum(-1) <= 1e-6)
    who = T.where(free.any(1), free.float().argmax(1), cls_sim.argmax(1))
    rows = T.as_tensor(y, device=DEV) * M + who
    empty = Wm[rows].sum(1) <= 1e-6
    if bool(empty.any()):
        Wm[rows[empty]] = A[empty]                     # an empty motor cell takes this way of doing it
    nz = ~empty
    if bool(nz.any()):
        rr = rows[nz]
        Wm.index_add_(0, rr, ETA * (A[nz] - Wm[rr]))   # otherwise it moves slowly toward it
    Wm.clamp_(min=0.0)


print('motor: %d actions x %d cells each' % (NC, M), flush=True)
g = T.Generator(device=DEV).manual_seed(5 + SEED)
for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for st in range(0, len(Atr), 256):
        sel = order[st:st + 256].cpu().numpy()
        teach(Atr[sel], ytr[sel])
    print('  epoch %2d: train %.3f | held-out %.3f' % (
        ep + 1, float((answer(Atr) == ytr).mean()), float((answer(Ate) == yte).mean())), flush=True)
