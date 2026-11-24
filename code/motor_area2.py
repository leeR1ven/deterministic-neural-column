"""Motor area built like the visual area, with competition INSIDE each action's region.

The designer's structure:

  visual area   Local3DStack over 28x28x16 (local volume, fixed weights, threshold)
  motor area    the same kind of stack, over a bigger surface; the prefrontal pattern arrives at the
                SURFACE through learned synapses, the surface is structurally like the visual input
                layer, and each action owns a REGION of that surface
  action        the region that ends up most active is the action, hence the digit

Two things were wrong in the first attempt, both measured:

  1. the similarity was not normalised by the prefrontal pattern's length, so every surface cell fired
     (100 % active) and nothing could be decided;
  2. every cell inside a region received the same teaching signal, so all of them converged on the same
     class mean and the region could not tell different writings apart (0.51 with dense inputs).

Fix for (2): competition inside the region, which is local inhibition rather than a global top-k.  When
the movement is performed, only the cells of that region that already match best learn the pattern;
the others stay as they are.  Different cells therefore end up holding different writings of the same
digit, while the action is still identified by WHICH REGION is active.

Usage: python motor_area2.py [train_n] [test_n] [epochs] [winners]
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
WIN = int(sys.argv[4]) if len(sys.argv) > 4 else 3
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
ROWS, COLS = 4, 40                       # each action owns a 4x40 region, ten regions side by side
FANM = int(os.environ.get('FANM', '800'))
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

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Vtr = vis.forward(Xt)[-1]
    Vte = vis.forward(Xe)[-1]
n1 = Vtr.shape[1]
print('visual area: %d cells, %.1f%% active' % (n1, 100 * float(Vtr.mean())), flush=True)

# motor surface: NC regions side by side, each ROWS x COLS cells
MSY, MSX = ROWS, COLS * NC
NCELLS = MSY * MSX
rng = np.random.default_rng(3)
mask = np.zeros((NCELLS, n1), bool)
for j in range(NCELLS):
    mask[j, rng.choice(n1, FANM, replace=False)] = True
MASK = T.as_tensor(mask, device=DEV)
W = T.zeros(NCELLS, n1, device=DEV)

# which columns belong to which action
reg = np.zeros(NCELLS, int)
for j in range(NCELLS):
    col = j % MSX
    reg[j] = min(col // COLS, NC - 1)
REG = T.as_tensor(reg, device=DEV)
print('motor surface: %dx%d = %d cells | %d regions of %d cells | %d inputs each' % (
    MSY, MSX, NCELLS, NC, ROWS * COLS, FANM), flush=True)


def surface(V):
    s = V @ W.T
    denom = (V.norm(dim=1, keepdim=True) * W.norm(dim=1).unsqueeze(0)) + 1e-9
    return s / denom                                   # similarity, no threshold yet


def answer(V, th=None):
    sim = surface(V)
    sc = T.stack([sim[:, REG == d].max(1).values for d in range(NC)], 1)
    return sc.argmax(1).cpu().numpy()


print('', flush=True)
g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Vtr), device=DEV, generator=g)
    for st in range(0, len(Vtr), 256):
        sel = order[st:st + 256]
        V = Vtr[sel]
        yb = ytr[sel.cpu().numpy()]
        sim = surface(V)
        # competition INSIDE the action's own region: only its best-matching cells learn this pattern
        for i, d in enumerate(yb):
            cols = REG == d
            idxs = T.nonzero(cols).squeeze(1)
            vals = sim[i, idxs]
            top = idxs[vals.topk(min(WIN, len(idxs))).indices]
            W[top] += ETA * (V[i].unsqueeze(0) - W[top])
        W.clamp_(min=0.0)
        W *= MASK.float()
        W /= W.norm(dim=1, keepdim=True).clamp(min=1e-9)
    print('  epoch %2d: train %.3f | held-out %.3f' % (
        ep + 1, float((answer(Vtr) == ytr).mean()), float((answer(Vte) == yte).mean())), flush=True)
