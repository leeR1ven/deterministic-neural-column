"""Look several times and collapse the MOTOR activity across the glances.

The designer's expectation: with several looks and the motor region collapsing over them, accuracy
should be higher than with one look. That is what this measures.

  each glance        the image shifted slightly (an eye movement); visual layer with a FIXED theta
  motor surface      the prefrontal pattern arrives at the surface, whose cells fire under a FIXED
                     theta, and the motor area's own local stack processes the result
  collapse           the motor activity of every glance is ACCUMULATED, and only then does the region
                     with the most activity decide -- no decision is taken per glance

No theta anywhere is calibrated to a target fraction: that was a disguised top-k and has been removed.
The activation ratios are reported so they can be seen to move with the weights.

Usage: python glances_collapse.py [train_n] [test_n]
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
S, NC, MZ = 28, 10, 16
PER, FANM = 40, 800
TH_V, TH_S = 0.6, float(os.environ.get("THS", "0.6"))   # fixed thresholds, never tuned
ETA, EPOCHS = 0.05, 10

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

vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH_V]
n1 = S * S * MZ


def visual(X):
    with T.no_grad():
        return vis.forward(T.as_tensor(X, device=DEV))[-1]


Vtr = visual(Xtr)
Vte = visual(Xte)
print('visual area: %.1f%% active (fixed theta %.1f)' % (100 * float(Vtr.mean()), TH_V), flush=True)

rng = np.random.default_rng(3)
mask = np.zeros((PER * NC, n1), bool)
for j in range(PER * NC):
    mask[j, rng.choice(n1, FANM, replace=False)] = True
MASK = T.as_tensor(mask, device=DEV).float()
W = T.zeros(PER * NC, n1, device=DEV)
REG = T.as_tensor(np.array([j % NC for j in range(PER * NC)]), device=DEV)   # region = column

mot = Local3DStack([(PER, NC, 1), (PER, NC, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
mot.theta = [0.9]                    # fixed, as the library's default


def surface(V):
    # the same rule as every other layer: score > theta * (the cell's own weight budget).
    # W rows are L1-normalised, so the budget is 1 and the threshold is simply theta.
    s = V @ W.T
    budget = W.abs().sum(1).unsqueeze(0) + 1e-9
    return (s > TH_S * budget).float()


g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    order = T.randperm(len(Vtr), device=DEV, generator=g)
    for st in range(0, len(Vtr), 256):
        sel = order[st:st + 256]
        V = Vtr[sel]
        yb = ytr[sel.cpu().numpy()]
        prof = T.zeros(len(sel), PER * NC, device=DEV)
        for i, d in enumerate(yb):
            prof[i, d::NC] = 1.0
        W += ETA * ((prof.T @ V) / max(float(prof.sum()), 1.0) * len(sel)) * MASK
        W.clamp_(min=0.0)
        W *= MASK
        W /= W.abs().sum(1, keepdim=True).clamp(min=1e-9)

print('motor surface: %.1f%% active (fixed theta %.2f)' % (
    100 * float(surface(Vtr).mean()), TH_S), flush=True)


def region_activity(V):
    """the motor area's own processing, returned as activity per region"""
    surf = surface(V)
    with T.no_grad():
        out = mot.forward(surf)[-1].reshape(len(surf), PER, NC, MZ)
    return out.sum(-1).sum(1)                     # (n, NC): sum over z and over y


def shifted(X, dy, dx):
    a = X.reshape(-1, S, S)
    b = np.zeros_like(a)
    y0, y1 = max(dy, 0), S + min(dy, 0)
    x0, x1 = max(dx, 0), S + min(dx, 0)
    b[:, y0:y1, x0:x1] = a[:, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return b.reshape(len(a), -1)


SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (2, 0), (-2, 0)]
print('', flush=True)
print('glances | accumulated motor activity -> held-out accuracy', flush=True)
acc = None
for k in range(len(SHIFTS)):
    dy, dx = SHIFTS[k]
    Xk = Xte if (dy == 0 and dx == 0) else shifted(Xte, dy, dx)
    act = region_activity(visual(Xk))
    acc = act if acc is None else acc + act
    pred = acc.argmax(1).cpu().numpy()
    print('  %d      | %.3f' % (k + 1, float((pred == yte).mean())), flush=True)

# control: decide from the LAST glance alone (no accumulation)
pred_last = region_activity(visual(Xte if False else shifted(Xte, 0, -1))).argmax(1).cpu().numpy()
print('', flush=True)
print('  one glance alone (no accumulation): %.3f' % float((pred_last == yte).mean()), flush=True)
