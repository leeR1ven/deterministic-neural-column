"""Three questions about why the accuracy is low, answered by measurement.

  1. When the model sees different pictures of the SAME digit, do the active neurons fall in one
     region?  Is the firing distributed like a bell?
  2. How similar are those patterns to each other, compared with patterns of other digits?
  3. Is the low accuracy explained by the MOTOR area, or by the representation before it?

Every number here comes from the same fixed local 3D layer used in the last section.
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC, Z = 28, 10, 16
ACT = 0.20
DIGIT = 3
N_PER = 300

from torchvision.datasets import MNIST

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)


def load(n_per_digit):
    X, Y = [], []
    for d in range(NC):
        idx = np.where(np.array(tr.targets) == d)[0][:n_per_digit]
        for k in idx:
            img, lab = tr[int(k)]
            X.append((np.asarray(img, np.float32) / 255.0).reshape(-1))
            Y.append(lab)
    return np.array(X, np.float32), np.array(Y)


X, Y = load(N_PER)
vis = Local3DStack([(S, S, 1), (S, S, Z)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(X[:1000], device=DEV), target=ACT)
with T.no_grad():
    P = (vis.forward(T.as_tensor(X, device=DEV))[-1] > 0.5)
P = P.cpu().numpy().astype(np.float32)
print('%d images (%d per digit), layer 1 = %d cells (28x28x%d), %.1f%% active' % (
    len(X), N_PER, P.shape[1], Z, 100 * P.mean()), flush=True)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


# ---------------- 1. is there a stable region, and is the firing bell-shaped? ----------------
sel = Y == DIGIT
Q = P[sel]
freq = Q.mean(0)                                   # how often each cell fires for this digit
active = freq[freq > 0]
print('', flush=True)
print('QUESTION 1  a fixed region, and a bell shape?', flush=True)
print('  cells that fire for every picture of the digit : %d of %d' % (
    int((freq > 0.9).sum()), P.shape[1]), flush=True)
print('  cells that fire for more than half            : %d' % int((freq > 0.5).sum()), flush=True)
print('  cells that fire for 10-90%% of the pictures    : %d' % int(((freq > 0.1) & (freq < 0.9)).sum()), flush=True)
qs = np.quantile(active, [0.1, 0.25, 0.5, 0.75, 0.9])
print('  firing-frequency deciles (10/25/50/75/90%%): %.2f %.2f %.2f %.2f %.2f' % tuple(qs), flush=True)
hist, edges = np.histogram(active, bins=5, range=(0, 1))
print('  histogram over [0,1]: %s' % (hist.tolist(),), flush=True)

# where in the volume do the active cells sit?  centre of mass per picture
idx = np.nonzero(Q)
yy = idx[1] // (S * Z)
xx = (idx[1] % (S * Z)) // Z
zz = idx[1] % Z
per_img = np.array([Q[i].reshape(S, S, Z).sum() for i in range(len(Q))])
cy = [(np.arange(S * S * Z).reshape(S, S, Z).sum(axis=(1, 2))[None] * 0)]  # placeholder
com = []
for i in range(len(Q)):
    v = Q[i].reshape(S, S, Z)
    tot = v.sum()
    if tot == 0:
        continue
    ys, xs, zs = np.mgrid[0:S, 0:S, 0:Z]
    com.append([(ys * v).sum() / tot, (xs * v).sum() / tot, (zs * v).sum() / tot])
com = np.array(com)
print('  centre of mass across pictures of the digit  : y %.1f+-%.1f  x %.1f+-%.1f  z %.1f+-%.1f' % (
    com[:, 0].mean(), com[:, 0].std(), com[:, 1].mean(), com[:, 1].std(),
    com[:, 2].mean(), com[:, 2].std()), flush=True)

# ---------------- 2. how similar are same-digit patterns? ----------------
print('', flush=True)
print('QUESTION 2  how similar are the patterns?', flush=True)
for d in range(NC):
    ii = np.where(Y == d)[0][:60]
    w = [ov(P[a], P[b]) for a in ii for b in ii if a < b]
    o = [ov(P[a], P[b]) for a in ii[:20] for b in np.where(Y != d)[0][:40]]
    print('  digit %d: same-digit overlap %.3f | other-digit %.3f' % (
        d, float(np.mean(w)), float(np.mean(o))), flush=True)

# ---------------- 3. is the motor the bottleneck? ----------------
print('', flush=True)
print('QUESTION 3  motor area or representation?', flush=True)
Ptr = T.as_tensor(P[Y != DIGIT] if False else P, device=DEV)
An = Ptr / Ptr.norm(dim=1, keepdim=True).clamp(min=1e-9)
ii = (An @ An.T).topk(6, dim=1).indices[:, 1:].cpu().numpy()
pred = np.array([np.bincount(Y[r], minlength=NC).argmax() for r in ii])
print('  same patterns, nearest-neighbour read-out : %.3f' % float((pred == Y).mean()), flush=True)
print('  the motor read-out measured before        : 0.764', flush=True)
