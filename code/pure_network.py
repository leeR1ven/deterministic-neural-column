"""The pure version: recognition done by cells and synapses, no external memory of pictures.

Every earlier number used k-nearest-neighbours as the read-out, and kNN keeps all 60000 training
pictures outside the network.  That is the fair criticism: the decision was never made by the network.

This runs the whole thing inside:

  layer 1   local 3D volume, fixed weights (the library's construction, untouched)
  layer 2   concept cells, each seeing a random subset of layer-1 cells, firing by the threshold rule
            (no top-k); their synapses are LEARNED by Hebb
  layer 3   M3 label cells per class, each connected to every concept cell; when the teacher presents
            a label, the closest label cell of that class moves slowly toward the concept pattern
  read-out  the class whose label cells are most similar to the current concept pattern

Two read-outs are reported side by side on exactly the same concept patterns: the label cells, and a
kNN control.  That separates "the network's representation is bad" from "the label cells are bad".

Usage: python pure_network.py [train_n] [test_n] [epochs] [concept_cells] [label_cells_per_class]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
N2 = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
M3 = int(sys.argv[5]) if len(sys.argv) > 5 else 50
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, A2 = 0.20, 0.10
FAN2 = 128

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

# ---------------- layer 1: fixed local 3D construction ----------------
vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Atr = vis.forward(Xt)[-1]
    Ate = vis.forward(Xe)[-1]
print('layer 1: %d cells, %.1f%% active, fixed weights' % (
    Atr.shape[1], 100 * float(Atr.mean())), flush=True)

# ---------------- layer 2: concept cells with learned synapses ----------------
n1 = Atr.shape[1]
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


# ---------------- layer 3: label cells ----------------
W3 = T.zeros(NC * M3, N2, device=DEV)


def sims(y2):
    yn = y2 / y2.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Wn = W3 / W3.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return yn @ Wn.T                                  # (n, NC*M3)


def classify(y2):
    # the closest label cell of each class decides for that class (summing 50 of them dilutes the
    # one that actually matches)
    s = sims(y2).reshape(len(y2), NC, M3).max(-1).values
    return s.argmax(1).cpu().numpy()


def knn_control(Ytr2, Yte2, k=5):
    An = Ytr2 / Ytr2.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = Yte2 / Yte2.norm(dim=1, keepdim=True).clamp(min=1e-9)
    ii = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
    pred = np.array([np.bincount(ytr[r], minlength=NC).argmax() for r in ii])
    return float((pred == yte).mean())


th = calibrate(Atr)
print('layer 2: %d concept cells, %d inputs each, %.0f%% active' % (N2, FAN2, 100 * A2), flush=True)
print('layer 3: %d label cells (%d per class)' % (NC * M3, M3), flush=True)

ETA2, ETA3 = 0.02, 0.05
g = T.Generator(device=DEV).manual_seed(3)
for ep in range(EPOCHS):
    order = T.randperm(len(Atr), device=DEV, generator=g)
    for st in range(0, len(Atr), 256):
        sel = order[st:st + 256]
        a = Atr[sel]
        y2 = concept(a, th)
        yb = ytr[sel.cpu().numpy()]

        # layer 2: co-active inputs grow (Hebb), then each cell is renormalised
        hebb = (y2.unsqueeze(-1) * a[:, IDX2]).sum(0) / max(float(y2.sum()), 1.0)
        W2 += ETA2 * (hebb - W2 * y2.mean())
        W2.clamp_(min=0.0)
        W2 /= W2.sum(1, keepdim=True).clamp(min=1e-9)

        # layer 3: the closest label cell of the correct class moves slowly toward this pattern
        flat = sims(y2)
        cls_idx = (T.as_tensor(yb, device=DEV) * M3).unsqueeze(1) + T.arange(M3, device=DEV).unsqueeze(0)
        cls_sim = flat.gather(1, cls_idx)
        cls_w = W3.reshape(NC, M3, -1)[T.as_tensor(yb, device=DEV)]
        free = (cls_w.sum(-1) <= 1e-6)
        who = T.where(free.any(1), free.float().argmax(1), cls_sim.argmax(1))
        rows = T.as_tensor(yb, device=DEV) * M3 + who
        empty = W3[rows].sum(1) <= 1e-6
        if bool(empty.any()):
            W3[rows[empty]] = y2[empty]
        nz = ~empty
        if bool(nz.any()):
            W3.index_add_(0, rows[nz], ETA3 * (y2[nz] - W3[rows[nz]]))

    th = calibrate(Atr)
    Ytr2, Yte2 = concept(Atr, th), concept(Ate, th)
    a_label = float((classify(Yte2) == yte).mean())
    a_knn = knn_control(Ytr2, Yte2)
    print('  epoch %2d: label cells %.3f | same patterns + kNN %.3f | active %.1f%%' % (
        ep + 1, a_label, a_knn, 100 * float(Ytr2.mean())), flush=True)
