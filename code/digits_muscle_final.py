"""See a digit, perform an action, and let an outsider read the action.

Everything follows the library's rule now:
  * visual side: 28x28x1 image through a local 3D stack (fixed weights, threshold sparsity)
  * motor side:  an action signal through its own local 3D stack -- the same mechanism, no
                 hand-drawn region.  Nearby cells in the motor volume are near each other by
                 construction, which is what makes a "region" exist at all.
  * binding:     Hebbian between the visual cells and the motor cells that are co-active
                 when the movement is performed.  No prototypes, no top-k, no chosen cells.
  * read-out:    the outsider asks which known action the produced motor population most
                 resembles.  That comparison is the only place a reference appears.

Usage: python digits_muscle_final.py [train_n] [test_n] [visual_act] [motor_act]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
VACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
MACT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28
NC = 10

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


Xtr, ytr = prep(tr, N_TRAIN, 1)
Xte, yte = prep(te, N_TEST, 2)

# ---------------- visual side ----------------
vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
vis.calibrate(T.as_tensor(Xtr[:400], device=DEV), target=VACT)
with T.no_grad():
    Vtr = vis.forward(T.as_tensor(Xtr, device=DEV))[-1]
    Vte = vis.forward(T.as_tensor(Xte, device=DEV))[-1]
print('visual: %d cells | active %.1f%%' % (Vtr.shape[1], 100 * float(Vtr.mean())), flush=True)

# ---------------- motor side: an action is a signal through its own local stack ----------------
ACT_SIGNALS = np.stack([np.clip(np.random.default_rng(100 + c).normal(0.5, 0.25, 64), 0, 1)
                        for c in range(NC)]).astype(np.float32)
mot = Local3DStack([(16, 16, 1), (16, 16, 4)], fan_in=4, seed=7, dev=DEV)
mot.calibrate(T.as_tensor(ACT_SIGNALS, device=DEV), target=MACT)
with T.no_grad():
    M = mot.forward(T.as_tensor(ACT_SIGNALS, device=DEV))[-1]     # (10, motor cells)
print('motor: %d cells | active %.1f%% | pairwise overlap %.2f' % (
    M.shape[1], 100 * float(M.mean()),
    float((M @ M.T).sum() - (M * M).sum()) / max(float((M @ M.T).sum() - (M * M).sum()) + 0.0, 1) if False else
    float(((M @ M.T).sum() - (M * M).sum()) / (NC * (NC - 1))) / max(float(M.sum(1).mean()), 1.0)), flush=True)

# ---------------- Hebbian binding between the co-active populations ----------------
W = T.zeros(M.shape[1], Vtr.shape[1], device=DEV)
ETA = 0.02
g = T.Generator(device=DEV).manual_seed(11)
for ep in range(8):
    order = T.randperm(len(Vtr), device=DEV, generator=g)
    for i in order.tolist():
        v = Vtr[i]
        m = M[ytr[i]]
        onv = T.nonzero(v > 0).squeeze(1)
        onm = T.nonzero(m > 0).squeeze(1)
        if onv.numel() == 0 or onm.numel() == 0:
            continue
        # no decay: accumulate the whole history, so each row reflects the class mean
        # rather than only the last few hundred samples
        W[onm.unsqueeze(1), onv.unsqueeze(0)] += 1.0


def drive(V):
    d = V @ W.T
    return d / (W.sum(1) + 1e-9)          # scale-free per motor cell


def calibrate_motor(V, target):
    lo, hi = 0.0, 2.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        act = float((drive(V) > mid).float().mean())
        if act > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


THM = calibrate_motor(Vtr, MACT)
print('motor read-out theta %.3f -> %.1f%% fire' % (
    THM, 100 * float((drive(Vtr) > THM).float().mean())), flush=True)


def readout(V):
    fire = (drive(V) > THM).float()
    inter = fire @ M.T
    union = fire.sum(1, keepdim=True) + M.sum(1).unsqueeze(0) - inter
    return (inter / (union + 1e-9)).argmax(1).cpu().numpy()


def concept_acc(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


pred = readout(Vte)
print('', flush=True)
print('visual population alone (nearest prototype) : %.3f' % concept_acc(Vtr, Vte), flush=True)
print('MOTOR read-out (what the outsider sees)     : %.3f' % float((pred == yte).mean()), flush=True)
