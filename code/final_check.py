"""One reproducible run of the whole chain.  Everything the objective asks for.

  visual : 28x28 image -> local 3D stack (fixed weights, threshold sparsity, no top-k)
  motor  : an action signal -> its own local 3D stack (same mechanism, no hand-drawn region)
  bind   : Hebbian between the cells that are co-active
  read   : an outsider scores the drive arriving in each action's REGION (region collapse)

Reports, for one seed: the raw-pixel ceiling, the visual accuracy, and the muscle read-out.
Run it with several seeds; the README quotes the mean, not the best.

Usage: python final_check.py [seed] [train_n] [test_n] [visual_act] [motor_act]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
VACT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
MACT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
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


Xtr, ytr = prep(tr, N_TRAIN, SEED * 101)
Xte, yte = prep(te, N_TEST, SEED * 101 + 7)
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)


def acc(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


ceiling = acc(Xt, Xe)

vis = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=SEED, dev=DEV)
vis.calibrate(Xt[:400], target=VACT)
with T.no_grad():
    Vtr = vis.forward(Xt)[-1]
    Vte = vis.forward(Xe)[-1]
visual = acc(Vtr, Vte)

ACT = np.stack([np.clip(np.random.default_rng(100 + c).normal(0.5, 0.25, 64), 0, 1)
                for c in range(NC)]).astype(np.float32)
MZ = int(os.environ.get("MZ", "8"))
mot = Local3DStack([(8, 8, 1), (8, 8, MZ)], fan_in=4, seed=SEED + 50, dev=DEV)
mot.calibrate(T.as_tensor(ACT, device=DEV), target=MACT)
with T.no_grad():
    M = mot.forward(T.as_tensor(ACT, device=DEV))[-1]

# Hebbian binding on co-active cells
W = T.zeros(M.shape[1], Vtr.shape[1], device=DEV)
g = T.Generator(device=DEV).manual_seed(SEED + 9)
for ep in range(6):
    for i in T.randperm(len(Vtr), device=DEV, generator=g).tolist():
        onv = T.nonzero(Vtr[i] > 0).squeeze(1)
        onm = T.nonzero(M[ytr[i]] > 0).squeeze(1)
        if onv.numel() == 0 or onm.numel() == 0:
            continue
        W[onm.unsqueeze(1), onv.unsqueeze(0)] += 1.0

d = Vte @ W.T
dn = d / (W.norm(dim=1) + 1e-9)   # L2, not L1: dividing by the row SUM
                                         # penalises classes with more active cells
score = dn @ M.T / (M.norm(dim=1).unsqueeze(0) + 1e-9)
muscle = float((score.argmax(1).cpu().numpy() == yte).mean())

print('seed %d | motor cells %d | visual act %.2f | motor act %.2f | ceiling %.3f | visual %.3f | MUSCLE %.3f' % (
    SEED, M.shape[1], VACT, MACT, ceiling, visual, muscle), flush=True)
