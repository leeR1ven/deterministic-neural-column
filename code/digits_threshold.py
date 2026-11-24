"""No top-k anywhere: sparsity comes from each cell's own weight budget.

The library's rule (sensory.py, and THRESHOLD_ACTIVATION.md there):

    active_i  <=>  score_i > theta * (|bias_i| + sum_j |w_ij|)

One theta per layer, calibrated so that about the requested fraction is active.  There is no
sorting and no winner-take-all: a cell fires or not according to its own weights.  Earlier
versions of this pipeline selected the top 3 scorers per input, which is exactly the top-k
the designer rejected.

Chain: 28x28 image -> local 3D stack (fixed weights) -> concept layer (threshold + Hebbian)
      -> nearest class prototype (the measuring instrument).

Usage: python digits_threshold.py [train_n] [test_n] [N2] [target_act]
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
N2 = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
ACT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28

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
NC = 10

# ---------------- first layer: the library's local 3D stack ----------------
DIMS = [(S, S, 1), (S, S, 2), (S, S, 4), (S, S, 8)]
stack = Local3DStack(DIMS, fan_in=4, seed=1, dev=DEV)
stack.calibrate(T.as_tensor(Xtr[:400], device=DEV), target=ACT)
with T.no_grad():
    Htr = stack.forward(T.as_tensor(Xtr, device=DEV))[-1]
    Hte = stack.forward(T.as_tensor(Xte, device=DEV))[-1]
print('3D stack | theta %s | last layer %d cells | active %.1f%%' % (
    ['%.2f' % t for t in stack.theta], Htr.shape[1], 100 * float(Htr.mean())), flush=True)

# ---------------- concept layer: threshold, no top-k ----------------
g = T.Generator(device=DEV).manual_seed(2)
n1 = Htr.shape[1]
W = T.randn(N2, n1, device=DEV, generator=g) * 0.1
mask = T.zeros_like(W, dtype=T.bool)
for j in range(N2):
    mask[j, T.randperm(n1, device=DEV, generator=g)[:100]] = True
W = (W * mask).abs()
W = W / W.sum(1, keepdim=True).clamp(min=1e-9)
B = (T.rand(N2, device=DEV, generator=g) * 0.5).abs()


def responses(codes, theta):
    s = codes @ W.T + B
    denom = B + W.sum(1)                       # each cell's own weight budget
    return (s > theta * denom).float()


def calibrate(codes, target, iters=18):
    lo, hi = 0.0, 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        act = float(responses(codes, mid).mean())
        if act > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


theta = calibrate(Htr, ACT)
print('concept theta %.3f -> %.1f%% active' % (theta, 100 * float(responses(Htr, theta).mean())), flush=True)

# Hebbian on the cells that fired (threshold, not top-k)
ETA = 0.15
for ep in range(3):
    order = T.randperm(len(Htr), device=DEV, generator=g)
    for i in order.tolist():
        x = Htr[i]
        s = x @ W.T + B
        denom = B + W.sum(1)
        fire = s > theta * denom
        if not bool(fire.any()):
            continue
        W[fire] += ETA * x.unsqueeze(0)          # pure Hebbian on the cells that fired
        W *= 0.998
        W.clamp_(min=0.0)
        W[fire] /= W[fire].sum(1, keepdim=True).clamp(min=1e-9)

# the threshold must be recalibrated AFTER learning: Hebbian growth raises every score,
# so a threshold set before training ends up letting more than half the cells fire
theta = calibrate(Htr, ACT)
print('concept theta after training %.3f -> %.1f%% active' % (
    theta, 100 * float(responses(Htr, theta).mean())), flush=True)
Atr, Ate = responses(Htr, theta), responses(Hte, theta)


def acc(A, B_, y):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B_ / B_.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == y).mean())


print('DIAG: after training, active per sample %.1f (%.1f%%) | W row sum mean %.3f | rows zero %d' % (
    Atr.sum(1).mean(), 100*float(Atr.mean()), float(W.sum(1).mean()), int((W.sum(1) < 1e-6).sum())), flush=True)
print('DIAG: mean response overlap same digit %.3f' % float(
    (Atr[T.as_tensor(ytr == 3, device=DEV)][:20] @ Atr[T.as_tensor(ytr == 3, device=DEV)][:20].T).mean() / max(float(Atr.sum(1).mean()), 1.0)), flush=True)
print('', flush=True)
print('raw pixels            : %.3f' % acc(T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV), yte), flush=True)
print('3D local first layer  : %.3f' % acc(Htr, Hte, yte), flush=True)
print('concept (no top-k)    : %.3f' % acc(Atr, Ate, yte), flush=True)
