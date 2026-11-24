"""The best configuration, measured properly: three seeds, visual and motor read-outs.

Best configuration found in the previous round:
    local 3D stack, activation 0.20, z = 16, depth 1, fan-in 4, k = 5 read-out

Here the same configuration is used for the whole chain -- see a digit, and say it with an
action -- and every number is reported per seed and as a mean, because a single seed can be
lucky.

Motor side: an action signal goes through its own local 3D stack (same mechanism), the binding is
Hebbian between co-active cells, and the outsider reads the CONTINUOUS motor population response
(thresholding it would make all patterns look alike).

Usage: python best_config.py [train_n] [test_n]
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
S, NC = 28, 10
ACT, Z, K = 0.20, 16, 5
MACT = 0.10

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


def knn(A, B, ytr, k):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    idx = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
    pred = np.array([np.bincount(ytr[idx[i]], minlength=NC).argmax() for i in range(len(B))])
    return float((pred == yte).mean())


results = []
print('train %d | test %d | activation %.2f | z %d | k %d' % (N_TRAIN, N_TEST, ACT, Z, K), flush=True)
print('seed | raw-pixel kNN | visual kNN | MOTOR kNN | MOTOR region', flush=True)

for seed in (1, 2, 3):
    Xtr, ytr = prep(tr, N_TRAIN, seed * 101)
    Xte, yte = prep(te, N_TEST, seed * 101 + 7)
    Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)

    raw = max(knn(Xt, Xe, ytr, k) for k in (1, 3))

    vis = Local3DStack([(S, S, 1), (S, S, Z)], fan_in=4, seed=seed, dev=DEV)
    vis.calibrate(Xt[:400], target=ACT)
    with T.no_grad():
        Vtr = vis.forward(Xt)[-1]
        Vte = vis.forward(Xe)[-1]
    visual = knn(Vtr, Vte, ytr, K)

    ACT_SIG = np.stack([np.clip(np.random.default_rng(100 + c).normal(0.5, 0.25, 64), 0, 1)
                        for c in range(NC)]).astype(np.float32)
    mot = Local3DStack([(8, 8, 1), (8, 8, 8)], fan_in=4, seed=seed + 50, dev=DEV)
    mot.calibrate(T.as_tensor(ACT_SIG, device=DEV), target=MACT)
    with T.no_grad():
        M = mot.forward(T.as_tensor(ACT_SIG, device=DEV))[-1]

    W = T.zeros(M.shape[1], Vtr.shape[1], device=DEV)
    g = T.Generator(device=DEV).manual_seed(seed + 9)
    for ep in range(4):
        for i in T.randperm(len(Vtr), device=DEV, generator=g).tolist():
            onv = T.nonzero(Vtr[i] > 0).squeeze(1)
            onm = T.nonzero(M[ytr[i]] > 0).squeeze(1)
            if onv.numel() == 0 or onm.numel() == 0:
                continue
            W[onm.unsqueeze(1), onv.unsqueeze(0)] += 1.0

    def produce(V):
        return (V @ W.T) / (W.norm(dim=1) + 1e-9)

    Ptr, Pte = produce(Vtr), produce(Vte)
    motor_knn = knn(Ptr, Pte, ytr, K)
    region = float(((Pte @ M.T / (M.sum(1).unsqueeze(0) + 1e-9)).argmax(1).cpu().numpy() == yte).mean())

    results.append((raw, visual, motor_knn, region))
    print('  %d  | %.3f         | %.3f      | %.3f     | %.3f' % (seed, raw, visual, motor_knn, region), flush=True)

r = np.array(results)
print('mean | %.3f         | %.3f      | %.3f     | %.3f' % tuple(r.mean(0)), flush=True)
print('std  | %.3f         | %.3f      | %.3f     | %.3f' % tuple(r.std(0)), flush=True)
