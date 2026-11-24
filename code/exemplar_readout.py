"""Was the prototype read-out the real bottleneck?  And what does the muscle path do without it?

Earlier rounds scored every representation with ONE class prototype.  The designer's suspicion was
that something else was wrong; a nearest-neighbour read-out (which keeps every training exemplar
rather than averaging them into one) shows what the same codes can actually support.

A prototype forces one template per digit, which is exactly what a digit with several handwriting
styles is not.  k-NN keeps all of them -- the "several concepts per digit" the designer described.

This reports, for the visual side and for the motor side:
    prototype / region read-out   (one template per class -- the old scoring)
    k-nearest exemplars           (every training sample kept)

Usage: python exemplar_readout.py [train_n] [test_n] [act] [k]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
K = int(sys.argv[4]) if len(sys.argv) > 4 else 5
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
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


Xtr, ytr = prep(tr, N_TRAIN, 1)
Xte, yte = prep(te, N_TEST, 2)
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)


def knn(A, B, k):
    """vote among the k nearest training patterns (every exemplar kept, not averaged)"""
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    idx = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
    pred = np.array([np.bincount(ytr[idx[i]], minlength=NC).argmax() for i in range(len(B))])
    return float((pred == yte).mean())


def proto(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


print('MNIST | train %d | test %d | activation %.2f | k = %d' % (N_TRAIN, N_TEST, ACT, K), flush=True)
print('', flush=True)
print('representation        | 1 prototype | %d-NN' % K, flush=True)
print('  raw 28x28 pixels    | %.3f       | %.3f' % (proto(Xt, Xe), knn(Xt, Xe, K)), flush=True)

st = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
st.calibrate(Xt[:400], target=ACT)
with T.no_grad():
    Vtr = st.forward(Xt)[-1]
    Vte = st.forward(Xe)[-1]
print('  local 3D stack      | %.3f       | %.3f' % (proto(Vtr, Vte), knn(Vtr, Vte, K)), flush=True)

# motor side: the same visual codes, but the observer listens to the action
ACT_SIG = np.stack([np.clip(np.random.default_rng(100 + c).normal(0.5, 0.25, 64), 0, 1) for c in range(NC)]).astype(np.float32)
mot = Local3DStack([(8, 8, 1), (8, 8, 8)], fan_in=4, seed=7, dev=DEV)
mot.calibrate(T.as_tensor(ACT_SIG, device=DEV), target=MACT)
with T.no_grad():
    M = mot.forward(T.as_tensor(ACT_SIG, device=DEV))[-1]

W = T.zeros(M.shape[1], Vtr.shape[1], device=DEV)
g = T.Generator(device=DEV).manual_seed(9)
for ep in range(4):
    for i in T.randperm(len(Vtr), device=DEV, generator=g).tolist():
        onv = T.nonzero(Vtr[i] > 0).squeeze(1)
        onm = T.nonzero(M[ytr[i]] > 0).squeeze(1)
        if onv.numel() == 0 or onm.numel() == 0:
            continue
        W[onm.unsqueeze(1), onv.unsqueeze(0)] += 1.0


def produce(V):
    """the motor population response, kept continuous: thresholding it here would make every
    pattern look the same (half the cells pass 0.5*max whatever the input is)"""
    d = V @ W.T
    return d / (W.norm(dim=1) + 1e-9)


Ptr, Pte = produce(Vtr), produce(Vte)
region = float(((Pte @ M.T / (M.sum(1).unsqueeze(0) + 1e-9)).argmax(1).cpu().numpy() == yte).mean())
print('  MUSCLE action       | %.3f       | %.3f' % (region, knn(Ptr, Pte, K)), flush=True)
