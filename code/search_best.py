"""Search for the highest accuracy the local 3D structure can reach.

The measuring instrument is the k-nearest-exemplar read-out: the previous round showed that one
prototype per class throws away about ten points, because a digit has several handwriting styles
and averaging them destroys exactly that.

Everything follows the main line: cells, synapses, fixed local weights, threshold sparsity, no
top-k, no hand-chosen cells, no symbolic layer.

Stages (each prints a table; later stages refine around the best earlier result):
    A  activation ratio x cells-per-layer z
    B  depth x fan-in          at the best (activation, z)
    C  k in k-NN               at the best structure
    D  training size           at the best structure
    E  several glances         at the best structure
    F  raw-pixel reference     k-NN on the pixels themselves

Usage: python search_best.py [train_n] [test_n] [stage]
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
STAGE = sys.argv[3] if len(sys.argv) > 3 else 'A'
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10

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


SEED = int(os.environ.get('SEED', '1'))
Xtr, ytr = prep(tr, N_TRAIN, SEED * 101)
Xte, yte = prep(te, N_TEST, SEED * 101 + 7)
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)


def knn(A, B, k):
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    idx = (Bn @ An.T).topk(k, dim=1).indices.cpu().numpy()
    pred = np.array([np.bincount(ytr[idx[i]], minlength=NC).argmax() for i in range(len(B))])
    return float((pred == yte).mean())


def knn_weighted(A, B, k):
    """nearer neighbours count more: weights 1/(rank+1)"""
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    sims, idx = (Bn @ An.T).topk(k, dim=1)
    sims = sims.cpu().numpy(); idx = idx.cpu().numpy()
    pred = []
    for i in range(len(B)):
        w = np.zeros(NC)
        for j in range(k):
            w[ytr[idx[i, j]]] += 1.0 / (j + 1)
        pred.append(int(w.argmax()))
    return float((np.array(pred) == yte).mean())


def run(act, z, depth, fan, k=5):
    dims = [(S, S, 1)] + [(S, S, z * (2 ** i)) for i in range(depth)]
    dims = [(a, b, min(c, 64)) for a, b, c in dims]
    st = Local3DStack(dims, fan_in=fan, seed=SEED, dev=DEV)
    st.calibrate(Xt[:400], target=act)
    with T.no_grad():
        h1 = st.forward(Xt)[-1]
        h2 = st.forward(Xe)[-1]
    return knn(h1, h2, k), float(h1.mean()), h1.shape[1]


if STAGE == 'A':
    print('stage A: activation x z (depth 1, fan-in 4, k=5), train %d' % N_TRAIN, flush=True)
    print('act  \\ z | ' + ' | '.join('z=%-2d' % z for z in (8, 16, 32)), flush=True)
    for act in (0.03, 0.05, 0.10, 0.15, 0.20, 0.35):
        row = []
        for z in (8, 16, 32):
            a, _, _ = run(act, z, 1, 4)
            row.append('%.3f' % a)
        print('  %.2f  | %s' % (act, ' | '.join(row)), flush=True)
elif STAGE == 'B':
    print('stage B: depth x fan-in (k=5)', flush=True)
    for act in (0.10, 0.20, 0.35):
        for depth in (1, 2, 3):
            for fan in (4, 6, 8):
                a, m, cells = run(act, 16, depth, fan)
                print('  act %.2f depth %d fan %d | cells %5d | acc %.3f' % (act, depth, fan, cells, a), flush=True)
elif STAGE == 'C':
    print('stage C: k in k-NN, plus distance weighting', flush=True)
    print('act  | ' + ' | '.join('k=%-2d' % k for k in (1, 3, 5, 9, 15)) + ' | weighted9', flush=True)
    for act in (0.10, 0.20, 0.35):
        st = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, seed=SEED, dev=DEV)
        st.calibrate(Xt[:400], target=act)
        with T.no_grad():
            h1 = st.forward(Xt)[-1]
            h2 = st.forward(Xe)[-1]
        row = ['%.3f' % knn(h1, h2, k) for k in (1, 3, 5, 9, 15)]
        print('  %.2f | %s | %.3f' % (act, ' | '.join(row), knn_weighted(h1, h2, 9)), flush=True)
elif STAGE == 'D':
    print('stage D: training size (act 0.20, z 16, k 5)', flush=True)
    for n in (2000, 4000, 10000):
        Xa, ya = prep(tr, n, SEED * 101)
        ytr = ya
        At = T.as_tensor(Xa, device=DEV)
        st = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, seed=SEED, dev=DEV)
        st.calibrate(At[:400], target=0.20)
        with T.no_grad():
            h1 = st.forward(At)[-1]
            h2 = st.forward(Xe)[-1]
        print('  train %5d | acc %.3f' % (n, knn(h1, h2, 5)), flush=True)
elif STAGE == 'E':
    print('stage E: glances (write-where-you-look), act 0.20, z 16, k 5', flush=True)
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
    FIX = [(14, 14), (9, 9), (19, 9), (9, 19), (19, 19)]
    SIG = 5.0
    MASKS = [(np.exp(-(((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * SIG ** 2))).reshape(-1) > 0.3).astype(np.float32)
             for fy, fx in FIX]
    st = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, seed=SEED, dev=DEV)
    st.calibrate(Xt[:400], target=0.20)
    for g in (1, 2, 3, 5):
        def buf(X):
            a = np.zeros_like(X)
            for t in range(g):
                m = MASKS[t % len(MASKS)]
                a = a * (1 - m) + X * m
            return a
        with T.no_grad():
            h1 = st.forward(T.as_tensor(buf(Xtr), device=DEV))[-1]
            h2 = st.forward(T.as_tensor(buf(Xte), device=DEV))[-1]
        print('  glances %d | acc %.3f' % (g, knn(h1, h2, 5)), flush=True)
elif STAGE == 'F':
    print('stage F: reference, k-NN on raw pixels', flush=True)
    for k in (1, 3, 5, 9):
        print('  raw pixels k=%d | %.3f' % (k, knn(Xt, Xe, k)), flush=True)
