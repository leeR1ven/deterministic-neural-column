"""Other parameters that were never varied.

So far only the volume shape, the activation ratio and the read-out k were swept.  These four were
left at the library's defaults the whole time and could each be responsible:

  A  weight mix      self weight 1.0 against neighbour weights alpha*(0.5..1.0)
  B  bias scale      the constant each cell adds before the threshold
  C  input form      grey levels vs binarised image vs contrast-normalised image
  D  read-out metric cosine vs euclidean vs correlation vs Jaccard

Each is swept alone, at the pinned configuration (activation 0.20, z=8, 10000 training images,
k=5).

Usage: python other_params.py [train_n] [test_n] [which(A|B|C|D|all)]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
S = 28
NC = 10
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
WHICH = sys.argv[3] if len(sys.argv) > 3 else 'all'

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


Xtr0, ytr = prep(tr, N_TRAIN, 101)
Xte0, yte = prep(te, N_TEST, 108)


# ---------------- the layer, built by hand so that every parameter can be varied ----------------
def build(z, alpha, bias_scale, fan=4, seed=1):
    rng = np.random.default_rng(seed)
    dim = (S, S, z)
    yy, xx, zz = np.meshgrid(np.arange(S), np.arange(S), np.arange(z), indexing='ij')
    oc = np.stack([yy.ravel(), xx.ravel(), zz.ravel()], -1)
    n = len(oc)
    idx = np.zeros((n, fan), np.int64)
    w = np.zeros((n, fan), np.float32)
    idx[:, 0] = np.arange(n)
    w[:, 0] = 1.0
    for j in range(1, fan):
        nb = oc.copy()
        nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, S - 1)
        nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, S - 1)
        nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, z - 1)
        idx[:, j] = np.ravel_multi_index(nb.T, dim)
        w[:, j] = (rng.random(n) * 0.5 + 0.5) * alpha
    b = (rng.random(n) * bias_scale).astype(np.float32)
    return idx, w, b


def forward(idx, w, b, z, X, theta):
    n = idx.shape[0]
    plane = S * S
    x = T.zeros(X.shape[0], n, device=DEV)
    x[:, :plane] = T.as_tensor(X, device=DEV)
    IDX = T.as_tensor(idx, device=DEV, dtype=T.long)
    W = T.as_tensor(w, device=DEV)
    B = T.as_tensor(b, device=DEV)
    out = []
    with T.no_grad():
        for i in range(0, len(X), 1000):
            xb = x[i:i + 1000]
            s = (xb[:, IDX] * W.unsqueeze(0)).sum(-1) + B.unsqueeze(0)
            denom = (B.abs() + W.abs().sum(1)).unsqueeze(0)
            out.append((s > theta * denom).float())
    return T.cat(out, 0)


def calibrate(idx, w, b, z, X, target=0.20):
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        act = float(forward(idx, w, b, z, X[:500], mid).mean())
        if act > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def knn(A, B, k, ytr, yte, metric='cos'):
    Af = A.float(); Bf = B.float()
    if metric == 'cos':
        An = Af / Af.norm(dim=1, keepdim=True).clamp(min=1e-9)
        Bn = Bf / Bf.norm(dim=1, keepdim=True).clamp(min=1e-9)
        S_ = Bn @ An.T
        idx = S_.topk(k, dim=1).indices
    else:
        pred = []
        An = Af / Af.norm(dim=1, keepdim=True).clamp(min=1e-9)
        Bn = Bf / Bf.norm(dim=1, keepdim=True).clamp(min=1e-9)
        for i in range(0, len(Bf), 200):
            bb = Bn[i:i + 200]
            if metric == 'euclid':
                d = ((bb[:, None, :] - An[None, :, :]) ** 2).sum(-1) if False else (bb.norm(dim=1, keepdim=True) ** 2 + An.norm(dim=1, keepdim=True).T ** 2 - 2 * bb @ An.T)
                ii = d.topk(k, dim=1, largest=False).indices
            elif metric == 'corr':
                cm = bb @ An.T
                ii = cm.topk(k, dim=1).indices
            else:  # jaccard
                inter = bb @ A[i:i + 200].float().T
                union = bb.sum(1, keepdim=True) + A[i:i + 200].float().sum(1).unsqueeze(0) - inter
                ii = (inter / (union + 1e-9)).topk(k, dim=1).indices
            for r in range(len(ii)):
                pred.append(int(np.bincount(ytr[ii[r].cpu().numpy()], minlength=NC).argmax()))
        return float((np.array(pred) == yte).mean())
    pred = [int(np.bincount(ytr[idx[r].cpu().numpy()], minlength=NC).argmax()) for r in range(len(idx))]
    return float((np.array(pred) == yte).mean())


def run(z=8, alpha=0.01, bias=0.5, Xa=None, Xb=None, metric='cos', k=5):
    Xa = Xtr0 if Xa is None else Xa
    Xb = Xte0 if Xb is None else Xb
    idx, w, b = build(z, alpha, bias)
    th = calibrate(idx, w, b, z, Xa)
    A = forward(idx, w, b, z, Xa, th)
    B = forward(idx, w, b, z, Xb, th)
    return knn(A, B, k, ytr, yte, metric)


print('baseline: activation 0.20, z=8, train %d, k=5' % N_TRAIN, flush=True)
base = run()
print('  baseline                       : %.3f' % base, flush=True)

if WHICH in ('A', 'all'):
    print('A  weight mix (self 1.0 vs neighbour alpha):', flush=True)
    for alpha in (0.001, 0.01, 0.1, 1.0):
        print('   alpha %-6s : %.3f' % (alpha, run(alpha=alpha)), flush=True)

if WHICH in ('B', 'all'):
    print('B  bias scale:', flush=True)
    for bs in (0.0, 0.1, 0.5, 1.0, 2.0):
        print('   bias %-5s : %.3f' % (bs, run(bias=bs)), flush=True)

if WHICH in ('C', 'all'):
    print('C  input form:', flush=True)
    print('   grey levels                   : %.3f' % base, flush=True)
    Xb = (Xtr0 > 0.2).astype(np.float32); Xbb = (Xte0 > 0.2).astype(np.float32)
    print('   binarised at 0.2              : %.3f' % run(Xa=Xb, Xb=Xbb), flush=True)
    Xc = Xtr0 / (Xtr0.mean(1, keepdims=True) + 1e-6); Xcc = Xte0 / (Xte0.mean(1, keepdims=True) + 1e-6)
    print('   contrast normalised           : %.3f' % run(Xa=Xc, Xb=Xcc), flush=True)

if WHICH in ('D', 'all'):
    print('D  read-out metric:', flush=True)
    idx, w, b = build(8, 0.01, 0.5)
    th = calibrate(idx, w, b, 8, Xtr0)
    A = forward(idx, w, b, 8, Xtr0, th)
    B = forward(idx, w, b, 8, Xte0, th)
    for m in ('cos', 'euclid', 'corr', 'jaccard'):
        print('   %-8s : %.3f' % (m, knn(A, B, 5, ytr, yte, m)), flush=True)
