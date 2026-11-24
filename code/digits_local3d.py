"""First layer built by the library's rule, not by a random projection.

local3d-similarity's construction (see its README and threednet/model.py):

    every neuron is a point in a 3D volume H x W x Z
    its inputs are ITSELF plus 3 random near neighbours (dx,dy in {-1,0,1}, dz in {-2..2})
    the weights are fixed, never trained
    a neuron fires iff  score > theta * (|bias| + sum |weights|),  one theta per layer

So the image becomes a 28x28x1 volume, and the stack grows it through 28x28x2, 28x28x4,
28x28x8 by local connections only.  The last layer's activation is the code that the
concept layer and the read-out then see.

Compare with the random projection used so far (an earlier round measured first layer 0.860,
concept 0.873 with it).

Usage: python digits_local3d.py [train_n] [test_n]
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
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S = 28
TARGET_ACT = float(os.environ.get('ACT', '0.10'))

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

DIMS = [(S, S, 1), (S, S, 2), (S, S, 4), (S, S, 8)]
stack = Local3DStack(DIMS, fan_in=4, seed=1, dev=DEV)
stack.calibrate(T.as_tensor(Xtr[:400], device=DEV), target=TARGET_ACT)
print('3D stack %s | theta %s | target activation %.0f%%' % (
    DIMS, ['%.2f' % t for t in stack.theta], 100 * TARGET_ACT), flush=True)

with T.no_grad():
    Ltr = stack.forward(T.as_tensor(Xtr, device=DEV))[-1]
    Lte = stack.forward(T.as_tensor(Xte, device=DEV))[-1]
print('last layer: %d cells | active %.0f (%.1f%%)' % (
    Ltr.shape[1], Ltr.sum(1).mean(), 100 * float(Ltr.mean())), flush=True)


def acc(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


def acc_pixels():
    A = T.as_tensor(Xtr, device=DEV)
    B = T.as_tensor(Xte, device=DEV)
    return acc(A, B)


# concept layer on top of the 3D code
N2 = int(os.environ.get('N2', '4096'))
g2 = T.Generator(device=DEV).manual_seed(2)
n1 = Ltr.shape[1]
W2 = T.randn(N2, n1, device=DEV, generator=g2) * 0.1
m2 = T.zeros_like(W2, dtype=T.bool)
for j in range(N2):
    m2[j, T.randperm(n1, device=DEV, generator=g2)[:100]] = True
W2 = W2 * m2
W2 = W2 / W2.abs().sum(1, keepdim=True).clamp(min=1e-9)
for ep in range(3):
    order = T.randperm(len(Ltr), device=DEV, generator=g2)
    for i in order.tolist():
        x = Ltr[i]
        s = W2 @ x
        w = T.topk(s, 3).indices
        W2[w] += 0.15 * (x.unsqueeze(0) - W2[w] * x.sum() / max(float(x.sum()), 1.0))
        W2 *= 0.998
        W2.clamp_(min=0.0)
        W2[w] /= W2[w].abs().sum(1, keepdim=True).clamp(min=1e-9)
TH2 = T.quantile(W2 @ Ltr.T, 0.90, dim=1, keepdim=True)

print('', flush=True)
print('raw pixels            : %.3f' % acc_pixels(), flush=True)
print('3D local layer (last) : %.3f' % acc(Ltr, Lte), flush=True)
print('concept layer on top  : %.3f' % acc((W2 @ Ltr.T > TH2).T.float(),
                                           (W2 @ Lte.T > TH2).T.float()), flush=True)
