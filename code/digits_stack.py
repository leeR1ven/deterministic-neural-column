"""The whole column as the library builds it: local 3D volumes, fixed weights, no top-k.

Earlier attempts mixed two things that do not belong together:
  * a RANDOM projection for the first layer (not the library's rule), and
  * a top-k winner rule inside the concept layer (rejected by the designer).
Replacing top-k with a plain threshold made every cell behave the same way, because without
competitive differentiation all cells learn the same average pattern.

The library's own answer is different: the weights are FIXED, and what separates cells is their
CONNECTIVITY -- each neuron sees itself plus three near neighbours in a 3D volume.  Sparsity
comes from each cell's own weight budget, with one theta per layer.  So this builds the whole
chain that way and reads the accuracy off each layer.

Usage: python digits_stack.py [train_n] [test_n] [target_act]
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
ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
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


def acc(A, B):
    P = T.stack([A[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Bn = B / B.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return float(((Bn @ P.T).argmax(1).cpu().numpy() == yte).mean())


print('raw pixels: %.3f' % acc(T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)), flush=True)
print('', flush=True)
print('depth | layer sizes                  | active%% | accuracy', flush=True)
for nlayers in (2, 3, 4, 5):
    zs = [1] + [2 ** i for i in range(1, nlayers)]
    dims = [(S, S, z) for z in zs]
    stack = Local3DStack(dims, fan_in=4, seed=1, dev=DEV)
    stack.calibrate(T.as_tensor(Xtr[:400], device=DEV), target=ACT)
    with T.no_grad():
        hs_tr = stack.forward(T.as_tensor(Xtr, device=DEV))
        hs_te = stack.forward(T.as_tensor(Xte, device=DEV))
    last = acc(hs_tr[-1], hs_te[-1])
    print('  %d   | %-28s | %5.1f%%  | %.3f' % (
        nlayers - 1, str(dims), 100 * float(hs_tr[-1].mean()), last), flush=True)
    if nlayers == 4:
        print('', flush=True)
        print('  per-layer accuracy for %s:' % str(dims), flush=True)
        for l in range(1, len(hs_tr)):
            print('    L%d: %d cells | active %.1f%% | acc %.3f' % (
                l, hs_tr[l].shape[1], 100 * float(hs_tr[l].mean()),
                acc(hs_tr[l], hs_te[l])), flush=True)
        print('', flush=True)
