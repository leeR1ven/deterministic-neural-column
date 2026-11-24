"""Show it the images it got wrong, let it learn them, and see whether it still gets them wrong.

The designer's instruction: when the model misrecognises a picture, let it learn that picture, then
check whether it still misrecognises it.  Nothing is added to the architecture -- the pipeline is the
plain one (local 3D layer + nearest-exemplar read-out), and "learning a picture" means the model
keeps that picture's pattern together with the right answer, the same way it keeps every training
example.

Also important: does learning the errors damage what it already knew?  So the report separates
  fixed      the error was corrected
  still wrong
  forgotten  something it used to get right became wrong
  overall    accuracy on the whole test set before and after

Usage: python learn_mistake.py [train_n] [test_n] [repeats]
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
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, K = 0.20, 5

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

vis = Local3DStack([(S, S, 1), (S, S, 16)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.calibrate(Xt[:1000], target=ACT)
with T.no_grad():
    Ptr = vis.forward(Xt)[-1]
    Pte = vis.forward(Xe)[-1]


def predict(mem, ymem, Q, k=K):
    """nearest exemplars in memory"""
    M = mem / mem.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Qn = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-9)
    out = []
    for i in range(0, len(Qn), 200):
        ii = (Qn[i:i + 200] @ M.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            out.append(int(np.bincount(ymem[ii[r]], minlength=NC).argmax()))
    return np.array(out)


mem, ymem = Ptr.clone(), ytr.copy()
before = predict(mem, ymem, Pte)
acc_before = float((before == yte).mean())
wrong = np.where(before != yte)[0]
print('train %d | test %d | before learning: %.3f | wrong: %d of %d' % (
    N_TRAIN, N_TEST, acc_before, len(wrong), N_TEST), flush=True)

# ---- let it learn exactly those pictures, once (or REPEATS times) ----
for rep in range(REPEATS):
    mem = T.cat([mem, Pte[wrong]], 0)
    ymem = np.concatenate([ymem, yte[wrong]])

after = predict(mem, ymem, Pte)
fixed = int(((before != yte) & (after == yte)).sum())
forgotten = int(((before == yte) & (after != yte)).sum())
still = int(((before != yte) & (after != yte)).sum())
print('after learning the %d mistakes:' % len(wrong), flush=True)
print('  corrected            : %d of %d  (%.0f%%)' % (
    fixed, len(wrong), 100.0 * fixed / max(len(wrong), 1)), flush=True)
print('  still wrong          : %d' % still, flush=True)
print('  forgotten (was right): %d' % forgotten, flush=True)
print('  overall              : %.3f -> %.3f' % (acc_before, float((after == yte).mean())), flush=True)
