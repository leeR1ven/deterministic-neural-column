"""Is low accuracy caused by SILENT outputs rather than wrong ones?

The designer's suspicion: at a low activation ratio the model may produce NO action at all.  An
empty pattern cannot be classified, and a nearest-prototype read-out returns class 0 for it, so
every silent case is scored as a wrong answer.  That would make the activation ratio look like the
cause when the real fault is the read-out (or a threshold that is simply too high).

This separates the three outcomes at each activation ratio:
    silent      the pattern is all zeros            -> the model said nothing
    wrong       it said something, and it was wrong
    correct     it said something, and it was right

Accuracy is then reported two ways: over all cases (silent counted as wrong) and over the cases
where the model actually spoke.

Usage: python silent_check.py [train_n] [test_n]
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


Xtr, ytr = prep(tr, N_TRAIN, 1)
Xte, yte = prep(te, N_TEST, 2)
Xt, Xe = T.as_tensor(Xtr, device=DEV), T.as_tensor(Xte, device=DEV)


def classify(Vtr, Vte):
    """nearest class prototype, with the silent cases separated out"""
    P = T.stack([Vtr[T.as_tensor(ytr == c, device=DEV)].mean(0) for c in range(NC)])
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    nz = Vte.sum(1) > 0
    Vn = Vte / Vte.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = (Vn @ P.T).argmax(1).cpu().numpy()
    silent = (~nz).cpu().numpy()
    correct = (pred == yte) & (~silent)
    wrong = (pred != yte) & (~silent)
    return silent.mean(), correct.sum() / len(yte), correct.sum() / max(int((~silent).sum()), 1), wrong.mean()


print('MNIST, local 3D stack, %d training images' % N_TRAIN, flush=True)
print('activation | mean cells | silent%% | correct (of all) | correct (of spoken) | wrong%%', flush=True)
for act in (0.02, 0.05, 0.10, 0.20, 0.35):
    st = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
    st.calibrate(Xt[:400], target=act)
    with T.no_grad():
        h1 = st.forward(Xt)[-1]
        h2 = st.forward(Xe)[-1]
    sil, acc_all, acc_spoken, wrong = classify(h1, h2)
    print('   %.2f    | %8.0f   | %5.1f%%  | %.3f            | %.3f               | %5.1f%%' % (
        act, float(h1.sum(1).mean()), 100 * sil, acc_all, acc_spoken, 100 * wrong), flush=True)

# Also: how many DISTINCT patterns does each layer produce?  A layer whose output barely varies
# with the input would look like low accuracy for a reason that has nothing to do with sparsity.
print('', flush=True)
print('how much does the pattern vary across DIFFERENT images?', flush=True)
for act in (0.02, 0.05, 0.10, 0.20, 0.35):
    st = Local3DStack([(S, S, 1), (S, S, 4), (S, S, 8)], fan_in=4, seed=1, dev=DEV)
    st.calibrate(Xt[:400], target=act)
    with T.no_grad():
        h = st.forward(Xe)[-1]
    u = (h.sum(0) > 0).sum()
    print('   act %.2f: cells ever active %d of %d | mean pairwise overlap %.3f' % (
        act, int(u), h.shape[1], float((h @ h.T).sum() - (h * h).sum()) / (len(h) * (len(h) - 1) * max(float(h.sum(1).mean()), 1.0))), flush=True)
