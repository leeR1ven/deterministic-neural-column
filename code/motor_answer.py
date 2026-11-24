"""The answer comes out of the MOTOR output, as the design says it should.

The designer's point: the model already has a motor pathway -- different actions mean different digits
-- so there is no need for anything outside the network (no kNN memory, no label cells I invented).

  layer 1   local 3D volume, fixed weights (the library's construction, untouched)
  motor     10 actions, each a distinct group of motor cells (the movement the teacher performs)
  binding   Hebbian: cells active in the concept population are connected to the motor cells that fire
            while the movement is performed
  answer    at test the movement is NOT shown; the concept activity drives the motor cells, and the
            action whose cells receive the most drive is what the model says it saw

The motor groups are given by the task (one movement per digit) -- that is the teacher's side, not a
structure added to the recognition pathway.  Nothing stores pictures; the only memory is the synaptic
weights between the concept population and the motor cells.

Usage: python motor_answer.py [train_n] [test_n] [epochs]
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
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT = 0.20
MOTOR = int(os.environ.get("MOTOR", "400"))
PER = MOTOR // NC                # cells per action

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
    Atr = vis.forward(Xt)[-1]
    Ate = vis.forward(Xe, )[ -1] if False else vis.forward(Xe)[-1]
n1 = Atr.shape[1]
print('layer 1: %d cells, %.1f%% active (fixed)' % (n1, 100 * float(Atr.mean())), flush=True)

# the teacher's movements: action d fires motor cells [d*PER, (d+1)*PER)
ACT_CELLS = [T.arange(d * PER, (d + 1) * PER, device=DEV) for d in range(NC)]

# binding: concept cells -> motor cells
Wm = T.zeros(MOTOR, n1, device=DEV)
ETA = 0.05
g = T.Generator(device=DEV).manual_seed(5)
for ep in range(EPOCHS):
    for i in T.randperm(len(Atr), device=DEV, generator=g).tolist():
        on = T.nonzero(Atr[i] > 0).squeeze(1)
        if on.numel() == 0:
            continue
        tgt = ACT_CELLS[ytr[i]]
        # slow, so that one presentation does not overwrite the last (the designer's point about
        # weights that change too fast)
        Wm *= (1.0 - 0.01)
        Wm[tgt.unsqueeze(1), on.unsqueeze(0)] += ETA


def answer(A):
    """no movement is shown; the drive into each action's region decides"""
    d = A @ Wm.T                                   # (n, motor cells)
    dn = d / (Wm.norm(dim=1) + 1e-9)               # L2, so classes with more weight do not dominate
    scores = T.stack([dn[:, ACT_CELLS[c]].sum(1) for c in range(NC)], 1)
    return scores.argmax(1).cpu().numpy()


tr_acc = float((answer(Atr) == ytr).mean())
te_acc = float((answer(Ate) == yte).mean())
print('', flush=True)
print('  trained on %d images | train %.3f | held-out %.3f' % (N_TRAIN, tr_acc, te_acc), flush=True)
print('  (chance is %.2f)' % (1.0 / NC), flush=True)
