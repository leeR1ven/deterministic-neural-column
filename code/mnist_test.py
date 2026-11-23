"""MNIST: real handwritten digits at scale, through the same chain.

Identical mechanisms to digits_test.py, but with the real MNIST set (60000 train / 10000
test) instead of sklearn's 1797 samples:

    28x28 -> 8x8 by box averaging -> signed complement
          -> FIRST LAYER: 1024 neurons, 40 inputs each, thresholded
          -> CONCEPT LAYER: 256 neurons, local Hebb + competition, no labels
          -> MUSCLES: 16 x 16 = 256 channels, 4 slots per digit
          -> an outside observer reads the digit off the muscles that fired

Speed note: the first layer's gain is calibrated ONCE on 200 images and then reused,
instead of bisecting per image (that would be 16x slower and pointless).

Usage: python mnist_test.py [seed] [train_images] [test_images]
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
IMG = int(sys.argv[4]) if len(sys.argv) > 4 else 8
rng = np.random.default_rng(SEED)

from torchvision.datasets import MNIST
from PIL import Image

root = r'D:\deepseekherness\data'
tr = MNIST(root=root, train=True, download=False)
te = MNIST(root=root, train=False, download=False)


def prep(ds, n, seed):
    r = np.random.default_rng(seed)
    idx = r.choice(len(ds), n, replace=False)
    X = np.zeros((n, IMG * IMG), np.float32)
    Y = np.zeros(n, int)
    for i, k in enumerate(idx):
        img, lab = ds[int(k)]
        small = np.asarray(img.resize((IMG, IMG), Image.BOX), np.float32) / 255.0
        X[i] = small.reshape(-1)
        Y[i] = lab
    return X, Y


Xtr, ytr = prep(tr, N_TRAIN, SEED + 1)
Xte, yte = prep(te, N_TEST, SEED + 2)
NC = 10
print('MNIST: train %d | test %d | %dx%d pooling' % (len(Xtr), len(Xte), IMG, IMG), flush=True)


# ---------------------------------------------------------------- first layer
class FirstLayer:
    def __init__(self, dim_in, n_cells=1024, k=40, seed=0):
        r = np.random.default_rng(seed + 1)
        W = r.normal(0, 1, (n_cells, dim_in))
        mask = np.zeros_like(W, bool)
        for j in range(n_cells):
            mask[j, r.choice(dim_in, k, replace=False)] = True
        W = W * mask
        self.W = W / np.abs(W).sum(1, keepdims=True)
        self.bud = np.abs(self.W).sum(1) + 1e-9
        self.gain = 1.0

    def calibrate(self, X, target=50, n=200):
        r = np.random.default_rng(0)
        idx = r.choice(len(X), min(n, len(X)), replace=False)
        best, bg = 1e9, 1.0
        for g in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
            tot = 0.0
            for i in idx:
                v = np.concatenate([2 * X[i] - 1, 1 - 2 * X[i]])
                tot += float((self.W @ (g * v) > 0.5 * self.bud).sum())
            m = tot / len(idx)
            if abs(m - target) < best:
                best, bg = abs(m - target), g
        self.gain = bg
        return bg

    def many(self, X):
        V = np.concatenate([2 * X - 1, 1 - 2 * X], axis=1)
        return (self.W @ (self.gain * V).T > 0.5 * self.bud[:, None]).T.astype(np.float32)


fl = FirstLayer(IMG * IMG * 2, seed=SEED)
g = fl.calibrate(Xtr)
Ctr = fl.many(Xtr)
Cte = fl.many(Xte)
print('first layer: gain %.1f | code size %.0f of 1024' % (g, Ctr.sum(1).mean()), flush=True)


# ---------------------------------------------------------------- concept layer
def train_layer(codes, n2=256, k2=100, epochs=4, seed=0):
    r = np.random.default_rng(seed + 2)
    n1 = codes.shape[1]
    W = r.normal(0, 1, (n2, n1)) * 0.1
    mask = np.zeros_like(W, bool)
    for j in range(n2):
        mask[j, r.choice(n1, k2, replace=False)] = True
    W = W * mask
    W = W / np.abs(W).sum(1, keepdims=True)
    eta, decay, win = 0.15, 0.002, 3
    order = np.arange(len(codes))
    for ep in range(epochs):
        r.shuffle(order)
        for i in order:
            x = codes[i]
            s = W @ x
            w = np.argpartition(s, -win)[-win:]
            W[w] += eta * (x[None, :] - W[w] * x.sum() / max(x.sum(), 1))
            W *= (1.0 - decay)
            np.maximum(W, 0.0, out=W)
            W[w] /= np.maximum(np.abs(W[w]).sum(1, keepdims=True), 1e-9)
    th = np.quantile(W @ codes.T, 0.90, axis=1, keepdims=True)
    return W, th


def respond(W, th, codes):
    return (W @ codes.T > th).T.astype(np.float32)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


def acc_proto(Atr, ytr, Ate, yte):
    P = np.stack([Atr[ytr == c].mean(0) for c in range(NC)])
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    An = Ate / np.maximum(np.linalg.norm(Ate, axis=1, keepdims=True), 1e-9)
    return float((np.argmax(An @ P.T, axis=1) == yte).mean())


NMUS, NSTEP = 16, 16
CH = NMUS * NSTEP
SYN = np.zeros((NC, CH), np.float32)
for c in range(NC):
    for i, mu in enumerate(rng.choice(NMUS, 4, replace=False)):
        SYN[c, i * NMUS + mu] = 1.0


def train_muscles(A, y, epochs=30, eta=0.08, bud=0.03):
    Wm = np.zeros((CH, A.shape[1]), np.float32)
    for ep in range(epochs):
        for i in range(len(A)):
            outer = SYN[y[i]][:, None] * A[i][None, :]
            Wm = np.clip(Wm + eta * (1.0 - Wm) * outer, 0.0, 1.0)
            Wm = Wm * (bud / np.maximum(Wm.sum(1, keepdims=True), 1e-9))
    return Wm


def muscle_acc(Wm, Ate, yte):
    ok = 0
    for i in range(len(Ate)):
        d = Wm @ Ate[i]
        if d.max() <= 0:
            continue
        pat = (d > 0.5 * d.max()).astype(np.float32)
        v = pat / max(np.linalg.norm(pat), 1e-9)
        sims = [float(v @ (s / max(np.linalg.norm(s), 1e-9))) for s in SYN]
        ok += int(int(np.argmax(sims)) == yte[i])
    return ok / float(len(Ate))


def raw_nn(Xtr, ytr, Xte, yte, k=200):
    """nearest neighbour over a sample of the training set (keeps it fast)"""
    r = np.random.default_rng(0)
    idx = r.choice(len(Xtr), min(k, len(Xtr)), replace=False)
    D = ((Xte[:, None, :] - Xtr[idx][None, :, :]) ** 2).sum(-1)
    return float((ytr[idx][np.argmin(D, axis=1)] == yte).mean())


# ---------------------------------------------------------------- run
W, th = train_layer(Ctr, epochs=4, seed=SEED)
Atr, Ate = respond(W, th, Ctr), respond(W, th, Cte)
a_proto = acc_proto(Atr, ytr, Ate, yte)
M = np.stack([Atr[ytr == c].mean(0) for c in range(NC)])
PROT = (M > 0.5 * M.max(1, keepdims=True)).astype(np.float32)
Wm = train_muscles(PROT, np.arange(NC))
a_mus = muscle_acc(Wm, Ate, yte)
a_nn = raw_nn(Xtr, ytr, Xte, yte)

perm = rng.permutation(len(ytr))
Wb, thb = train_layer(Ctr, epochs=4, seed=SEED + 99)
Ab_tr, Ab_te = respond(Wb, thb, Ctr), respond(Wb, thb, Cte)
Mb = np.stack([Ab_tr[perm == c].mean(0) if (perm == c).any() else M[c] for c in range(NC)])
PROTb = (Mb > 0.5 * Mb.max(1, keepdims=True)).astype(np.float32)
Wmb = train_muscles(PROTb, np.arange(NC))
a_blind = muscle_acc(Wmb, Ab_te, yte)

print('', flush=True)
print('train | concept | MUSCLE | rawNN | shuffled', flush=True)
print('  %5d | %.3f   | %.3f  | %.3f | %.3f' % (N_TRAIN, a_proto, a_mus, a_nn, a_blind), flush=True)
