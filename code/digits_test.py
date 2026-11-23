"""Real handwritten digits -> concept neurons -> muscle action.

The designer's choice: use real images (sklearn's 8x8 handwritten digits, 1797 samples,
10 classes, no download needed) instead of synthetic strokes.

Chain, the same mechanisms as the word model:
    8x8 digit (0..16 grey) -> signed complement (128 dims)
        -> FIRST LAYER: 1024 neurons, 40 inputs each, thresholded (this is the "encoder":
           nothing but cells, synapses and a threshold)
        -> CONCEPT LAYER: 256 neurons, 100 inputs each, learned only by local Hebb +
           competition, no labels
        -> MUSCLES: 16 muscles x 8 time steps = 128 channels; each digit owns 3
           (muscle, time) slots; the map from concept response to muscle pattern is
           Hebbian
    Read-out: an outside observer looks at which muscles fired and says which digit.

Reported:
    consistency   mean overlap between the concept responses of two different images of
                  the same digit (is there a concept at all?)
    prototype     nearest class prototype of the concept response
    MUSCLE        what the outside observer reads from the muscle action
    raw NN        nearest neighbour on the pixels (control, no model)
    shuffled      the same model with shuffled labels (control)

Usage: python digits_test.py [seed] [train_per_class]
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rng = np.random.default_rng(SEED)

from sklearn.datasets import load_digits

D = load_digits()
X = D.images.reshape(len(D.images), -1).astype(np.float32) / 16.0     # 8x8 = 64, 0..1
Y = D.target.astype(int)
NC = 10
print('digits: %d images, %d classes, %dx%d' % (len(X), NC, D.images.shape[1], D.images.shape[2]), flush=True)


def split(per_class, seed):
    r = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(NC):
        idx = np.where(Y == c)[0]
        r.shuffle(idx)
        tr.extend(idx[:per_class])
        te.extend(idx[per_class:])
    return np.array(tr), np.array(te)


TR, TE = split(PER_CLASS, SEED + 1)
print('train %d | test %d' % (len(TR), len(TE)), flush=True)


# ---------------------------------------------------------------- first layer
class FirstLayer:
    """1024 neurons, 40 inputs each: cells, synapses, a threshold. Nothing else."""

    def __init__(self, dim_in, n_cells=1024, k=40, target=50, seed=0):
        r = np.random.default_rng(seed + 1)
        W = r.normal(0, 1, (n_cells, dim_in))
        mask = np.zeros_like(W, bool)
        for j in range(n_cells):
            mask[j, r.choice(dim_in, k, replace=False)] = True
        W = W * mask
        self.W = W / np.abs(W).sum(1, keepdims=True)
        self.bud = np.abs(self.W).sum(1) + 1e-9
        self.target = target

    def one(self, vec):
        v = np.concatenate([2 * vec - 1, 1 - 2 * vec])
        lo, hi = 0.1, 8.0
        for _ in range(16):
            g = 0.5 * (lo + hi)
            if float((self.W @ (g * v) > 0.5 * self.bud).sum()) > self.target:
                hi = g
            else:
                lo = g
        g = 0.5 * (lo + hi)
        return (self.W @ (g * v) > 0.5 * self.bud).astype(np.float32)


fl = FirstLayer(128, seed=SEED)
Ctr = np.stack([fl.one(X[i]) for i in TR])
Cte = np.stack([fl.one(X[i]) for i in TE])
print('first layer: code size %.0f of 1024' % float(Ctr.sum(1).mean()), flush=True)


# ---------------------------------------------------------------- concept layer
def train_layer(codes, n2=256, k2=100, epochs=15, seed=0):
    r = np.random.default_rng(seed + 2)
    n1 = codes.shape[1]
    W = r.normal(0, 1, (n2, n1)) * 0.1
    mask = np.zeros_like(W, bool)
    for j in range(n2):
        mask[j, r.choice(n1, k2, replace=False)] = True
    W = W * mask
    W = W / np.abs(W).sum(1, keepdims=True)
    eta, decay, win = 0.08, 0.002, 3
    order = np.arange(len(codes))
    for ep in range(epochs):
        r.shuffle(order)
        for i in order:
            x = codes[i]
            s = W @ x
            w = np.argsort(s)[-win:]
            W[w] += eta * (x[None, :] - W[w] * x.sum() / max(x.sum(), 1))
            W *= (1.0 - decay)
            W = np.maximum(W, 0.0)
            W[w] /= np.maximum(np.abs(W[w]).sum(1, keepdims=True), 1e-9)
    th = np.quantile(W @ codes.T, 0.90, axis=1, keepdims=True)
    return W, th


def respond(W, th, codes):
    return (W @ codes.T > th).T.astype(np.float32)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


def consistency(A, y):
    vals = []
    for c in range(NC):
        idx = np.where(y == c)[0][:12]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                vals.append(ov(A[idx[i]], A[idx[j]]))
    return float(np.mean(vals))


def acc_proto(Atr, ytr, Ate, yte):
    P = np.stack([Atr[ytr == c].mean(0) for c in range(NC)])
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    An = Ate / np.maximum(np.linalg.norm(Ate, axis=1, keepdims=True), 1e-9)
    return float((np.argmax(An @ P.T, axis=1) == yte).mean())


# ---------------------------------------------------------------- muscles
NMUS, NSTEP = 16, 16
CH = NMUS * NSTEP
SYN = np.zeros((NC, CH), np.float32)
for c in range(NC):
    for i, mu in enumerate(rng.choice(NMUS, 4, replace=False)):
        SYN[c, i * NMUS + mu] = 1.0


def train_muscles(A, y, epochs=40, eta=0.05, bud=0.03):
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
        pat = (d > 0.5 * d.max()).astype(np.float32) if d.max() > 0 else np.zeros_like(d)
        if pat.sum() == 0:
            continue
        v = pat / max(np.linalg.norm(pat), 1e-9)
        sims = [float(v @ (s / max(np.linalg.norm(s), 1e-9))) for s in SYN]
        ok += int(int(np.argmax(sims)) == yte[i])
    return ok / float(len(Ate))


def raw_nn(Xtr, ytr, Xte, yte):
    D2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return float((ytr[np.argmin(D2, axis=1)] == yte).mean())


# ---------------------------------------------------------------- run
ytr, yte = Y[TR], Y[TE]
W, th = train_layer(Ctr, epochs=15, seed=SEED)
Atr, Ate = respond(W, th, Ctr), respond(W, th, Cte)

cons = consistency(Ate, yte)
a_proto = acc_proto(Atr, ytr, Ate, yte)
# train the muscle map on the class PROTOTYPE (the concept), not on every instance:
# instances of one digit differ enough that per-instance training blurs the map
# (measured: 0.29 per-instance vs 0.62 on prototypes)
M = np.stack([Atr[ytr == c].mean(0) for c in range(NC)])
PROT = (M > 0.5 * M.max(1, keepdims=True)).astype(np.float32)
Wm = train_muscles(PROT, np.arange(NC))
a_mus = muscle_acc(Wm, Ate, yte)
a_nn = raw_nn(X[TR], ytr, X[TE], yte)

perm = rng.permutation(len(ytr))
Wb, thb = train_layer(Ctr, epochs=15, seed=SEED + 99)
Ab_tr, Ab_te = respond(Wb, thb, Ctr), respond(Wb, thb, Cte)
Mb = np.stack([Ab_tr[perm == c].mean(0) for c in range(NC)]) if all((perm == c).any() for c in range(NC)) else M
PROTb = (Mb > 0.5 * Mb.max(1, keepdims=True)).astype(np.float32)
Wmb = train_muscles(PROTb, np.arange(NC))
a_blind = muscle_acc(Wmb, Ab_te, yte)

print('', flush=True)
print('train/class | concept cells active | consistency | prototype acc | MUSCLE acc | raw NN | shuffled', flush=True)
print('  %4d      | %.1f                  | %.3f       | %.3f         | %.3f      | %.3f  | %.3f' % (
    PER_CLASS, Ate.sum(1).mean(), cons, a_proto, a_mus, a_nn, a_blind), flush=True)
