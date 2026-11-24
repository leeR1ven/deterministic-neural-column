"""Continuous viewing during TRAINING: each image is a short sequence, integrated in the loop.

All earlier training used single static frames.  Here every training image becomes a sequence of
frames (small rotations), and the sample's representation is the trace accumulated inside the loop:

    m_t = m_{t-1} + (V_t - m_{t-1}) / TAU

where V_t is the pattern produced by the t-th frame.  The same procedure is applied to the test
images, using freshly drawn frames from the same distribution.

The control gets exactly the same frames but accumulates them in PIXEL space instead:

    b_t = b_{t-1} + (frame_t - b_{t-1}) / TAU,   then one pass through the layer

So the only difference is where the integration happens.  If continuous viewing is worth something,
the loop-integrated version should hold up better on frames drawn differently from training.

Usage: python continuous_train.py [train_n] [test_n] [frames] [tau]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 500
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 5
TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, NC = 28, 10
ACT, Z, K = 0.20, 16, 5
SPREAD = 15.0        # frames are rotated within +-15 degrees

from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
te = MNIST(root=r'D:\deepseekherness\data', train=False, download=False)


def frame(ds, k, angle):
    img, lab = ds[int(k)]
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return (np.asarray(img, np.float32) / 255.0).reshape(-1), lab


def sequence(ds, idx, rng, frames=FRAMES):
    """a sequence of frames for every index, with its own random angles"""
    out = np.zeros((len(idx), frames, S * S), np.float32)
    labs = np.zeros(len(idx), int)
    for i, k in enumerate(idx):
        angs = rng.uniform(-SPREAD, SPREAD, frames)
        for t in range(frames):
            out[i, t], labs[i] = frame(ds, k, angs[t])
    return out, labs


rng = np.random.default_rng(0)
train_idx = rng.choice(len(tr), N_TRAIN, replace=False)
test_idx = rng.choice(len(te), N_TEST, replace=False)

print('continuous viewing: %d frames per sample, rotations within +-%.0f deg, TAU %.1f' % (
    FRAMES, SPREAD, TAU), flush=True)


def trace(X, tau=TAU):
    """the loop's slow state over the frames of each sequence"""
    m = np.zeros_like(X[:, 0])
    for t in range(X.shape[1]):
        m = m + (X[:, t] - m) / tau
    return m


def make_layer():
    vis = Local3DStack([(S, S, 1), (S, S, Z)], fan_in=4, seed=1, dev=DEV)
    return vis


def patterns(vis, X, bs=1000):
    out = []
    with T.no_grad():
        for i in range(0, len(X), bs):
            out.append(vis.forward(T.as_tensor(X[i:i + bs], device=DEV))[-1])
    return T.cat(out, 0)


def knn(A, B, yref, k=K, bs=200):
    Af = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pred = []
    for i in range(0, len(B), bs):
        Bf = B[i:i + bs] / B[i:i + bs].norm(dim=1, keepdim=True).clamp(min=1e-9)
        ii = (Bf @ Af.T).topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(ii)):
            pred.append(int(np.bincount(yref[ii[r]], minlength=NC).argmax()))
    return np.array(pred)


# ---------------- the loop-integrated representation ----------------
vis = make_layer()
calib = sequence(tr, train_idx[:200], np.random.default_rng(9))[0][:, 0]
vis.calibrate(T.as_tensor(calib, device=DEV), target=ACT)

S_tr, y_tr = sequence(tr, train_idx, np.random.default_rng(1))
S_te, y_te = sequence(te, test_idx, np.random.default_rng(2))

# (a) integrate in pixel space, then one pass  -- the control
pix_tr = T.as_tensor(trace(S_tr), device=DEV)
pix_te = T.as_tensor(trace(S_te), device=DEV)
acc_pix = float((knn(pix_tr, pix_te, y_tr) == y_te).mean())

# (b) integrate in the loop: pattern of each frame, then the trace of the patterns
loop_tr, loop_te = [], []
for t in range(FRAMES):
    A = patterns(vis, S_tr[:, t])
    B = patterns(vis, S_te[:, t])
    loop_tr.append(A)
    loop_te.append(B)
mtr, mte = loop_tr[0].clone(), loop_te[0].clone()
for t in range(1, FRAMES):
    mtr = mtr + (loop_tr[t] - mtr) / TAU
    mte = mte + (loop_te[t] - mte) / TAU
acc_loop = float((knn(mtr, mte, y_tr) == y_te).mean())

# (c) no integration at all: a single frame (the old regime)
acc_single = float((knn(loop_tr[0], loop_te[0], y_tr) == y_te).mean())

print('', flush=True)
print('  single frame, no integration      : %.3f' % acc_single, flush=True)
print('  pixels integrated, one pass       : %.3f' % acc_pix, flush=True)
print('  LOOP-integrated over %d frames     : %.3f' % (FRAMES, acc_loop), flush=True)
