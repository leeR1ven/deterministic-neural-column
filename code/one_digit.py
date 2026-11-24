"""For ONE digit: how often does each concept cell fire across its instances?

The designer's expectation: the activated neurons form a normal distribution -- a group in
the middle fires almost always, and the probability falls off as you move away, rather than
"these exact neurons and no others".

This measures it directly.  For one digit, every image is shown, every concept cell's
firing count is recorded, and the counts are summarised as a distribution.
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
DIGIT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = np.random.default_rng(SEED)

from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
idx = np.where(np.array(tr.targets) == DIGIT)[0][:N]
X = np.zeros((len(idx), 64), np.float32)
for i, k in enumerate(idx):
    img, _ = tr[int(k)]
    X[i] = np.asarray(img.resize((8, 8), Image.BOX), np.float32).reshape(-1) / 255.0
print('digit %d: %d images' % (DIGIT, len(X)), flush=True)

# first layer
r = np.random.default_rng(SEED + 1)
N1, K1, GAIN = 1024, 40, 2.0
W1 = r.normal(0, 1, (N1, 128))
mask = np.zeros_like(W1, bool)
for j in range(N1):
    mask[j, r.choice(128, K1, replace=False)] = True
W1 = W1 * mask
W1 = W1 / np.abs(W1).sum(1, keepdims=True)
BUD1 = np.abs(W1).sum(1) + 1e-9
C = (W1 @ (GAIN * np.concatenate([2 * X - 1, 1 - 2 * X], axis=1)).T > 0.5 * BUD1[:, None]).T.astype(np.float32)

# concept layer learned only on THESE images (so the cells are specialised to this digit)
N2, K2 = 512, 100
r2 = np.random.default_rng(SEED + 2)
W2 = r2.normal(0, 1, (N2, N1)) * 0.1
m2 = np.zeros_like(W2, bool)
for j in range(N2):
    m2[j, r2.choice(N1, K2, replace=False)] = True
W2 = W2 * m2
W2 = W2 / np.abs(W2).sum(1, keepdims=True)
for ep in range(20):
    order = r2.permutation(len(C))
    for i in order:
        x = C[i]
        s = W2 @ x
        w = np.argpartition(s, -3)[-3:]
        W2[w] += 0.15 * (x[None, :] - W2[w] * x.sum() / max(x.sum(), 1))
        W2 *= 0.998
        np.maximum(W2, 0.0, out=W2)
        W2[w] /= np.maximum(np.abs(W2[w]).sum(1, keepdims=True), 1e-9)
TH = np.quantile(W2 @ C.T, 0.90, axis=1, keepdims=True)
A = (W2 @ C.T > TH).T.astype(np.float32)          # (images, cells)

freq = A.mean(0)                                   # firing frequency of every cell
active = freq[freq > 0]
print('cells that never fire: %d of %d' % (int((freq == 0).sum()), N2), flush=True)
print('cells that fire in >80%% of the images: %d' % int((freq > 0.8).sum()), flush=True)
print('cells that fire in 20-80%%: %d' % int(((freq >= 0.2) & (freq <= 0.8)).sum()), flush=True)
print('cells that fire in <20%%: %d' % int(((freq > 0) & (freq < 0.2)).sum()), flush=True)
print('', flush=True)
print('frequency distribution (deciles of the firing frequency among active cells):', flush=True)
qs = np.quantile(active, [0.1, 0.25, 0.5, 0.75, 0.9])
print('  10%%: %.2f | 25%%: %.2f | median: %.2f | 75%%: %.2f | 90%%: %.2f' % tuple(qs), flush=True)
print('  mean %.2f | std %.2f  (a bell shape has mean ~= median and the deciles evenly '
      'spread)' % (float(active.mean()), float(active.std())), flush=True)

hist, edges = np.histogram(active, bins=10, range=(0, 1))
print('', flush=True)
print('histogram of firing frequency:', flush=True)
for i in range(10):
    bar = '#' * int(60 * hist[i] / max(hist.max(), 1))
    print('  %.1f-%.1f : %4d %s' % (edges[i], edges[i + 1], hist[i], bar), flush=True)

# how much do two instances of the same digit overlap?
def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


pairs = [(i, j) for i in range(40) for j in range(i + 1, 40)]
print('', flush=True)
print('overlap between two different images of the SAME digit: %.3f' % (
    float(np.mean([ov(A[i], A[j]) for i, j in pairs]))), flush=True)
