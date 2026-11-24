"""Is the dataset itself sound?  And does one digit split into several concepts?

Two checks the designer asked for.

1. DATASET SANITY.  If a digit's images are genuinely different from each other, that is not a
   fault -- the model is supposed to cope.  But if the labels are mixed up, that IS a fault.
   So:
     * the average image of each digit is printed as ASCII -- it must look like the digit
     * for a sample of images, the nearest other image in the whole set is found and its
       label compared: a high disagreement rate would mean duplicates with conflicting labels
     * how similar images of the same digit are to each other, and to other digits

2. ONE DIGIT, SEVERAL CONCEPTS.  If the same digit activates different cell groups for
   different writings (different fonts, different paper), that is fine -- several concepts can
   all mean "this digit".  This measures how many separate response clusters a digit has.
"""
import os
import sys

import numpy as np

import numpy as np
from torchvision.datasets import MNIST
from PIL import Image

tr = MNIST(root=r'D:\deepseekherness\data', train=True, download=False)
rng = np.random.default_rng(0)

# ---------------- 1a  average image per digit ----------------
print('average image of each digit (training set):', flush=True)
imgs = {d: [] for d in range(10)}
for k in range(20000):
    img, lab = tr[k]
    a = np.asarray(img.resize((8, 8), Image.BOX), np.float32) / 255.0
    imgs[lab].append(a)
for d in range(10):
    m = np.mean(imgs[d], axis=0)
    print('  digit %d  (n=%d)' % (d, len(imgs[d])), flush=True)
    for row in m:
        print('    ' + ''.join('#' if v > 0.35 else ('+' if v > 0.12 else '.') for v in row), flush=True)

# ---------------- 1b  do duplicates disagree on the label? ----------------
print('', flush=True)
print('label consistency: for 600 random images, the nearest other image and its label', flush=True)
X, Y = [], []
for k in range(len(tr)):
    img, lab = tr[k]
    X.append(np.asarray(img.resize((8, 8), Image.BOX), np.float32).reshape(-1) / 255.0)
    Y.append(lab)
    if len(X) >= 20000:
        break
X = np.array(X, np.float32)
Y = np.array(Y)
idx = rng.choice(len(X), 600, replace=False)
same = 0
for i in idx:
    D = ((X - X[i]) ** 2).sum(1)
    D[i] = np.inf
    j = int(np.argmin(D))
    same += int(Y[j] == Y[i])
print('  nearest neighbour has the same label: %d/600 (%.1f%%)' % (same, 100.0 * same / 600), flush=True)

# ---------------- 1c  within-class similarity ----------------
def bin_img(v):
    return (v > 0.15).astype(np.float32)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


B = np.stack([bin_img(x) for x in X])
print('', flush=True)
print('image similarity (binary, threshold 0.15):', flush=True)
for d in range(10):
    ii = np.where(Y == d)[0][:40]
    w = [ov(B[i], B[j]) for a, i in enumerate(ii) for j in ii[a + 1:]]
    o = []
    for a, i in enumerate(ii):
        jj = np.where(Y != d)[0][:40]
        o.extend([ov(B[i], B[j]) for j in jj[:10]])
    print('  digit %d: same-class %.3f | other-class %.3f | ratio %.2f' % (
        d, float(np.mean(w)), float(np.mean(o)), float(np.mean(w)) / max(float(np.mean(o)), 1e-9)), flush=True)
