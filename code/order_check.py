"""Does the prefrontal state hold the sentence IN ORDER, or does it only hold the last word?

Suspicion: the recurrence is too weak. A cell fires when its own input passes the threshold. The input is
    word_t + ALPHA * state_{t-1}
so a cell that was on at t-1 but is off in the current word receives only ALPHA, and with ALPHA below the
threshold it is switched off again. That would mean the state equals the current word -- no memory at all.

Measured here: after reading a five-word sentence, how much of each word is still present in the state.

Usage: python order_check.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH, NV = 28, 16, 0.9, 12


def glyph(i):
    r = np.random.default_rng(5000 + i)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(3, S - 8, 2)
        h, w = r.integers(3, 11, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


GL = np.stack([glyph(i) for i in range(NV)]).astype(np.float32)
sens = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
sens.theta = [0.6]
pfc = Local3DStack([(S, S, MZ), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
pfc.theta = [TH]
with T.no_grad():
    PAT = sens.forward(T.as_tensor(GL, device=DEV))[-1]

sentence = [0, 3, 7, 1, 9]
print('sentence: %s' % sentence, flush=True)
print('', flush=True)
print('ALPHA | state overlap with each word, after the whole sentence', flush=True)
for alpha in (0.3, 0.9, 1.1, 1.5):
    st = T.zeros(PAT.shape[1], device=DEV)
    with T.no_grad():
        for t in sentence:
            st = pfc.forward((PAT[t] + alpha * st).unsqueeze(0))[-1][0]
    ov = []
    for t in sentence:
        a, b = (st > 0), (PAT[t] > 0)
        u = float((a | b).sum())
        ov.append(float((a & b).sum()) / u if u else 0.0)
    print('  %.1f | %s   (last word %.2f, first word %.2f)' % (
        alpha, '  '.join('%.2f' % v for v in ov), ov[-1], ov[0]), flush=True)
print('', flush=True)
print('if the overlap with the FIRST word is near 0 and with the LAST word near 1, the state holds only', flush=True)
print('the last word -- no memory of the order at all.', flush=True)
