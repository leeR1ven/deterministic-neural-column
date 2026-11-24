"""Make the prefrontal population bigger than the visual one, and see whether the context separates.

sentence_ctx2.py got the repeated-word sentence exactly right [3,0,7,1] with unit-length keys, but by only 0.06,
and the ten-word sentence failed after three steps. Both failures have the same shape: two states that share the
current word (3 after 0, and 3 after 7) are only told apart by the context half, and the context half is
smaller than the visual half, so the keys still correlate about 0.6.

The designer said it directly: to hold a longer context, make the prefrontal part bigger. So here the context
field lives in its own population, K times the size of the visual one, each visual cell driving K prefrontal
cells with fixed random wiring. The key is [visual word ; prefrontal context], and the drive of a learned
transition is its cosine with that key -- the same synapses as before, just not stored as a dense matrix:
W k = sum over transitions of (next word) (key_t . k).

Nothing is calibrated. The threshold is 0.7 because one observed transition has cosine 1.0 with itself.

Usage: python sentence_ctx3.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 4, 0.6
TH_A = 0.7
NW = 12


def glyphs():
    g = np.zeros((NW, S * S), np.float32)
    rng = np.random.default_rng(7)
    for i in range(NW):
        g[i, rng.choice(S * S, 6, replace=False)] = 1.0
    return g


vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    P = vis.forward(T.as_tensor(glyphs(), device=DEV))[-1].float()
D = P.shape[1]
Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)


def make_pfc(k, seed=11):
    rng = np.random.default_rng(seed)
    return T.as_tensor(rng.integers(0, k * D, (D, k)), device=DEV)


def pfc_code(V, idx, k):
    out = T.zeros(k * D, device=DEV)
    for i in range(k):
        out.index_add_(0, idx[:, i], V)
    return out.clamp(max=1.0)


def keys_of(sentence, gamma, idx, k):
    F = T.zeros(k * D, device=DEV)
    ks = []
    for t in range(len(sentence)):
        V = P[sentence[t]]
        kk = T.cat([V, F])
        ks.append(kk / kk.norm().clamp(min=1e-9))
        F = gamma * F + pfc_code(V, idx, k)
    return ks


def walk(sentence, ks, gamma, idx, k, steps):
    V, F = P[sentence[0]].clone(), T.zeros(k * D, device=DEV)
    out, mar = [sentence[0]], []
    for i in range(steps):
        kk = T.cat([V, F])
        kk = kk / kk.norm().clamp(min=1e-9)
        act = T.zeros(D, device=DEV)
        for t in range(len(sentence) - 1):
            act += P[sentence[t + 1]] * float((ks[t] * kk).sum())
        if i + 1 < len(sentence):
            want = P[sentence[i + 1]]
            tgt = float(act[want > 0.5].min())
            o = act.clone()
            o[want > 0.5] = 0.0
            mar.append(tgt - float(o.max()))
        fired = (act > TH_A).float()
        if fired.sum() == 0:
            out.append(-1)
            break
        out.append(int((Pn @ (fired / fired.norm())).argmax()))
        F = gamma * F + pfc_code(V, idx, k)
        V = fired
    return out, mar


for sentence in ([0, 3, 0, 7, 1], [0, 3, 0, 7, 1, 7, 3, 9, 5, 2]):
    steps = len(sentence) - 1
    print('sentence: %s' % sentence, flush=True)
    for k in (1, 2, 4, 8, 16):
        idx = make_pfc(k)
        for gamma in (0.8, 0.9):
            ks = keys_of(sentence, gamma, idx, k)
            got, mar = walk(sentence, ks, gamma, idx, k, steps)
            hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
            print('  pfc x%-2d gamma %.1f  %s  %d/%d  worst margin %+.2f'
                  % (k, gamma, got[1:], hit, steps, min(mar) if mar else 0.0), flush=True)
    print('', flush=True)
