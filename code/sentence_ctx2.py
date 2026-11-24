"""Two changes after the first context run: unit-length keys, and every context cell its own decay.

What the first run showed (sentence_ctx.py): delaying the context field helped -- the repeated 0 stopped being
confused on step 2, and the short sentence went from 1/4 to 3/4. It still failed on step 3, and the long sentence
collapsed to one word and stayed there.

Two causes, both visible in the numbers:
  1  the keys were scaled by 1/|key|^2 for the transition that made them, not by the length of the key being
     read, so a short key (the cold start, no context yet) dumped its whole successor at full strength. Using
     unit-length keys makes the drive a cosine, which is what a synapse set can actually do locally.
  2  every context cell faded with the same GAMMA, so "seen one frame ago" and "seen three frames ago" differ
     only in size, and a linear readout cannot separate a size from a pattern. Real prefrontal cells have very
     different time constants, so the same word at different ages leaves a different DIRECTION. Here every
     context cell draws its own decay from a fixed random range.

Usage: python sentence_ctx2.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 8, 0.6
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
rng = np.random.default_rng(3)
GAM = T.as_tensor(rng.uniform(0.55, 0.95, D).astype(np.float32), device=DEV)   # one decay per cell


def fade(F, V, gamma, hetero):
    return (GAM * F if hetero else gamma * F) + V


def learn(sentence, gamma, hetero, cos):
    F = T.zeros(D, device=DEV)
    ks = []
    for t in range(len(sentence)):
        V = P[sentence[t]]
        ks.append(T.cat([V, F]))
        F = fade(F, V, gamma, hetero)
    W = T.zeros(D, 2 * D, device=DEV)
    for t in range(len(sentence) - 1):
        k = ks[t]
        if cos:
            k = k / k.norm().clamp(min=1e-9)
            W += T.outer(P[sentence[t + 1]], k)
        else:
            W += T.outer(P[sentence[t + 1]], k) / k.pow(2).sum().clamp(min=1e-9)
    return W


def walk(sentence, W, gamma, hetero, cos, steps):
    Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    V, F = P[sentence[0]].clone(), T.zeros(D, device=DEV)
    out, mar = [sentence[0]], []
    for i in range(steps):
        k = T.cat([V, F])
        if cos:
            k = k / k.norm().clamp(min=1e-9)
        act = W @ k
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
        V, F = fired, fade(F, V, gamma, hetero)
    return out, mar


for sentence in ([0, 3, 0, 7, 1], [0, 3, 0, 7, 1, 7, 3, 9, 5, 2]):
    steps = len(sentence) - 1
    print('sentence: %s' % sentence, flush=True)
    for cos in (True, False):
        for hetero in (False, True):
            for gamma in (0.8, 0.9):
                W = learn(sentence, gamma, hetero, cos)
                got, mar = walk(sentence, W, gamma, hetero, cos, steps)
                hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
                print('  %s %s gamma %.1f  %s  %d/%d  worst margin %+.2f'
                      % ('cos ' if cos else 'l2  ', 'hetero' if hetero else 'same  ',
                         gamma, got[1:], hit, steps,
                         min(mar) if mar else 0.0), flush=True)
    print('', flush=True)
