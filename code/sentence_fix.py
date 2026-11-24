"""Find the detail that breaks the sentence memory, and test the fix.

The failing sentence is 0 -> 3 -> 0 -> 7 -> 1: the word 0 appears twice and is followed by a different word
each time, so only the context carried by the warm cells can get it right.

The chain in sentence_recall.py learned its synapses from the CLEAN word patterns but was recalled from the
WARM mixture. That is a mismatch: what is learned must be what is used. Here both passes use the same
dynamics -- every frame the input word is added to the graded residual warmth, cells over threshold fire,
and the synapses are grown from the state that was actually active.

Variants:
  glyph    dense  = the three-rectangle glyphs used before (a word covers ~30% of the field)
           sparse = six pixels per word, the designer's "6 neurons per frame"
  learn    clean  = synapses from the bare word pattern
           warm   = synapses from the context-carrying state
  recall   gives the first word only and lets the learned synapses walk the sentence; no lookup is used to
           drive the walk, the decoded word is only a score

Fixed constants (never calibrated): TH 0.6 visual, TH_A 0.5 association, ALPHA 0.5 (< TH_A, so a warm cell
never fires without its word), GAMMA 0.8, BETA 0.2, DECAY 0.8.

Usage: python sentence_fix.py
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
TH_A, ALPHA, GAMMA, BETA, DECAY = 0.5, 0.5, 0.8, 0.2, 0.8
NW = 12


def glyphs(kind):
    g = np.zeros((NW, S * S), np.float32)
    if kind == 'sparse':
        rng = np.random.default_rng(7)
        for i in range(NW):
            g[i, rng.choice(S * S, 6, replace=False)] = 1.0
    else:
        for i in range(NW):
            r = np.random.default_rng(5000 + i)
            a = np.zeros((S, S), np.float32)
            for _ in range(3):
                y0, x0 = r.integers(3, S - 8, 2)
                h, w = r.integers(3, 11, 2)
                a[y0:y0 + h, x0:x0 + w] = 1.0
            g[i] = a.reshape(-1)
    return g


vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]


def patterns(kind):
    with T.no_grad():
        return vis.forward(T.as_tensor(glyphs(kind), device=DEV))[-1].float()


def overlap(P):
    Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    G = Pn @ Pn.T
    off = G - T.diag(T.diag(G))
    n = P.shape[0]
    return float(P.mean()), float(off.abs().sum() / (n * (n - 1)))


def learn(P, sentence, mode):
    D = P.shape[1]
    W = T.zeros(D, D, device=DEV)
    r = T.zeros(D, device=DEV)
    f = T.zeros(D, device=DEV)
    keys = []
    for t in range(len(sentence) - 1):
        if mode == 'clean':
            key = P[sentence[t]].clone()
        else:
            drive = P[sentence[t]] + ALPHA * r - f
            fired = (drive > TH_A).float()
            r = T.maximum(r * GAMMA, fired)
            f = DECAY * f + BETA * fired
            key = r.clone()
        keys.append(key)
        W += T.outer(P[sentence[t + 1]], key) / key.pow(2).sum().clamp(min=1e-9)
    return W, keys


def walk(P, sentence, W, key0, steps):
    Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r, f = key0.clone(), T.zeros(P.shape[1], device=DEV)
    out = []
    for _ in range(steps):
        act = W @ r - f
        fired = (act > TH_A).float()
        a = fired / fired.norm().clamp(min=1e-9)
        sims = Pn @ a
        out.append((int(sims.argmax()), float(sims.max())))
        r = T.maximum(r * GAMMA, fired)
        f = DECAY * f + BETA * fired
    return out


sentence = [0, 3, 0, 7, 1]
need = len(sentence) - 1
print('sentence: %s   (0 repeats and leads to a different word each time)' % sentence, flush=True)
print('target next words: %s' % sentence[1:], flush=True)
print('', flush=True)

for gkind in ('dense', 'sparse'):
    P = patterns(gkind)
    dens, ov = overlap(P)
    print('%s glyphs: %d cells, activation %.1f%%, mean |cos| between words %.3f'
          % (gkind, P.shape[1], 100 * dens, ov), flush=True)
    for mode in ('clean', 'warm'):
        W, keys = learn(P, sentence, mode)
        key0 = keys[0]
        got = walk(P, sentence, W, key0, need)
        words = [w for w, _ in got]
        hit = sum(1 for (w, _), t in zip(got, sentence[1:]) if w == t)
        print('  learn %-5s  ->  %s   correct %d/%d   %s'
              % (mode, words, hit, need, 'FULL' if hit == need else ''), flush=True)
    print('', flush=True)
