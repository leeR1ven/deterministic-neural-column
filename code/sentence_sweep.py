"""The final configuration on 20 random sentences, so the 6/6 above cannot be a lucky sample.

Configuration that survives the sweep (sentence_ctx4.py):
  key          [ the word being read now ; the context field ]
  context      a decaying sum of the past words, total activity held at 5 words' worth
  drive        cosine of the key with each learned key, threshold 0.7, cells over it fire
  synapses     one per (context cell, next-word cell), adjusted by the error at the cell it lands on
  recall       only the first word is given; the walk is driven by the learned synapses alone

Usage: python sentence_sweep.py
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
TH_A, GAMMA, SCALE, ETA, EPOCHS = 0.7, 0.9, 5.0, 0.25, 500
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
NA = float(P.sum(1).mean())


def keys_of(sentence):
    F = T.zeros(D, device=DEV)
    ks = []
    for w in sentence:
        k = T.cat([P[w], F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = GAMMA * F + P[w]
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return ks


def learn(sentence, ks):
    W = T.zeros(D, 2 * D, device=DEV)
    for _ in range(EPOCHS):
        for t in range(len(sentence) - 1):
            W += ETA * T.outer(P[sentence[t + 1]] - W @ ks[t], ks[t])
    return W


def walk(sentence, W):
    V, F = P[sentence[0]].clone(), T.zeros(D, device=DEV)
    out = [sentence[0]]
    for _ in range(len(sentence) - 1):
        k = T.cat([V, F])
        k = k / k.norm().clamp(min=1e-9)
        fired = ((W @ k) > TH_A).float()
        if fired.sum() == 0:
            out.append(-1)
            break
        out.append(int((Pn @ (fired / fired.norm())).argmax()))
        F = GAMMA * F + V
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
        V = fired
    return out


rng = np.random.default_rng(2024)
SENTS = [[int(w) for w in rng.integers(0, NW, int(n))]
         for n in rng.integers(5, 13, 20)]
tot_steps = tot_ok = exact = 0
for sentence in SENTS:
    got = walk(sentence, learn(sentence, keys_of(sentence)))
    hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
    total = len(sentence) - 1
    exact += (hit == total)
    tot_steps += total
    tot_ok += hit
    print('%-46s %s  %d/%d' % (sentence, 'ok ' if hit == total else '   ', hit, total), flush=True)
print('', flush=True)
print('exact sentences %d/%d   steps %d/%d (%.1f%%)'
      % (exact, len(SENTS), tot_ok, tot_steps, 100.0 * tot_ok / tot_steps), flush=True)
