"""How much context activity is needed, and does the designer's "make the prefrontal part bigger" hold up?

The delta rule now recalls 4 of 5 test sentences exactly. The one it misses, 5 1 5 1 3 7 3 7 0 0 11 6, fails for a
stated reason: with the context field normalised to the activity of one word, the newest word always takes about
half of it, so "1 after 5" and "1 after 5,1,5" end up with nearly the same key -- the same key with two different
answers, which no set of synapses can satisfy.

So the context population has to hold MORE than one word's worth of activity. That is the designer's lever,
"to keep a longer context, make the prefrontal activation proportion larger". Here it is swept: the context
field's total activity is held at SCALE times the activity of one word.

Still local: one synapse per (input cell, output cell), adjusted by the error at the output cell it lands on.

Usage: python sentence_ctx4.py
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
TH_A, ETA, EPOCHS = 0.7, 0.25, 500
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


def step(F, V, gamma, scale):
    F = gamma * F + V
    return F * (scale * NA / F.sum().clamp(min=1e-9))


def keys_of(sentence, gamma, scale):
    F = T.zeros(D, device=DEV)
    ks = []
    for t in range(len(sentence)):
        V = P[sentence[t]]
        k = T.cat([V, F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = step(F, V, gamma, scale)
    return ks


def learn(sentence, ks, eta=ETA, epochs=EPOCHS):
    W = T.zeros(D, 2 * D, device=DEV)
    for _ in range(epochs):
        for t in range(len(sentence) - 1):
            W += eta * T.outer(P[sentence[t + 1]] - W @ ks[t], ks[t])
    return W


def walk(sentence, W, gamma, scale):
    V, F = P[sentence[0]].clone(), T.zeros(D, device=DEV)
    out = [sentence[0]]
    for i in range(len(sentence) - 1):
        k = T.cat([V, F])
        k = k / k.norm().clamp(min=1e-9)
        fired = ((W @ k) > TH_A).float()
        if fired.sum() == 0:
            out.append(-1)
            break
        out.append(int((Pn @ (fired / fired.norm())).argmax()))
        F = step(F, V, gamma, scale)
        V = fired
    return out


SENTS = ([0, 3, 0, 7, 1],
         [0, 3, 0, 7, 1, 7, 3, 9, 5, 2],
         [1, 4, 1, 4, 9, 2, 9, 6, 8, 3, 8, 5],
         [2, 2, 8, 8, 4, 2, 4, 6, 6, 9],
         [5, 1, 5, 1, 3, 7, 3, 7, 0, 0, 11, 6],
         [7, 0, 7, 0, 6, 6, 11, 4, 4, 1, 9])

for gamma in (0.8, 0.9, 0.95):
    for scale in (1.0, 2.0, 3.0, 5.0):
        ok, worst = 0, ''
        for sentence in SENTS:
            ks = keys_of(sentence, gamma, scale)
            got = walk(sentence, learn(sentence, ks), gamma, scale)
            hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
            ok += (hit == len(sentence) - 1)
            if hit < len(sentence) - 1:
                worst = '%s -> %s' % (sentence, got[1:])
        print('gamma %.2f  context x%.0f   exact %d/%d   %s'
              % (gamma, scale, ok, len(SENTS), worst if ok < len(SENTS) else ''), flush=True)
