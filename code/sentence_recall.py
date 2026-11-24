"""Remember one sentence -- with a repeated word, so only real context can get it right.

Sentence: 0 -> 3 -> 0 -> 7 -> 1.  The word 0 appears twice and is followed by a different word each time.
A memory that only knows the current word cannot tell the two apart, and will repeat what followed the first
0. Only if the activity carries the previous frames (the designer's residual warmth) can the second 0 lead
to 7.

Two conditions, same learned chain:
  no warmth    the state is just the current word's pattern
  warmth       every cell keeps firing for a few frames after it fired and gets harder to fire again, so the
               state is the current word plus the recent past (ALPHA 1.0, BETA 0.2, DECAY 0.8)

Usage: python sentence_recall.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 16, 0.6
ALPHA, BETA, DECAY = 1.0, 0.2, 0.8


def glyph(i):
    r = np.random.default_rng(5000 + i)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(3, S - 8, 2)
        h, w = r.integers(3, 11, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


GL = np.stack([glyph(i) for i in range(12)]).astype(np.float32)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GL, device=DEV))[-1]
D = PAT.shape[1]
Pn = PAT / PAT.norm(dim=1, keepdim=True).clamp(min=1e-9)

sentence = [0, 3, 0, 7, 1]
print('sentence (word 0 repeats): %s' % sentence, flush=True)

# the chain: each word grows synapses onto the word that follows it
W = T.zeros(D, D, device=DEV)
for a, b in zip(sentence[:-1], sentence[1:]):
    W += T.outer(PAT[b], PAT[a])
W.clamp_(min=0.0)


def predict(state):
    act = W @ state
    an = act / act.norm().clamp(min=1e-9)
    return int((Pn @ an).argmax()), Pn @ an


def run(with_warmth):
    state = T.zeros(D, device=DEV)
    fatigue = T.zeros(D, device=DEV)
    out = [sentence[0]]
    for t in sentence:
        if with_warmth:
            drive = PAT[t] + ALPHA * state - fatigue
            state = (drive > TH).float()
            fatigue = fatigue * DECAY + BETA * state
        else:
            state = PAT[t].clone()
            fatigue = T.zeros(D, device=DEV)
        nxt, _ = predict(state)
        out.append(nxt)
    return out


for label, ww in (('no warmth', False), ('warmth', True)):
    got = run(ww)
    print('  %-10s recall: %s   %s' % (
        label, got[1:], 'OK' if got[1:] == sentence[1:] else '--'), flush=True)
print('', flush=True)
print('target:  [3, 0, 7, 1]  (what follows each word; the second 0 must lead to 7, not to 3)', flush=True)
