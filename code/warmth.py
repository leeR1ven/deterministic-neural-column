"""Residual warmth: a cell that has just fired is harder to fire again, and recovers slowly.

The designer's mechanism: each frame's assembly keeps reverberating but the loop gets harder every cycle,
so it fades; several earlier frames are therefore still active when the next frame arrives, with the
earliest ones weakest. That is what lets connections form between frames that are close in time and only
rarely between frames far apart.

The missing piece in every earlier attempt was exactly this: with a binary threshold and a constant
recurrent weight the activity either stays on forever or dies in one step -- there was no gradual fading.
Here every cell carries a fatigue value:

    fire_i    if  input_i + ALPHA * state_i - fatigue_i  >  TH
    fatigue_i += BETA   when the cell fires
    fatigue_i *= DECAY  every step   (it recovers)

Measured: after reading a five-word sentence, how much of each word is still present. A useful memory
shows a gradient -- the most recent word strong, earlier ones progressively weaker -- instead of only the
last word (amnesia) or all of them equally (union, no order).

Usage: python warmth.py
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
ALPHA = float(os.environ.get('ALPHA', '1.0'))     # recurrent weight
BETA = float(os.environ.get('BETA', '0.5'))      # how much harder a cell is to fire after firing
DECAY = float(os.environ.get('DECAY', '0.6'))    # how fast that hardness recovers


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

sentence = [0, 3, 7, 1, 9]
state = T.zeros(PAT.shape[1], device=DEV)
fatigue = T.zeros(PAT.shape[1], device=DEV)
hist = []
with T.no_grad():
    for t in sentence:
        drive = PAT[t] + ALPHA * state - fatigue
        state = (drive > TH).float()
        fatigue = fatigue * DECAY + BETA * state
        hist.append(state.clone())

print('sentence %s | ALPHA %.1f BETA %.1f DECAY %.1f' % (sentence, ALPHA, BETA, DECAY), flush=True)
print('', flush=True)
print('overlap of the state at each step with each word of the sentence:', flush=True)
for step, st in enumerate(hist):
    ov = []
    for t in sentence:
        a, b = (st > 0), (PAT[t] > 0)
        u = float((a | b).sum())
        ov.append(float((a & b).sum()) / u if u else 0.0)
    print('  after word %d: %s' % (step + 1, '  '.join('%.2f' % v for v in ov)), flush=True)
print('', flush=True)
print('final state: active %.1f%% | overlap with the words, oldest to newest: %s' % (
    100 * float(hist[-1].mean()),
    '  '.join('%.2f' % v for v in [
        float(((hist[-1] > 0) & (PAT[t] > 0)).sum()) / max(float(((hist[-1] > 0) | (PAT[t] > 0)).sum()), 1)
        for t in sentence])), flush=True)
print('', flush=True)
print('what to look for: a descending profile (newest strong, oldest weak).  All equal means union (no', flush=True)
print('order); only the last one non-zero means it forgot everything.', flush=True)
