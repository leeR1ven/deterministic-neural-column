"""What the model can say about a run of the same word: a rough duration, not a count.

The designer's reading is that three 0s in a row is like feeding one sine wave and asking how many cycles it
made -- the right answer is "about this long", not "three". This measures exactly that.

The context field is a leaky sum that is rescaled back to about SCALE words of activity whenever a word arrives.
During a run of one word it therefore fills up and then stops changing: the state keeps moving for the first few
frames of the run and then saturates. Frames inside the saturated part are indistinguishable -- which is why
7,0,0,0,2,11 cannot be recalled, and why asking for it is the wrong question.

Usage: python run_length.py
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
GAMMA, SCALE = 0.9, 5.0
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
NA = float(P.sum(1).mean())

sentence = [5] + [0] * 8 + [9]
F = T.zeros(D, device=DEV)
keys, fills = [], []
for w in sentence:
    k = T.cat([P[w], F])
    keys.append(k / k.norm().clamp(min=1e-9))
    F = GAMMA * F + P[w]
    F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    fills.append(float(F.sum()) / (SCALE * NA))

print('sentence %s   (the run of 0s is at positions 1..8)' % sentence, flush=True)
print('', flush=True)
print('  frame   how much the state moved   distance from the first 0', flush=True)
print('          since the previous frame    (0 = the run is invisible)', flush=True)
moves = []
for i in range(1, 9):
    pos = i
    same = float(keys[pos] @ keys[1])
    if i == 1:
        print('  %2d      --                          %.3f' % (i, same), flush=True)
    else:
        mv = float(keys[pos] @ keys[pos - 1])
        moves.append(mv)
        print('  %2d      %.3f                        %.3f' % (i, mv, same), flush=True)
n = next((i for i, m in enumerate(moves, 2) if m > 0.99), len(moves) + 1)
print('', flush=True)
print('by frame %d each new frame of the run changes the state by less than 1%%,' % n, flush=True)
print('so the model can say "this has been going on a while" but cannot count the frames.', flush=True)
