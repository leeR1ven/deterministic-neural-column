"""The designer's chain: teach one sentence, then give only the first word and let it walk the chain.

Mechanism, exactly as described:

  frame t      input word w_t is active -> output pattern of that word is active
  between      the output that was active at t grows synapses onto the INPUT of the word at t+1,
               because it fired just before it (D -> B, E -> C in the designer's example)
  recall       activate only the first word; each step, the active pattern drives the next input through
               those learned synapses.  The chain A -> D -> B -> E -> C -> F unrolls by itself.

Nothing accumulates: each step the active set is just the current word (plus what the chain has just
driven), so there is nothing to blur.

Usage: python chain.py [sentence_len]
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
LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def glyph(i):
    r = np.random.default_rng(5000 + i)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(3, S - 8, 2)
        h, w = r.integers(3, 11, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


NW = 12
GL = np.stack([glyph(i) for i in range(NW)]).astype(np.float32)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GL, device=DEV))[-1]
D = PAT.shape[1]
Pn = PAT / PAT.norm(dim=1, keepdim=True).clamp(min=1e-9)

sentence = [0, 3, 7, 1, 9, 5][:LEN]
print('sentence to learn: %s' % sentence, flush=True)

# ---- learn the chain: the pattern active at step t grows synapses onto the pattern of the word at t+1 ----
W = T.zeros(D, D, device=DEV)
G = T.zeros(D, D, device=DEV)                      # global: word -> following word
for t in range(len(sentence) - 1):
    G += T.outer(PAT[sentence[t + 1]], PAT[sentence[t]])       # what fires now drives what fires next
W = G


def next_word(state, exclude):
    act = W @ state
    an = act / act.norm().clamp(min=1e-9)
    sims = Pn @ an
    order = sims.argsort(descending=True).cpu().numpy()
    for k in order:
        if int(k) not in exclude:
            return int(k), float(sims[k])
    return int(order[0]), float(sims[order[0]])


# ---- recall: only the first word is given ----
state = PAT[sentence[0]].clone()
out = [sentence[0]]
seen = {sentence[0]}
for step in range(len(sentence) * 2):
    w, sc = next_word(state, seen)
    out.append(w)
    seen.add(w)
    if w == sentence[-1]:
        break
    state = PAT[w].clone()

print('recalled:          %s' % out, flush=True)
print('target:            %s' % sentence, flush=True)
ok = out[1:] == sentence[1:]
print('', flush=True)
print('walks the sentence in order: %s' % ('YES' if ok else 'NO'), flush=True)
