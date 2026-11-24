"""Dialogue as a chain: learn transitions through question+answer streams, then let the chain continue.

This follows the designer's mechanism instead of the accumulated-state version that blurred:

  a word's active pattern grows synapses onto the pattern of the word that follows it, everywhere in the
  data (monologue and every dialogue).  Nothing is accumulated; the state each step is just the word that
  is currently active plus whatever the chain has just driven.

  teaching    for each dialogue, the stream is question tokens then answer tokens, and every consecutive
              pair grows a connection
  answering   feed the question word by word; when the question ends, the chain continues on its own and
              the first word it produces is the answer

Usage: python chain_dialogue.py [vocab]
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 16, 0.6
DATA = r'D:\deepseekherness\data'
NVOC = int(sys.argv[1]) if len(sys.argv) > 1 else 300

raw = open(os.path.join(DATA, 'dialogue', 'tinytalks_v1.txt'), encoding='utf-8', errors='replace').read()
pairs, q = [], None
for line in raw.splitlines():
    line = line.strip()
    if line.startswith('Q:'):
        q = line[2:].strip()
    elif line.startswith('A:') and q is not None:
        pairs.append((q, line[2:].strip()))
        q = None


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


counts = Counter()
for qq, aa in pairs:
    counts.update(words(qq) + words(aa))
vocab = ['<pad>'] + [w for w, _ in counts.most_common(NVOC)]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
pairs = [(qq, aa) for qq, aa in pairs if all(w in w2i and w2i[w] > 0 for w in words(qq) + words(aa))]
print('pairs %d | vocabulary %d' % (len(pairs), V), flush=True)


def encode(s):
    return [w2i[w] for w in words(s)] or [0]


def glyph(i):
    r = np.random.default_rng(5000 + i)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(3, S - 8, 2)
        h, w = r.integers(3, 11, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


GL = np.stack([glyph(i) for i in range(V)]).astype(np.float32)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GL, device=DEV))[-1]
D = PAT.shape[1]
Pn = PAT / PAT.norm(dim=1, keepdim=True).clamp(min=1e-9)

rng = np.random.default_rng(0)
order = rng.permutation(len(pairs))
n_hold = min(60, len(pairs) // 4)
held = [pairs[i] for i in order[:n_hold]]
train = [pairs[i] for i in order[n_hold:]]

# ---- learn the chain over every stream ----
W = T.zeros(D, D, device=DEV)
streams = [encode(qq) + encode(aa) for qq, aa in train]
for s in streams:
    for a, b in zip(s[:-1], s[1:]):
        W += T.outer(PAT[b], PAT[a])          # what is active now drives what is active next
W.clamp_(min=0.0)
print('chain learned from %d streams | %d non-zero synapses' % (len(streams), int((W > 0).sum())), flush=True)


def next_word(state):
    act = W @ state
    an = act / act.norm().clamp(min=1e-9)
    sims = Pn @ an
    order2 = sims.argsort(descending=True).cpu().numpy()
    return int(order2[0]), float(sims[order2[0]]), order2[:5]


ok = top5 = 0
KW = int(os.environ.get('KW', '3'))     # how many recent frames stay active together
for qq, aa in held:
    hist = []
    for t in encode(qq):
        hist.append(PAT[t])
        hist = hist[-KW:]
    state = T.stack(hist).sum(0)
    w, sc, top = next_word(state)
    want = encode(aa)[0]
    ok += int(w == want)
    top5 += int(want in top)
print('', flush=True)
print('held-out pairs %d' % len(held), flush=True)
print('  first answer word right : %d/%d' % (ok, len(held)), flush=True)
print('  within the top five     : %d/%d' % (top5, len(held)), flush=True)
print('  (chance: 1/%d and 5/%d)' % (V, V), flush=True)
