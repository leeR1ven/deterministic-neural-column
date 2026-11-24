"""Two things measured here, both following the designer's last point.

1  The model is always running, so the same word read at two different moments is already a different state --
   but only if time really passed and something happened in between. Measured as the cosine between the states:
   the same word twice in a row is almost the same state, the same word with anything in between is not.

2  "A person does not produce the same sound twice either: there is a different sound, or a pause." So the
   vocabulary gets a thirteenth item, silence (an empty picture, nothing fires), and sentences are tested with
   pauses in them, including a pause between two identical words. The model has to produce the pause in the
   right place too.

Sentences here never repeat a word back to back -- the designer's point is that a word repeated with nothing in
between is the same cells cycling and is not the case worth testing. The earlier failures of exactly that kind
are left where they were, in sentence_sweep.py.

Usage: python sentence_repeats.py
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
TH_A, GAMMA, SCALE, ETA, EPOCHS = 0.7, 0.9, 5.0, 0.25, 400
NW, SIL = 12, 12          # words 0..11, item 12 is silence


def glyphs():
    g = np.zeros((NW + 1, S * S), np.float32)
    rng = np.random.default_rng(7)
    for i in range(NW):
        g[i, rng.choice(S * S, 6, replace=False)] = 1.0
    return g                              # row 12 stays all zero: silence


vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    P = vis.forward(T.as_tensor(glyphs(), device=DEV))[-1].float()
D = P.shape[1]
Pn = P[:NW] / P[:NW].norm(dim=1, keepdim=True).clamp(min=1e-9)
NA = float(P[:NW].sum(1).mean())
print('silence pattern has %d active cells (want 0)' % int(P[SIL].sum()), flush=True)


def ctx_step(F, V):
    """the context field: decay, add what just fired, and hold its total activity near SCALE words.
    During a pause nothing is added and nothing is rescaled, so the field just fades -- time passes."""
    F = GAMMA * F + V
    if float(V.sum()) > 0.5:
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return F


def keys_of(sentence):
    F = T.zeros(D, device=DEV)
    ks = []
    for w in sentence:
        k = T.cat([P[w], F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = ctx_step(F, P[w])
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
            w = SIL                      # nothing fires: the pause
        else:
            w = int((Pn @ (fired / fired.norm())).argmax())
        out.append(w)
        F = ctx_step(F, V)
        V = (P[w] if w == SIL else fired)
    return out


# ---- 1  how different is the state of the same word at two moments? ----
print('', flush=True)
for label, sent in (('0 twice in a row     ', [0, 0]),
                    ('0 with 3 in between  ', [0, 3, 0]),
                    ('0 with a pause after 3', [0, 3, SIL, 0]),
                    ('0 with 3,7 in between', [0, 3, 7, 0])):
    ks = keys_of(sent)
    same = [i for i in range(1, len(sent)) if sent[i] == 0]
    c = float(ks[0] @ ks[same[0]])
    print('%-24s state of the two 0s: cosine %.3f  %s'
          % (label, c, '(identical)' if c > 0.95 else '(different)'), flush=True)

# ---- 2  sentences without back-to-back repeats, some with pauses ----
rng = np.random.default_rng(99)


def make(n, allow_sil):
    s = [int(rng.integers(0, NW))]
    while len(s) < n:
        pool = list(range(NW + (1 if allow_sil else 0)))
        pool = [w for w in pool if w != s[-1] and not (w == SIL and s[-1] == SIL)]
        s.append(int(rng.choice(pool)))
    if s[0] == SIL:
        s = s[1:]
    return s


for label, allow_sil, trials in (('no pause', False, 20), ('with pauses', True, 20)):
    exact = steps = ok = 0
    bad = []
    for _ in range(trials):
        sent = make(int(rng.integers(5, 13)), allow_sil)
        if not allow_sil:
            sent = [w if w != SIL else 0 for w in sent]
        got = walk(sent, learn(sent, keys_of(sent)))
        hit = sum(1 for a, b in zip(got[1:], sent[1:]) if a == b)
        exact += (hit == len(sent) - 1)
        steps += len(sent) - 1
        ok += hit
        if hit != len(sent) - 1:
            bad.append((sent, got[1:], hit, len(sent) - 1))
    print('', flush=True)
    print('%-12s exact %2d/%d   steps %d/%d (%.1f%%)'
          % (label, exact, trials, ok, steps, 100.0 * ok / max(steps, 1)), flush=True)
    for sent, got, hit, tot in bad:
        print('   miss %d/%d  %s -> %s' % (hit, tot, sent, got), flush=True)
