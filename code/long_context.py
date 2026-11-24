"""A longer, graded context: every context cell with its own decay, so a run of one word keeps changing.

The designer's suggestion: as the model says the second, third, ... 0, its preceding context keeps changing, and
that change is what tells it which 0 it is at. With one single decay rate that cannot work: a decaying sum of a
constant input converges to the same direction, so after a few frames every 0 in a run looks the same. It only
works if the context has a spread of time constants, because then the fast cells fill up first and the slow ones
later, and the shape of the state keeps moving for much longer.

So: every context cell draws its own decay from [0.60, 0.99] (fixed at wiring time, same as any other fixed
weight). Everything else is unchanged -- delayed context field, cosine drive, threshold 0.7, synapses adjusted by
the error at the cell they land on.

Tests
  counting   [3, 0 x k, 9, 4] for k = 1..5, each learned on its own: can it say the right number of 0s?
  repeats    the sentence the designer says is hard, 7,0,0,0,2,11
  suite      20 random sentences with no back-to-back repeats
  pauses     20 random sentences with silence in them

Usage: python long_context.py
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
TH_A, SCALE, ETA, EPOCHS = 0.7, 5.0, 0.25, 500
NW, SIL = 12, 12


def glyphs():
    g = np.zeros((NW + 1, S * S), np.float32)
    rng = np.random.default_rng(7)
    for i in range(NW):
        g[i, rng.choice(S * S, 6, replace=False)] = 1.0
    return g


vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    P = vis.forward(T.as_tensor(glyphs(), device=DEV))[-1].float()
D = P.shape[1]
Pn = P[:NW] / P[:NW].norm(dim=1, keepdim=True).clamp(min=1e-9)
NA = float(P[:NW].sum(1).mean())
GAM = T.as_tensor(np.random.default_rng(5).uniform(0.60, 0.99, D).astype(np.float32), device=DEV)
print('context cells: %d, decay spread %.2f..%.2f' % (D, float(GAM.min()), float(GAM.max())), flush=True)


def ctx_step(F, V):
    F = GAM * F + V
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
            w = SIL
        else:
            w = int((Pn @ (fired / fired.norm())).argmax())
        out.append(w)
        F = ctx_step(F, V)
        V = P[w] if w == SIL else fired
    return out


def score(sentence):
    got = walk(sentence, learn(sentence, keys_of(sentence)))
    hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
    return got, hit, len(sentence) - 1


print('', flush=True)
print('counting a run of 0s', flush=True)
for k in range(1, 6):
    sent = [3] + [0] * k + [9, 4]
    got, hit, tot = score(sent)
    print('  %d zero(s)  %s  %d/%d  %s' % (k, sent, hit, tot, 'ok' if hit == tot else got[1:]), flush=True)

print('', flush=True)
sent = [7, 0, 0, 0, 2, 11]
got, hit, tot = score(sent)
print('  %s -> %s  %d/%d' % (sent, got[1:], hit, tot), flush=True)

rng = np.random.default_rng(99)


def make(n, allow_sil):
    s = [int(rng.integers(0, NW))]
    while len(s) < n:
        pool = [w for w in range(NW + (1 if allow_sil else 0))
                if w != s[-1] and not (w == SIL and s[-1] == SIL)]
        s.append(int(rng.choice(pool)))
    return s[1:] if s[0] == SIL else s


print('', flush=True)
for label, allow_sil in (('no repeats back to back', False), ('with pauses', True)):
    exact = steps = ok = 0
    bad = []
    for _ in range(20):
        sent = make(int(rng.integers(5, 13)), allow_sil)
        got, hit, tot = score(sent)
        exact += (hit == tot)
        steps += tot
        ok += hit
        if hit != tot:
            bad.append((sent, got[1:]))
    print('%-24s exact %2d/20   steps %d/%d (%.1f%%)'
          % (label, exact, ok, steps, 100.0 * ok / max(steps, 1)), flush=True)
    for sent, got in bad[:3]:
        print('   miss  %s -> %s' % (sent, got), flush=True)
