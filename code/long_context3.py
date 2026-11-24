"""Check that the longer context used for counting does not cost anything on ordinary sentences.

Configuration that counts runs exactly (long_context2.py): context cells decay in [0.40, 0.999], total context
activity held at 10 words' worth, learning 2000 passes.

Usage: python long_context3.py
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
TH_A, ETA, EPOCHS = 0.7, 0.25, 2000
GMIN, GMAX, SCALE = 0.40, 0.999, 10.0
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
GAM = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, D).astype(np.float32), device=DEV)


def step(F, V):
    F = GAM * F + V
    if float(V.sum()) > 0.5:
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return F


def score(s):
    F = T.zeros(D, device=DEV)
    ks = []
    for w in s:
        k = T.cat([P[w], F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = step(F, P[w])
    W = T.zeros(D, 2 * D, device=DEV)
    for _ in range(EPOCHS):
        for t in range(len(s) - 1):
            W += ETA * T.outer(P[s[t + 1]] - W @ ks[t], ks[t])
    V, F = P[s[0]].clone(), T.zeros(D, device=DEV)
    out = [s[0]]
    for _ in range(len(s) - 1):
        k = T.cat([V, F])
        k = k / k.norm().clamp(min=1e-9)
        fired = ((W @ k) > TH_A).float()
        w = int((Pn @ (fired / fired.norm())).argmax()) if fired.sum() > 0 else SIL
        out.append(w)
        F = step(F, V)
        V = P[w] if w == SIL else fired
    hit = sum(1 for a, b in zip(out[1:], s[1:]) if a == b)
    return out, hit, len(s) - 1


rng = np.random.default_rng(99)


def make(n, allow_sil):
    s = [int(rng.integers(0, NW))]
    while len(s) < n:
        pool = [w for w in range(NW + (1 if allow_sil else 0))
                if w != s[-1] and not (w == SIL and s[-1] == SIL)]
        s.append(int(rng.choice(pool)))
    return s[1:] if s[0] == SIL else s


for label, allow_sil, trials in (('no repeats back to back', False, 6), ('with pauses', True, 6)):
    exact = steps = ok = 0
    for _ in range(trials):
        s = make(int(rng.integers(5, 13)), allow_sil)
        got, hit, tot = score(s)
        exact += (hit == tot)
        steps += tot
        ok += hit
        print('  %-46s %d/%d %s' % (s, hit, tot, '' if hit == tot else got[1:]), flush=True)
    print('%-24s exact %d/%d  steps %d/%d (%.1f%%)'
          % (label, exact, trials, ok, steps, 100.0 * ok / max(steps, 1)), flush=True)
