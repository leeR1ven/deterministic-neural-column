"""Does a longer context let the model count a run exactly, or only roughly?

long_context.py showed the graded context already gets 7,0,0,0,2,11 right, but [3, 0,0,0, 9,4] comes out one
zero short and then falls silent: the state at the third zero is not distinct enough for its learned transition
to reach the threshold. Two things could be responsible and they are easy to tell apart:

  the learning did not converge   -> the residual (what the cells should have done minus what they did) stays high
  the state is not distinct       -> the residual goes to zero and the recall still slips, or the keys themselves
                                     are nearly equal

So: residual printed next to the result, and the context made longer (slower cells, more capacity).

Usage: python long_context2.py
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
TH_A, ETA = 0.7, 0.25
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


def run(gmin, gmax, scale, epochs, key_spread=False):
    gam = T.as_tensor(np.random.default_rng(5).uniform(gmin, gmax, D).astype(np.float32), device=DEV)

    def step(F, V):
        F = gam * F + V
        if float(V.sum()) > 0.5:
            F = F * (scale * NA / F.sum().clamp(min=1e-9))
        return F

    def keys_of(s):
        F = T.zeros(D, device=DEV)
        ks = []
        for w in s:
            k = T.cat([P[w], F])
            ks.append(k / k.norm().clamp(min=1e-9))
            F = step(F, P[w])
        return ks

    def score(s):
        ks = keys_of(s)
        W = T.zeros(D, 2 * D, device=DEV)
        for _ in range(epochs):
            for t in range(len(s) - 1):
                W += ETA * T.outer(P[s[t + 1]] - W @ ks[t], ks[t])
        res = max(float((P[s[t + 1]] - W @ ks[t]).abs().max()) for t in range(len(s) - 1))
        V, F = P[s[0]].clone(), T.zeros(D, device=DEV)
        out = [s[0]]
        for _ in range(len(s) - 1):
            k = T.cat([V, F])
            k = k / k.norm().clamp(min=1e-9)
            fired = ((W @ k) > TH_A).float()
            w = int((Pn @ (fired / fired.norm())).argmax()) if fired.sum() > 0 else 12
            out.append(w)
            F = step(F, V)
            V = fired if fired.sum() > 0 else T.zeros(D, device=DEV)
        hit = sum(1 for a, b in zip(out[1:], s[1:]) if a == b)
        spread = 0.0
        if key_spread and len(ks) > 3:
            spread = float(ks[2] @ ks[3])
        return out, hit, res, spread

    counts = []
    for k in (1, 2, 3, 4):
        s = [3] + [0] * k + [9, 4]
        out, hit, res, _ = score(s)
        counts.append('%d:%d/%d' % (k, hit, len(s) - 1))
    s = [7, 0, 0, 0, 2, 11]
    out, hit, res, _ = score(s)
    print('  decay %.2f-%.3f  scale %.0f  epochs %4d   count %s   hard %d/%d  residual %.3f'
          % (gmin, gmax, scale, epochs, ' '.join(counts), hit, len(s) - 1, res), flush=True)


print('counting a run of 0s  (k: correct steps / total)', flush=True)
for gmin, gmax, scale, epochs in ((0.60, 0.99, 5.0, 500),
                                  (0.60, 0.99, 5.0, 1500),
                                  (0.40, 0.999, 10.0, 800),
                                  (0.40, 0.999, 10.0, 2000),
                                  (0.20, 0.9995, 20.0, 2000)):
    run(gmin, gmax, scale, epochs)
