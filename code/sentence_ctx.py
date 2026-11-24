"""The context field must be delayed: it carries what came BEFORE, never the word being read now.

Why the earlier attempt failed (numbers from diag_walk.py): the state used as the key had the current word added
to it at full strength, so the two occurrences of word 0 produced almost the same key, and the chain had to
average what followed each of them -- 3 and 7 came out together (0.71 vs 0.69 similarity, both above the
threshold, so both fired).

Split the state in two, as the prefrontal cortex does:
  V   the word being seen now          (the visual pattern, transient)
  F   the context                       (a graded trace of the words that came before, nothing of the word now)
The synapses read [V ; F] and drive the next V.  The same past therefore lands in F with the same weight whether
or not the current word is a repeat, and "0 cold" and "0 after 3" are different keys.

F fades with GAMMA.  That fade is the context window: the slower it fades, the further back the model can see.

Constants, none calibrated to an activation fraction:
  TH_A 0.7   a synapse set by one observed transition gives exactly 1.0, a superposition of two gives less than
             0.7, so an unambiguous transition fires and a blurred one does not
  GAMMA      fade of the context trace (swept)
  TH 0.6     visual layer, as in the other scripts

Usage: python sentence_ctx.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 8, 0.6
TH_A = 0.7
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


def keys_of(sentence, gamma, mode):
    """mode cur = the word alone, ctx = word + delayed context, blend = word + context that contains the word."""
    F = T.zeros(D, device=DEV)
    ks = []
    for t in range(len(sentence)):
        V = P[sentence[t]]
        if mode == 'cur':
            k = V
        elif mode == 'ctx':
            k = T.cat([V, F])
        else:
            k = T.cat([V, F + V])
        ks.append(k)
        F = gamma * F + V
    return ks


def learn(sentence, gamma, mode):
    ks = keys_of(sentence, gamma, mode)
    W = T.zeros(D, ks[0].shape[0], device=DEV)
    for t in range(len(sentence) - 1):
        W += T.outer(P[sentence[t + 1]], ks[t]) / ks[t].pow(2).sum().clamp(min=1e-9)
    return W, ks


def walk(sentence, W, ks, gamma, mode, steps):
    Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
    V = P[sentence[0]].clone()
    F = T.zeros(D, device=DEV)
    out, margins = [sentence[0]], []
    for _ in range(steps):
        k = V if mode == 'cur' else (T.cat([V, F]) if mode == 'ctx' else T.cat([V, F + V]))
        act = W @ k
        want = P[sentence[len(out)]] if len(out) < len(sentence) else None
        if want is not None:
            tgt = float(act[want > 0.5].min())
            other = act.clone()
            other[want > 0.5] = 0.0
            margins.append((tgt, float(other.max())))
        fired = (act > TH_A).float()
        if fired.sum() == 0:
            out.append(-1)
            break
        sims = Pn @ (fired / fired.norm())
        out.append(int(sims.argmax()))
        V, F = fired, gamma * F + V
    return out, margins


for sentence in ([0, 3, 0, 7, 1], [0, 3, 0, 7, 1, 7, 3, 9, 5, 2]):
    steps = len(sentence) - 1
    print('sentence: %s' % sentence, flush=True)
    for mode in ('cur', 'blend', 'ctx'):
        for gamma in (0.6, 0.8, 0.9):
            W, ks = learn(sentence, gamma, mode)
            got, mar = walk(sentence, W, ks, gamma, mode, steps)
            hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
            weak = min((t - o) for t, o in mar) if mar else 0.0
            print('  %-5s gamma %.1f  %s  %d/%d  worst margin %+.2f'
                  % (mode, gamma, got[1:], hit, steps, weak), flush=True)
    print('', flush=True)
