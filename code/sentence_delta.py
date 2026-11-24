"""The long sentence fails for a reason the short one hides: when a word is the answer more than once, the two
learned transitions ADD UP on that word's cells, so a word that was predicted twice gets a double drive and fires
in the wrong place. The short sentence never repeats an answer, which is why it worked.

Fix, still local: each synapse adjusts by the error at the cell it lands on -- err = (what should have fired) -
(what did fire), and the synapse moves by err times the activity of its own input. That is the Widrow-Hoff rule,
which is a local rule (the postsynaptic cell's error is the only extra signal), and it drives the summed
superposition down to the correct answer.

Compared here, same keys and same walk:
  chain   one observed transition, one fixed synapse growth (the rule used so far)
  delta   the same synapses, adjusted 30 times by the postsynaptic error

Usage: python sentence_delta.py
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
TH_A, GAMMA, ETA, EPOCHS = 0.7, 0.8, 0.2, 800
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
T.manual_seed(0)


NORMF = True


def keys_of(sentence, gamma=GAMMA, normf=NORMF):
    """key at frame t = [the word being read now ; the graded trace of the words before it]"""
    F = T.zeros(D, device=DEV)
    ks = []
    for t in range(len(sentence)):
        V = P[sentence[t]]
        k = T.cat([V, F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = gamma * F + V
        if normf:
            F = F * (V.sum().clamp(min=1) / F.sum().clamp(min=1e-9))
    return ks


def learn_chain(sentence, ks, gamma=GAMMA):
    W = T.zeros(D, 2 * D, device=DEV)
    for t in range(len(sentence) - 1):
        W += T.outer(P[sentence[t + 1]], ks[t])
    return W


def learn_delta(sentence, ks, eta=ETA, epochs=EPOCHS, gamma=GAMMA):
    W = T.zeros(D, 2 * D, device=DEV)
    for _ in range(epochs):
        for t in range(len(sentence) - 1):
            err = P[sentence[t + 1]] - W @ ks[t]
            W += eta * T.outer(err, ks[t])
    return W


def walk(sentence, W, steps, gamma=GAMMA, normf=NORMF):
    V, F = P[sentence[0]].clone(), T.zeros(D, device=DEV)
    out, mar = [sentence[0]], []
    for i in range(steps):
        k = T.cat([V, F])
        k = k / k.norm().clamp(min=1e-9)
        act = W @ k
        if i + 1 < len(sentence):
            want = P[sentence[i + 1]]
            tgt = float(act[want > 0.5].min())
            o = act.clone()
            o[want > 0.5] = -9
            mar.append(tgt - float(o.max()))
        fired = (act > TH_A).float()
        if fired.sum() == 0:
            out.append(-1)
            break
        out.append(int((Pn @ (fired / fired.norm())).argmax()))
        F = gamma * F + V
        if normf:
            F = F * (V.sum().clamp(min=1) / F.sum().clamp(min=1e-9))
        V = fired
    return out, mar


SENTS = ([0, 3, 0, 7, 1],
         [0, 3, 0, 7, 1, 7, 3, 9, 5, 2],
         [1, 4, 1, 4, 9, 2, 9, 6, 8, 3, 8, 5],
         [2, 2, 8, 8, 4, 2, 4, 6, 6, 9],
         [5, 1, 5, 1, 3, 7, 3, 7, 0, 0, 11, 6])

exact = {'chain': 0, 'delta': 0}
for sentence in SENTS:
    steps = len(sentence) - 1
    print('sentence: %s' % sentence, flush=True)
    ks = keys_of(sentence)
    for name, W in (('chain', learn_chain(sentence, ks)), ('delta', learn_delta(sentence, ks))):
        got, mar = walk(sentence, W, steps)
        hit = sum(1 for a, b in zip(got[1:], sentence[1:]) if a == b)
        exact[name] += (hit == steps)
        res = max(float((P[sentence[t + 1]] - W @ ks[t]).abs().max()) for t in range(steps))
        print('  %-5s  %s  %d/%d  worst margin %+.2f  residual %.3f'
              % (name, got[1:], hit, steps, min(mar) if mar else 0.0, res), flush=True)
    print('', flush=True)
print('sentences recalled exactly:   chain %d/%d   delta %d/%d'
      % (exact['chain'], len(SENTS), exact['delta'], len(SENTS)), flush=True)
