"""The motor area built exactly like the visual one -- two regions, same layer, same rules, same threshold.

The designer's point: the motor area should be designed just like the visual area, with the same activation; the
only difference is that one receives information and the other sends it. So it is literally the same layer type
with its own random wiring and its own cells:

  visual region  cells 0 .. D0-1        driven by the picture of the word when the model hears it
  motor region   cells D0 .. 2*D0-1     active when the model says the word
  a word code    the visual pattern for a word it hears, the motor pattern for a word it says
  the key        [ word code ; context field ]   (D = 2*D0 cells, so the two regions cannot be confused)
  the readout    20 cells per word, each taking four of that word's motor cells (its own region) and the
                 basal-ganglia inhibition of weight -3; total weight 7, threshold 3.5, so
                   BG active  -> 4 - 3 = 1 < 3.5, silent
                   BG quiet   -> 4 > 3.5, spoken

Everything else is the memory that already works: delayed context with one decay per cell, cosine drive with a
fixed threshold 0.7, synapses adjusted by the error at the cell they land on.

Usage: python dialogue_motor2.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 2, 0.6
TH_A, ETA = 0.7, 0.25
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
DATA = r'D:\deepseekherness\data\dialogue\tinytalks_v1.txt'
D0 = S * S * MZ
DT = 2 * D0


def tok(s):
    for ch in '?.,!':
        s = s.replace(ch, ' ' + ch + ' ')
    return s.lower().split()


def load(n):
    out = []
    for b in open(DATA, encoding='utf-8').read().split('\n\n'):
        lines = [l for l in b.strip().split('\n') if l.strip()]
        if len(lines) < 2 or not lines[0].startswith('Q:') or not lines[1].startswith('A:'):
            continue
        q, a = tok(lines[0][2:]), tok(lines[1][2:])
        if 0 < len(q) <= 7 and 0 < len(a) <= 8:
            out.append((q, a))
        if len(out) == n:
            break
    return out


PAIRS = load(8)
CONV = [PAIRS[0], PAIRS[1]]
vocab = ['<sil>']
for q, a in PAIRS + CONV:
    for w in q + a:
        if w not in vocab:
            vocab.append(w)
NV, SIL = len(vocab), 0
IDX = {w: i for i, w in enumerate(vocab)}
PAIRS = [([IDX[w] for w in q], [IDX[w] for w in a]) for q, a in PAIRS]
CONV = [PAIRS[0], PAIRS[1]]

rng = np.random.default_rng(7)
G = np.zeros((NV, S * S), np.float32)
for i in range(1, NV):
    G[i, rng.choice(S * S, 10, replace=False)] = 1.0

vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
mot = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
mot.theta = [TH]                                       # same layer, own wiring, own cells
with T.no_grad():
    P = vis.forward(T.as_tensor(G, device=DEV))[-1].float()
    M = mot.forward(T.as_tensor(G, device=DEV))[-1].float()
NA = float(P[1:].sum(1).mean())
GAM = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, DT).astype(np.float32), device=DEV)
print('%d words; visual patterns %.0f cells, motor patterns %.0f cells; %d + %d = %d cells'
      % (NV, NA, float(M[1:].sum(1).mean()), D0, D0, DT), flush=True)

MOT = T.zeros(NV, 20, 4, dtype=T.long)                 # readout cells, inside the motor region
mrng = np.random.default_rng(13)
for w in range(1, NV):
    idx = (M[w] > 0.5).nonzero().flatten() + D0
    MOT[w] = T.as_tensor(idx[mrng.integers(0, len(idx), 80)].reshape(20, 4), device=DEV)
MOT = MOT.to(DEV)


def heard(w):
    v = T.zeros(DT, device=DEV)
    v[:D0] = P[w]
    return v


def spoken(w):
    v = T.zeros(DT, device=DEV)
    v[D0:] = M[w]
    return v


def code(w, phase):
    if w == SIL:
        return T.zeros(DT, device=DEV)
    return spoken(w) if phase == 'speak' else heard(w)


def speak(fired, bg):
    best, conf = SIL, 0.0
    for w in range(1, NV):
        s = fired[MOT[w]].sum(1) - (3.0 if bg else 0.0)
        act = float((s > 3.5).float().mean())
        if act > conf:
            best, conf = w, act
    return (best if conf > 0.4 else None), conf


def step(F, V):
    F = GAM * F + V
    if float(V.sum()) > 0.5:
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return F


def keys_of(seq):
    F, ks = T.zeros(DT, device=DEV), []
    for w, ph in seq:
        k = T.cat([code(w, ph), F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = step(F, code(w, ph))
    return ks


def learn(seq, epochs):
    ks = keys_of(seq)
    W = T.zeros(DT, 2 * DT, device=DEV)
    for _ in range(epochs):
        for t in range(len(seq) - 1):
            W += ETA * T.outer(code(seq[t + 1][0], seq[t + 1][1]) - W @ ks[t], ks[t])
    res = max(float((code(seq[t + 1][0], seq[t + 1][1]) - W @ ks[t]).abs().max())
              for t in range(len(seq) - 1))
    return W, res


def run(seq, turns, epochs):
    W, res = learn(seq, epochs)
    F, said, quiet = T.zeros(DT, device=DEV), [], True
    for words, maxlen in turns:
        for w in words:
            if speak(heard(w), 1)[0] is not None:
                quiet = False
            F = step(F, heard(w))
        F = step(F, T.zeros(DT, device=DEV))            # the outside pauses
        V = T.zeros(DT, device=DEV)
        for _ in range(maxlen):
            k = T.cat([V, F])
            k = k / k.norm().clamp(min=1e-9)
            fired = ((W @ k) > TH_A).float()
            if fired.sum() == 0:
                said.append(SIL)
                break
            w, _ = speak(fired, 0)
            said.append(w if w is not None else SIL)
            F = step(F, V)
            V = fired
    return said, quiet, res


correct = 0
for q, a in PAIRS:
    seq = [(w, 'listen') for w in q] + [(SIL, 'listen')] + [(w, 'speak') for w in a] + [(SIL, 'speak')]
    said, quiet, res = run(seq, [(q, len(a) + 2)], 800)
    got = [x for x in said if x != SIL]
    ok = got == a
    correct += ok
    print('  Q: %-34s A: %-28s want %-28s %s'
          % (' '.join(vocab[w] for w in q), ' '.join(vocab[w] for w in got),
             ' '.join(vocab[w] for w in a), 'ok' if ok else '<- wrong'), flush=True)
print('', flush=True)
print('answers exactly right: %d/%d' % (correct, len(PAIRS)), flush=True)

(q0, a0), (q1, a1) = CONV
seq = ([(w, 'listen') for w in q0] + [(SIL, 'listen')] + [(w, 'speak') for w in a0] + [(SIL, 'speak')]
       + [(w, 'listen') for w in q1] + [(SIL, 'listen')] + [(w, 'speak') for w in a1] + [(SIL, 'speak')])
said, quiet, res = run(seq, [(q0, len(a0) + 2), (q1, len(a1) + 2)], 2500)
print('', flush=True)
print('two exchanges (residual %.2f):' % res, flush=True)
print('  outside: %s' % ' '.join(vocab[w] for w in q0), flush=True)
print('  outside: %s' % ' '.join(vocab[w] for w in q1), flush=True)
print('  model  : %s' % ' '.join(vocab[w] for w in said if w != SIL), flush=True)
print('  want   : %s' % ' '.join(vocab[w] for w in a0 + a1 if w != SIL), flush=True)
print('  silent while the outside spoke: %s' % quiet, flush=True)
