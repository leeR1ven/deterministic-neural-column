"""The designer's correction: a word the model says is not the same thing as a word it hears.

In dialogue_turns.py both used the same code -- the picture. So "hear bye, then say bye" was two identical states
in a row, which is the case that cannot work. But a word produced through the motor area is a different event
from the same word arriving through the ears, and the brain gives them different representations.

So every word now has two codes:
  heard   the visual pattern of its picture
  spoken  its own block of cells in the memory space, one block per word, fed back from the motor area
The key is still [current code ; context] and the synapses are still adjusted by the error at the cell they land
on.  What changes is only that during its own turn the current code is the spoken one.

Motor area: 20 cells per word, each reading one cell of that word's block (weight 4) plus the basal-ganglia
inhibition (weight -3).  Total weight 7, threshold 3.5:
  BG active  -> 4 - 3 = 1, no motor cell fires, the model is silent
  BG quiet   -> 4, the word is spoken

Usage: python dialogue_motor.py
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
TH_A, ETA, EPOCHS = 0.7, 0.25, 1500
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
BLK = 20
DATA = r'D:\deepseekherness\data\dialogue\tinytalks_v1.txt'


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
mot.theta = [TH]                                     # the motor area: same design as the visual one
with T.no_grad():
    P = vis.forward(T.as_tensor(G, device=DEV))[-1].float()
    M = mot.forward(T.as_tensor(G, device=DEV))[-1].float()
D = P.shape[1]
NA = float(P[1:].sum(1).mean())
GAM = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, D).astype(np.float32), device=DEV)

SPK = M                                # the spoken code of a word is its motor region's pattern
print('vocabulary %d words, %d cells; motor patterns %d cells on average'
      % (NV, D, int(M[1:].sum(1).mean())), flush=True)

# readout cells: 20 per word, each taking four of that word's motor cells, plus the basal-ganglia inhibition
MPW = 20
mrng = np.random.default_rng(13)
MOT = T.zeros(NV, MPW, 4, dtype=T.long)
for w in range(1, NV):
    idx = (SPK[w] > 0.5).nonzero().flatten()
    MOT[w] = T.as_tensor(idx[mrng.integers(0, len(idx), MPW * 4)].reshape(MPW, 4), device=DEV)
MOT = MOT.to(DEV)


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


def code(w, phase):
    if w == SIL:
        return T.zeros(D, device=DEV)
    return SPK[w] if phase == 'speak' else P[w]


def keys_of(seq):
    F, ks = T.zeros(D, device=DEV), []
    for w, ph in seq:
        k = T.cat([code(w, ph), F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = step(F, code(w, ph))
    return ks


def learn(seq, epochs=EPOCHS):
    ks = keys_of(seq)
    W = T.zeros(D, 2 * D, device=DEV)
    for _ in range(epochs):
        for t in range(len(seq) - 1):
            tgt = code(seq[t + 1][0], seq[t + 1][1])
            W += ETA * T.outer(tgt - W @ ks[t], ks[t])
    res = max(float((code(seq[t + 1][0], seq[t + 1][1]) - W @ ks[t]).abs().max())
              for t in range(len(seq) - 1))
    return W, res


def run(seq, turns, epochs=EPOCHS):
    W, res = learn(seq, epochs)
    F, said, quiet = T.zeros(D, device=DEV), [], True
    for words, maxlen in turns:
        for w in words:
            if speak(P[w], 1)[0] is not None:
                quiet = False
            F = step(F, P[w])
        F = step(F, T.zeros(D, device=DEV))          # the outside pauses before handing over
        V = T.zeros(D, device=DEV)
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
    said, quiet, res = run(seq, [(q, len(a) + 2)])
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
said, quiet, res = run(seq, [(q0, len(a0) + 2), (q1, len(a1) + 2)], epochs=1500)
print('', flush=True)
print('two exchanges (residual %.2f):' % res, flush=True)
print('  outside: %s' % ' '.join(vocab[w] for w in q0), flush=True)
print('  outside: %s' % ' '.join(vocab[w] for w in q1), flush=True)
print('  model  : %s' % ' '.join(vocab[w] for w in said if w != SIL), flush=True)
print('  want   : %s' % ' '.join(vocab[w] for w in a0 + a1 if w != SIL), flush=True)
print('  silent while the outside spoke: %s' % quiet, flush=True)
