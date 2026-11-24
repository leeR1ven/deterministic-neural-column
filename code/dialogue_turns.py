"""A first dialogue: the model listens to a question, then takes its turn and answers.

Built out of the pieces that now work: the graded context field (each cell its own decay), the key
[current word ; context], the cosine drive with a fixed threshold, and synapses adjusted by the error at the cell
they land on.  The only new part is the motor area and the turn.

  motor area   20 cells per word.  Each motor cell takes four of its word's active cells (weight 1 each) and one
               inhibitory input from the basal ganglia (weight -3).  With the total weight 7 and threshold 0.5 it
               needs a score above 3.5, so:
                 BG active (the outside is speaking)  -> 4 - 3 = 1, no motor cell can fire: the model is silent
                 BG quiet  (the model's turn)         -> 4, the word is spoken
               That is the whole gate: the basal ganglia are an inhibitory input to the motor cells, no symbol.
  turn          the experimenter sets it.  While BG is active the model's own output is not fed back, the outside
               word is; when BG goes quiet the model's own predicted word is fed back and spoken.
  end of turn   the model is taught a silence at the end of its answer and stops speaking there, so the turn
               goes back to the outside.

Tests: eight short questions from tinytalks, then one conversation with two exchanges in a row.

Usage: python dialogue_turns.py
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
TH_A, ETA, EPOCHS = 0.7, 0.25, 600
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
DATA = r'D:\deepseekherness\data\dialogue\tinytalks_v1.txt'


def tok(s):
    for ch in '?.,!':
        s = s.replace(ch, ' ' + ch + ' ')
    return s.lower().split()


def load(n):
    txt = open(DATA, encoding='utf-8').read()
    out = []
    for b in txt.split('\n\n'):
        lines = [l for l in b.strip().split('\n') if l.strip()]
        if len(lines) < 2 or not lines[0].startswith('Q:') or not lines[1].startswith('A:'):
            continue
        q = tok(lines[0][2:])
        a = tok(lines[1][2:])
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
NV = len(vocab)
SIL = 0
IDX = {w: i for i, w in enumerate(vocab)}
PAIRS = [([IDX[w] for w in q], [IDX[w] for w in a]) for q, a in PAIRS]
CONV = [PAIRS[0], PAIRS[1]]
print('vocabulary %d words; %d single questions' % (NV, len(PAIRS)), flush=True)

rng = np.random.default_rng(7)
G = np.zeros((NV, S * S), np.float32)
for i in range(1, NV):
    G[i, rng.choice(S * S, 10, replace=False)] = 1.0
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    P = vis.forward(T.as_tensor(G, device=DEV))[-1].float()
D = P.shape[1]
NA = float(P[1:].sum(1).mean())
GAM = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, D).astype(np.float32), device=DEV)

# motor area: 20 cells per word, four of that word's cells each, plus one inhibitory basal-ganglia input
MPW = 20
mrng = np.random.default_rng(13)
MOT = T.zeros(NV, MPW, 4, dtype=T.long)
for w in range(1, NV):
    idx = (P[w] > 0.5).nonzero().flatten()
    pick = idx[mrng.integers(0, len(idx), MPW * 4)].reshape(MPW, 4)
    MOT[w] = pick.to(DEV)
MOT = MOT.to(DEV)


def speak(fired, bg):
    """which word the motor area is saying, and how sure it is; bg 1 means the outside is speaking."""
    best, conf = SIL, 0.0
    for w in range(1, NV):
        s = fired[MOT[w]].sum(1) - (3.0 if bg else 0.0)
        act = (s > 3.5).float().mean()
        if float(act) > conf:
            best, conf = w, float(act)
    return (best if conf >= 0.5 else None), conf


def step(F, V):
    F = GAM * F + V
    if float(V.sum()) > 0.5:
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return F


def keys_of(seq):
    F, ks = T.zeros(D, device=DEV), []
    for w in seq:
        k = T.cat([P[w], F])
        ks.append(k / k.norm().clamp(min=1e-9))
        F = step(F, P[w])
    return ks


def learn(seq, epochs=EPOCHS):
    ks = keys_of(seq)
    W = T.zeros(D, 2 * D, device=DEV)
    for _ in range(epochs):
        for t in range(len(seq) - 1):
            W += ETA * T.outer(P[seq[t + 1]] - W @ ks[t], ks[t])
    res = max(float((P[seq[t + 1]] - W @ ks[t]).abs().max()) for t in range(len(seq) - 1))
    return W, res


def run(seq, turns, epochs=EPOCHS):
    """turns: list of (listen_words, max_answer) --- BG active while listening, quiet while answering."""
    W, res = learn(seq, epochs)
    F, said = T.zeros(D, device=DEV), []
    quiet_during_listen = True
    for words, maxlen in turns:
        for w in words:
            if speak(P[w], 1)[0] is not None:
                quiet_during_listen = False
            F = step(F, P[w])
            V = P[w]
        F = step(F, T.zeros(D, device=DEV))      # the outside pauses before the turn changes hands
        V = T.zeros(D, device=DEV)
        for _ in range(maxlen):
            k = T.cat([V, F])
            k = k / k.norm().clamp(min=1e-9)
            fired = ((W @ k) > TH_A).float()
            w, _ = speak(fired, 0)
            if fired.sum() == 0:
                said.append(SIL)
                break
            said.append(w if w is not None else SIL)
            F = step(F, V)
            V = fired
    return said, quiet_during_listen, res


correct = 0
for q, a in PAIRS:
    said, quiet, res = run(q + [SIL] + a + [SIL], [((q), len(a) + 2)])
    ok = [x for x in said if x != SIL] == a
    correct += ok
    print('  Q: %-34s A: %-30s want %-30s %s'
          % (' '.join(vocab[w] for w in q), ' '.join(vocab[w] for w in said if w != SIL),
             ' '.join(vocab[w] for w in a), 'ok' if ok else '<- wrong'), flush=True)
print('', flush=True)
print('answers exactly right: %d/%d' % (correct, len(PAIRS)), flush=True)

(q0, a0), (q1, a1) = CONV
seq = q0 + [SIL] + a0 + [SIL] + q1 + [SIL] + a1 + [SIL]
said, quiet, res = run(seq, [(q0, len(a0) + 2), (q1, len(a1) + 2)], epochs=1500)
print('', flush=True)
print('two exchanges in one conversation (learning residual %.2f):' % res, flush=True)
print('  outside: %s' % ' '.join(vocab[w] for w in q0), flush=True)
print('  model  : %s' % ' '.join(vocab[w] for w in said if w != SIL), flush=True)
print('  want   : %s' % ' '.join(vocab[w] for w in a0 + a1 if w != SIL), flush=True)
print('  outside then: %s' % ' '.join(vocab[w] for w in q1), flush=True)
print('  motor area stayed silent while the outside was speaking: %s' % quiet, flush=True)
