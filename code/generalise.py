"""Does it generalise when there is much more data -- dialogue and non-dialogue both?

Up to now every conversation had its own connection matrix: the model remembered one dialogue. Here there is ONE
matrix for all of them, and the question is whether the answer to a question it has never seen starts to come out.

Data
  dialogue     tinytalks, 301 question/answer pairs; 40 held out for testing
  monologue    the-verdict.txt, fed as a plain stream of heard words (no turns)
Training set size is swept so a curve over data size can be seen.

Model: unchanged. Two regions of the same layer type (visual and motor), a word's code is the visual pattern when
heard and the motor pattern when spoken, the key is [code ; context field], the drive is a cosine with threshold
0.7, and the synapses are adjusted by the error at the cell they land on. The only difference from before is that
the same error rule is applied to all pairs at once instead of one sentence at a time -- the batch form of the
same local rule (every synapse moves by the summed error of the cells it lands on times its own input).

Usage: python generalise.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 40, 2, 0.6
TH_A, ETA, ITERS = 0.7, 0.5, 400
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
D0 = S * S * MZ
DT = 2 * D0
DIA = r'D:\deepseekherness\data\dialogue\tinytalks_v1.txt'
MONO = r'D:\deepseekherness\data\monologue\corpus.txt'
MAXV = 500


def tok(s):
    for ch in '?.,!;:"':
        s = s.replace(ch, ' ' + ch + ' ')
    return s.lower().split()


pairs = []
for b in open(DIA, encoding='utf-8').read().split('\n\n'):
    lines = [l for l in b.strip().split('\n') if l.strip()]
    if len(lines) >= 2 and lines[0].startswith('Q:') and lines[1].startswith('A:'):
        q, a = tok(lines[0][2:]), tok(lines[1][2:])
        if 0 < len(q) <= 12 and 0 < len(a) <= 14:
            pairs.append((q, a))

freq = {}
for q, a in pairs:
    for w in q + a:
        freq[w] = freq.get(w, 0) + 1
keep = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:MAXV]]
vocab = ['<sil>'] + keep
IDX = {w: i for i, w in enumerate(vocab)}
SIL, NV = 0, len(vocab)
pairs = [([IDX[w] for w in q], [IDX[w] for w in a]) for q, a in pairs
         if all(w in IDX for w in q + a)]
mono = [IDX[w] for w in tok(open(MONO, encoding='utf-8').read()) if w in IDX]
print('%d pairs usable, %d words in the vocabulary, %d monologue words' % (len(pairs), NV, len(mono)), flush=True)

rng = np.random.default_rng(7)
G = np.zeros((NV, S * S), np.float32)
for i in range(1, NV):
    G[i, rng.choice(S * S, 10, replace=False)] = 1.0
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
mot = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
mot.theta = [TH]
with T.no_grad():
    P = vis.forward(T.as_tensor(G, device=DEV))[-1].float()
    M = mot.forward(T.as_tensor(G, device=DEV))[-1].float()
NA = float(P[1:].sum(1).mean())
GAMP = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, DT).astype(np.float32), device=DEV)
MVP = M / M.norm(dim=1, keepdim=True).clamp(min=1e-9)


def heard(w):
    v = T.zeros(DT, device=DEV)
    v[:D0] = P[w]
    return v


def spoken(w):
    v = T.zeros(DT, device=DEV)
    v[D0:] = M[w]
    return v


def code(w, ph):
    if w == SIL:
        return T.zeros(DT, device=DEV)
    return spoken(w) if ph else heard(w)


def step(F, V):
    F = GAMP * F + V
    if float(V.sum()) > 0.5:
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return F


def seqs(pairs_, with_mono):
    out = []
    for q, a in pairs_:
        out.append([(w, 0) for w in q] + [(SIL, 0)] + [(w, 1) for w in a] + [(SIL, 1)])
    if with_mono and len(mono) > 20:
        L = 14
        for i in range(0, len(mono) - L, L):
            out.append([(w, 0) for w in mono[i:i + L]])
    return out


def batch(ss):
    ks, vs = [], []
    for s in ss:
        F = T.zeros(DT, device=DEV)
        for t in range(len(s) - 1):
            v = code(s[t][0], s[t][1])
            k = T.cat([v, F])
            ks.append(k / k.norm().clamp(min=1e-9))
            vs.append(code(s[t + 1][0], s[t + 1][1]))
            F = step(F, v)
    return T.stack(ks), T.stack(vs)


def train(ss, lam=1e-3):
    """the same rule, solved instead of iterated: the fixed point of the delta rule is the least-squares fit.
    lam is the weight decay relative to the number of transitions."""
    K, V = batch(ss)
    A = K.T @ K
    B = V.T @ K
    lam = lam * K.shape[0]
    A = A + lam * T.eye(A.shape[0], device=DEV)
    return T.linalg.solve(A, B.T).T


def answer(W, q, maxlen=18, th=TH_A):
    F = T.zeros(DT, device=DEV)
    for w in q:
        F = step(F, heard(w))
    V = T.zeros(DT, device=DEV)          # the key at the pause is built before the fade, exactly as in training
    out, peak = [], 0.0
    for _ in range(maxlen):
        k = T.cat([V, F])
        k = k / k.norm().clamp(min=1e-9)
        act = W @ k
        fired = (act > th).float()
        peak = max(peak, float(act.max()))
        if fired.sum() == 0:
            break
        out.append(int((MVP @ (fired[D0:] / fired[D0:].norm().clamp(min=1e-9))).argmax()))
        F = step(F, V)
        V = fired
    return out, peak


idx = np.random.default_rng(2024).permutation(len(pairs))
test = [pairs[i] for i in idx[:40]]
pool = [pairs[i] for i in idx[40:]]

print('', flush=True)
print('train size | dialogue only        | dialogue + monologue', flush=True)
print('           | exact  word-acc      | exact  word-acc', flush=True)
for k in (200,):
    row = []
    for with_mono in (False, True):
        ss = seqs(pool[:k], with_mono)
        W = train(ss)
        ex = acc = tot = 0
        for q, a in test:
            got, _ = answer(W, q)
            ex += (got == a)
            acc += sum(1 for x, y in zip(got, a) if x == y)
            tot += max(len(a), 1)
        row.append('%2d/40  %.1f%%' % (ex, 100.0 * acc / tot))
    print('   %3d     | %-26s | %s' % (k, row[0], row[1]), flush=True)

print('', flush=True)
print('weight decay (ridge) sweep, 200 pairs, one connection set:', flush=True)
ss = seqs(pool[:200], False)
seq0 = seqs([pool[0]], False)[0]
ip = len(pool[0][0])
F = T.zeros(DT, device=DEV)
ktr = []
for t in range(len(seq0) - 1):
    v = code(seq0[t][0], seq0[t][1])
    k = T.cat([v, F])
    ktr.append(k / k.norm().clamp(min=1e-9))
    F = step(F, v)
tgt = code(seq0[ip + 1][0], seq0[ip + 1][1])
print('  decay    mean err   drive on the right next word at the pause   trained   held-out', flush=True)
for lam in (1e-7,):
    W = train(ss, lam)
    K, V = batch(ss)
    err = float((V - K @ W.T).abs().mean())
    dr = (W @ ktr[ip])[tgt > 0.5]
    tr = sum(answer(W, q)[0] == a for q, a in pool[:40])
    te = sum(answer(W, q)[0] == a for q, a in test)
    print('  %.0e   %.4f     %.2f .. %.2f   %2d/40     %2d/40'
          % (lam, err, float(dr.min()), float(dr.max()), tr, te), flush=True)
