"""How far does making each layer wider actually go?  Two layers always, only the width changes.

The dense connection set is not stored: the ridge solution that the delta rule converges to can be written
W = V^T (K K^T + lam)^-1 K, so a prediction is V^T (Ck k).  That is the same number the dense W would give,
it just never builds the big matrix, which is what lets the layer width go far past 40x40.

For each width the whole pipeline is rebuilt (new glyphs, new both layers, same 2 layers) and the split is
re-drawn, so the scatter across seeds is visible -- the designer's point is that a 19-vs-17 difference may
be nothing but noise.

Usage: python generalise2.py
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
MZ, TH = 2, 0.6
TH_A, LAM = 0.7, 1e-7
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
DIA = r'D:\deepseekherness\data\dialogue\tinytalks_v1.txt'
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
vocab = ['<sil>'] + [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:MAXV]]
IDX = {w: i for i, w in enumerate(vocab)}
SIL, NV = 0, len(vocab)
pairs = [([IDX[w] for w in q], [IDX[w] for w in a]) for q, a in pairs
         if all(w in IDX for w in q + a)]
print('%d pairs, %d words' % (len(pairs), NV), flush=True)


def run(S, seed):
    D0 = S * S * MZ
    DT = 2 * D0
    rng = np.random.default_rng(seed)
    G = np.zeros((NV, S * S), np.float32)
    for i in range(1, NV):
        G[i, rng.choice(S * S, 10, replace=False)] = 1.0
    vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1 + seed, dev=DEV)
    vis.theta = [TH]
    mot = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=101 + seed, dev=DEV)
    mot.theta = [TH]
    with T.no_grad():
        P = vis.forward(T.as_tensor(G, device=DEV))[-1].float()
        M = mot.forward(T.as_tensor(G, device=DEV))[-1].float()
    NA = float(P[1:].sum(1).mean())
    GAMP = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, DT).astype(np.float32), device=DEV)
    MVP = M / M.norm(dim=1, keepdim=True).clamp(min=1e-9)
    zero = T.zeros(DT, device=DEV)

    def heard(w):
        v = zero.clone()
        v[:D0] = P[w]
        return v

    def spoken(w):
        v = zero.clone()
        v[D0:] = M[w]
        return v

    def code(w, ph):
        if w == SIL:
            return zero.clone()
        return spoken(w) if ph else heard(w)

    def step(F, V):
        F = GAMP * F + V
        if float(V.sum()) > 0.5:
            F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
        return F

    def seqof(pairs_):
        return [[(w, 0) for w in q] + [(SIL, 0)] + [(w, 1) for w in a] + [(SIL, 1)] for q, a in pairs_]

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

    idx = np.random.default_rng(2024 + seed).permutation(len(pairs))
    test = [pairs[i] for i in idx[:40]]
    pool = [pairs[i] for i in idx[40:]]
    ss = seqof(pool[:200])
    K, V = batch(ss)
    Gm = K @ K.T
    Gm += LAM * K.shape[0] * T.eye(K.shape[0], device=DEV)
    Ck = T.linalg.solve(Gm, K)                           # (N, 2*DT)
    Vt = V.T

    def answer(q, maxlen=18):
        F = T.zeros(DT, device=DEV)
        for w in q:
            F = step(F, heard(w))
        Vv, out = zero.clone(), []
        for _ in range(maxlen):
            k = T.cat([Vv, F])
            k = k / k.norm().clamp(min=1e-9)
            act = Vt @ (Ck @ k)
            fired = (act > TH_A).float()
            if fired.sum() == 0:
                break
            out.append(int((MVP @ (fired[D0:] / fired[D0:].norm().clamp(min=1e-9))).argmax()))
            F = step(F, Vv)
            Vv = fired
        return out

    tr = sum(answer(q) == a for q, a in pool[:40])
    te = sum(answer(q) == a for q, a in test)
    err = float((V - (Vt @ (Ck @ K.T)).T).abs().mean())
    return D0 * 2, tr, te, err


print('', flush=True)
print(' width   cells   seed   trained   unseen   mean training error', flush=True)
for S in (28, 32, 40, 48, 64):
    for seed in (0, 1):
        cells, tr, te, err = run(S, seed)
        print('  %2dx%-3d %5d   %2d     %2d/40     %2d/40     %.4f'
              % (S, S, cells, seed, tr, te, err), flush=True)
