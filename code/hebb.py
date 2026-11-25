"""Plain Hebbian growth with one local rule on top: each cell's total input weight is kept fixed.

No equations, no least squares.  Two cells that are active at the same moment grow a synapse; afterwards every
cell's incoming weights are rescaled to the same total, which is the homeostatic rule that a cell can check
locally.  With that, a cell that is the target of one state fires fully (the whole weight sits on that one key)
and a cell that several different states try to drive only gets a fraction of it -- so competing successors
suppress each other without any equation being solved.

Everything else is unchanged: two regions of the same layer type, heard/spoken codes, key = [code ; context],
threshold 0.7, silence ends the turn.

Usage: python hebb.py
"""
import sys

import numpy as np
import scipy.sparse as sp
import torch as T

sys.path.insert(0, r'D:\deepseekherness\repo_publish\deterministic-neural-column\code')
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 96, 2, 0.6
TH_A = 0.7
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
D0, DT = S * S * MZ, 2 * S * S * MZ
MAXV = 500
DIA = r'D:\deepseekherness\data\dialogue\tinytalks_v1.txt'


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
pairs = [([IDX[w] for w in q], [IDX[w] for w in a]) for q, a in pairs if all(w in IDX for w in q + a)]
print('%d pairs, %d words, %d cells per region' % (len(pairs), NV, D0), flush=True)

vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
mot = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)
mot.theta = [TH]
rng = np.random.default_rng(0)
P, M = {}, {}
with T.no_grad():
    for w in range(1, NV):
        x = np.zeros((1, S * S), np.float32)
        x[0, rng.choice(S * S, 10, replace=False)] = 1.0
        xt = T.as_tensor(x, device=DEV)
        P[w] = vis.forward(xt)[-1][0].nonzero().flatten().cpu().numpy()
        M[w] = mot.forward(xt)[-1][0].nonzero().flatten().cpu().numpy()
NA = float(np.mean([len(P[w]) for w in range(1, NV)]))
GAM = np.random.default_rng(5).uniform(GMIN, GMAX, DT).astype(np.float32)
Ms = sp.csr_matrix((np.ones(sum(len(M[w]) for w in range(1, NV))),
                    (np.concatenate([np.full(len(M[w]), w) for w in range(1, NV)]),
                     np.concatenate([M[w] for w in range(1, NV)]))), shape=(NV, D0))


def stepf(F, idx):
    out = {}
    for i, v in F.items():
        nv = v * GAM[i]
        if nv > 1e-4:
            out[i] = nv
    for i in idx:
        out[i] = 1.0
    tot = sum(out.values())
    if tot > 0:
        sc = SCALE * NA / tot
        for i in out:
            out[i] *= sc
    return out


def kdict(Vidx, F):
    ci = np.concatenate([np.asarray(Vidx, np.int64), np.fromiter(F.keys(), np.int64, len(F)) + DT]) \
        if len(F) else np.asarray(Vidx, np.int64)
    cv = np.concatenate([np.ones(len(Vidx), np.float32), np.fromiter(F.values(), np.float32, len(F))]) \
        if len(F) else np.ones(len(Vidx), np.float32)
    if len(ci) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return ci, cv / np.sqrt((cv ** 2).sum())


# ---- plain Hebbian: every transition grows a synapse from each active key cell onto each target cell ----
rows, cols, data, Tcols, TW = [], [], [], [], []
F = {}
for pi, (q, a) in enumerate(pairs):
    seq = [(P[w], 0) for w in q] + [(np.zeros(0), 0)] + [(M[w], D0) for w in a] + [(np.zeros(0), D0)]
    F = {}
    for t in range(len(seq) - 1):
        idx, off = seq[t]
        nidx, noff = seq[t + 1]
        ci, cv = kdict(idx, F)
        for j in nidx + noff:
            rows.append(np.full(len(ci), int(j), np.int32))
            cols.append(ci.astype(np.int32))
        Tcols.append(nidx + noff)
        F = stepf(F, idx + off)
rows = np.concatenate(rows); cols = np.concatenate(cols)
rs = np.bincount(rows, minlength=DT).astype(np.float32)
W = sp.csr_matrix((1.0 / rs[rows], (rows, cols)), shape=(DT, 2 * DT))
print('%d synapses; a cell drives one state fully, two competing states at half' % len(rows), flush=True)


def answer(q, maxlen=18):
    F = {}
    for w in q:
        F = stepf(F, P[w])
    V, out = np.zeros(0, np.int64), []
    for _ in range(maxlen):
        ci, cv = kdict(V, F)
        kd = np.zeros(2 * DT, np.float32)
        kd[ci] = cv
        act = np.asarray(W @ kd).ravel()
        fired = act > TH_A
        if fired.sum() == 0:
            break
        sc = Ms @ fired[D0:].astype(np.float32)
        out.append(int(sc.argmax()) if sc.max() > 0 else SIL)
        F = stepf(F, V)
        V = np.nonzero(fired)[0]
    return out


ex = sum(answer(q) == a for q, a in pairs[:40])
print('', flush=True)
print('sentences exactly right: %d/40' % ex, flush=True)
for q, a in pairs[3:6]:
    print('  want: %-42s' % ' '.join(vocab[w] for w in a), flush=True)
    print('  got : %-42s' % ' '.join(vocab[w] for w in answer(q)), flush=True)
