"""960x960 per layer, two layers: 1,843,200 cells.  Keys are 99.99% zeros, so they are stored sparse.

Same model as generalise2.py -- two regions of the same layer type, heard/spoken codes, key = [code ; context],
threshold 0.7, and the connection set is the least-squares fixed point of the error rule (dual form, so the
dense matrix is never built).  Only the storage changes, and it is exact: a key row holds ~150 non-zero cells
out of 3.7M.

Usage: python huge.py
"""
import sys

import numpy as np
import scipy.sparse as sp
import torch as T

sys.path.insert(0, r'D:\deepseekherness\repo_publish\deterministic-neural-column\code')
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 960, 2, 0.6
TH_A, LAM = 0.7, 1e-7
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
Pidx, Midx = [None] * NV, [None] * NV
with T.no_grad():
    for w in range(1, NV):
        x = np.zeros((1, S * S), np.float32)
        x[0, rng.choice(S * S, 10, replace=False)] = 1.0
        xt = T.as_tensor(x, device=DEV)
        Pidx[w] = vis.forward(xt)[-1][0].nonzero().flatten().cpu().numpy().astype(np.int32)
        Midx[w] = mot.forward(xt)[-1][0].nonzero().flatten().cpu().numpy().astype(np.int32)
NA = float(np.mean([len(Pidx[w]) for w in range(1, NV)]))
print('word patterns: %d cells' % int(NA), flush=True)
GAM = np.random.default_rng(5).uniform(GMIN, GMAX, DT).astype(np.float32)
nz = [np.array([], np.int32)] * NV


def stepf(F, idx):
    """F: dict index->value.  Add the code's cells (value 1), fade the rest, hold the total at SCALE words."""
    out = {}
    for i, v in F.items():
        nv = v * GAM[i]
        if nv > 1e-4:
            out[i] = nv
    for i in idx:
        out[i] = 1.0
    tot = sum(out.values())
    if tot > 0 and len(idx) > 0:
        sc = SCALE * NA / tot
        for i in out:
            out[i] *= sc
    return out


def build(seq):
    """seq: list of (code index array, offset) ; returns key rows, target rows"""
    rows, cols, vals, trows, tcols = [], [], [], [], []
    F = {}
    r = 0
    for t in range(len(seq) - 1):
        idx, off = seq[t]
        nidx, noff = seq[t + 1]
        ci = np.concatenate([idx + off, np.fromiter(F.keys(), np.int32, len(F)) + D0]) if F else idx + off
        cv = np.concatenate([np.ones(len(idx), np.float32), np.fromiter(F.values(), np.float32, len(F))]) if F else np.ones(len(idx), np.float32)
        nrm = np.sqrt((cv ** 2).sum())
        rows.append(np.full(len(ci), r, np.int32)); cols.append(ci); vals.append(cv / nrm)
        trows.append(np.full(len(nidx), r, np.int32)); tcols.append(nidx + noff)
        F = stepf(F, idx + off)
        r += 1
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals),
            np.concatenate(trows), np.concatenate(tcols), r)


HEARDOFF, SPOKENOFF = 0, D0
Krows, Kcols, Kvals, Trows, Tcols, N = [], [], [], [], [], 0
for q, a in pairs:
    seq = [(Pidx[w], HEARDOFF) for w in q] + [(nz[0], HEARDOFF)] + \
          [(Midx[w], D0) for w in a] + [(nz[0], D0)]
    a1, a2, a3, a4, a5, r = build(seq)
    Krows.append(a1 + N); Kcols.append(a2); Kvals.append(a3)
    Trows.append(a4 + N); Tcols.append(a5); N += r
    if N == 0:
        break
print('%d transitions, %d non-zero key entries (dense would be %.1f GB)'
      % (N, sum(len(x) for x in Kcols), N * 2 * DT * 4 / 1e9), flush=True)

K = sp.csr_matrix((np.concatenate(Kvals), (np.concatenate(Krows), np.concatenate(Kcols))),
                  shape=(N, 2 * DT))
A = (K @ K.T).toarray().astype(np.float64)
A += LAM * N * np.eye(N)
L = np.linalg.cholesky(A)
Kc = K.tocsc()
Wt_rows = np.concatenate(Trows); Wt_cols = np.concatenate(Tcols)
Ms = sp.csr_matrix((np.ones(sum(len(Pidx[w]) for w in range(1, NV))),
                    (np.concatenate([np.full(len(Midx[w]), w) for w in range(1, NV)]),
                     np.concatenate([Midx[w] for w in range(1, NV)]))), shape=(NV, D0))


def solve_b(kk):
    b = np.zeros(N)
    for i, v in kk.items():
        col = Kc[:, i]
        b[col.indices] += v * col.data
    return b


def predict(kk):
    import scipy.linalg as sla
    al = sla.cho_solve((L, True), solve_b(kk))
    return np.bincount(Wt_cols, weights=al[Wt_rows], minlength=DT)[:DT]


def keyfrom(Vidx, F):
    ci = np.concatenate([Vidx, np.fromiter(F.keys(), np.int32, len(F)) + D0]) if F else Vidx.astype(np.int32)
    cv = np.concatenate([np.ones(len(Vidx), np.float32), np.fromiter(F.values(), np.float32, len(F))]) if F else np.ones(len(Vidx), np.float32)
    k = {}
    for i, v in zip(ci, cv / np.sqrt((cv ** 2).sum())):
        k[int(i)] = float(v)
    return k


def answer(q, maxlen=18):
    F = {}
    for w in q:
        F = stepf(F, Pidx[w])
    Vidx, out = nz[0], []
    for _ in range(maxlen):
        act = predict(keyfrom(Vidx, F))
        fired = act > TH_A
        if not fired.any():
            break
        sc = Ms @ fired[D0:].astype(np.float32)
        out.append(int(sc.argmax()) if sc.max() > 0 else SIL)
        F = stepf(F, Vidx)
        Vidx = np.nonzero(fired)[0].astype(np.int32)
    return out


test = pairs[:40]
pool = pairs[40:]
tr = 0
for q, a in pool[:10]:
    tr += (answer(q) == a)
te = 0
for q, a in test[:40]:
    te += (answer(q) == a)
print('', flush=True)
print('trained exact: %d/10     unseen exact: %d/40' % (tr, te), flush=True)
