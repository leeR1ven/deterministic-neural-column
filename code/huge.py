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
S, MZ, TH = 96, 2, 0.6
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
    if tot > 0:
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
        ci = np.concatenate([idx + off, np.fromiter(F.keys(), np.int32, len(F)) + DT]) if F else idx + off
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
TW, PAUSE = [], {}
for pi, (q, a) in enumerate(pairs):
    seq = [(Pidx[w], HEARDOFF) for w in q] + [(nz[0], HEARDOFF)] + \
          [(Midx[w], D0) for w in a] + [(nz[0], D0)]
    wl = [w for w in q] + [SIL] + [w for w in a] + [SIL]
    PAUSE[pi] = N + len(q)
    TW.append(np.array(wl[1:len(seq)], np.int32))
    a1, a2, a3, a4, a5, r = build(seq)
    Krows.append(a1 + N); Kcols.append(a2); Kvals.append(a3)
    Trows.append(a4 + N); Tcols.append(a5); N += r
TW = np.concatenate(TW)
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

# every transition gets its own strength: how hard it drives its own target, then scale it to 1.
Wt = sp.csr_matrix((np.ones(len(Wt_rows), np.float32), (Wt_rows, Wt_cols)), shape=(N, DT))
Tov = np.asarray((Wt @ Wt.T).todense()).astype(np.float64)
nV = np.asarray(Wt.sum(1)).ravel().astype(np.float64)
Ainv = np.linalg.inv(A + 1e-3 * N * np.eye(N))
sd = np.ones(N)
for w in range(1, NV):
    S = np.nonzero(TW == w)[0]
    if len(S):
        sd[S] = Ainv[np.ix_(S, S)].sum(1)
SWt = np.ones(N)   # per-transition scaling amplifies the split transitions (13x) and makes it worse: 3/40
print('every transition drives its own target at exactly 1.0 (scales %.2f..%.2f)'
      % (SWt.min(), SWt.max()), flush=True)


def solve_b(kk):
    b = np.zeros(N)
    for i, v in kk.items():
        col = Kc[:, i]
        b[col.indices] += v * col.data
    return b


def predict(kk):
    import scipy.linalg as sla
    al = sla.cho_solve((L, True), solve_b(kk)) * SWt
    return np.bincount(Wt_cols, weights=al[Wt_rows], minlength=DT)[:DT]


def keyfrom(Vidx, F):
    ci = np.concatenate([Vidx, np.fromiter(F.keys(), np.int32, len(F)) + DT]) if F else Vidx.astype(np.int32)
    cv = np.concatenate([np.ones(len(Vidx), np.float32), np.fromiter(F.values(), np.float32, len(F))]) if F else np.ones(len(Vidx), np.float32)
    k = {}
    for i, v in zip(ci, cv / np.sqrt((cv ** 2).sum())):
        k[int(i)] = float(v)
    return k


def answer(q, maxlen=18, s=1.0):
    F = {}
    for w in q:
        F = stepf(F, Pidx[w])
    Vidx, out = nz[0], []
    for _ in range(maxlen):
        act = predict(keyfrom(Vidx, F)) * s
        fired = act > TH_A
        if not fired.any():
            break
        sc = Ms @ fired[D0:].astype(np.float32)
        out.append(int(sc.argmax()) if sc.max() > 0 else SIL)
        F = stepf(F, Vidx)
        Vidx = np.nonzero(fired)[0].astype(np.int32)
    return out


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


print('', flush=True)
pi = 3
q, a = pairs[pi]
F = {}
for w in q:
    F = stepf(F, Pidx[w])
b = solve_b(keyfrom(np.array([], np.int32), F))
top = np.argsort(-b)[:6]
print('the pause state of "%s" wants "%s"; the states closest to it are:'
      % (' '.join(vocab[w] for w in q)[:44], vocab[TW[PAUSE[pi]]]), flush=True)
for i in top:
    print('   cosine %.3f   that state wanted "%s"%s'
          % (b[i], vocab[TW[i]], '   <-- the same' if TW[i] == TW[PAUSE[pi]] else ''), flush=True)

print('', flush=True)
M = 2000
rr = np.random.default_rng(7)
r_rows = np.repeat(np.arange(M), 4)
r_cols = rr.integers(0, 2 * DT, M * 4)
R = sp.csr_matrix((np.ones(M * 4, np.float32), (r_rows, r_cols)), shape=(M, 2 * DT))
Fk = np.asarray((R @ K.T).todense())
kd = np.zeros(2 * DT, np.float32)
for i, v in keyfrom(np.array([], np.int32), F).items():
    kd[i] = v
fp = np.asarray((R @ kd)).ravel()
cos = (Fk / np.linalg.norm(Fk, axis=0)) .T @ (fp / np.linalg.norm(fp))
top2 = np.argsort(-cos)[:6]
print('the same states, after a random expansion with %d fixed random connections:' % (M * 4), flush=True)
for i in top2:
    print('   cosine %.3f   that state wanted "%s"%s'
          % (cos[i], vocab[TW[i]], '   <-- the same' if TW[i] == TW[PAUSE[pi]] else ''), flush=True)

print('', flush=True)
for s in (1.0, 1.2, 1.4, 1.6, 2.0):
    ex = sum(answer(q, s=s) == a for q, a in pairs[:40])
    print('weights times %.1f   exact %d/40' % (s, ex), flush=True)
print('', flush=True)
for q, a in pairs[3:6]:
    print('  want: %-42s' % ' '.join(vocab[w] for w in a), flush=True)
    print('  got : %-42s' % ' '.join(vocab[w] for w in answer(q, s=1.4)), flush=True)
 