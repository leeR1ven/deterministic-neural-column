"""Local Hebbian learning, one transition at a time.  No matrix solve anywhere.

The rule, per synapse, per frame (designer's rule, negatives allowed):
    change = rate * (what the target cell should have done - what it did) * (was my input cell active)
That only needs the synapse's own input cell and the cell it lands on, so it is local.

Everything else is the architecture: two regions of the same local layer (visual for heard, motor for spoken),
a context field with one decay per cell that stays alive through pauses, key = [code ; context],
fixed threshold 0.7, silence ends the turn.  No gain is tuned: the threshold is used as is.

Training set: pairs 40..234.  Test set: pairs 0..39 (never trained on).

Usage: python local_hebb.py
"""
import sys

import numpy as np
import torch as T

sys.path.insert(0, r'D:\deepseekherness\repo_publish\deterministic-neural-column\code')
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 16, 2, 0.6
TH_A = 0.7
GMIN, GMAX, SCALE = 0.60, 0.99, 5.0
D0, DT = S * S * MZ, 2 * S * S * MZ
LR, EPOCHS = 0.05, 150
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
TEST, TRAIN = pairs[:40], pairs[40:]
print('%d pairs (%d train, %d test), %d words, %d cells per region'
      % (len(pairs), len(TRAIN), len(TEST), NV, D0), flush=True)

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
GAM = T.as_tensor(np.random.default_rng(5).uniform(GMIN, GMAX, DT).astype(np.float32), device=DEV)
print('word patterns %d cells' % int(NA), flush=True)


def stepf(F, V):
    F = GAM * F + V
    if float(V.sum()) > 0.5:
        F = F * (SCALE * NA / F.sum().clamp(min=1e-9))
    return F


def codes_pair(q, a):
    """all frames of one dialogue: heard question, pause, spoken answer, pause"""
    e = np.zeros(0, np.int64)
    fr = [(P[w], 0) for w in q] + [(e, 0)] + [(M[w], D0) for w in a] + [(e, D0)]
    return fr


def build(pairs_, ):
    keys, tgts = [], []
    for q, a in pairs_:
        fr = codes_pair(q, a)
        F = T.zeros(DT, device=DEV)
        for t in range(len(fr) - 1):
            v = T.zeros(DT, device=DEV)
            v[T.as_tensor(fr[t][1] + fr[t][0], device=DEV)] = 1.0
            v = v / v.norm().clamp(min=1e-9)
            k = T.cat([v, F])
            k = k / k.norm().clamp(min=1e-9)
            tv = T.zeros(DT, device=DEV)
            tv[T.as_tensor(fr[t + 1][1] + fr[t + 1][0], device=DEV)] = 1.0
            keys.append(k)
            tgts.append(tv)
            F = stepf(F, v)
    return T.stack(keys), T.stack(tgts)


K, V = build(TRAIN)
N = K.shape[0]
print('%d transitions; connection set %d x %d (%.1f M synapses)'
      % (N, DT, 2 * DT, DT * 2 * DT / 1e6), flush=True)

W = T.zeros(DT, 2 * DT, device=DEV)
order = np.arange(N)
for ep in range(EPOCHS):
    np.random.default_rng(ep).shuffle(order)
    tot = 0.0
    for i in order:
        pred = W @ K[i]
        err = V[i] - pred
        W += LR * T.outer(err, K[i])          # one synapse at a time: err[post] * key[pre]
        tot += float(err.abs().sum())
    W /= (W.norm(dim=1, keepdim=True) / 1.0).clamp(min=1e-9)
    if ep % 10 == 0 or ep == EPOCHS - 1:
        print('  epoch %2d   mean error per transition %.3f' % (ep, tot / N), flush=True)


def answer(q, maxlen=18):
    F = T.zeros(DT, device=DEV)
    for w in q:
        Vv = T.zeros(DT, device=DEV)
        Vv[T.as_tensor(P[w], device=DEV)] = 1.0
        Vv = Vv / Vv.norm().clamp(min=1e-9)
        F = stepf(F, Vv)
    Vcur, out, rows = T.zeros(DT, device=DEV), [], []
    for _ in range(maxlen):
        k = T.cat([Vcur, F])
        k = k / k.norm().clamp(min=1e-9)
        act = W @ k
        fired = act > TH_A
        if int(fired.sum()) == 0:
            break
        sc = T.zeros(NV, device=DEV)
        for w in range(1, NV):
            cells = T.as_tensor(M[w], device=DEV) + D0
            sc[w] = fired[cells].float().mean()
        w_best = int(sc.argmax())
        out.append(w_best if float(sc[w_best]) > 0 else SIL)
        rows.append(fired)
        F = stepf(F, Vcur)
        Vcur = fired.float()
    return out, rows


ex = sum(answer(q)[0] == a for q, a in TEST)
print('', flush=True)
print('held-out sentences exactly right: %d/%d' % (ex, len(TEST)), flush=True)
tr = sum(answer(q)[0] == a for q, a in TRAIN[:40])
print('trained sentences exactly right : %d/%d' % (tr, 40), flush=True)
for q, a in TRAIN[3:6]:
    print('  want: %-40s got: %s' % (' '.join(vocab[w] for w in a),
                                     ' '.join(vocab[w] for w in answer(q)[0])), flush=True)
