"""Does a longer context actually pull the pause states apart?  Measure before training.

For every training question the pause state is [nothing ; context].  Different questions must not look alike,
or no learning rule can tell whose turn it is.  This prints, for several context settings, how similar those
pause states are to each other.

Usage: python context_width.py
"""
import sys

import numpy as np
import torch as T

sys.path.insert(0, r'D:\deepseekherness\repo_publish\deterministic-neural-column\code')
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 16, 2, 0.6
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
questions = [q for q, a in pairs[40:]]

vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
rng = np.random.default_rng(0)
P = {}
with T.no_grad():
    for w in range(1, NV):
        x = np.zeros((1, S * S), np.float32)
        x[0, rng.choice(S * S, 10, replace=False)] = 1.0
        P[w] = vis.forward(T.as_tensor(x, device=DEV))[-1][0].nonzero().flatten().cpu().numpy()
NA = float(np.mean([len(P[w]) for w in range(1, NV)]))


def pause_states(gmin, gmax, scale, seed=5):
    gam = T.as_tensor(np.random.default_rng(seed).uniform(gmin, gmax, DT).astype(np.float32), device=DEV)
    out = []
    for q in questions:
        F = T.zeros(DT, device=DEV)
        for w in q:
            V = T.zeros(DT, device=DEV)
            V[T.as_tensor(P[w], device=DEV)] = 1.0
            F = gam * F + V
            F = F * (scale * NA / F.sum().clamp(min=1e-9))
        out.append(F / F.norm().clamp(min=1e-9))
    return T.stack(out)


print('%d training questions' % len(questions), flush=True)
print('', flush=True)
print('  decay range    offset   mean cosine   p95 nearest   pairs above 0.9', flush=True)
for gmin, gmax, scale in ((0.60, 0.99, 5.0), (0.90, 0.999, 5.0), (0.60, 0.99, 20.0),
                          (0.90, 0.999, 20.0), (0.99, 0.9999, 50.0), (0.99, 0.9999, 5.0)):
    Sst = pause_states(gmin, gmax, scale)
    C = (Sst @ Sst.T).cpu().numpy()
    m = C[~np.eye(len(C), dtype=bool)]
    nn = np.array([np.max(np.delete(C[i], i)) for i in range(len(C))])
    print('  %.2f-%.4f   %5.1f   %.3f          %.3f         %.1f%%'
          % (gmin, gmax, scale, m.mean(), np.percentile(nn, 95), 100.0 * (m > 0.9).mean()), flush=True)
