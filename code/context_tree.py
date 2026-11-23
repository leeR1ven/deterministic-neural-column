"""Adding context so the graph becomes a tree, not a funnel.

Measured on the current model (graph2.py): all 40 words flow into ONE fixed point
(basin 40/40) but that trunk leads nowhere (reachable = 1).  That is the convergence
half of the designer's picture; the divergence half needs the successor to depend on
more than the current word.

So: a context pattern is added to the L2/3 state.  A context is just a fixed sparse
pattern of cells (a few "situation" neurons being active).  Then

    state = content + context

and the same word in different contexts can lead to different words -- the branching
the designer described ("the same start can lead to different ends").

Trained tree (12 words):
    trunk  T=0
    from T: context 0 -> 1, context 1 -> 2, context 2 -> 3
    from 1, 2, 3 (any context) -> back to T        (branches converge to the trunk)

Measured: from T, how many distinct successors do the contexts produce; and from the
branches, does everything return to the trunk.

Usage: python context_tree.py <seed>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', '12', '8']
src = open(os.path.join(BASE, 'scale_more2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more2.py')}
exec(compile(src[:cut], 'ct', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CODES_T = ns['CODES_T']
held_state = ns['held_state']; DEV = ns['DEV']; W_m = ns['W_m']
ETA = ns['ETA']; MOT_BUD = ns['MOT_BUD']; T_ = ns['T']
ov = ns['ov']

# ---- context patterns: fixed sparse sets of L2/3 cells
rng = np.random.default_rng(int(SEED) + 99)
NCTX = 3
CTX = []
for c in range(NCTX):
    v = T_.zeros(ns['N23'], device=DEV)
    v[T_.as_tensor(rng.choice(ns['N23'], 60, replace=False), device=DEV)] = 1.0
    CTX.append(v)

TREE = [(0, 0, 1), (0, 1, 2), (0, 2, 3)]      # (from, context, to)
for b in (1, 2, 3):
    for c in range(NCTX):
        TREE.append((b, c, 0))

# ---- retrain the shared map on (state + context) -> next code
W_m = T_.zeros_like(W_m)
print('training a tree over %d edges (word + context -> next word) ...' % len(TREE), flush=True)
for ep in range(60):
    for a, c, b in TREE:
        x23, _ = held_state(ns['SIN'][a])
        x23 = T_.clamp(x23 + CTX[c], 0, 1)
        x5 = ((0.02 + ns['W_q'] @ x23) > ns['TH5'] * ns['bud5']).float()
        W_m = T_.clamp(W_m + ETA * (1 - W_m) * (CODES_T[b].unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))

cv = [c.cpu().numpy() for c in CODES_T]
cn = [max(float(np.linalg.norm(v)), 1e-9) for v in cv]


def read(word, ctx):
    x23, _ = held_state(ns['SIN'][word])
    x23 = T_.clamp(x23 + CTX[ctx], 0, 1)
    x5 = ((0.02 + ns['W_q'] @ x23) > ns['TH5'] * ns['bud5']).float()
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ cv[j]) / cn[j] for j in range(NITEMS)]
    return int(np.argmax(sims)), float(np.max(sims))


print('\nDIVERGENCE from the trunk (word 0, different contexts):', flush=True)
outs = []
for c in range(NCTX):
    nxt, s = read(0, c)
    outs.append(nxt)
    print('  context %d -> %s (sim %.2f) | want %s' % (
        c, WORDS[nxt], s, WORDS[c + 1]), flush=True)
print('  distinct successors from the same word: %d/%d' % (len(set(outs)), NCTX), flush=True)

print('\nCONVERGENCE from the branches (words 1,2,3, every context):', flush=True)
back = 0
tot = 0
for b in (1, 2, 3):
    row = []
    for c in range(NCTX):
        nxt, s = read(b, c)
        row.append(WORDS[nxt][:6])
        tot += 1
        back += int(nxt == 0)
    print('  word %s: %s' % (WORDS[b][:6], ' '.join(row)), flush=True)
print('  branches returning to the trunk: %d/%d' % (back, tot), flush=True)
