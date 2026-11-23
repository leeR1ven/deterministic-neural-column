"""How much outside input does it take to change what the model is thinking?

The designer's prediction: a little irrelevant input cannot change the thought, but enough
of it should, because the neurons the outside input activates become the dominant context.

Design: hold item A, then deliver k distractor items at once, then let it settle, then ask
which item the state is closest to.  Two versions, because they answer different questions:

  ADDITIVE   the distractors' drives are summed, so both the NUMBER of activated neurons and
             the total drive grow with k.
  NORMALISED the summed drive is rescaled to the same total as one item, so only the NUMBER
             of activated neurons changes.  If the crossover still happens here, it is the
             population that matters, not the strength.

Usage: python wm_competition.py [seed]
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
sys.argv = ['local_adapt2.py', str(SEED)]

import numpy as np
import torch as T
import local_adapt2 as la

DEV = la.DEV
print('', flush=True)
print('=== how much outside input changes the thought ===', flush=True)


def step(x23, x5, s_in_drive, th23):
    """s_in_drive is already a 1024-d drive (or None)"""
    bud = la.bud23 if s_in_drive is not None else la.BUD_REC
    drive = la.W_p @ x5 + la.bias + (s_in_drive if s_in_drive is not None else 0)
    n23 = (drive > th23 * bud).float()
    d5 = la.W_q @ n23
    tot = d5.sum()
    if float(tot) > 0:
        d5 = d5 / tot * (la.N5 * 0.15)
    n5 = (d5 > la.th5 * la.bud5).float()
    nm = (la.W_l5m1 @ n5 > la.THM * la.bud_m1).float()
    return n23, n5, nm


def run(x23, x5, drive, steps, th23):
    for _ in range(steps):
        x23, x5, m = step(x23, x5, drive, th23)
    return x23, x5


def item_settle(k, steps=14):
    lo, hi = 0.05, 0.95
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV),
                      la.W_in @ la.S[k], steps, mid)
        best = (mid, x23, x5)
        n = float(x23.sum())
        if n > 400:
            lo = mid
        elif n < 100:
            hi = mid
        else:
            return best
    return best


ITEM = [item_settle(k) for k in range(len(la.ITEMS))]
TH = [i[0] for i in ITEM]
FP = [(i[1], i[2]) for i in ITEM]


def closest(x23):
    best, bi = -1.0, -1
    for i, (f, _) in enumerate(FP):
        v = float((f * x23).sum() / max(float(((f + x23) > 0).sum()), 1))
        if v > best:
            best, bi = v, i
    return bi, best


A = 0                                   # the thought being held
others = [k for k in range(len(la.ITEMS)) if k != A]


def drive_of(kinds):
    d = T.zeros(la.N23, device=DEV)
    for k in kinds:
        d = d + (la.W_in @ la.S[k])
    return d


print('\nthought being held: %s' % la.ITEMS[A], flush=True)

# each distractor is individually WEAK (scale alpha); the question is whether enough of
# them accumulate to take over -- the designer's prediction
print('\nweak distractors, accumulating (alpha = strength of each one)', flush=True)
print('  alpha \\ k | 1 | 2 | 4 | 6 | 8', flush=True)
for alpha in (0.10, 0.20, 0.35, 0.60):
    row = []
    for k in (1, 2, 4, 6, 8):
        kinds = others[:k]
        d = T.zeros(la.N23, device=DEV)
        for kk in kinds:
            d = d + alpha * (la.W_in @ la.S[kk])
        x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV),
                      la.W_in @ la.S[A], 14, TH[A])
        x23, x5 = run(x23, x5, None, 2, TH[A])
        x23, x5 = run(x23, x5, d, 14, TH[A])
        x23, x5 = run(x23, x5, None, 10, TH[A])
        got, sc = closest(x23)
        row.append('A' if got == A else la.ITEMS[got][:3])
    print('  %5.2f     | %s' % (alpha, ' | '.join('%-3s' % r for r in row)), flush=True)
