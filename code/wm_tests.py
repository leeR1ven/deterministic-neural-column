"""The two tests that decide whether the "hold" is a memory or just a fixed point.

README section 3.7 admits the current hold test passes even if the hold loop is deleted.
These probes ask sharper questions:

  DISTRACTOR      present item A, then item B, then hold.  Does A survive, or does B
                  overwrite it?
  PERTURBATION    present A, corrupt a fraction of the cells, let the network run.  Does it
                  fall back to A's state (attractor), or stay corrupted?

Both use the same matrices AND the same per-item threshold that local_adapt2.py uses
(the first version of this script used the global threshold and produced nonsense,
because the demo calibrates theta separately for every item).

Usage: python wm_tests.py [seed]
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
print('=== working-memory probes (same matrices and per-item theta as the demo) ===', flush=True)


def step(x23, x5, s_in, th23):
    bud = la.bud23 if s_in is not None else la.BUD_REC
    drive = la.W_p @ x5 + la.bias + (la.W_in @ s_in if s_in is not None else 0)
    n23 = (drive > th23 * bud).float()
    d5 = la.W_q @ n23
    tot = d5.sum()
    if float(tot) > 0:
        d5 = d5 / tot * (la.N5 * 0.15)
    n5 = (d5 > la.th5 * la.bud5).float()
    nm = (la.W_l5m1 @ n5 > la.THM * la.bud_m1).float()
    return n23, n5, nm


def run(x23, x5, s_in, steps, th23):
    for _ in range(steps):
        x23, x5, m = step(x23, x5, s_in, th23)
    return x23, x5


def item_settle(k, steps=14):
    """the same per-item bisection the demo uses; returns (theta, x23, x5)"""
    lo, hi = 0.05, 0.95
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV),
                      la.S[k], steps, mid)
        best = (mid, x23, x5)
        n = float(x23.sum())
        if n > 400:
            lo = mid
        elif n < 100:
            hi = mid
        else:
            return best
    return best


INPUTS = [item_settle(k) for k in range(len(la.ITEMS))]
TH = [INPUTS[k][0] for k in range(len(la.ITEMS))]
FP = [(INPUTS[k][1], INPUTS[k][2]) for k in range(len(la.ITEMS))]
print('  per-item theta: %s' % ['%.2f' % t for t in TH], flush=True)
print('  presentation sizes: %s' % [int(f[0].sum()) for f in FP], flush=True)


def closest(x23):
    best, bi = -1.0, -1
    for i, (f, _) in enumerate(FP):
        v = float((f * x23).sum() / max(float(((f + x23) > 0).sum()), 1))
        if v > best:
            best, bi = v, i
    return bi, best


# sanity: presenting item k and holding it should keep item k
print('\n0 SANITY: present k, hold, does it stay k?', flush=True)
ok = 0
for k in range(len(la.ITEMS)):
    x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[k], 14, TH[k])
    x23, x5 = run(x23, x5, None, 10, TH[k])
    got, s = closest(x23)
    ok += int(got == k)
    if got != k:
        print('   %s -> %s (%.2f) MISMATCH' % (la.ITEMS[k], la.ITEMS[got], s), flush=True)
print('   stayed: %d/%d' % (ok, len(la.ITEMS)), flush=True)

# ---------------- 1  distractor ----------------
print('\n1 DISTRACTOR: present A, then B, then hold -- which one is still there?', flush=True)
kept_n = 0
for a, b in [(0, 1), (1, 0), (2, 3)]:
    x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[a], 14, TH[a])
    x23, x5 = run(x23, x5, None, 2, TH[a])
    (x23, x5) = run(x23, x5, la.S[b], 14, TH[b])     # B arrives on top of A
    x23, x5 = run(x23, x5, None, 10, TH[b])
    got, s = closest(x23)
    surv = (got == a)
    kept_n += int(surv)
    print('   A=%-8s B=%-8s -> final state is %-8s (%.2f) : %s' % (
        la.ITEMS[a], la.ITEMS[b], la.ITEMS[got], s,
        'A SURVIVED' if surv else ('B overwrote A' if got == b else 'neither')), flush=True)
print('   A survived in %d/3 cases' % kept_n, flush=True)

# ---------------- 2  perturbation ----------------
print('\n2 PERTURBATION: corrupt a fraction of the held cells, then let it run', flush=True)
rng = np.random.default_rng(int(SEED))
for frac in (0.05, 0.2, 0.5):
    rec = []
    for k in range(len(la.ITEMS)):
        x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[k], 14, TH[k])
        x23, x5 = run(x23, x5, None, 2, TH[k])
        before = x5.clone()          # x5 is the state that carries the memory: x23 is
        idx = T.as_tensor(rng.choice(la.N5, int(frac * la.N5), replace=False), device=DEV)
        x5[idx] = 1.0 - x5[idx]      # recomputed from x5 every step, so perturbing x23
                                     # does nothing (that was a bug in the first version)
        x23, x5 = run(x23, x5, None, 10, TH[k])
        rec.append(float((before * x5).sum() / max(float(((before + x5) > 0).sum()), 1)))
    print('   flipped %2d%% -> overlap with the pre-perturbation state after 10 steps: %.3f' % (
        int(frac * 100), float(np.mean(rec))), flush=True)
