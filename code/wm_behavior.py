"""Two behaviours that matter for a system without a goal (designer's framing).

The designer's correction: a simple model has no purpose, so drifting, being overwritten,
and not recovering are NORMAL, not defects.  What can be asked of it instead:

  RECALL      tell it the beginning again -- does the continuation come back?
  HOLD        give it a "stay on this" signal (the basal ganglia in this architecture) --
              can that keep a new input from overwriting the state?

Both use the same matrices and per-item thresholds as local_adapt2.py.

Usage: python wm_behavior.py [seed]
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
print('=== behaviours (same matrices as the demo) ===', flush=True)


def step(x23, x5, s_in, th23, hold_gain=1.0):
    """hold_gain scales the incoming sensory drive: 1 = normal, 0 = input blocked"""
    bud = la.bud23 if s_in is not None else la.BUD_REC
    drive = la.W_p @ x5 + la.bias + (hold_gain * (la.W_in @ s_in) if s_in is not None else 0)
    n23 = (drive > th23 * bud).float()
    d5 = la.W_q @ n23
    tot = d5.sum()
    if float(tot) > 0:
        d5 = d5 / tot * (la.N5 * 0.15)
    n5 = (d5 > la.th5 * la.bud5).float()
    nm = (la.W_l5m1 @ n5 > la.THM * la.bud_m1).float()
    return n23, n5, nm


def run(x23, x5, s_in, steps, th23, hold_gain=1.0):
    for _ in range(steps):
        x23, x5, m = step(x23, x5, s_in, th23, hold_gain)
    return x23, x5


def item_settle(k, steps=14):
    lo, hi = 0.05, 0.95
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[k], steps, mid)
        best = (mid, x23, x5)
        n = float(x23.sum())
        if n > 400:
            lo = mid
        elif n < 100:
            hi = mid
        else:
            return best
    return best


INP = [item_settle(k) for k in range(len(la.ITEMS))]
TH = [i[0] for i in INP]
FP = [(i[1], i[2]) for i in INP]


def closest(x23):
    best, bi = -1.0, -1
    for i, (f, _) in enumerate(FP):
        v = float((f * x23).sum() / max(float(((f + x23) > 0).sum()), 1))
        if v > best:
            best, bi = v, i
    return bi, best


def heard(m, x5):
    sums = [float((la.W_l5m1[i * 280:(i + 1) * 280] @ x5).sum()) for i in range(la.NPAIR)]
    if max(sums) <= 0:
        return None
    i = int(np.argmax(sums))
    return m[i * 280:(i + 1) * 280]


# ---------------- 1  RECALL: tell it the beginning again ----------------
print('\n1 RECALL: walk from item 0, get distracted, then tell it 0 again', flush=True)
x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[0], 14, TH[0])
path_before = [closest(x23)[0]]
for _ in range(2):
    h = heard(None, x5) if False else None
    m = la.W_l5m1 @ x5
    h = heard(m, x5)
    if h is None:
        path_before.append(-1); break
    hin = la.st.input_from_motor(h) if hasattr(la.st, 'input_from_motor') else None
    # the demo re-encodes the heard motor pattern through the sensory stack:
    heard_in = la.st.forward(h.unsqueeze(0))[-1][0]
    heard_in = heard_in / heard_in.sum().clamp(min=1) * la.S.sum(1, keepdim=True).mean()
    x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), heard_in, 14, TH[min(2, len(TH) - 1)])
    path_before.append(closest(x23)[0])
print('   before distraction the chain is: %s' % ' '.join(la.ITEMS[i] for i in path_before), flush=True)

# distract with item 3, then give item 0 again
x23, x5 = run(x23, x5, la.S[3], 14, TH[3])
print('   after presenting %s the state is %s' % (la.ITEMS[3], la.ITEMS[closest(x23)[0]]), flush=True)
x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[0], 14, TH[0])
print('   telling it %s again -> state is %s' % (la.ITEMS[0], la.ITEMS[closest(x23)[0]]), flush=True)
got, _ = closest(x23)
print('   RECALL %s' % ('OK: the beginning brings back the beginning, and the chain can restart'
                        if got == 0 else 'FAILED'), flush=True)

# ---------------- 2  HOLD: a "stay on this" signal ----------------
print('\n2 HOLD: present A, then B while a hold signal scales the new input down', flush=True)
print('   hold_gain 1.00 = no suppression, 0.00 = new input fully blocked', flush=True)
for gain in (1.0, 0.6, 0.3, 0.0):
    kept = 0
    for a, b in [(0, 1), (1, 0)]:
        x23, x5 = run(T.zeros(la.N23, device=DEV), T.zeros(la.N5, device=DEV), la.S[a], 14, TH[a])
        x23, x5 = run(x23, x5, None, 2, TH[a])
        x23, x5 = run(x23, x5, la.S[b], 14, TH[b], hold_gain=gain)
        x23, x5 = run(x23, x5, None, 10, TH[b])
        kept += int(closest(x23)[0] == a)
    print('   hold_gain %.2f -> the original item survived in %d/2 cases' % (gain, kept), flush=True)
