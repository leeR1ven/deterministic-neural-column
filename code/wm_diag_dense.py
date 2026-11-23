"""Diagnostic (new file): why the dense sensory drive cannot drive walk_rel2's trace.

Compares, with the SAME loop matrices and the SAME trace rule, two input regimes:

  A. dense drive   W_in @ S_k fed straight into the loop, threshold calibrated by
                   density to ~200 cells (this is local_adapt2 / wm_demo.py)
  B. binarized     the same drive cut at the same calibrated threshold to form a 0/1
                   code, which is what walk_rel2 feeds its loop

For both it prints the quantities that decide whether the trace can tell cells apart:

  rel_on / rel_off                 drive relative to the cell's budget, on vs off the
                                   item's cells
  m_ref(on) / m_ref(off)           the per-cell trace level at the end of the input
  mn(on) / mn(off) at cue end      (= 1 for BOTH groups by construction -- this is the
                                   point: mn is normalised per cell)
  mn after 10 hold steps           the only thing that separates the groups, because m
                                   drifts by (rel_hold - m)/TAU in ABSOLUTE units
  fire counts over 22 steps        what the loop actually does with it

Usage: python wm_diag_dense.py [seed]
"""
import os
import sys
import numpy as np
import torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_encoder as te
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ITEMS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
DIMS = [(28, 10, 1), (28, 10, 10), (28, 10, 10), (28, 10, 10)]
NS, N23, N5, NI = 2800, 1024, 512, 256
CUE, HOLD = 10, 12
TAU, KAPPA = 400.0, 1.5

codes = T.as_tensor(np.stack([np.clip(te.pattern(w), 0, 1) for w in ITEMS]),
                    device=DEV, dtype=T.float32)
st = Local3DStack(DIMS, seed=SEED, dev=DEV)
st.calibrate(codes, target=0.12)
S = st.forward(codes)[-1].round()
S = S / S.sum(1, keepdim=True).clamp(min=1) * S.sum(1, keepdim=True).mean()

SX, SY, SZ = DIMS[-1]
sx, sy, sz = np.unravel_index(np.arange(NS), (SX, SY, SZ))
NX = NY = 32
cx = (np.arange(N23) % NX) * (SX - 1) / (NX - 1)
cy = (np.arange(N23) // NX) * (SY - 1) / (NY - 1)
dd = ((sx[None, :] - cx[:, None]) ** 2 / 9.0 + (sy[None, :] - cy[:, None]) ** 2 / 4.0
      + (sz[None, :] * 0.3) ** 2)
W_in = T.zeros(N23, NS, device=DEV)
for j in range(N23):
    W_in[j, np.argsort(dd[j])[:80]] = 1.0 / 80.0
DRIVE = W_in @ S.T

rng2 = np.random.default_rng(SEED + 7)


def sp(post, pre, k, budget):
    M = T.zeros(post, pre, device=DEV)
    idx = np.stack([rng2.choice(pre, k, replace=False) for _ in range(post)])
    M[T.as_tensor(np.repeat(np.arange(post), k), device=DEV),
      T.as_tensor(idx.reshape(-1), device=DEV)] = 1.0
    return M * (budget / M.sum(1, keepdim=True).clamp(min=1e-6))


W_q = sp(N5, N23, 20, 0.30)
W_ei = sp(NI, N23, 100, 0.34)
W_ie = sp(N23, NI, 100, 0.65)
bias = T.full((N23,), 0.03, device=DEV)
bias_i = T.full((NI,), 0.02, device=DEV)
TH_I = T.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T.float32)
bud_i = T.full((NI,), 0.34, device=DEV)
bud5 = T.full((N5,), 0.30, device=DEV)


def run(inp_fn, BUD, TH23V, cue=None):
    """inp_fn(t) -> the L2/3 drive vector (already in the loop's units)"""
    m = T.zeros(N23, device=DEV)
    xi = T.zeros(NI, device=DEV)
    fire = T.zeros(N23, device=DEV)
    m_ref = None
    fires, mns = [], []
    for t in range(CUE + HOLD):
        d23 = bias + inp_fn(t) - W_ie @ xi
        rel = d23 / BUD
        if t == CUE - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23V > TH23V).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I * bud_i).float()
        fires.append(int(fire.sum()))
        mns.append(mn.clone())
    return fires, mns, m_ref, fire


def bisect_density(v, target, bud):
    lo, hi = -1.0, float((v / bud).max()) + 1e-6
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if int((v / bud > mid).sum()) >= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print('seed %d | dense drive vs 0/1 code, same loop' % SEED)
for label, dense in (('A dense drive (local_adapt2 wiring)', True),
                     ('B binarized code (this port)', False)):
    # threshold from the drive, calibrated once on item 0's density target
    bud_dense = W_in.abs().sum(1) + bias.abs()
    rel0 = ((DRIVE[:, 0] + bias) / bud_dense).cpu().numpy()
    TARGET = 150
    TH_D = float(np.sort(rel0)[::-1][min(TARGET, len(rel0) - 1)])
    ON = rel0 >= TH_D                      # the item-0 cells, whichever regime
    # per-item 0/1 code at the same density target, for regime B
    codes_k = []
    for k in range(6):
        v = DRIVE[:, k]
        r = ((v + bias) / bud_dense).cpu().numpy()
        th = float(np.sort(r)[::-1][min(TARGET, len(r) - 1)])
        codes_k.append((v >= float(np.sort(v.cpu().numpy())[::-1][min(TARGET, N23 - 1)])).float())
    if dense:
        BUD = bud_dense
        RELS = [((DRIVE[:, k] + bias) / BUD) for k in range(6)]
    else:
        BUD = T.ones(N23, device=DEV) + bias.abs()
        RELS = [((codes_k[k] + bias) / BUD) for k in range(6)]
    TH23V = T.as_tensor(np.quantile(np.stack([r.cpu().numpy() for r in RELS]), 0.90, axis=0),
                        device=DEV, dtype=T.float32)
    on = T.as_tensor(ON, device=DEV)
    re = RELS[0].cpu().numpy()
    print('\n%s' % label)
    print('  threshold (fraction of budget) %.3f | presentation cells %d' % (
        TH_D, int(on.sum())))
    print('  rel on item %.2f vs off item %.2f  (ratio %.2f)' % (
        float(re[ON].mean()), float(re[~ON].mean()),
        float(re[ON].mean() / re[~ON].mean())))
    if dense:
        inp = lambda t, k=0: (DRIVE[:, k] if t < CUE else T.zeros(N23, device=DEV))
    else:
        inp = lambda t, k=0: (codes_k[k] if t < CUE else T.zeros(N23, device=DEV))
    fires, mns, m_ref, fire = run(inp, BUD, TH23V)
    mr = m_ref.cpu().numpy()
    mn_end = mns[-1].cpu().numpy()
    print('  m_ref on item %.4f vs off %.4f  (ratio %.1f)' % (
        float(mr[ON].mean()), float(mr[~ON].mean()),
        float(mr[ON].mean() / max(mr[~ON].mean(), 1e-12))))
    print('  mn at cue end   on %.3f vs off %.3f   (both ~1: mn is per-cell normalised)' % (
        float(mns[CUE - 1][on].mean()), float(mns[CUE - 1][~on].mean())))
    print('  mn after 12 hold steps on %.3f vs off %.3f' % (
        float(mn_end[on.cpu().numpy()].mean()), float(mn_end[~on.cpu().numpy()].mean())))
    print('  fire counts over %d steps: %s' % (CUE + HOLD, fires))
    print('  held count %d of %d' % (fires[-1], N23))
