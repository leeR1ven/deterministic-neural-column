"""Instrumentation probe (new file, does not modify anything).

Replica of walk_rel2.py's construction with the internals of held_state() printed, so
the working mechanism can be described numerically before it is transplanted:

  * TH23v distribution (per-cell 90th percentile of the presentation drive)
  * presentation cell count and the contrast rel_active vs rel_inactive
  * m_ref for cells that were active during the cue vs cells that were not
  * mn (= m / m_ref) after the cue, for both groups
  * held cell count, and how many of the held cells were in the presentation state

Usage: python wm_diag_ref.py [seed]
"""
import os
import sys
import numpy as np
import torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_encoder as te

DEV = 'cuda'
N23, N5, NI, NW = 1024, 512, 256, 280
WORDS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
         'iota', 'kappa', 'lambda', 'mu']
NITEMS = 12
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
KFF, KQ = 40, 20
TAU, KAPPA, CUE = 400.0, 1.5, 10
TH5 = 0.17

base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
mean_in = comp.mean(0)
comp_c = comp - mean_in
rng = np.random.default_rng(SEED)
W_ff = rng.normal(0, 1, (N23, comp.shape[1]))
mask = np.zeros_like(W_ff, bool)
for j in range(N23):
    mask[j, rng.choice(comp.shape[1], KFF, replace=False)] = True
W_ff = W_ff * mask
W_ff = W_ff / np.abs(W_ff).sum(1, keepdims=True)
bud_ff = np.abs(W_ff).sum(1) + 1e-9
best = None
for th in (0.05, 0.10, 0.15, 0.20):
    m = float(np.mean([float((W_ff @ comp_c[k] > th * bud_ff).mean()) for k in range(NITEMS)]))
    if best is None or abs(m - 0.08) < abs(best[1] - 0.08):
        best = (th, m)
TH_A = best[0]
GAIN = []
for k in range(NITEMS):
    lo, hi = 0.1, 6.0
    for _ in range(22):
        g = 0.5 * (lo + hi)
        n = float((W_ff @ (g * comp_c[k]) > TH_A * bud_ff).sum())
        if n > 70:
            hi = g
        else:
            lo = g
    GAIN.append(0.5 * (lo + hi))
CODES = [(W_ff @ (GAIN[k] * comp_c[k]) > TH_A * bud_ff).astype(np.float32) for k in range(NITEMS)]
print('code sizes: %s' % [int(c.sum()) for c in CODES])
SIN = [T.as_tensor(c, device=DEV, dtype=T.float32) for c in CODES]

rng2 = np.random.default_rng(SEED + 7)


def sp(post, pre, k, budget):
    M = T.zeros(post, pre, device=DEV)
    for j in range(post):
        M[j, T.as_tensor(rng2.choice(pre, k, replace=False), device=DEV)] = 1.0
    return M * (budget / M.sum(1, keepdim=True).clamp(min=1e-6))


W_q = sp(N5, N23, KQ, 0.30)
W_ei = sp(NI, N23, 100, 0.34)
W_ie = sp(N23, NI, 100, 0.65)
bias23 = T.zeros(N23, device=DEV) + 0.03
bias_i = T.zeros(NI, device=DEV) + 0.02
BUD = T.ones(N23, device=DEV) + bias23
TH_I_CELL = T.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T.float32)
bud_i = T.full((NI,), 0.34, device=DEV)
bud5 = T.full((N5,), 0.30, device=DEV)
rels = [((SIN[k] + bias23) / BUD).cpu().numpy() for k in range(NITEMS)]
TH23v = T.as_tensor(np.quantile(np.stack(rels), 0.90, axis=0), device=DEV, dtype=T.float32)
thv = TH23v.cpu().numpy()
print('TH23v: zeros %d | in (0,1) %d | ones %d | mean %.3f' % (
    int((thv == 0).sum()), int(((thv > 0) & (thv < 1)).sum()), int((thv == 1).sum()), float(thv.mean())))
print('rel at presentation: active %.3f | inactive %.3f' % (
    float(np.mean([rels[k][CODES[k] > 0.5].mean() for k in range(NITEMS)])),
    float(np.mean([rels[k][CODES[k] < 0.5].mean() for k in range(NITEMS)]))))


def held_diag(code, cue=CUE, extra=12):
    m = T.zeros(N23, device=DEV)
    xi = T.zeros(NI, device=DEV)
    m_ref = None
    fires, xis, rels_h, mns = [], [], [], []
    for t in range(cue + extra):
        inp = code if t < cue else None
        d23 = bias23 + (inp if inp is not None else 0) - W_ie @ xi
        rel = d23 / BUD
        if t == cue - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23v > TH23v).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I_CELL * bud_i).float()
        fires.append(fire)
        xis.append(float(xi.sum()))
        rels_h.append(rel)
        mns.append(mn)
    return fires, xis, rels_h, mns, m_ref


def ov(a, b):
    u = ((a + b) > 0).sum()
    return float((a * b).sum() / u) if u else 0.0


print('\ncounts per step for item 0 (cue %d + hold 12)' % CUE)
fires, xis, rels_h, mns, m_ref = held_diag(SIN[0])
print('  fire   : %s' % [int(f.sum()) for f in fires])
print('  xi     : %s' % [int(x) for x in xis])
print('  rel    : %s' % ['%.2f' % float(r.mean()) for r in rels_h])
mn = mns[-1].cpu().numpy()
thv = TH23v.cpu().numpy()
held0 = fires[-1].cpu().numpy() > 0.5
pres0 = fires[CUE - 1].cpu().numpy() > 0.5
for name, sel in [('TH23v==0', thv == 0), ('TH23v==1', thv == 1)]:
    if sel.sum() == 0:
        continue
    print('   %-9s n=%4d mn %.2f..%.2f | held %d' % (
        name, int(sel.sum()), float(mn[sel].min()), float(mn[sel].max()),
        int(held0[sel].sum())))
print('  m_ref active (in code) %.4f | m_ref inactive %.4f' % (
    float(m_ref.cpu().numpy()[CODES[0] > 0.5].mean()), float(m_ref.cpu().numpy()[CODES[0] < 0.5].mean())))

print('\nper item: presentation -> held')
allh, allp = [], []
for k in range(NITEMS):
    fires, xis, rels_h, mns, m_ref = held_diag(SIN[k])
    p = fires[CUE - 1].cpu().numpy()
    h = fires[-1].cpu().numpy()
    allp.append(p)
    allh.append(h)
    c = CODES[k]
    print('  %-8s pres %4d | held %3d | held in code %3d | ov(held,pres) %.2f | ov(held,code) %.2f' % (
        WORDS[k], int(p.sum()), int(h.sum()), int(((h * c) > 0).sum()),
        ov(h, p), ov(h, c)))
print('\npairwise held overlap %.3f | pairwise presentation overlap %.3f' % (
    float(np.mean([ov(allh[i], allh[j]) for i in range(NITEMS) for j in range(i + 1, NITEMS)])),
    float(np.mean([ov(allp[i], allp[j]) for i in range(NITEMS) for j in range(i + 1, NITEMS)]))))
