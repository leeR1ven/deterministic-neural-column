"""wm_port.py -- transplant walk_rel2.py's trace hold into local_adapt2.py's sensory demo.

WHAT IS TRANSPLANTED, AND INTO WHAT
-----------------------------------
local_adapt2.py's sensory path is kept verbatim:
    word -> Local3DStack.forward -> S (2800-d sparse code) -> W_in -> dense drive (1024-d)
walk_rel2.py's holding mechanism is kept verbatim (this is the only memory in the system):
    m      per-cell slow variable (a trace), integrating the cell's own relative drive
    m_ref  per-cell trace level at the last input step
    mn     = m / m_ref        (per-cell, history-dependent excitability)
    fire   = 1[ rel + KAPPA*mn*TH23v > TH23v ]     with rel = (bias + input - W_ie@xi)/BUD
    xi     256-cell inhibitory pool driven by `fire` (W_ei), feeding back through W_ie
    TH23v  per-cell threshold = 90th percentile of the presentation drive over the items
No ad-hoc threshold recursion: the only per-item calibration is the sensory->code stage,
done once before the loop (see below), exactly like walk_rel2's per-item GAIN calibration.

THE ONE CONCEPTUAL CHANGE (and why it is forced)
------------------------------------------------
walk_rel2 feeds the loop a **sparse 0/1 code**: rel = 1.00 on the item's cells and 0.03
off them.  local_adapt2 feeds a **dense drive** W_in @ s whose cells all sit inside a
narrow band (measured: 0.10-0.15 on the item, 0.03 background).  The trace cannot
separate cells in that regime, for a reason internal to the mechanism:

    mn = m / m_ref is normalised PER CELL, so mn is ~1 for every cell at the moment the
    input is removed, whatever its drive was.  What separates cells afterwards is the
    ABSOLUTE drift of m: m += (rel_hold - m)/TAU.  A cell with m_ref = 0.021 loses ~0.6 %
    of its trace per hold step; a cell with m_ref = 0.0007 (never driven) loses ~10 % per
    step and is silenced in ~10 steps.  Discrimination needs m_ref(active)/m_ref(off) to
    be large.  walk_rel2 measures 30-3000; the raw dense drive gives ~1.4, which is why
    the previous attempt (wm_demo.py) held nothing: sanity 3/6 and held sizes 0-5 of 1024.

So the sensory drive is passed through ONE calibrated threshold to form the L2/3 input
code c_k -- the same L2/3 input nonlinearity local_adapt2 already has (`drive > th*bud`).
c_k is 0/1 in the same 1024-d space as the L2/3 state, exactly like walk_rel2's SIN[k].
Its size is calibrated per item by bisection on the drive (walk_rel2 does the same with
its per-item GAIN), so the loop then runs unchanged.  The threshold is the only new
quantity, it is calibrated on the drive (not on the loop), and its cell counts are
printed below.

MEASUREMENTS
    SANITY      present k, hold 10, read out the item the state matches (Jaccard vs the
                presentation-state templates).  Must be 6/6.
    PERTURB     present k, hold 2, flip 5/20/50 %, run 10 more, overlap with the
                pre-flip state.  Every cell population is reported separately:
                  x23 = the L2/3 state that carries the trace (the memory)
                  x5  = the L5 readout layer (see `fb`: in walk_rel2's wiring it has no
                        path back into L2/3, so flipping it is erased in one step -- that
                        is reported as such and is NOT used as evidence)
                Three flip rules: uniform (wm_tests.py: frac of ALL cells, mixed on/off),
                active (frac of the ACTIVE cells go off) and swap (count-preserving).
                Each is compared with a no-flip control run and with the state one step
                after the flip, so "recovered" cannot be confused with "never applied".
    DISTRACTOR  present A, then B, then hold: which item is present at the end.

Usage:
    python wm_port.py [seed] [code_size] [kappa] [tau] [fb] [inh] [l5_target]

    seed       sensory stack seed and loop-matrix seed (loop uses seed + 7, walk_rel2)
    code_size  target cells in the sensory 0/1 input code (default 150)
    kappa      trace gain (default 1.5 = walk_rel2)
    tau        trace time constant (default 400 = walk_rel2)
    fb         L5 -> L2/3 feedback gain (default 0 = walk_rel2's wiring; 1.0 restores
               local_adapt2's W_p feedback, which makes the L5 population part of the loop
               and therefore makes an L5 perturbation meaningful)
    inh        inhibitory synaptic gain, scales W_ie (default 1.0 = walk_rel2's value).
               Lowering it makes the hold stationary at denser codes but weakens the
               distractor (measured; see WM_PORT_REPORT.md).
    l5_target  L5 density target of 512 cells at presentation (default 70)
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import torch as T

import text_encoder as te
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CODE_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 150
KAPPA = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 400.0
FB = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
INH = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
L5_TARGET = int(sys.argv[7]) if len(sys.argv) > 7 else 70   # L5 density target of 512

ITEMS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
NITEMS = len(ITEMS)
DIMS = [(28, 10, 1), (28, 10, 10), (28, 10, 10), (28, 10, 10)]
NS, N23, N5, NI = 2800, 1024, 512, 256
CUE, HOLD = 10, 10                      # walk_rel2: cue 10, then 12 free steps
TH5_WALK = 0.17                         # walk_rel2's value, printed for comparison

print('=' * 78)
print('wm_port.py | seed %d | code_size %d | kappa %.2f | tau %.0f | fb %.2f | inh %.2f' % (
    SEED, CODE_SIZE, KAPPA, TAU, FB, INH))
print('=' * 78, flush=True)

# ---------------------------------------------------------------- sensory path
# identical to local_adapt2.py lines 40-57
codes0 = T.as_tensor(np.stack([np.clip(te.pattern(w), 0, 1) for w in ITEMS]),
                     device=DEV, dtype=T.float32)
st = Local3DStack(DIMS, seed=SEED, dev=DEV)
st.calibrate(codes0, target=0.12)
S = st.forward(codes0)[-1].round()
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
DRIVE = W_in @ S.T                       # (N23, NITEMS) dense sensory drive per word
dr = DRIVE.cpu().numpy()

print('\n[1] sensory stack (local_adapt2 path, unchanged)')
print('    raw last-layer counts  %s of %d  (before the per-item rescaling below)' % (
    [int(x.sum()) for x in st.forward(codes0)[-1]], NS))
print('    S counts per item      %s of %d  (layer theta %s)' % (
    [int(x.sum()) for x in S], NS, ['%.2f' % t for t in st.theta]))
print('    W_in row sum %.2f | drive per item: max %.3f mean %.3f std-over-cells %.3f' % (
    float(W_in.abs().sum(1).max()), dr.max(), dr.mean(), float(np.mean(dr.std(0)))))
print('    drive correlation between items %.2f | drive total range %.3f..%.3f' % (
    float(np.mean([np.corrcoef(dr[:, i], dr[:, j])[0, 1]
                   for i in range(NITEMS) for j in range(i + 1, NITEMS)])),
    float(dr.min()), float(dr.max())))

# ------------------------------------------------------- loop matrices (walk_rel2)
rng2 = np.random.default_rng(SEED + 7)


def sp(post, pre, k, budget):
    """sparse random projection with a fixed row budget (walk_rel2's sp)"""
    M = T.zeros(post, pre, device=DEV)
    idx = np.stack([rng2.choice(pre, k, replace=False) for _ in range(post)])
    M[T.as_tensor(np.repeat(np.arange(post), k), device=DEV),
      T.as_tensor(idx.reshape(-1), device=DEV)] = 1.0
    return M * (budget / M.sum(1, keepdim=True).clamp(min=1e-6))


W_q = sp(N5, N23, 20, 0.30)              # L2/3 -> L5 readout
W_ei = sp(NI, N23, 100, 0.34)            # L2/3 -> inhibition
W_ie = sp(N23, NI, 100, 0.65) * INH      # inhibition -> L2/3 (gain knob, see report)
bias = T.full((N23,), 0.03, device=DEV)  # walk_rel2's bias (local_adapt2 uses 0.05)
bias_i = T.full((NI,), 0.02, device=DEV)
TH_I = T.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T.float32)
bud_i = T.full((NI,), 0.34, device=DEV)
bud5 = T.full((N5,), 0.30, device=DEV)

# optional L5 -> L2/3 feedback.  walk_rel2 has none (L5 is a pure readout); local_adapt2
# has one (random sparse W_p, untrained).  fb=0 is the faithful walk_rel2 wiring.
W_p = T.zeros(N23, N5, device=DEV)
if FB > 0:
    rngp = np.random.default_rng(SEED)
    NBLK = 16
    for blk in range(NBLK):
        a = slice(blk * N23 // NBLK, (blk + 1) * N23 // NBLK)
        b0 = slice(blk * N5 // NBLK, (blk + 1) * N5 // NBLK)
        W_p[a, b0] = T.as_tensor(rngp.random((a.stop - a.start, b0.stop - b0.start)),
                                 device=DEV)
    W_p *= 0.30 / W_p.sum(1, keepdim=True).clamp(min=1e-9)   # local_adapt2's WL

BUD = T.ones(N23, device=DEV) + bias.abs()                   # walk_rel2's budget: 1 + bias
if FB > 0:
    BUD = BUD + FB * W_p.abs().sum(1)


# --------------------------------------------------------------- input code stage
def bisect_threshold(v, target):
    """largest threshold with count(v > th) >= target"""
    lo, hi = float(v.min()) - 1e-6, float(v.max()) + 1e-6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if int((v > mid).sum()) >= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


CUE_CODES = []
CODE_TH = []
for k in range(NITEMS):
    v = DRIVE[:, k]
    th = bisect_threshold(v, CODE_SIZE)
    CUE_CODES.append((v > th).float())
    CODE_TH.append(th)

CODE_NP = [c.cpu().numpy() > 0.5 for c in CUE_CODES]
REL_NP = [((CUE_CODES[k] + bias) / BUD).cpu().numpy() for k in range(NITEMS)]
TH23V = T.as_tensor(np.quantile(np.stack(REL_NP), 0.90, axis=0),
                    device=DEV, dtype=T.float32)
thv = TH23V.cpu().numpy()
NEVER = float((bias / BUD).mean().item()) if hasattr((bias / BUD).mean(), 'item') \
    else float((bias / BUD).mean())

print('\n[2] sensory -> L2/3 input code (one threshold per item, calibrated on the drive)')
print('    threshold per item     %s' % ['%.4f' % t for t in CODE_TH])
print('    code counts per item   %s of %d  (target %d)' % (
    [int(c.sum()) for c in CUE_CODES], N23, CODE_SIZE))
print('    code pairwise Jaccard  %.2f' % float(np.mean(
    [float((CUE_CODES[i] * CUE_CODES[j]).sum() /
           max(float(((CUE_CODES[i] + CUE_CODES[j]) > 0).sum()), 1))
     for i in range(NITEMS) for j in range(i + 1, NITEMS)])))
print('    rel on code %.2f vs off code %.2f   (walk_rel2: 1.00 vs 0.03)' % (
    float(np.mean([REL_NP[k][CODE_NP[k]].mean() for k in range(NITEMS)])),
    float(np.mean([REL_NP[k][~CODE_NP[k]].mean() for k in range(NITEMS)]))))
print('    TH23v: never-active %d | one-item %d | multi-item %d | mean %.2f' % (
    int((np.abs(thv - NEVER) < 1e-3).sum()), int(((thv > NEVER + 1e-3) & (thv < 0.99)).sum()),
    int((thv >= 0.99).sum()), float(thv.mean())))


# ------------------------------------------------------------------------- loop
class WM:
    """walk_rel2's held_state loop with explicit phases (so a perturbation can be
    inserted between phases).  State: x23 (output/firing), m (trace), xi (inhibition)."""

    def __init__(self, kappa=KAPPA, tau=TAU, th5=None):
        self.kappa, self.tau = kappa, tau
        self.th5 = th5
        self.x23 = T.zeros(N23, device=DEV)
        self.x5 = T.zeros(N5, device=DEV)
        self.m = T.zeros(N23, device=DEV)
        self.xi = T.zeros(NI, device=DEV)
        self.mref = None
        self.mn = T.zeros(N23, device=DEV)
        self.rel = T.zeros(N23, device=DEV)

    def phase(self, cue, steps, setref=True):
        """cue: 0/1 code or None.  m_ref is taken at the last input step (walk_rel2);
        setref=False keeps the previous reference (used only by the distractor probe)."""
        counts = []
        for t in range(steps):
            if cue is not None and setref and t == steps - 1:
                self.mref = self.m.clone().clamp(min=1e-6)
            inp = cue if cue is not None else T.zeros(N23, device=DEV)
            d23 = bias + inp - W_ie @ self.xi
            if FB > 0:
                d23 = d23 + FB * (W_p @ self.x5)
            rel = d23 / BUD
            if self.mref is None:
                self.mn = self.m / self.m.abs().max().clamp(min=1e-6)
            else:
                self.mn = self.m / self.mref
            fire = (rel + self.kappa * self.mn * TH23V > TH23V).float()
            self.m = self.m + (rel - self.m) / self.tau
            self.xi = ((bias_i + W_ei @ fire) > TH_I * bud_i).float()
            self.x23 = fire
            self.rel = rel
            self.x5 = ((0.02 + W_q @ fire) > self.th5 * bud5).float()
            counts.append(int(fire.sum()))
        return counts

    def apply_flip(self, x23_new, pool=True):
        """Install a flipped L2/3 pattern at the same point in the cycle as the loop's own
        update.  In this architecture x23 is NOT a recurrent variable: the next step
        recomputes it from xi and the trace, so a flip is erased before it can do anything
        unless the inhibitory pool that the flipped pattern drives is recomputed too.
        pool=True does exactly what the loop itself does at the end of a step:
        xi = 1[bias_i + W_ei @ x23 > TH_I * bud_i]."""
        self.x23 = x23_new
        if pool:
            self.xi = ((bias_i + W_ei @ x23_new) > TH_I * bud_i).float()


def ov(a, b):
    """the repo's overlap: |a AND b| / |a OR b| (Jaccard)"""
    a = a.cpu().numpy() if hasattr(a, 'cpu') else a
    b = b.cpu().numpy() if hasattr(b, 'cpu') else b
    u = ((a + b) > 0).sum()
    return float((a * b).sum() / u) if u else 0.0


def present(k, wm=None):
    wm = WM(th5=TH5) if wm is None else wm
    cnt = wm.phase(CUE_CODES[k], CUE)
    return wm, cnt


def hold(k, steps=HOLD):
    wm, _ = present(k)
    wm.phase(None, steps)
    return wm


# ---- calibrate the L5 readout threshold so L5 stays sparse (walk_rel2: 20-47 of 512) --
def l5_mean_count(th):
    c = 0
    for k in range(NITEMS):
        wm = WM(th5=th)
        wm.phase(CUE_CODES[k], CUE)
        c += int(wm.x5.sum())
    return c / NITEMS


_lo, _hi = 0.02, 0.95
for _ in range(16):
    _mid = 0.5 * (_lo + _hi)
    if l5_mean_count(_mid) > L5_TARGET:
        _lo = _mid
    else:
        _hi = _mid
TH5 = 0.5 * (_lo + _hi)

# ---- presentation templates and held states ---------------------------------------
TEMPL, TEMPL_X5 = [], []
PRES_SIZES, PRES_COUNTS = [], []
for k in range(NITEMS):
    wm, cnt = present(k)
    TEMPL.append(wm.x23.clone())
    TEMPL_X5.append(wm.x5.clone())
    PRES_SIZES.append(int(wm.x23.sum()))
    PRES_COUNTS.append(cnt[-1])

HELD = [hold(k) for k in range(NITEMS)]
HELD_SIZES = [int(w.x23.sum()) for w in HELD]
HELD_X5 = [int(w.x5.sum()) for w in HELD]

mrefs = []
for k in range(NITEMS):
    wm, _ = present(k)
    mrefs.append(wm.mref.cpu().numpy())
RATIOS = [float(mrefs[k][CODE_NP[k]].mean() / max(mrefs[k][~CODE_NP[k]].mean(), 1e-12))
          for k in range(NITEMS)]
MN_ON = float(np.mean([float(HELD[k].mn[CODE_NP[k]].mean()) for k in range(NITEMS)]))
MN_OFF = float(np.mean([float(HELD[k].mn[~CODE_NP[k]].mean()) for k in range(NITEMS)]))

print('\n[3] the loop, stage by stage')
print('    presentation counts    %s of %d' % (PRES_SIZES, N23))
print('    held counts (10 free)  %s of %d' % (HELD_SIZES, N23))
print('    L5 threshold %.3f -> %d of %d at presentation (walk_rel2 0.17 would give %d);'
      ' L5 at present %s of %d; L5 at hold %s' % (
          TH5, int(l5_mean_count(TH5)), N5, int(l5_mean_count(TH5_WALK)),
          [int(t.sum()) for t in TEMPL_X5], N5, HELD_X5))
print('    TRACE CONTRAST  m_ref(code)/m_ref(off-code) = %s  (walk_rel2: 30-3000)'
      % ['%.0f' % r for r in RATIOS])
print('    mn at hold: on-code %.2f | off-code %.2f | gap %.2f' % (MN_ON, MN_OFF, MN_ON - MN_OFF))
print('    inhibition pool: present %s of %d | hold %s of %d | rel at hold %s (mean %.3f)'
      % ([int(present(k)[0].xi.sum()) for k in range(NITEMS)], NI,
         [int(w.xi.sum()) for w in HELD], NI,
         ['%.2f' % float(w.rel.mean()) for w in HELD],
         float(np.mean([float(w.rel.mean()) for w in HELD]))))
print('    held self-vs-template %.2f | held pairwise Jaccard %.2f' % (
    float(np.mean([ov(HELD[k].x23, TEMPL[k]) for k in range(NITEMS)])),
    float(np.mean([ov(HELD[i].x23, HELD[j].x23) for i in range(NITEMS)
                   for j in range(i + 1, NITEMS)]))))

# ---- is the held state stationary?  follow item 0 for 30 free steps ---------------
_wm = WM(th5=TH5)
_wm.phase(CUE_CODES[0], CUE)
_traj = [int(_wm.x23.sum())]
_snap = {}
for _t in range(1, 31):
    _wm.phase(None, 1)
    _traj.append(int(_wm.x23.sum()))
    if _t in (10, 20, 30):
        _snap[_t] = _wm.x23.clone()
print('    hold trajectory item 0 counts: %s' % _traj)
print('    held-state stability: ov(step10,step20) %.2f | ov(step10,step30) %.2f' % (
    ov(_snap[10], _snap[20]), ov(_snap[10], _snap[30])))

# ---- control: trace deleted (KAPPA = 0) -------------------------------------------
CTRL_HELD, CTRL_OK = [], 0
for k in range(NITEMS):
    wm = WM(kappa=0.0, th5=TH5)
    wm.phase(CUE_CODES[k], CUE)
    wm.phase(None, HOLD)
    CTRL_HELD.append(int(wm.x23.sum()))
    CTRL_OK += int(int(np.argmax([ov(wm.x23, t) for t in TEMPL])) == k)
print('    control KAPPA=0: held %s of %d, sanity %d/%d  (the trace carries the hold)'
      % (CTRL_HELD, N23, CTRL_OK, NITEMS))


# ---------------------------------------------------------------- SANITY -----------
def readout(state, templ):
    sims = [ov(state, t) for t in templ]
    i = int(np.argmax(sims))
    return i, sims[i], sims


print('\n[SANITY] present k, hold 10 steps, read out the matching item')
SAN_OK = 0
SAN_SIMS = []
for k in range(NITEMS):
    got, sim, sims = readout(HELD[k].x23, TEMPL)
    SAN_OK += int(got == k)
    SAN_SIMS.append(sim)
    print('    %-8s held %3d | read %-8s %.2f | sims %s' % (
        ITEMS[k], HELD_SIZES[k], ITEMS[got], sim, ' '.join('%.2f' % s for s in sims)))
print('    SANITY %d/%d | matching ov %.2f..%.2f | unreadable (best ov < 0.40) %d/%d' % (
    SAN_OK, NITEMS, min(SAN_SIMS), max(SAN_SIMS),
    sum(1 for s in SAN_SIMS if s < 0.40), NITEMS), flush=True)

# ---------------------------------------------------------------- PERTURB ----------
def flip_uniform(x, frac, rng):
    """wm_tests.py rule: flip `frac` of ALL cells (mixed off->on and on->off)"""
    j = T.as_tensor(rng.choice(len(x), int(frac * len(x)), replace=False), device=DEV)
    y = x.clone()
    y[j] = 1.0 - y[j]
    return y


def flip_active(x, frac, rng):
    """literal reading: `frac` of the ACTIVE cells are flipped off"""
    act = T.nonzero(x > 0.5).flatten()
    n = max(1, int(round(frac * len(act))))
    j = act[T.as_tensor(rng.choice(len(act), n, replace=False), device=DEV)]
    y = x.clone()
    y[j] = 0.0
    return y


def flip_swap(x, frac, rng):
    """count-preserving: frac of the active cells off, as many inactive cells on"""
    act = T.nonzero(x > 0.5).flatten()
    off = T.nonzero(x < 0.5).flatten()
    n = max(1, int(round(frac * len(act))))
    ja = act[T.as_tensor(rng.choice(len(act), n, replace=False), device=DEV)]
    jo = off[T.as_tensor(rng.choice(len(off), n, replace=False), device=DEV)]
    y = x.clone()
    y[ja] = 0.0
    y[jo] = 1.0
    return y


FLIPS = {'uniform': flip_uniform, 'active': flip_active, 'swap': flip_swap}


def perturb_run(k, frac, rule, pop):
    """present k, hold 2, flip `frac` of the `pop` state, run 10 more.

    Returns (pre, post, ctrl, imm, pre5, post5, ctrl5) where `ctrl` is the same run
    without any flip and `imm` is the state one step after the flip."""
    rng = np.random.default_rng(SEED * 1000 + k * 37 + int(frac * 100)
                                + (0 if pop != 'x5' else 500))
    wmr, _ = present(k)                      # reference run, no flip
    wmr.phase(None, 2)
    pre, pre5 = wmr.x23.clone(), wmr.x5.clone()
    wmr.phase(None, HOLD)
    ctrl, ctrl5 = wmr.x23.clone(), wmr.x5.clone()

    wm, _ = present(k)
    wm.phase(None, 2)
    f = FLIPS[rule]
    if pop == 'x23':
        wm.apply_flip(f(wm.x23, frac, rng), pool=True)
    elif pop == 'x23_nopool':
        wm.apply_flip(f(wm.x23, frac, rng), pool=False)
    else:
        wm.x5 = f(wm.x5, frac, rng)
    wm.phase(None, 1)
    imm, imm5 = wm.x23.clone(), wm.x5.clone()
    wm.phase(None, HOLD - 1)
    return pre, wm.x23.clone(), ctrl, imm, pre5, wm.x5.clone(), ctrl5, imm5


print('\n[PERTURB] present k, hold 2, flip, run 10 more steps')
print('    ov_pre  = overlap with the pre-flip state (the number the task asks for)')
print('    ov_base = overlap of the NO-FLIP control with the pre-flip state (state drift)')
print('    ov_imm  = overlap 1 step after the flip (proves the flip entered the loop)')
print('    ov_ctrl = overlap with the no-flip control after the same 10 steps')
print('              (1.00 = the perturbation was completely corrected)')
PERT = {}


def report_pop(pop):
    print('    --- population %s (%s) ---' % (
        pop, 'L2/3 state that carries the trace' if pop.startswith('x23')
        else 'L5 readout layer'))
    for rule in ('uniform', 'active', 'swap'):
        for frac in (0.05, 0.20, 0.50):
            r_pre, r_imm, r_ctrl, r_base, sz = [], [], [], [], []
            for k in range(NITEMS):
                pre, post, ctrl, imm, pre5, post5, ctrl5, imm5 = perturb_run(k, frac, rule, pop)
                if pop == 'x5':
                    r_pre.append(ov(pre5, post5))
                    r_imm.append(ov(pre5, imm5))
                    r_ctrl.append(ov(ctrl5, post5))
                    r_base.append(ov(pre5, ctrl5))
                    sz.append((int(pre5.sum()), int(post5.sum()), int(ctrl5.sum())))
                else:
                    r_pre.append(ov(pre, post))
                    r_imm.append(ov(pre, imm))
                    r_ctrl.append(ov(ctrl, post))
                    r_base.append(ov(pre, ctrl))
                    sz.append((int(pre.sum()), int(post.sum()), int(ctrl.sum())))
            PERT[(pop, rule, frac)] = (float(np.mean(r_pre)), float(np.mean(r_ctrl)),
                                       float(np.mean(r_imm)), float(np.mean(r_base)))
            print('    %-8s %-5s | ov_pre %-6.3f ov_base %-6.3f ov_imm %-6.3f ov_ctrl %-6.3f | %d/%d/%d'
                  % (rule, '%d%%' % int(frac * 100), float(np.mean(r_pre)),
                     float(np.mean(r_base)), float(np.mean(r_imm)), float(np.mean(r_ctrl)),
                     int(np.mean([s[0] for s in sz])), int(np.mean([s[1] for s in sz])),
                     int(np.mean([s[2] for s in sz]))))


report_pop('x23')
report_pop('x23_nopool')
report_pop('x5')
print('    local_adapt2 baseline (wm_tests.py, L5 state): 0.68 / 0.25 / 0.13 at 5/20/50 %')
print('    PERTURB summary (task metric = ov_pre, with the no-flip drift ov_base for scale):')
print('      L2/3 state, uniform flip: ov_pre 5%% %.2f 20%% %.2f 50%% %.2f | ov_base %.2f | ov_ctrl %.2f/%.2f/%.2f'
      % (PERT[('x23', 'uniform', 0.05)][0], PERT[('x23', 'uniform', 0.20)][0],
         PERT[('x23', 'uniform', 0.50)][0], PERT[('x23', 'uniform', 0.05)][3],
         PERT[('x23', 'uniform', 0.05)][1], PERT[('x23', 'uniform', 0.20)][1],
         PERT[('x23', 'uniform', 0.50)][1]))
print('      L5 readout layer, uniform flip: ov_pre 5%% %.2f 20%% %.2f 50%% %.2f | ov_base %.2f | ov_ctrl %.2f/%.2f/%.2f'
      % (PERT[('x5', 'uniform', 0.05)][0], PERT[('x5', 'uniform', 0.20)][0],
         PERT[('x5', 'uniform', 0.50)][0], PERT[('x5', 'uniform', 0.05)][3],
         PERT[('x5', 'uniform', 0.05)][1], PERT[('x5', 'uniform', 0.20)][1],
         PERT[('x5', 'uniform', 0.50)][1]))

# ---------------------------------------------------------------- DISTRACTOR -------
print('\n[DISTRACTOR] present A (10), hold 2, present B (10), hold 10; which item is left?')
ALL_RES = {}
for setref in (True, False):
    res = []
    for a in range(NITEMS):
        for b in range(NITEMS):
            if a == b:
                continue
            wm, _ = present(a)
            wm.phase(None, 2)
            wm.phase(CUE_CODES[b], CUE, setref=setref)
            wm.phase(None, HOLD)
            got, sim, sims = readout(wm.x23, TEMPL)
            res.append((a, b, got, sim, sims[a], sims[b], int(wm.x23.sum())))
    ALL_RES[setref] = res
    surv = sum(1 for r in res if r[2] == r[0])
    take = sum(1 for r in res if r[2] == r[1])
    other = len(res) - surv - take
    weak = sum(1 for r in res if r[3] < 0.40)
    print('    m_ref re-referenced at the end of B: %-5s | A survived %2d/%d | B took over %2d/%d'
          ' | neither %d/%d | unreadable (best ov < 0.40) %d/%d | held %d-%d | mean ovA %.2f ovB %.2f'
          % (setref, surv, len(res), take, len(res), other, len(res), weak, len(res),
             min(r[6] for r in res), max(r[6] for r in res),
             float(np.mean([r[4] for r in res])), float(np.mean([r[5] for r in res]))))
print('    per-pair detail (m_ref re-referenced at the end of B):')
for (a, b, got, sim, ova, ovb, held) in ALL_RES[True]:
    print('      A=%-8s B=%-8s -> %-8s %.2f | ovA %.2f ovB %.2f | held %3d%s' % (
        ITEMS[a], ITEMS[b], ITEMS[got], sim, ova, ovb, held,
        '   <- unreadable' if sim < 0.40 else ''))

print('\nDONE seed %d code %d kappa %.2f tau %.0f fb %.2f' % (
    SEED, CODE_SIZE, KAPPA, TAU, FB), flush=True)
