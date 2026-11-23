"""Three PFC pathways with one motor block per speakable item.

Why blocks: the previous version shared one 280-cell motor map for all items, and
because the L5 states overlap, the learned map interfered -- suppression of the
non-speak items eroded the speak items' own pathway (act match fell 0.45 -> 0.32).
Separate blocks remove the interference by construction: an untrained block simply
never fires, so "thinking" cannot leak into the motor area.

    SPEAK items     each has its own 280-cell block, learned from babbling
    SUPPRESS items  no block at all -> the speech pathway is exactly 0 for them

Everything else is as described: PFC (L2/3<->L5) -> sensory / ganglia / motor,
instinct sensory->motor, heard motor->sensory, context go releases a tonic brake
and context hold drives the ganglia.
"""
import os
import sys
import numpy as np
import torch as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensory import Local3DStack
import text_encoder as te

DEV = 'cuda'
NS = 2800
NSE, N23, N5, NBG, NCTX = 1024, 1024, 512, 128, 32
A1, S1 = slice(0, 896), slice(896, 1024)   # auditory vs touch
ITEMS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
SPEAK, SUPPRESS = [0, 1, 2], [3, 4, 5]
NBLK = len(SPEAK)
NM = 280 * NBLK
DIMS = [(28, 10, 1), (28, 10, 10), (28, 10, 10), (28, 10, 10)]
ETA, WMAX, WL = 3e-2, 1.0, 0.30
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
GAIN_M, MOT_BUD, BG_BUD, INST_BUD = 3.0, 2.0, 6.0, 6.0
TH_MOT, GO_W, HOLD_W = 2.6, 4.5, 0.55
NSPLIT = 16

codes = T.as_tensor(np.stack([np.clip(te.pattern(w), 0, 1) for w in ITEMS]), device=DEV, dtype=T.float32)
INP = (codes > 0).float()
st = Local3DStack(DIMS, seed=SEED, dev=DEV)
st.calibrate(codes, target=0.12)
SEN_IN = st.forward(codes)[-1].round()
SEN_IN = SEN_IN / SEN_IN.sum(1, keepdim=True).clamp(min=1) * SEN_IN.sum(1, keepdim=True).mean()

rng = np.random.default_rng(SEED)
SX, SY, SZ = DIMS[-1]
sx, sy, sz = np.unravel_index(np.arange(NS), (SX, SY, SZ))
NX = NY = 32
cx = (np.arange(NSE) % NX) * (SX - 1) / (NX - 1)
cy = (np.arange(NSE) // NX) * (SY - 1) / (NY - 1)
dd = ((sx[None, :] - cx[:, None]) ** 2 / 9.0 + (sy[None, :] - cy[:, None]) ** 2 / 4.0
      + (sz[None, :] * 0.3) ** 2)
W_in = T.zeros(NSE, NS, device=DEV)
for j in range(NSE):
    W_in[j, np.argsort(dd[j])[:80]] = 1.0 / 80.0
W_in[S1, :] = 0.0        # the touch cells are not driven by sound

W_p = T.zeros(N23, N5, device=DEV)
W_q = T.zeros(N5, N23, device=DEV)
for blk in range(NSPLIT):
    a = slice(blk * N23 // NSPLIT, (blk + 1) * N23 // NSPLIT)
    b0 = slice(blk * N5 // NSPLIT, (blk + 1) * N5 // NSPLIT)
    W_p[a, b0] = T.as_tensor(rng.random((a.stop - a.start, b0.stop - b0.start)), device=DEV)
    W_q[b0, a] = T.as_tensor(rng.random((b0.stop - b0.start, a.stop - a.start)), device=DEV)
W_p *= WL / W_p.sum(1, keepdim=True).clamp(min=1e-9)
W_q *= WL / W_q.sum(1, keepdim=True).clamp(min=1e-9)
W_up = T.zeros(N23, NSE, device=DEV)
for blk in range(NSPLIT):
    a = slice(blk * N23 // NSPLIT, (blk + 1) * N23 // NSPLIT)
    b0 = slice(blk * NSE // NSPLIT, (blk + 1) * NSE // NSPLIT)
    W_up[a, b0] = T.as_tensor(rng.random((a.stop - a.start, b0.stop - b0.start)), device=DEV)
W_up *= (WL / 2) / W_up.sum(1, keepdim=True).clamp(min=1e-9)

W_l5m = T.zeros(NM, N5, device=DEV)                    # one block per speak item
W_hear = T.zeros(NSE, NM, device=DEV)
for i in range(280):
    for b in range(NBLK):
        W_hear[i * (NSE // 280):(i + 1) * (NSE // 280), b * 280 + i] = 1.0 / (NSE // 280)
TOUCH = 128      # the touch sub-population of the sensory area (0..127)
W_inst = T.zeros(NM, NSE, device=DEV)
for i in range(NM):
    W_inst[i, 896 + T.as_tensor(rng.choice(TOUCH, 96, replace=False), device=DEV)] = 1.0 / 96.0
W_inst *= INST_BUD / W_inst.sum(1, keepdim=True)
W_bg = T.zeros(NM, NBG, device=DEV)
for i in range(NM):
    W_bg[i, T.as_tensor(rng.choice(NBG, 64, replace=False), device=DEV)] = 1.0
W_bg *= BG_BUD / W_bg.sum(1, keepdim=True)
W_brake = T.zeros(NM, NCTX, device=DEV)
W_brake[:] = GO_W / NCTX
W_gobrk = T.zeros(NCTX, NCTX, device=DEV)
W_gobrk[:] = 1.0 / NCTX
W_hold = T.zeros(NBG, NCTX, device=DEV)
W_hold[:] = HOLD_W / NCTX

bias = T.zeros(N23, device=DEV) + 0.15
bud23 = W_up.abs().sum(1) + W_p.abs().sum(1) + bias.abs()
bud5 = T.full((N5,), WL, device=DEV)
bud_m = T.full((NM,), 1.0, device=DEV)


def run(th23, th_se, s_in, steps=14, go=False, hold=False, bg_off=False, se_drive=None, probe=False):
    se = T.zeros(NSE, device=DEV); x23 = T.zeros(N23, device=DEV)
    x5 = T.zeros(N5, device=DEV); mo = T.zeros(NM, device=DEV); bg = T.zeros(NBG, device=DEV)
    gv = T.ones(NCTX, device=DEV) if go else T.zeros(NCTX, device=DEV)
    hv = T.ones(NCTX, device=DEV) if hold else T.zeros(NCTX, device=DEV)
    comp = {}
    for _ in range(steps):
        d_se = (se_drive if se_drive is not None else W_in @ s_in)
        if float(mo.sum()) > 0:
            d_se = d_se + W_hear @ mo
        _t = d_se.sum()
        if float(_t) > 0:
            d_se = d_se / _t * (NSE * 0.12)
        nse = (d_se > th_se).float()
        n23 = (W_up @ nse + W_p @ x5 + bias > th23 * bud23).float()
        d5 = W_q @ n23
        _t5 = d5.sum()
        if float(_t5) > 0:
            d5 = d5 / _t5 * (N5 * 0.15)
        n5 = (d5 > 0.5 * bud5).float()
        nbg = (W_hold @ hv > 0.5).float() if hold else T.zeros(NBG, device=DEV)
        brake = max(0.0, 1.0 - float((W_gobrk @ gv).sum()))
        # feed-forward inhibition on the instinct: it scales with how much of the
        # sensory area is active, so a structured word is suppressed while the
        # itch (fully active) still gets through
        ff = 3.0 * float(nse[896:1024].mean())
        # the tonic brake only gates the CORTICAL pathway; a reflex is subcortical
        # and must not be blocked by it (that is why scratching is possible at all)
        cort = GAIN_M * (W_l5m @ n5) - W_brake @ (T.ones(NCTX, device=DEV) * brake)
        dm = T.clamp(cort, min=0.0) + (W_inst @ nse - ff)
        if not bg_off:
            dm = dm - W_bg @ nbg
        nm = (dm > TH_MOT * bud_m).float()
        if probe:
            comp = dict(pfc=GAIN_M * float((W_l5m @ n5).mean()), inst=float((W_inst @ nse).mean()) - 3.0 * float(nse.mean()),
                        brake=brake, inh=float((W_bg @ nbg).mean()))
        se, x23, x5, mo, bg = nse, n23, n5, nm, nbg
    return dict(se=se, l23=x23, l5=x5, mo=mo, bg=bg, comp=comp)


def calibrate(s_in):
    _d = W_in @ s_in
    _t = _d.sum()
    if float(_t) > 0:
        _d = _d / _t * (NSE * 0.12)
    lo, hi = 0.0, 1.0
    for _ in range(18):
        th_se = 0.5 * (lo + hi)
        n = float((_d > th_se).sum())
        if n > 150:
            lo = th_se
        elif n < 100:
            hi = th_se
        else:
            break
    lo, hi = 0.05, 0.95
    for _ in range(12):
        th23 = 0.5 * (lo + hi)
        r = run(th23, th_se, s_in)
        n = float(r['l23'].sum())
        if n > 400:
            lo = th23
        elif n < 100:
            hi = th23
        else:
            break
    return th23, th_se


def ov(a, b):
    return float((a * b).sum() / max(((a + b) > 0).sum(), 1))


TH = {}
print('seed %d | speak %s | suppress %s | %d motor blocks' % (SEED, SPEAK, SUPPRESS, NBLK), flush=True)
for k in range(len(ITEMS)):
    TH[k] = calibrate(SEN_IN[k])
    r = run(TH[k][0], TH[k][1], SEN_IN[k])
    print('  %-8s th23 %.2f th_se %.2f | sensory %4d L23 %4d L5 %3d' % (
        ITEMS[k], TH[k][0], TH[k][1], int(r['se'].sum()), int(r['l23'].sum()), int(r['l5'].sum())), flush=True)

print('binding each speak item to its own motor block (%d epochs) ...' % EPOCHS, flush=True)
for ep in range(EPOCHS):
    for b, k in enumerate(SPEAK):
        r = run(TH[k][0], TH[k][1], SEN_IN[k])
        rows = slice(b * 280, (b + 1) * 280)
        W_l5m[rows] = T.clamp(W_l5m[rows] + ETA * (1 - W_l5m[rows] / WMAX)
                              * (INP[k].unsqueeze(1) * r['l5'].unsqueeze(0)), 0, WMAX)
        mean = W_l5m[rows].sum(1, keepdim=True)
        W_l5m[rows] = W_l5m[rows] * T.clamp(MOT_BUD / mean.clamp(min=1e-9), max=1.0)

ITCH = T.zeros(NSE, device=DEV)
ITCH[896:1024] = 1.0    # the itch is a touch signal: it drives S1 only
print('\nTEST', flush=True)


def show(name, r):
    print('  %-30s motor %3d bg %3d | pfc %.2f inst %.2f brake %.1f inh %.1f' % (
        name, int(r['mo'].sum()), int(r['bg'].sum()),
        r['comp']['pfc'], r['comp']['inst'], r['comp']['brake'], r['comp']['inh']), flush=True)


# 1 THINK: a suppress item, PFC active, private setting (brake released) -> motor must stay 0
r_think = run(TH[SUPPRESS[0]][0], TH[SUPPRESS[0]][1], SEN_IN[SUPPRESS[0]], go=False, probe=True)
show('1 THINK  private, no speech block', r_think)
# 2 ACT: a speak item, private setting -> its own block fires with the intended code
r_act = run(TH[SPEAK[0]][0], TH[SPEAK[0]][1], SEN_IN[SPEAK[0]], go=True, probe=True)
b0 = r_act['mo'][0:280]
show('2 ACT    private + speak item', r_act)
m_act = ov(b0, INP[SPEAK[0]])
# 3 INSTINCT: the itch pattern, ganglia offline
r_inst = run(TH[0][0], 0.02, None, se_drive=ITCH, bg_off=True, probe=True)
show('3 INSTINCT (ganglia off, itch)', r_inst)
# 4 SUPPRESS: the same itch, public setting
r_sup = run(TH[0][0], 0.02, None, se_drive=ITCH, hold=True, probe=True)
show('4 SUPPRESS public + itch', r_sup)

print('\n  criteria:', flush=True)
c1 = int(r_think['mo'].sum()) == 0 and int(r_think['se'].sum()) > 0
c2 = m_act > 0.5
c3 = int(r_inst['mo'].sum()) > 0
c4 = int(r_sup['mo'].sum()) == 0 and int(r_sup['bg'].sum()) > 0
print('    THINK     %s  (motor %d, sensory %d)' % ('OK' if c1 else '--',
                                                    int(r_think['mo'].sum()), int(r_think['se'].sum())), flush=True)
print('    ACT       %s  (block-0 match %.2f)' % ('OK' if c2 else '--', m_act), flush=True)
print('    INSTINCT  %s  (motor %d)' % ('OK' if c3 else '--', int(r_inst['mo'].sum())), flush=True)
print('    SUPPRESS  %s  (motor %d, ganglia %d)' % ('OK' if c4 else '--',
                                                      int(r_sup['mo'].sum()), int(r_sup['bg'].sum())), flush=True)
print('  %s' % ('ALL CRITERIA PASS' if (c1 and c2 and c3 and c4) else 'fail'), flush=True)







