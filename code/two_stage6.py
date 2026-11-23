"""Two stages, cleanly separated.

The previous attempt bolted the redundancy-reduction rule onto the RECURRENT loop and
the operating point drifted (some items died: L23 = 0).  Here the two jobs are split:

  STAGE A  feedforward orthogonalisation, exactly as the reference library does it:
           卤1 complementary input  ->  subtract the mean input (adaptation)
           ->  local signed projection  ->  score > theta * own budget
           one global theta, no recurrence, no per-item tuning.
           Verified in isolation: overlap 0.033 at density 0.073 (the random bound).

  STAGE B  the recurrent loop on top of that code: L5, motor blocks per transition,
           heard feedback that re-enters through stage A.

Measured: stage-A overlap, then the long chain / held-out / branching tests.
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
N23, N5 = 1024, 512
NW = 280
WORDS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
         'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
         'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
NITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ITEMS = WORDS[:NITEMS]
CLEN = int(sys.argv[7]) if len(sys.argv) > 7 else 8
CHAIN = list(range(min(CLEN, NITEMS)))
TRANS = list(zip(CHAIN[:-1], CHAIN[1:]))
HELD = (4, 5) if (NITEMS > 5 and (len(sys.argv) <= 6 or sys.argv[6] != 'all')) else None
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ETA, WMAX = 3e-2, 1.0
KFF = int(sys.argv[4]) if len(sys.argv) > 4 else 40
DIMS = [(28, 10, 1), (28, 10, 10), (28, 10, 10), (28, 10, 10)]

# ---------- stage A: 卤1 complementary code with adaptation
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in ITEMS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)          # 卤1, 560
DC = comp.shape[1]
mean_in = comp.mean(0)
comp_c = comp - mean_in                                              # adaptation
print('items %d | code %d dims (卤1) | input overlap before/after centring: %.2f / %.2f' % (
    NITEMS, DC,
    float(np.mean([(comp[i] * comp[j]).sum() / DC for i in range(NITEMS) for j in range(i + 1, NITEMS)])),
    float(np.mean([(comp_c[i] * comp_c[j]).sum() / DC for i in range(NITEMS) for j in range(i + 1, NITEMS)]))),
    flush=True)

rng = np.random.default_rng(SEED)
W_ff = rng.normal(0, 1, (N23, DC))
if KFF < DC:
    mask = np.zeros_like(W_ff, bool)
    for j in range(N23):
        mask[j, rng.choice(DC, KFF, replace=False)] = True
    W_ff = W_ff * mask
W_ff = W_ff / np.abs(W_ff).sum(1, keepdims=True)
bud_ff = np.abs(W_ff).sum(1) + 1e-9


def stage_a(vec):
    """single feedforward pass: return the sparse code as a 0/1 tensor"""
    v = T.as_tensor(vec - mean_in, device=DEV, dtype=T.float32)
    score = T.as_tensor(W_ff, device=DEV, dtype=T.float32) @ v
    b = T.as_tensor(bud_ff, device=DEV, dtype=T.float32)
    return (score > TH_A * b).float()


# one global theta for stage A, chosen so the mean density is ~10 %
best = None
for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    ds = []
    for k in range(NITEMS):
        v = comp_c[k]
        s = W_ff @ v
        ds.append(float((s > th * bud_ff).mean()))
    m = float(np.mean(ds))
    if best is None or abs(m - 0.10) < abs(best[1] - 0.10):
        best = (th, m)
TH_A = best[0]
codes = []
for k in range(NITEMS):
    codes.append((W_ff @ comp_c[k] > TH_A * bud_ff).astype(float))
dens = float(np.mean([c.mean() for c in codes]))
ovA = float(np.mean([(codes[i] * codes[j]).sum() / max(((codes[i] + codes[j]) > 0).sum(), 1)
                     for i in range(NITEMS) for j in range(i + 1, NITEMS)]))
print('stage A: theta %.2f | density %.3f | overlap %.3f | bound %.3f | ratio %.1f' % (
    TH_A, dens, ovA, dens / (2 - dens), ovA / (dens / (2 - dens))), flush=True)
print('  %s' % ('ORTHOGONAL (overlap < 0.10)' if ovA < 0.10 else 'still correlated'), flush=True)

# ---------- stage B: the recurrent loop and one motor block per transition
SIN = T.as_tensor(np.stack(codes), device=DEV, dtype=T.FloatTensor.dtype)
NM = 280 * len(TRANS)
rng2 = np.random.default_rng(SEED + 100)
# sparse fan-in: a large fan-in gives every L5 cell almost the same drive, which makes
# the density a step function of the threshold (measured: 1.00 -> 0.03 with nothing in
# between).  The coefficient of variation is ~1/sqrt(f*K), so a small K restores a
# graded curve.
KQ = int(sys.argv[5]) if len(sys.argv) > 5 else 20
W_q = T.zeros(N5, N23, device=DEV)
for j in range(N5):
    idx = T.as_tensor(rng2.choice(N23, KQ, replace=False), device=DEV)
    W_q[j, idx] = 1.0
W_q *= 0.30 / W_q.sum(1, keepdim=True).clamp(min=1e-9)
W_l5m = T.zeros(NM, N5, device=DEV)
bias5 = T.zeros(N5, device=DEV) + 0.02
bud5 = T.full((N5,), 0.30, device=DEV)
TH5 = 0.30
bud_m1 = T.full((NM,), 0.075, device=DEV)
THM = 0.25


def loop(x23, steps=10, th5=None):
    x5 = T.zeros(N5, device=DEV)
    for _ in range(steps):
        d5 = W_q @ x23 + bias5
        n5 = (d5 > (TH5 if th5 is None else th5) * bud5).float()
        x5 = n5
    return x5


def ov(a, b):
    return float((a * b).sum() / max(((a + b) > 0).sum(), 1))


_b5 = None
for th in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25):
    ds = [float(loop(SIN[k], th5=th).mean()) for k in range(NITEMS)]
    m = float(np.mean(ds))
    if _b5 is None or abs(m - 0.10) < abs(_b5[1] - 0.10):
        _b5 = (th, m)
TH5 = _b5[0]
print('theta5 = %.2f -> mean L5 density %.2f' % (TH5, _b5[1]), flush=True)
print('training %d transitions ...' % len(TRANS), flush=True)
for ep in range(EPOCHS):
    for bi, (a, b) in enumerate(TRANS):
        if (a, b) == HELD:
            continue
        x5 = loop(SIN[a])
        rows = slice(bi * 280, (bi + 1) * 280)
        W_l5m[rows] = T.clamp(W_l5m[rows] + ETA * (1 - W_l5m[rows] / WMAX)
                              * (T.as_tensor((np.clip(te.pattern(ITEMS[b]), 0, 1) > 0.5).astype(float),
                                             device=DEV).unsqueeze(1) * x5.unsqueeze(0)), 0, WMAX)
        mean = W_l5m[rows].sum(1, keepdim=True)
        W_l5m[rows] = W_l5m[rows] * T.clamp(0.075 / mean.clamp(min=1e-9), max=1.0)


def blocks_drive(x5):
    return [float((W_l5m[b * 280:(b + 1) * 280] @ x5).sum()) for b in range(NM // 280)]


def walk(start, steps=8):
    path = [start]
    x5 = loop(SIN[start])
    for _ in range(steps):
        drives = blocks_drive(x5)
        bi = int(np.argmax(drives))
        if max(drives) <= 0:
            path.append(-1); break
        m = (W_l5m[bi * 280:(bi + 1) * 280] @ x5 > THM * bud_m1[:280]).float()
        heard = m.cpu().numpy()
        # the heard code re-enters stage A as a 卤1 pattern on the same 280 channels
        hb = (heard > 0.5).astype(float)
        vec = np.concatenate([2 * hb - 1, 1 - 2 * hb])
        code = (W_ff @ (vec - mean_in) > TH_A * bud_ff).astype(float)
        # which item does this code look like?
        sims = [ov(code, c) for c in codes]
        path.append(int(np.argmax(sims)))
        x5 = loop(T.as_tensor(code, device=DEV, dtype=T.FloatTensor.dtype))
    return path


p = walk(0, steps=len(CHAIN))
print('\nS1 LONG CHAIN from item 0:', flush=True)
print('  %s' % ' '.join(ITEMS[i][:3] if i >= 0 else '---' for i in p), flush=True)
s1 = p[:len(CHAIN)] == CHAIN
print('  -> %s (reached %d/%d)' % ('full chain' if s1 else 'breaks', sum(1 for i, v in enumerate(p[:len(CHAIN)]) if v == CHAIN[i]), len(CHAIN)), flush=True)

pa = walk(HELD[0], steps=3) if HELD else [0, -1]
print('\nS2 HELD OUT %s: cue -> %s' % (
    (HELD if HELD else 'disabled'),
    ' '.join(ITEMS[i][:3] if i >= 0 else '---' for i in pa)), flush=True)
s2 = bool(HELD) and len(pa) > 1 and pa[1] == HELD[1]
print('  -> %s' % ('produced the untrained transition' if s2 else 'memorisation only'), flush=True)

print('\nVERDICT', flush=True)
print('  stage A overlap %.3f (target < 0.10) : %s' % (ovA, 'OK' if ovA < 0.10 else 'FAIL'), flush=True)
print('  S1 %d-step chain : %s' % (len(CHAIN), 'OK' if s1 else 'FAIL'), flush=True)
print('  S2 held-out      : %s' % ('generalises' if s2 else 'memorises only'), flush=True)







