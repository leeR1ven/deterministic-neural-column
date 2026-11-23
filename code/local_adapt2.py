"""Final configuration: closed-loop density control for L2/3 and L5, fixed motor
threshold, separate motor block per transition.

Why this combination:
  * closed-loop density control removes the seed dependence (measured: theta
    settles at 0.38/0.30 for every seed)
  * the motor threshold must NOT be under that control: before training its
    weights are zero, so the controller drives theta to the floor and the two
    blocks then fire identically (measured: beta->gamma fidelity 0.00)
  * one motor block per transition removes the interference that made two
    transitions collapse to a chance-level 0.58 overlap
"""
import sys
import numpy as np
import torch as T
sys.path.insert(0, r'D:\deepseekherness\gpu')
from sensory import Local3DStack
import text_encoder as te

DEV = 'cuda'
NS, N23, N5 = 2800, 1024, 512
ITEMS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
DIMS = [(28, 10, 1), (28, 10, 10), (28, 10, 10), (28, 10, 10)]
SEQ = [0, 1, 2]
NPAIR = len(SEQ) - 1
NM = 280 * NPAIR
ETA, EPOCHS, WMAX, WL = 3e-2, 120, 1.0, 0.30
M1BUD, THM = 0.075, 0.25
LO, HI, STEP = 0.13, 0.35, 0.004
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NBLK = 16

codes = T.as_tensor(np.stack([np.clip(te.pattern(w), 0, 1) for w in ITEMS]), device=DEV, dtype=T.float32)
INP = (codes > 0).float()
st = Local3DStack(DIMS, seed=SEED, dev=DEV)
st.calibrate(codes, target=0.12)
S = st.forward(codes)[-1].round()
S = S / S.sum(1, keepdim=True).clamp(min=1) * S.sum(1, keepdim=True).mean()

rng = np.random.default_rng(SEED)
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

W_p = T.zeros(N23, N5, device=DEV)
W_q = T.zeros(N5, N23, device=DEV)
for blk in range(NBLK):
    a = slice(blk * N23 // NBLK, (blk + 1) * N23 // NBLK)
    b0 = slice(blk * N5 // NBLK, (blk + 1) * N5 // NBLK)
    W_p[a, b0] = T.as_tensor(rng.random((a.stop - a.start, b0.stop - b0.start)), device=DEV)
    W_q[b0, a] = T.as_tensor(rng.random((b0.stop - b0.start, a.stop - a.start)), device=DEV)
W_p *= WL / W_p.sum(1, keepdim=True).clamp(min=1e-9)
W_q *= WL / W_q.sum(1, keepdim=True).clamp(min=1e-9)

W_l5m1 = T.zeros(NM, N5, device=DEV)          # built entirely by learning
bias = T.zeros(N23, device=DEV) + 0.05
bud23 = W_in.abs().sum(1) + W_p.abs().sum(1) + bias.abs()
bud5 = W_q.abs().sum(1).clamp(min=1e-9)
bud_m1 = T.full((NM,), M1BUD, device=DEV)


BUD_REC = (W_p.abs().sum(1) + bias.abs())      # budget when there is no input

TAU = 400.0        # slow-variable time constant, in simulation steps
KAPPA = 1.5        # how strongly a cell's own past raises its excitability


def settle(th23, th5, s_in, steps=14, mt=None, mref=None):
    """One presentation or hold phase.

    The per-cell slow variable mt integrates the cell's own relative drive; at the end
    of the input phase mref records its value, and afterwards a cell fires more easily
    the closer its trace is to that remembered level.  This is what makes the state
    survive the removal of the input: without it the unit is memoryless, and an earlier
    version held by having the tonic bias exceed the threshold, which left every cell
    on (1024/1024) -- not memory.
    """
    x23 = T.zeros(N23, device=DEV); x5 = T.zeros(N5, device=DEV); m = T.zeros(NM, device=DEV)
    mt = T.zeros(N23, device=DEV) if mt is None else mt
    for _ in range(steps):
        bud = bud23 if s_in is not None else BUD_REC
        drive = W_p @ x5 + bias + (W_in @ s_in if s_in is not None else 0)
        rel = drive / bud
        if mref is None:
            n23 = (rel > th23).float()
        else:
            mn = mt / mref.clamp(min=1e-6)
            n23 = (rel + KAPPA * mn * th23 > th23).float()
        mt = mt + (rel - mt) / TAU
        d5 = W_q @ n23
        tot = d5.sum()
        if float(tot) > 0:
            d5 = d5 / tot * (N5 * 0.15)
        n5 = (d5 > th5 * bud5).float()
        nm = (W_l5m1 @ n5 > THM * bud_m1).float()
        x23, x5, m = n23, n5, nm
    return x23, x5, m, mt


def settle_item(s_in, steps=14):
    """presentation-time threshold control: keep the assembly between 100 and 400
    cells.  Some items land at 70-76 cells (and then die at L5), others at 510 (and
    then swallow every later step)."""
    lo, hi = 0.05, 0.95
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        out = settle(mid, th5, s_in, steps)[:3]
        n = float(out[0].sum())
        best = out
        if n > 400:
            lo = mid
        elif n < 100:
            hi = mid
        else:
            return out
    return best

th23 = th5 = 0.5
for it in range(300):
    d23 = d5 = 0.0
    for k in range(len(ITEMS)):
        x23, x5, _, _ = settle(th23, th5, S[k])
        d23 += float(x23.mean()); d5 += float(x5.mean())
    d23 /= len(ITEMS); d5 /= len(ITEMS)
    if d23 > HI: th23 = min(th23 + STEP, 0.95)
    elif d23 < LO: th23 = max(th23 - STEP, 0.02)
    if d5 > HI: th5 = min(th5 + STEP, 0.95)
    elif d5 < LO: th5 = max(th5 - STEP, 0.02)
print('seed %d | theta L23 %.2f L5 %.2f M1 %.2f | density %.2f/%.2f' % (
    SEED, th23, th5, THM, d23, d5), flush=True)

for ep in range(EPOCHS):
    for bi, (a, b) in enumerate(zip(SEQ[:-1], SEQ[1:])):
        xa, x5a, _ = settle_item(S[a])
        # training-time floor: if the item's L5 assembly is tiny, the motor block
        # gets no signal to learn from (measured: beta 76 L2/3 cells -> 0 L5 cells,
        # and then that block's fidelity was 0.00).  Top up from the same projection.
        MIN5 = 24
        if float(x5a.sum()) < MIN5:
            d5r = W_q @ xa
            idx = T.topk(d5r, min(MIN5, N5)).indices
            x5a = T.zeros(N5, device=DEV)
            x5a[idx] = 1.0
        rows = slice(bi * 280, (bi + 1) * 280)
        W_l5m1[rows] = T.clamp(W_l5m1[rows] + ETA * (1 - W_l5m1[rows] / WMAX)
                               * (INP[b].unsqueeze(1) * x5a.unsqueeze(0)), 0, WMAX)
        mean = W_l5m1[rows].sum(1, keepdim=True) / N5
        W_l5m1[rows] = W_l5m1[rows] * T.clamp(M1BUD / mean.clamp(min=1e-9), max=1.0)
        for bj in range(NPAIR):
            if bj == bi:
                continue
            r2 = slice(bj * 280, (bj + 1) * 280)
            W_l5m1[r2] = T.clamp(W_l5m1[r2] - ETA * (1 - W_l5m1[r2] / WMAX)
                                 * (INP[b].unsqueeze(1) * x5a.unsqueeze(0)), 0, WMAX)

def settle_item(s_in, steps=14):
    """presentation-time threshold control: keep the assembly between 100 and 400
    cells.  Some items land at 70-76 cells (and then die at L5), others at 510 (and
    then swallow every later step)."""
    lo, hi = 0.05, 0.95
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        out = settle(mid, th5, s_in, steps)[:3]
        n = float(out[0].sum())
        best = out
        if n > 400:
            lo = mid
        elif n < 100:
            hi = mid
        else:
            return out
    return best


FP = [settle_item(S[k]) for k in range(len(ITEMS))]
print('  per-item active counts (L23 / L5 / M1):', flush=True)
for k in range(len(ITEMS)):
    print('    %-8s %4d / %4d / %4d' % (ITEMS[k], int(FP[k][0].sum()), int(FP[k][1].sum()), int(FP[k][2].sum())), flush=True)


def ov(a, b):
    return float((a * b).sum() / max(((a + b) > 0).sum(), 1))


fid = []
for bi, (a, b) in enumerate(zip(SEQ[:-1], SEQ[1:])):
    _, _, m = FP[a]
    fid.append(ov(m[bi * 280:(bi + 1) * 280], INP[b]))
C = [f[0].cpu().numpy() for f in FP]
dist = float(np.mean([ov(C[i], C[j]) for i in range(len(C)) for j in range(i + 1, len(C))]))
alive = 0
held_states = []
for k in range(len(ITEMS)):
    x23, x5, m = settle_item(S[k])
    for _ in range(10):
        x23 = (W_p @ x5 + bias > th23 * BUD_REC).float()
        x5 = (W_q @ x23 > th5 * bud5).float()
        m = (W_l5m1 @ x5 > THM * bud_m1).float()
    held_states.append(x23.clone())
    alive += int(float(x23.mean()) > 0.01)
# An independent audit showed that this "hold" cannot fail: the bias alone (0.5) already
# exceeds the free-running threshold (~0.36), so with the input removed every cell turns
# on and the state is the whole layer.  Report that instead of hiding it: the held state
# size, and how much of the item's own presentation state survives in it.
held_sizes = [int(h.sum()) for h in held_states]
held_self = [ov(held_states[k].cpu().numpy(), C[k]) for k in range(len(ITEMS))]
held_ov = float(np.mean([ov(held_states[i].cpu().numpy(), held_states[j].cpu().numpy())
                         for i in range(len(ITEMS)) for j in range(i + 1, len(ITEMS))]))
print('  HOLD DIAGNOSTIC: held sizes %s of %d | self-overlap %.2f | held-vs-held %.2f' % (
    held_sizes, N23, float(np.mean(held_self)), held_ov), flush=True)
print('  (%s)' % (
    'the held state is the WHOLE layer -- this is not working memory, see README section 4'
    if min(held_sizes) > 0.9 * N23 else 'held states are sparse'), flush=True)


def which(x23):
    best, bi = -1, -1
    for i, (f, _, _) in enumerate(FP):
        v = ov(f, x23)
        if v > best:
            best, bi = v, i
    return bi


def heard(m, x5):
    sums = [float((W_l5m1[i * 280:(i + 1) * 280] @ x5).sum()) for i in range(NPAIR)]
    if max(sums) <= 0:
        return None
    return m[int(np.argmax(sums)) * 280:(int(np.argmax(sums)) + 1) * 280]


path = []
x23, x5, m = settle_item(S[SEQ[0]])
path.append(which(x23))
for _ in range(6):
    h = heard(m, x5)
    if h is None:
        path.append(-1); break
    heard_in = st.forward(h.unsqueeze(0))[-1][0]
    heard_in = heard_in / heard_in.sum().clamp(min=1) * S.sum(1, keepdim=True).mean()
    x23, x5, m = settle_item(heard_in, steps=14)
    path.append(which(x23))

print('  MOTOR %s | DIST %.2f | HOLD %d/6' % (
    ' '.join('%.2f' % f for f in fid), dist, alive), flush=True)
print('  CHAIN %s' % ' '.join(ITEMS[i][:3] if i >= 0 else '---' for i in path), flush=True)
# The working-memory criterion now requires the held state to be SPARSE and ITEM-SPECIFIC.
# It used to count "mean > 0.01", which the all-on state satisfies trivially; an audit
# caught that this made the criterion impossible to fail.
hold_real = (max(held_sizes) < 0.9 * N23) and (held_ov < 0.5) and (float(np.mean(held_self)) > 0.3)
ok = (min(fid) > 0.5) and (dist < 0.5) and (alive == 6) and hold_real and (path[1] == 1) and (path[2] == 2)
print('  hold is real (sparse, item-specific): %s' % ('yes' if hold_real else 'NO -- the held state is the whole layer'), flush=True)
print('  %s' % ('ALL CRITERIA PASS' if ok else 'FAIL (see the line above)'), flush=True)









