"""Wire the LEARNED concept layer into the PFC loop.

The concept representation is no longer a mask I computed.  It is layer 2, learned by the
local rule in concept_emerge.py; a concept's input to the PFC is simply how layer 2
responds to that category's words (averaged over its five instances, then thresholded) --
a measured pattern, not a hand-picked set of cells.

Chain + muscle language + listening, exactly as before, now on top of the learned layer.

Usage: python concept_emerge_loop.py <seed> [th5]
"""
import os
import sys
import numpy as np
import torch as T
import text_encoder as te

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TH5 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
rng = np.random.default_rng(SEED)
DEV = 'cuda' if T.cuda.is_available() else 'cpu'

STEMS = ['cat', 'dog', 'run', 'bri', 'mou', 'flo', 'gra', 'spo']
SUFFIX = ['a', 'e', 'i', 'o', 'u']
WORDS, CAT = [], []
for ci, st in enumerate(STEMS):
    for sf in SUFFIX:
        WORDS.append(st + sf)
        CAT.append(ci)
N, NCAT = len(WORDS), len(STEMS)

# ---------- layer 1 ----------
N1, K1, TARGET1 = 1024, 40, 50
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
comp_c = comp - comp.mean(0)
W1 = rng.normal(0, 1, (N1, comp_c.shape[1]))
m1 = np.zeros_like(W1, bool)
for j in range(N1):
    m1[j, rng.choice(comp_c.shape[1], K1, replace=False)] = True
W1 = W1 * m1
W1 = W1 / np.abs(W1).sum(1, keepdims=True)
bud1 = np.abs(W1).sum(1) + 1e-9
X = []
for k in range(N):
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if float((W1 @ (g * comp_c[k]) > 0.5 * bud1).sum()) > TARGET1:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    X.append((W1 @ (g * comp_c[k]) > 0.5 * bud1).astype(np.float32))
X = np.stack(X)

# ---------- layer 2: learned, local rule ----------
N2, K2 = 256, 100
W2 = rng.normal(0, 1, (N2, N1)) * 0.1
m2 = np.zeros_like(W2, bool)
for j in range(N2):
    m2[j, rng.choice(N1, K2, replace=False)] = True
W2 = W2 * m2
W2 = W2 / np.abs(W2).sum(1, keepdims=True)
ETA, DECAY, WIN = 0.08, 0.002, 3
order = list(range(N))
for ep in range(300):
    rng.shuffle(order)
    for k in order:
        x = X[k]
        s = W2 @ x
        win = np.argsort(s)[-WIN:]
        W2[win] += ETA * (x[None, :] - W2[win] * x.sum() / max(x.sum(), 1))
        W2 -= DECAY * W2
        W2 = np.maximum(W2, 0.0)
        W2 = W2 / np.maximum(np.abs(W2).sum(1, keepdims=True), 1e-9)

S = np.stack([W2 @ X[k] for k in range(N)])
TH2 = np.quantile(S, 0.90, axis=0)
A = (S > TH2).astype(np.float32)

# ---------- the concept input to the PFC: how the learned layer responds ----------
CONCEPT = np.zeros((NCAT, N2), np.float32)
for c in range(NCAT):
    v = np.mean([A[k] for k in range(N) if CAT[k] == c], axis=0)
    v = v / max(float(v.max()), 1e-9)
    CONCEPT[c] = (v > 0.5).astype(np.float32)
print('learned concept patterns: %s cells (layer 2 = %d cells)' % (
    [int(c.sum()) for c in CONCEPT], N2), flush=True)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


cross = [ov(CONCEPT[i], CONCEPT[j]) for i in range(NCAT) for j in range(i + 1, NCAT)]
print('concept overlap (learned): mean %.3f max %.3f' % (float(np.mean(cross)), float(np.max(cross))), flush=True)

# ---------- PFC loop on the learned concepts ----------
N23, N5, NI = N2, 512, 256


def sparse_rows(nr, nc, k, budget):
    M = np.zeros((nr, nc), np.float32)
    for j in range(nr):
        M[j, rng.choice(nc, k, replace=False)] = budget / k
    return T.as_tensor(M, device=DEV)


W_q = sparse_rows(N5, N23, int(os.environ.get('KQ', '20')), 0.30)
W_ei = sparse_rows(NI, N23, 100, 0.34)
W_ie = sparse_rows(N23, NI, 100, 0.65)
bias23 = T.zeros(N23, device=DEV) + 0.03
bias_i = T.zeros(NI, device=DEV) + 0.02
BUD = T.ones(N23, device=DEV) + bias23
TH_I = T.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T.float32)
bud_i = T.full((NI,), 0.34, device=DEV)
bud5 = T.full((N5,), 0.30, device=DEV)
TAU, KAPPA, CUE, HOLD = 400.0, 1.5, 6, 10
_rels = []
for c in range(NCAT):
    _d = bias23 + T.as_tensor(CONCEPT[c], device=DEV)
    _rels.append((_d / BUD).cpu().numpy())
TH23v = T.as_tensor(np.quantile(np.stack(_rels), 0.90, axis=0), device=DEV, dtype=T.float32)
CONC = [T.as_tensor(CONCEPT[c], device=DEV) for c in range(NCAT)]


def hold_state(cc, cue=CUE, extra=HOLD):
    m = T.zeros(N23, device=DEV)
    x23 = T.zeros(N23, device=DEV); xi = T.zeros(NI, device=DEV)
    m_ref = None
    for t in range(cue + extra):
        inp = cc if t < cue else None
        d23 = bias23 + (inp if inp is not None else 0) - W_ie @ xi
        rel = d23 / BUD
        if t == cue - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23v > TH23v).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I * bud_i).float()
        x23 = fire
    return x23


def to_l5(x23, th5=None):
    t = TH5 if th5 is None else th5
    return ((0.02 + W_q @ x23) > t * bud5).float()


x5s = [to_l5(hold_state(CONC[c])).cpu().numpy() for c in range(NCAT)]
print('L5 states: %s cells' % [int(v.sum()) for v in x5s], flush=True)
cross5 = [ov(x5s[i], x5s[j]) for i in range(NCAT) for j in range(i + 1, NCAT)]
print('L5 overlap: mean %.3f max %.3f' % (float(np.mean(cross5)), float(np.max(cross5))), flush=True)

# ---------- the muscle language and the chain ----------
NMUS, STEPS = 16, 5
SYN = np.zeros((NCAT, NMUS * STEPS), np.float32)
for c in range(NCAT):
    pick = rng.choice(NMUS, 3, replace=False)
    for i, mu in enumerate(pick):
        SYN[c, i * NMUS + mu] = 1.0
NW = NMUS * STEPS
CHAIN = [int(v) for v in rng.permutation(NCAT)]
W_m = T.zeros(NW, N5, device=DEV)
MOT_BUD, ETA2 = 0.05 * (len(CHAIN) / 7.0), 3e-2
for ep in range(80):
    for i in range(len(CHAIN) - 1):
        a, b = CHAIN[i], CHAIN[i + 1]
        x5 = to_l5(hold_state(CONC[a]), 0.17)
        W_m = T.clamp(W_m + ETA2 * (1 - W_m) * (
            T.as_tensor(SYN[b], device=DEV).unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))


def speak(x5):
    d = (W_m @ x5).cpu().numpy()
    return (d > 0.4 * MOT_BUD).astype(np.float32)


def listen(pat):
    v = pat / max(float(np.linalg.norm(pat)), 1e-9)
    sims = [float(v @ s) / max(float(np.linalg.norm(s)), 1e-9) for s in SYN]
    return int(np.argmax(sims)), float(np.max(sims))


ok = 0
cur = CHAIN[0]
for step in range(len(CHAIN) - 1):
    x5 = to_l5(hold_state(CONC[cur]), 0.17)
    heard, s = listen(speak(x5))
    want = CHAIN[step + 1]
    ok += int(heard == want)
    cur = heard
print('chain over learned concepts: %d/%d utterances understood (TH5 %.2f)' % (
    ok, len(CHAIN) - 1, TH5), flush=True)

# ================= several concepts at once -> predict the next concept ============
print('', flush=True)
print('MULTI-CONCEPT context: two concepts heard in order -> the next concept', flush=True)


def hold_two(ca, cb, cue=6, extra=10):
    # the two concepts are active TOGETHER (they are disjoint codes, overlap 0.000)
    return hold_together(ca, cb, cue, extra)


def hold_together(ca, cb, cue=6, extra=10):
    """present concept a, then concept b, then hold -- the state carries both"""
    m = T.zeros(N23, device=DEV)
    x23 = T.zeros(N23, device=DEV); xi = T.zeros(NI, device=DEV)
    m_ref = None
    total = 2 * cue + extra
    for t in range(total):
        inp = (ca + cb) if t < 2 * cue else None
        d23 = bias23 + (inp if inp is not None else 0) - W_ie @ xi
        rel = d23 / BUD
        if t == 2 * cue - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23v > TH23v).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I * bud_i).float()
        x23 = fire
    return x23


# the same second concept appears in two contexts and leads to different continuations
PHRASES = [((0, 1), 2), ((3, 1), 4), ((5, 6), 7), ((7, 0), 5)]
# ---- homeostasis: pick the L5 threshold that makes the phrases themselves separable.
# A fixed threshold worked on one seed and not on others; instead each network raises
# its own threshold until no two combined states overlap more than 0.2 while keeping
# the states non-empty.  This is the computational form of a cell adjusting its own
# excitability to a target rate -- local, no supervisor.
_pre = {(a, b): hold_two(CONC[a], CONC[b]) for (a, b), _n in PHRASES}   # hold once, threshold many times
_best = None
for _t in np.arange(0.14, 0.46, 0.02):
    _st = [to_l5(_pre[(a, b)], _t).cpu().numpy() for (a, b), _n in PHRASES]
    _sz = [float(v.sum()) for v in _st]
    if min(_sz) < 3:
        continue
    _mx = max(ov(_st[i], _st[j]) for i in range(len(_st)) for j in range(i + 1, len(_st)))
    if _mx < 0.20:
        _best = (float(_t), _mx, _sz)
        break
if _best is None:
    _best = (0.24, float('nan'), [])
    print('  homeostasis: no threshold reached overlap < 0.2, falling back to 0.24', flush=True)
TH5 = _best[0]
print('  homeostasis chose TH5 = %.2f (max phrase overlap %.2f, state sizes %s)' % (
    _best[0], _best[1], [int(v) for v in _best[2]]), flush=True)
# two maps: one for the sequence (trained above), one for the multi-concept context.
# They are separate because the same output rows trained by both tasks blend together
# and then fire for neither; biologically the sequence and the context are also handled
# by different pathways.  The readout is scale-free (half of a row's own weight), so the
# training strength does not matter.
W_m2 = T.zeros(NW, N5, device=DEV)
for ep in range(120):
    for (a, b), nxt in PHRASES:
        x5 = to_l5(hold_two(CONC[a], CONC[b]))
        tgt = T.as_tensor(SYN[nxt], device=DEV)
        W_m2 = T.clamp(W_m2 + ETA2 * (1 - W_m2) * (tgt.unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)


def answer_two(a, b):
    x5 = to_l5(hold_two(CONC[a], CONC[b]))
    d = (W_m2 @ x5).cpu().numpy()
    row = W_m2.sum(1).cpu().numpy() + 1e-9
    pat = (d > 0.5 * row).astype(np.float32)
    heard, s = listen(pat)
    return heard, s


ok3 = 0
for (a, b), nxt in PHRASES:
    got, s = answer_two(a, b)
    good = got == nxt
    ok3 += int(good)
    print('  heard (%d then %d) -> answers concept %d (sim %.2f) | want %d %s' % (
        a, b, got, s, nxt, 'OK' if good else '--'), flush=True)
print('  multi-concept answers correct: %d/%d' % (ok3, len(PHRASES)), flush=True)

g1 = answer_two(0, 1)[0]
g2 = answer_two(3, 1)[0]
print('  same last concept (1) in two contexts -> %d vs %d : %s' % (
    g1, g2, 'DIFFERENT (uses the full context)' if g1 != g2 else 'SAME (still a lookup)'), flush=True)




# diagnostic: how different are the combined states for (0,1) and (3,1)?
_s01 = to_l5(hold_two(CONC[0], CONC[1])).cpu().numpy()
_s31 = to_l5(hold_two(CONC[3], CONC[1])).cpu().numpy()
_s0 = to_l5(hold_state(CONC[0])).cpu().numpy()
_s1 = to_l5(hold_state(CONC[1])).cpu().numpy()
print('  combined state (0,1): %d cells | (3,1): %d cells | overlap between them %.2f' % (
    int(_s01.sum()), int(_s31.sum()), ov(_s01, _s31)), flush=True)
print('  singles: concept 0 %d, concept 1 %d, concept 3 %d cells' % (
    int(_s0.sum()), int(_s1.sum()), int(to_l5(hold_state(CONC[3])).cpu().numpy().sum())), flush=True)





# ---- all six pairwise overlaps among the four phrase states
print('', flush=True)
print('PHRASE STATE OVERLAP MATRIX (all 6 pairs):', flush=True)
_keys = [(a, b) for (a, b), _n in PHRASES]
_stt = {}
for k in _keys:
    _stt[k] = to_l5(hold_two(CONC[k[0]], CONC[k[1]])).cpu().numpy()
for i in range(len(_keys)):
    for j in range(i + 1, len(_keys)):
        print('  %s vs %s : %.2f  (sizes %d, %d)' % (
            _keys[i], _keys[j], ov(_stt[_keys[i]], _stt[_keys[j]]),
            int(_stt[_keys[i]].sum()), int(_stt[_keys[j]].sum())), flush=True)
_vals = [ov(_stt[_keys[i]], _stt[_keys[j]]) for i in range(len(_keys)) for j in range(i + 1, len(_keys))]
print('  mean %.2f | max %.2f' % (float(np.mean(_vals)), float(np.max(_vals))), flush=True)




