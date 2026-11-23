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


W_q = sparse_rows(N5, N23, 20, 0.30)
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


def to_l5(x23):
    return ((0.02 + W_q @ x23) > TH5 * bud5).float()


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
        x5 = to_l5(hold_state(CONC[a]))
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
    x5 = to_l5(hold_state(CONC[cur]))
    heard, s = listen(speak(x5))
    want = CHAIN[step + 1]
    ok += int(heard == want)
    cur = heard
print('chain over learned concepts: %d/%d utterances understood (TH5 %.2f)' % (
    ok, len(CHAIN) - 1, TH5), flush=True)
