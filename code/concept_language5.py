"""Concept-level loop + a movement language made of muscle combinations.

Two changes from the word-level model:

  1. the unit is a CONCEPT, not a word.  Concepts are extracted from structured words
     (members of a category share a stem), and a concept's code is exactly the set of
     cells that answer to >=3 of the 5 members and to nothing outside (the strict
     criterion from concepts4.py, where chance level was 1.1 cells).
  2. the output is a MOVEMENT: each concept maps to a synergy -- a fixed combination of
     8 muscles over 3 time steps.  A concept therefore "says" something with a muscle
     pattern and a temporal order, and a listener can read it back.

Measured:
  A  concept codes: are the 8 concepts separable, and are they fewer than the words?
  B  concept chain: can the loop walk from one concept to the next?
  C  speaking: does the model emit the right synergy for the concept it is thinking of?
  D  listening: can the concept be recovered from the emitted synergy alone?

Usage: python concept_language.py <seed>
"""
import os
import sys
import numpy as np
import torch as T
import text_encoder as te

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(SEED)
DEV = 'cuda' if T.cuda.is_available() else 'cpu'

STEMS = ['cat', 'dog', 'run', 'bri', 'mou', 'flo', 'gra', 'spo']
SUFFIX = ['a', 'e', 'i', 'o', 'u']
WORDS, CAT = [], []
for ci, st in enumerate(STEMS):
    for sf in SUFFIX:
        WORDS.append(st + sf)
        CAT.append(ci)
N, NCAT, PER = len(WORDS), len(STEMS), len(SUFFIX)

N23, K, TARGET = 1024, 40, 50
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
comp_c = comp - comp.mean(0)

W_ff = rng.normal(0, 1, (N23, comp_c.shape[1]))
mask = np.zeros_like(W_ff, bool)
for j in range(N23):
    mask[j, rng.choice(comp_c.shape[1], K, replace=False)] = True
W_ff = W_ff * mask
W_ff = W_ff / np.abs(W_ff).sum(1, keepdims=True)
bud = np.abs(W_ff).sum(1) + 1e-9


def code_of(k, ncells=TARGET):
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        n = float((W_ff @ (g * comp_c[k]) > 0.5 * bud).sum())
        if n > ncells:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    return (W_ff @ (g * comp_c[k]) > 0.5 * bud).astype(np.float32)


WORDS_CODE = [code_of(k) for k in range(N)]
M = np.stack(WORDS_CODE)

# ---- concept cells, strict criterion: >=3 of 5 members, 0 outside
concept_mask = np.zeros((NCAT, N23), np.float32)
for cell in range(N23):
    on = [k for k in range(N) if M[k, cell] > 0]
    if not on:
        continue
    cats = set(CAT[k] for k in on)
    if len(cats) != 1:
        continue
    c = cats.pop()
    members = sum(1 for k in on if CAT[k] == c)
    if members >= 3 and len(on) - members == 0:
        concept_mask[c, cell] = 1.0

STRICT = concept_mask.copy()   # the 5-17 cells that answer to >=3 of 5 members
# the concept activated by ANY member: the union of the members' codes.  This is the
# same size class as a word code, so the working model's threshold calibration applies.
UNION = np.zeros((NCAT, N23), np.float32)
for c in range(NCAT):
    for k in range(N):
        if CAT[k] == c:
            UNION[c] = np.maximum(UNION[c], WORDS_CODE[k])
concept_mask = UNION
sizes = [int(concept_mask[c].sum()) for c in range(NCAT)]
print('concepts: %d | concept code sizes %s | words: %d' % (NCAT, sizes, N), flush=True)
print('  concept units (%d) < words (%d): %s' % (NCAT, N, 'YES' if NCAT < N else 'NO'), flush=True)

# ---- concept separability (no learning involved)
def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


cross = [ov(concept_mask[i], concept_mask[j]) for i in range(NCAT) for j in range(i + 1, NCAT)]
print('  pairwise overlap between concepts: mean %.3f | max %.3f' % (
    float(np.mean(cross)), float(np.max(cross))), flush=True)

# ---- the concept chain: a random order over the 8 concepts
CHAIN = [int(v) for v in rng.permutation(NCAT)]
print('\nconcept chain: %s' % ' -> '.join(str(c) for c in CHAIN), flush=True)

# ---- PFC small loop over concept codes (same mechanism as the word model)
N5, NI = 512, 256
def _sparse_rows(nrows, ncols, k, budget):
    """same construction as the working model: k inputs per row, row sum = budget"""
    Mx = np.zeros((nrows, ncols), np.float32)
    for j in range(nrows):
        Mx[j, rng.choice(ncols, k, replace=False)] = budget / k
    return T.as_tensor(Mx, device=DEV)


W_q = _sparse_rows(N5, N23, 20, 0.30)
W_ei = _sparse_rows(NI, N23, 100, 0.34)
W_ie = _sparse_rows(N23, NI, 100, 0.65)   # was 4x too weak: only 25 inputs/row, row sum 0.17
bias23 = T.zeros(N23, device=DEV) + 0.03
bias_i = T.zeros(NI, device=DEV) + 0.02
BUD = T.ones(N23, device=DEV) + bias23
# calibrate per-cell thresholds to the 90th percentile of each cell's own relative drive
# (a flat 0.35 saturated the layer: 90% of the cells fired)
_rels = []
for c in range(NCAT):
    _d = bias23 + T.as_tensor(concept_mask[c], device=DEV)
    _rels.append((_d / BUD).cpu().numpy())
Q = float(os.environ.get('QTH', '0.5'))
TH23v = T.as_tensor(np.quantile(np.stack(_rels), 0.90, axis=0), device=DEV, dtype=T.float32)
TH_I = T.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T.float32)
bud_i = T.full((NI,), 0.34, device=DEV)
bud5 = T.full((N5,), 0.30, device=DEV)
TH5 = float(os.environ.get('TH5O', '0.17'))
TAU, KAPPA, CUE, HOLD = 400.0, 1.5, 6, 10
GAIN = 1.0   # the concept codes are only 5-17 cells; without a gain the bias dominates the state
CONCEPT = [T.as_tensor(concept_mask[c] * GAIN, device=DEV) for c in range(NCAT)]


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


# ---- the movement language: 8 muscles x 3 steps per concept
NMUS = int(os.environ.get('NMUS', '12'))
STEPS = int(os.environ.get('NST', '4'))
SYN = np.zeros((NCAT, NMUS * STEPS), np.float32)
for c in range(NCAT):
    pick = rng.choice(NMUS, 3, replace=False)
    for i, mu in enumerate(pick):
        SYN[c, i * NMUS + mu] = 1.0
print('\nmovement language: %d muscles x %d steps = %d channels | 3 muscles per concept' % (
    NMUS, STEPS, NMUS * STEPS), flush=True)
so = [ov(SYN[i], SYN[j]) for i in range(NCAT) for j in range(i + 1, NCAT)]
print('  synergy overlap between concepts: mean %.3f max %.3f' % (
    float(np.mean(so)), float(np.max(so))), flush=True)

# ---- learn: held concept -> the synergy of the NEXT concept (the loop speaks the future)
NW = NMUS * STEPS
W_m = T.zeros(NW, N5, device=DEV)
MOT_BUD, ETA = 0.05 * (len(CHAIN) / 7.0), 3e-2
for ep in range(80):
    for i in range(len(CHAIN) - 1):
        a, b = CHAIN[i], CHAIN[i + 1]
        x5 = to_l5(hold_state(CONCEPT[a]))
        W_m = T.clamp(W_m + ETA * (1 - W_m) * (
            T.as_tensor(SYN[b], device=DEV).unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))


def speak(x5):
    d = (W_m @ x5).cpu().numpy()
    return (d > 0.4 * MOT_BUD).astype(np.float32)


def listen(pattern):
    """a listener recovers the concept from the muscle pattern alone"""
    v = pattern / max(float(np.linalg.norm(pattern)), 1e-9)
    sims = [float(v @ s) / max(float(np.linalg.norm(s)), 1e-9) for s in SYN]
    return int(np.argmax(sims)), float(np.max(sims))


print('\nA concept chain, spoken with muscles and read back:', flush=True)
cur, ok, ok_listen = CHAIN[0], 0, 0
for step in range(len(CHAIN) - 1):
    x5 = to_l5(hold_state(CONCEPT[cur]))
    pat = speak(x5)
    heard, s = listen(pat)
    want = CHAIN[step + 1]
    good = heard == want
    ok_listen += int(good)
    print('  thinking concept %d -> muscles %s -> listener hears concept %d (sim %.2f) | want %d %s' % (
        cur, ''.join('#' if v > 0 else '.' for v in pat), heard, s, want,
        'OK' if good else '--'), flush=True)
    cur = heard
print('  utterances correctly understood: %d/%d' % (ok_listen, len(CHAIN) - 1), flush=True)


# ---- diagnostic: are the L5 states for different concepts actually different?
print('\nL5 states per concept:', flush=True)
x5s = [to_l5(hold_state(CONCEPT[c])).cpu().numpy() for c in range(NCAT)]
for c in range(NCAT):
    print('  concept %d: %d cells' % (c, int(x5s[c].sum())), flush=True)
def _ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0
cross = [_ov(x5s[i], x5s[j]) for i in range(NCAT) for j in range(i + 1, NCAT)]
print('  pairwise overlap of L5 states: mean %.3f | max %.3f' % (float(np.mean(cross)), float(np.max(cross))), flush=True)










# ================= reverse direction: hear a muscle sequence, answer with one =====
# The external pattern is treated like any other sensory input: it goes through a sparse
# random projection, the loop holds it, a map reads which CONCEPT it is, and then the
# model answers with the synergy of the next concept in its chain.
print('', flush=True)
print('REVERSE: an outside speaker sends a muscle sequence, the model answers', flush=True)
NWIN = NMUS * STEPS
W_hear = rng.normal(0, 1, (N23, NWIN))
mask_h = np.zeros_like(W_hear, bool)
for j in range(N23):
    mask_h[j, rng.choice(NWIN, 10, replace=False)] = True
W_hear = W_hear * mask_h
W_hear = W_hear / np.abs(W_hear).sum(1, keepdims=True)
bud_h = np.abs(W_hear).sum(1) + 1e-9


def hear_code(pat):
    lo, hi = 0.1, 8.0
    for _ in range(18):
        g = 0.5 * (lo + hi)
        n = float((W_hear @ (g * pat) > 0.5 * bud_h).sum())
        if n > 200:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    return (W_hear @ (g * pat) > 0.5 * bud_h).astype(np.float32)


HEARD = [hear_code(SYN[c]) for c in range(NCAT)]
print('  heard codes: %s cells' % [int(h.sum()) for h in HEARD], flush=True)
hov = [ov(HEARD[i], HEARD[j]) for i in range(NCAT) for j in range(i + 1, NCAT)]
print('  pairwise overlap of heard codes: mean %.3f max %.3f' % (
    float(np.mean(hov)), float(np.max(hov))), flush=True)
W_rec = T.zeros(N23, N5, device=DEV)
for ep in range(80):
    for c in range(NCAT):
        x5 = to_l5(hold_state(T.as_tensor(HEARD[c] * 6.0, device=DEV)))
        W_rec = T.clamp(W_rec + ETA * (1 - W_rec) * (
            T.as_tensor(STRICT[c], device=DEV).unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_rec = W_rec * (MOT_BUD / W_rec.sum(1, keepdim=True).clamp(min=1e-6))


def recognize_heard(pat):
    x5 = to_l5(hold_state(T.as_tensor(hear_code(pat) * 6.0, device=DEV)))
    d = (W_rec @ x5).cpu().numpy()
    sims = [float(d @ STRICT[c]) / max(float(np.linalg.norm(STRICT[c])), 1e-9) for c in range(NCAT)]
    return int(np.argmax(sims)), float(np.max(sims))


ok2 = 0
for i in range(len(CHAIN) - 1):
    sent = CHAIN[i]
    want = CHAIN[i + 1]
    heard_concept, s1 = recognize_heard(SYN[sent])
    x5 = to_l5(hold_state(CONCEPT[heard_concept]))
    reply = speak(x5)
    said, s2 = listen(reply)
    good = (heard_concept == sent) and (said == want)
    ok2 += int(good)
    print('  sent concept %d -> model hears %d (sim %.2f), replies concept %d (sim %.2f) | want %d %s' % (
        sent, heard_concept, s1, said, s2, want, 'OK' if good else '--'), flush=True)
print('  exchanges answered correctly: %d/%d' % (ok2, len(CHAIN) - 1), flush=True)


