"""Predict the next word from the words that came BEFORE it, not from one word.

The current model reads one word, holds it, and emits the next.  That is single-item
prediction.  Here the state is built from a SEQUENCE: words are presented one after
another and the slow variable naturally accumulates them, so the held state is a
mixture -- the context.

Two things are measured:

  CONTEXT      (a, b) -> c and (d, e) -> f : can it use both preceding words?
  SAME-END     (0, 1) -> 2  and  (3, 1) -> 4 : the SAME last word in different
               contexts must give different predictions.  Without this there is no
               context, only a lookup from the last word.

Usage: python seq_predict.py <seed>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', '12', '8']
src = open(os.path.join(BASE, 'scale_more2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more2.py')}
exec(compile(src[:cut], 'sp', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CODES_T = ns['CODES_T']
SIN = ns['SIN']; T_ = ns['T']; DEV = ns['DEV']
W_q = ns['W_q']; W_ei = ns['W_ei']; W_ie = ns['W_ie']
bias23 = ns['bias23']; bias_i = ns['bias_i']; BUD = ns['BUD']
TH23v = ns['TH23v']; TH_I_CELL = ns['TH_I_CELL']; bud_i = ns['bud_i']
bud5 = ns['bud5']; TH5 = ns['TH5']; N23 = ns['N23']; N5 = ns['N5']; NI = ns['NI']
TAU, KAPPA = 400.0, 1.5
CUE_ONE, HOLD = 5, 10

# sentences: (context words) -> next word.  word 1 appears in two contexts with two
# different continuations on purpose.
SENTS = [
    ((0, 1), 2),
    ((3, 1), 4),      # same last word, different context -> different next
    ((5, 6), 7),
    ((8, 9), 10),
    ((11, 0), 5),
]


def state_of(seq):
    """present the words one after another, then hold; return the held L2/3 state"""
    m = T_.zeros(N23, device=DEV)
    x23 = T_.zeros(N23, device=DEV); xi = T_.zeros(NI, device=DEV)
    m_ref = None
    total = CUE_ONE * len(seq) + HOLD
    for t in range(total):
        idx = t // CUE_ONE
        inp = SIN[seq[idx]] if idx < len(seq) else None
        d23 = bias23 + (inp if inp is not None else 0) - W_ie @ xi
        rel = d23 / BUD
        if t == CUE_ONE * len(seq) - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23v > TH23v).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I_CELL * bud_i).float()
        x23 = fire
    return x23


def to_l5(x23):
    return ((0.02 + W_q @ x23) > TH5 * bud5).float()


W_m = T_.zeros(ns['NW'], N5, device=DEV)
MOT_BUD, ETA = ns['MOT_BUD'], ns['ETA']
print('training %d sentence transitions (context -> next word) ...' % len(SENTS), flush=True)
for ep in range(80):
    for ctx, nxt in SENTS:
        x23 = state_of(ctx)
        x5 = to_l5(x23)
        W_m = T_.clamp(W_m + ETA * (1 - W_m) * (CODES_T[nxt].unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))

cv = [c.cpu().numpy() for c in CODES_T]
cn = [max(float(np.linalg.norm(v)), 1e-9) for v in cv]


def predict(ctx):
    x5 = to_l5(state_of(ctx))
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ cv[j]) / cn[j] for j in range(NITEMS)]
    return int(np.argmax(sims)), float(np.max(sims))


print('\nCONTEXT test (predict the next word from the two preceding ones):', flush=True)
ok = 0
for ctx, nxt in SENTS:
    got, s = predict(ctx)
    good = got == nxt
    ok += int(good)
    print('  (%s, %s) -> %-8s (sim %.2f) | want %-8s %s' % (
        WORDS[ctx[0]][:6], WORDS[ctx[1]][:6], WORDS[got][:8], s, WORDS[nxt][:8],
        'OK' if good else '--'), flush=True)
print('  correct: %d/%d' % (ok, len(SENTS)), flush=True)

print('\nSAME LAST WORD, different context:', flush=True)
a = predict((0, 1))
b = predict((3, 1))
print('  (w0, w1) -> %s   vs   (w3, w1) -> %s' % (WORDS[a[0]][:8], WORDS[b[0]][:8]), flush=True)
print('  different predictions from the same ending: %s' % ('YES' if a[0] != b[0] else 'NO -- still a lookup on the last word'), flush=True)

print('\nLAST-WORD-ONLY baseline (ignore the context):', flush=True)
ok2 = 0
for ctx, nxt in SENTS:
    got, _ = predict((ctx[1],))
    ok2 += int(got == nxt)
print('  correct: %d/%d (a pure last-word lookup cannot exceed the best single answer)' % (
    ok2, len(SENTS)), flush=True)
