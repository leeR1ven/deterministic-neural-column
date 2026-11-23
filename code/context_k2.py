"""How long a context can the loop actually use?  (fixed construction)

For k context words, the sentences share their last k-1 words and differ ONLY in the
first word, and that first word determines the continuation.  So a lookup on the last
1, 2, ... k-1 words cannot tell them apart -- only a model that keeps the whole context
can.

Usage: python context_k2.py <seed> <k>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', '12', '8']
src = open(os.path.join(BASE, 'scale_more2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more2.py')}
exec(compile(src[:cut], 'ck2', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CODES_T = ns['CODES_T']
SIN = ns['SIN']; DEV = ns['DEV']; T_ = ns['T']; N23 = ns['N23']
W_q = ns['W_q']; W_ei = ns['W_ei']; W_ie = ns['W_ie']
bias23 = ns['bias23']; bias_i = ns['bias_i']; BUD = ns['BUD']
TH23v = ns['TH23v']; TH_I_CELL = ns['TH_I_CELL']; bud_i = ns['bud_i']
bud5 = ns['bud5']; TH5 = ns['TH5']; N5 = ns['N5']; NI = ns['NI']
TAU, KAPPA = 400.0, 1.5
CUE_ONE, HOLD = 5, 10

# k-1 shared trailing words (filler), 3 discriminating first words
TAIL = tuple(range(4, 3 + K))          # words 4..3+K-1 as filler
SENTS = []
for i in range(3):
    SENTS.append(((i,) + TAIL, 8 + i))  # (first word, filler...) -> a distinct next word
SENTS.append(((11,) + TAIL, 9))         # one more, same tail


def state_of(seq):
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
for ep in range(100):
    for ctx, nxt in SENTS:
        x5 = to_l5(state_of(ctx))
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


ok = sum(1 for ctx, nxt in SENTS if predict(ctx)[0] == nxt)
print('k = %d | full context %d/%d' % (K, ok, len(SENTS)), flush=True)
for m in (1, 2, K - 1):
    if m >= K:
        continue
    okb = sum(1 for ctx, nxt in SENTS if predict(ctx[-m:])[0] == nxt)
    print('  baseline: last %d word(s) -> %d/%d' % (m, okb, len(SENTS)), flush=True)
