"""All three dimensions at once: 160 words, a two-word context, a 60-step chain.

Previous rounds scaled one dimension at a time.  Here the next word depends on the last
TWO words, so the held state must keep both, and the chain is 60 steps long.

Construction: a random sequence of 162 words over a 160-word vocabulary.  The pair
table is defined BY the sequence: (w_t, w_t+1) -> w_t+2.  The model is trained on every
pair of the sequence, then started from the first two words and walked 60 steps; it is
correct if it reproduces the stored sequence exactly.

Capacity: 60 transitions x 70 cells / 4096 output channels = 1.02, inside the rule from
the previous rounds.

Usage: python seq2.py <seed>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
NWORDS, LSEQ, STEPS = 160, 162, 60
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', str(NWORDS), str(LSEQ),
            '4096', '70', '4096']
src = open(os.path.join(BASE, 'scale_more8.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more8.py')}
exec(compile(src[:cut], 's2', 'exec'), ns)

WORDS = ns['WORDS']; CODES_T = ns['CODES_T']; SIN = ns['SIN']
DEV = ns['DEV']; T_ = ns['T']; N23 = ns['N23']
W_q = ns['W_q']; W_ei = ns['W_ei']; W_ie = ns['W_ie']
bias23 = ns['bias23']; bias_i = ns['bias_i']; BUD = ns['BUD']
TH23v = ns['TH23v']; TH_I_CELL = ns['TH_I_CELL']; bud_i = ns['bud_i']
bud5 = ns['bud5']; TH5 = ns['TH5']; N5 = ns['N5']; NI = ns['NI']
NITEMS = ns['NITEMS']; ETA = ns['ETA']; NW = ns['NW']
TAU, KAPPA = 400.0, 1.5
CUE_ONE, HOLD = 5, 10
MOT_BUD = 0.05 * (STEPS / 7.0)
CODE_T = [T_.as_tensor(np.asarray(c, dtype=np.float32), device=DEV) for c in ns['CODES']]

rng = np.random.default_rng(int(SEED) + 5)
SEQ = [int(v) for v in rng.integers(0, NWORDS, LSEQ)]


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


W_m = T_.zeros(N23, N5, device=DEV)   # motor = concept code
print('%d words | %d-step chain | each step depends on the previous TWO words' % (
    NWORDS, STEPS), flush=True)
for ep in range(80):
    for i in range(LSEQ - 2):
        ctx = (SEQ[i], SEQ[i + 1])
        x5 = to_l5(state_of(ctx))
        W_m = T_.clamp(W_m + ETA * (1 - W_m) * (CODE_T[SEQ[i + 2]].unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))

cbuf = [np.asarray(c, dtype=np.float32) for c in ns['CODES']]
cnorm = [max(float(np.linalg.norm(v)), 1e-9) for v in cbuf]


def predict(ctx):
    x5 = to_l5(state_of(ctx))
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ cbuf[j]) / cnorm[j] for j in range(NITEMS)]
    return int(np.argmax(sims)), float(np.max(sims))


cur = [SEQ[0], SEQ[1]]
got = []
ok = 0
for step in range(STEPS):
    nxt, s = predict(tuple(cur))
    got.append(nxt)
    if nxt == SEQ[step + 2]:
        ok += 1
    cur = [cur[1], nxt]
print('  walked %d steps | correct %d/%d' % (STEPS, ok, STEPS), flush=True)
print('  first 10 predicted: %s' % [WORDS[i][:6] for i in got[:10]], flush=True)
print('  first 10 true     : %s' % [WORDS[i][:6] for i in SEQ[2:12]], flush=True)

# baseline: same vocabulary and chain, but context of ONE word only
print('\n  control: same sequence with a ONE-word context (must do worse if two matter):', flush=True)
W1 = T_.zeros(N23, N5, device=DEV)
for ep in range(80):
    for i in range(LSEQ - 1):
        x5 = to_l5(state_of((SEQ[i],)))
        W1 = T_.clamp(W1 + ETA * (1 - W1) * (CODE_T[SEQ[i + 1]].unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W1 = W1 * (MOT_BUD / W1.sum(1, keepdim=True).clamp(min=1e-6))


def predict1(w):
    x5 = to_l5(state_of((w,)))
    dv = (W1 @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ cbuf[j]) / cnorm[j] for j in range(NITEMS)]
    return int(np.argmax(sims))


cur, ok1 = SEQ[0], 0
for step in range(STEPS):
    nxt = predict1(cur)
    if nxt == SEQ[step + 1]:
        ok1 += 1
    cur = nxt
print('  one-word context: %d/%d' % (ok1, STEPS), flush=True)


