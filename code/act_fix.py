"""Concentrate the shared map's weights so the muscle pattern stays clean at scale.

Diagnosed last round at 36 words: the target rows get only ~0.007 of drive against a
maximum of 0.05, and only 4 motor cells cross the threshold -- so the spoken word is
wrong even though the continuous drive still matches the right word.

Hypothesis: the rows spread their weight over many cells instead of concentrating on
the ones that were active when they should fire.  Fix that with a standard local
mechanism -- mild decay of the synapses that were NOT active during a Hebbian event
(synaptic competition):

    W <- W * (1 - decay * (1 - outer(active_pre, target)))
    ... then the usual soft-bound growth, then the row normalisation

Measured for decay in {0, 0.05, 0.2}:
    row concentration   fraction of a target row's weight sitting on its top-20 cells
    ACT                 does the thresholded muscle pattern read as the intended word

Usage: python act_fix.py <seed> <n_items> <chain_len>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
NIT = sys.argv[2] if len(sys.argv) > 2 else '36'
CLEN = sys.argv[3] if len(sys.argv) > 3 else '16'
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', NIT, CLEN]
src = open(os.path.join(BASE, 'scale_more2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more2.py')}
exec(compile(src[:cut], 'af', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CHAIN = ns['CHAIN']
held_state = ns['held_state']; CODES_T = ns['CODES_T']; SIN = ns['SIN']
DEV = ns['DEV']; T_ = ns['T']; TRANS = ns['TRANS']
MOT_BUD = ns['MOT_BUD']; ETA = ns['ETA']; NW = ns['NW']; N5 = ns['N5']
recog = ns['recog']; ov = ns['ov']

W0 = T_.zeros(NW, N5, device=DEV)


def train(decay, epochs=80):
    W = T_.zeros(NW, N5, device=DEV)
    for ep in range(epochs):
        for a, b in TRANS:
            _, x5 = held_state(SIN[a])
            tgt = CODES_T[b]
            outer = tgt.unsqueeze(1) * x5.unsqueeze(0)          # (280,512) where both active
            if decay > 0:
                W = W * (1.0 - decay * (1.0 - outer))            # weaken everything not co-active
            W = T_.clamp(W + ETA * (1 - W) * outer, 0, 1.0)
            W = W * (MOT_BUD / W.sum(1, keepdim=True).clamp(min=1e-6))
    return W


def concentration(W):
    """fraction of a target row's weight sitting on its 20 strongest cells"""
    w = W.cpu().numpy()
    rows = (CODES_T[CHAIN[1]].cpu().numpy() > 0.5)
    vals = []
    for r in np.where(rows)[0][:40]:
        row = w[r]
        top = np.sort(row)[-20:].sum()
        vals.append(top / max(row.sum(), 1e-9))
    return float(np.mean(vals))


def act_test(W):
    x23, x5 = held_state(SIN[CHAIN[0]])
    drive = W @ x5
    d = drive.cpu().numpy()
    m = (d > 0.4 * MOT_BUD).astype(np.float32)
    if m.sum() == 0:
        return 0, -1, 0.0
    spoken, s = recog(T_.as_tensor(m, device=DEV))
    return int(m.sum()), spoken, s


print('items %s | transitions %d | MOT_BUD %.3f' % (NITEMS, len(TRANS), MOT_BUD), flush=True)
print('decay | row concentration | motor cells | spoken (sim) | correct?', flush=True)
for decay in (0.0, 0.05, 0.2, 0.4):
    W = train(decay)
    conc = concentration(W)
    n, sp, s = act_test(W)
    ok = (sp == CHAIN[1])
    print('  %.2f | %.2f | %3d | %-8s (%.2f) | %s' % (
        decay, conc, n, WORDS[sp] if sp >= 0 else '---', s, 'OK' if ok else '--'), flush=True)
    if ok:
        break
