"""Is the shared map's drive aligned with the target code at 36 words?

If the strongest drive cells are NOT the target's cells, no thresholding or lateral
inhibition can fix the muscle pattern -- the association itself would be off.
This prints the overlap of the top-k drive cells with the target code for several k,
and the same for a 12-word run as a control.

Usage: python align_check.py <seed> <n_items> <chain_len>
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
exec(compile(src[:cut], 'ac', 'exec'), ns)

WORDS = ns['WORDS']; CHAIN = ns['CHAIN']; NITEMS = ns['NITEMS']
held_state = ns['held_state']; W_m = ns['W_m']; CODES_T = ns['CODES_T']
SIN = ns['SIN']; ov = ns['ov']

x23, x5 = held_state(SIN[CHAIN[0]])
d = (W_m @ x5).cpu().numpy()
tgt = (CODES_T[CHAIN[1]].cpu().numpy() > 0.5)
order = np.argsort(d)[::-1]
print('items %s | x5 %d cells | target code %d cells | drive max %.4f' % (
    NITEMS, int(x5.sum()), int(tgt.sum()), float(d.max())), flush=True)
for k in (10, 20, 40, 60, 100):
    top = np.zeros_like(tgt)
    top[order[:k]] = True
    print('  top-%3d drive cells: %3d of them are in the target (precision %.2f, recall %.2f)' % (
        k, int((top & tgt).sum()),
        float((top & tgt).sum() / k), float((top & tgt).sum() / max(tgt.sum(), 1))), flush=True)
# how much drive mass sits on target vs off-target cells
on = float(d[tgt].sum()); off = float(d[~tgt].sum())
print('  drive mass: on-target %.3f | off-target %.3f (ratio %.2f)' % (
    on, off, on / max(off, 1e-9)), flush=True)
