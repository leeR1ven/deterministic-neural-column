"""Why does the muscle pattern degrade at 36 words?

Prints, for the first transition of the chain:
  the size of x5 (the readout stage) at test time and at training time
  the drive the shared map produces for the target rows
  how the thresholded drive compares with the target's input code

Usage: python why_act.py <seed> <n_items> <chain_len>
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
exec(compile(src[:cut], 'wa', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CHAIN = ns['CHAIN']
held_state = ns['held_state']; W_m = ns['W_m']; CODES_T = ns['CODES_T']
SIN = ns['SIN']; MOT_BUD = ns['MOT_BUD']; THM = ns['THM']
TRANS = ns['TRANS']; ov = ns['ov']

x23, x5 = held_state(SIN[CHAIN[0]])
drive = W_m @ x5
want = CODES_T[CHAIN[1]]
wc = (want.cpu().numpy() > 0.5)
print('items %s | transitions %d | MOT_BUD %.3f' % (NITEMS, len(TRANS), MOT_BUD), flush=True)
print('x5 at test: %d cells (of %d)' % (int(x5.sum()), x5.numel()), flush=True)
print('target code: %d cells' % int(wc.sum()), flush=True)
d = drive.cpu().numpy()
print('drive: max %.4f | median %.4f | 90pct of target rows %.4f | 99pct all %.4f' % (
    float(d.max()), float(np.median(d)), float(np.quantile(d[wc], 0.9)), float(np.quantile(d, 0.99))), flush=True)
for th in (0.1, 0.2, 0.4, 0.8):
    m = (d > th * MOT_BUD)
    print('  threshold %.1f x budget (=%.4f): %3d cells | overlap with the target code %.2f' % (
        th, th * MOT_BUD, int(m.sum()), ov(m.astype(float), wc.astype(float))), flush=True)
# how many training rows exist at all
trained = (W_m.sum(1) > 0).cpu().numpy()
print('rows with any weight: %d of %d' % (int(trained.sum()), W_m.shape[0]), flush=True)
