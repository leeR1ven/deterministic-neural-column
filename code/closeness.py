"""How close is the answer, not whether it is word-perfect.

For every answer step of the 40 taught sentences this measures three things:
  rank        where the right word sits among all 501 words in the motor readout (1 = chosen)
  overlap     what fraction of the right word's own motor cells actually fired
  cells       how many cells fired in that step in total (is it sparse?)

Usage: python closeness.py
"""
import os
import runpy
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(r'D:\deepseekherness\repo_publish\deterministic-neural-column\code\closeness.py'))
g = runpy.run_path(os.path.join(BASE, 'huge.py'))
D0, TH_A = g['D0'], g['TH_A']
stepf, predict, keyfrom = g['stepf'], g['predict'], g['keyfrom']
Ms, Midx, Pidx, pairs, nz = g['Ms'], g['Midx'], g['Pidx'], g['pairs'], g['nz']


def closeness(q, ref, s=1.6, maxlen=18):
    F = {}
    for w in q:
        F = stepf(F, Pidx[w])
    V, res = nz[0].astype(np.int32), []
    for k in range(min(maxlen, len(ref))):
        act = predict(keyfrom(V, F)) * s
        fired = act > TH_A
        if not fired.any():
            break
        sc = np.asarray(Ms @ fired[D0:].astype(np.float32)).ravel()
        tw = ref[k]
        rank = int((sc > sc[tw]).sum()) + 1
        cells = Midx[tw] + D0
        res.append((rank, float(fired[cells].mean()), int(fired.sum())))
        F = stepf(F, V)
        V = np.nonzero(fired)[0]
    return res


rows = []
for q, a in pairs[:40]:
    rows += closeness(q, a)
r = np.array([x[0] for x in rows])
ov = np.array([x[1] for x in rows])
n = np.array([x[2] for x in rows])
print('', flush=True)
print('answer steps measured: %d' % len(r), flush=True)
print('the right word is the model\'s first choice      : %5.1f%%' % (100 * (r == 1).mean()), flush=True)
print('the right word is in its top 3                   : %5.1f%%' % (100 * (r <= 3).mean()), flush=True)
print('the right word is in its top 5                   : %5.1f%%' % (100 * (r <= 5).mean()), flush=True)
print('median rank of the right word                    : %5.1f' % np.median(r), flush=True)
print('mean fraction of the right word\'s cells that fired: %5.1f%%' % (100 * ov.mean()), flush=True)
print('median cells firing per step                     : %5.1f (of 36864)' % np.median(n), flush=True)
