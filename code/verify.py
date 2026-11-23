"""Final verification of the PFC small loop as a working-memory engine.

For each seed it reports the three things the objective asks for:
  HOLD     every item keeps its own state after the cue (self similarity, and how many
           of the 12 are most like themselves)
  OVERLAP  pairwise overlap of the held states (target < 0.10)
  CHAIN    8-step walk with ONE shared map and a threshold-free template readout

Usage: python verify.py <first_seed> <count>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
s0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
src = open(os.path.join(BASE, 'walk_rel2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
head, tail = src[:cut], src[cut:]

print('seed | hold(all-12) | mean self | overlap | chain steps correct', flush=True)
res = []
for sd in range(s0, s0 + n):
    ns = {'__file__': os.path.join(BASE, 'walk_rel2.py')}
    sys.argv = ['x', str(sd), '0.17', '0.05', '0.02']
    exec(compile(head, 'verify', 'exec'), ns)
    ov = ns['ov']; CODES = ns['CODES']; SIN = ns['SIN']; WORDS = ns['WORDS']
    held_state = ns['held_state']
    states, selfs, best_ok = [], [], 0
    for k in range(len(WORDS)):
        x23, _ = held_state(SIN[k])
        s = x23.cpu().numpy()
        states.append(s)
        sims = [ov(s, CODES[j]) for j in range(len(WORDS))]
        selfs.append(sims[k])
        best_ok += int(int(np.argmax(sims)) == k)
    live = [s for s in states if s.sum() > 0]
    ovl = float(np.mean([(live[i] * live[j]).sum() / max(((live[i] + live[j]) > 0).sum(), 1)
                         for i in range(len(live)) for j in range(i + 1, len(live))])) if len(live) > 1 else float('nan')
    # walk, with the trained map
    bufs = {}
    old = sys.stdout
    import io
    buf = io.StringIO()
    sys.stdout = buf
    try:
        exec(compile(tail, 'verify_walk', 'exec'), ns)
    finally:
        sys.stdout = old
    text = buf.getvalue()
    steps = [ln for ln in text.splitlines() if ln.strip().startswith('step ')]
    correct = sum(1 for ln in steps if 'read ' in ln and 'want ' in ln and
                  ln.split('read ')[1].split()[0] == ln.split('want ')[1][:len(ln.split('read ')[1].split()[0])])
    print('  %2d | %2d/12 | %.2f | %.3f | %d/%d' % (
        sd, best_ok, float(np.mean(selfs)), ovl, correct, len(ns['CHAIN']) - 1), flush=True)
    res.append((best_ok, float(np.mean(selfs)), ovl, correct))

print('\nmean: hold %d/12 | self %.2f | overlap %.3f | chain %.1f' % (
    int(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res])),
    float(np.mean([r[2] for r in res])), float(np.mean([r[3] for r in res]))), flush=True)
