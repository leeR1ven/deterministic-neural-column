"""Structure of the associative web: trunks, basins, reachability.

The picture being tested (the designer's): like a tree -- from any branch you eventually
climb to the trunk, and from the trunk you can reach any branch.  In a deterministic
transition graph that means:

  * attractors   (cycles / fixed points) = the trunks
  * basin sizes  how many starting words flow into each trunk
  * transient    how many steps it takes to reach the trunk
  * reachability from the trunk / from the chain start = the branches it can reach

Measured on the graph the model actually forms, for every word.

Usage: python graph2.py <seed> <n_items>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
NIT = sys.argv[2] if len(sys.argv) > 2 else '40'
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', NIT, '20']
src = open(os.path.join(BASE, 'scale_more2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more2.py')}
exec(compile(src[:cut], 'g2', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CODES_T = ns['CODES_T']
held_state = ns['held_state']; W_m = ns['W_m']; CHAIN = ns['CHAIN']

cv = [c.cpu().numpy() for c in CODES_T]
cn = [max(float(np.linalg.norm(v)), 1e-9) for v in cv]
succ = {}
for k in range(NITEMS):
    _, x5 = held_state(ns['SIN'][k])
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    succ[k] = int(np.argmax([float(dv @ cv[j]) / cn[j] for j in range(NITEMS)]))

# ---- attractors and basins
attr_of, attractors = {}, {}
for s in range(NITEMS):
    path, cur, idx = [], s, {}
    while cur not in idx and cur not in attr_of:
        idx[cur] = len(path)
        path.append(cur)
        cur = succ[cur]
    if cur in idx:                     # found a new cycle
        cyc = tuple(path[idx[cur]:])
        attractors.setdefault(cyc, len(attractors))
    a = attr_of.get(cur, attractors.get(tuple(path[idx[cur]:]), None) if cur in idx else None)
    if a is None:
        a = attr_of[cur]
    for node in path:
        attr_of[node] = a

basin = {}
for s in range(NITEMS):
    basin[attr_of[s]] = basin.get(attr_of[s], 0) + 1

inv = {v: k for k, v in attractors.items()}
print('seed %s | %d words' % (SEED, NITEMS), flush=True)
print('  attractors (trunks): %d' % len(attractors), flush=True)
for cyc, i in sorted(attractors.items(), key=lambda kv: -basin.get(kv[1], 0)):
    print('    trunk %s (length %d) <- basin %d words' % (
        ' -> '.join(WORDS[x][:6] for x in cyc), len(cyc), basin.get(i, 0)), flush=True)

# ---- transient depth to the trunk
depths = []
for s in range(NITEMS):
    d, cur = 0, s
    while cur not in [x for cyc in attractors for x in cyc] and d <= NITEMS:
        cur = succ[cur]
        d += 1
    depths.append(d)
print('  transient depth: mean %.1f | max %d' % (float(np.mean(depths)), int(np.max(depths))), flush=True)

# ---- reachability from each node
def reach(s, limit=60):
    seen, cur = set(), s
    for _ in range(limit):
        if cur in seen:
            break
        seen.add(cur)
        cur = succ[cur]
    return seen

r_chain = reach(CHAIN[0])
print('  reachable from the chain start: %d/%d words' % (len(r_chain), NITEMS), flush=True)
biggest = max(attractors, key=lambda c: basin.get(attractors[c], 0))
trunk_node = biggest[0]
r_trunk = reach(trunk_node)
print('  reachable from the biggest trunk: %d/%d words' % (len(r_trunk), NITEMS), flush=True)

# ---- how many distinct ends from every start
ends = sorted({tuple(sorted(reach(s))) for s in range(NITEMS)})
print('  distinct reachable sets over all starts: %d' % len(ends), flush=True)

# ---- does every start converge to a trunk within NITEMS steps?
conv = sum(1 for s in range(NITEMS) if depths[s] <= NITEMS)
print('  starts that converge to a trunk: %d/%d' % (conv, NITEMS), flush=True)
