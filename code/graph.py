"""Map the transition graph the model has actually built.

The old criterion asked "does it follow MY chain".  At scale the activity may go from
one chain to another -- that is association, not necessarily an error.  So instead of
judging my chain, this measures the graph the model itself forms:

    for every word: cue it, hold, read what comes next

and then reports
    coverage     how many words lead somewhere
    successors   how many distinct successors there are (1 = collapse, N = noise)
    self_loops   words that lead back to themselves
    cycles       the cycles in the graph (chains of association)
    my_chain     whether the trained chain appears as a path in that graph

A coherent web shows up as: high coverage, many 2-cycles/chains, few isolated nodes.
Noise shows up as: coverage random, each read unrelated to the previous one.

Usage: python graph.py <seed> <n_items>
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
exec(compile(src[:cut], 'g', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CODES_T = ns['CODES_T']
held_state = ns['held_state']; W_m = ns['W_m']; DEV = ns['DEV']
T_ = ns['T']; CHAIN = ns['CHAIN']

code_vecs = [c.cpu().numpy() for c in CODES_T]
code_norms = [max(float(np.linalg.norm(v)), 1e-9) for v in code_vecs]

succ = {}
for k in range(NITEMS):
    x23, x5 = held_state(ns['SIN'][k])
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ code_vecs[j]) / code_norms[j] for j in range(NITEMS)]
    succ[k] = int(np.argmax(sims))

print('seed %s | %d words | the graph the model has built:' % (SEED, NITEMS), flush=True)
cov = sum(1 for v in succ.values())
distinct = len(set(succ.values()))
selfl = sum(1 for k, v in succ.items() if v == k)
print('  coverage %d/%d | distinct successors %d | self loops %d' % (
    cov, NITEMS, distinct, selfl), flush=True)

# cycles
seen_global = set()
cycles = []
for start in range(NITEMS):
    if start in seen_global:
        continue
    path, cur, seen = [], start, {}
    while cur not in seen and cur not in seen_global:
        seen[cur] = len(path)
        path.append(cur)
        cur = succ[cur]
    if cur in seen:
        cycles.append(path[seen[cur]:])
    seen_global.update(path)
cycles = [c for c in cycles if len(c) >= 2]
cycles.sort(key=len, reverse=True)
print('  cycles (length>=2): %d | longest %d' % (
    len(cycles), len(cycles[0]) if cycles else 0), flush=True)
for c in cycles[:3]:
    print('    %s' % ' -> '.join(WORDS[i] for i in c + [c[0]]), flush=True)

# does my trained chain appear as a path?
want = [CHAIN[i] for i in range(len(CHAIN) - 1)]
hits = sum(1 for i in range(len(want)) if succ[want[i]] == want[i] + 1)
print('  my trained chain: %d/%d edges present in the graph' % (hits, len(want)), flush=True)

# the walks that follow the graph from the chain's start
cur = CHAIN[0]
walk, seen = [cur], set()
for _ in range(min(12, NITEMS)):
    cur = succ[cur]
    if cur in seen:
        walk.append(cur)
        break
    seen.add(cur)
    walk.append(cur)
print('  graph walk from item 0: %s' % ' -> '.join(WORDS[i] for i in walk), flush=True)
