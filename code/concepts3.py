"""Concepts, take two: learn them on the INPUT projection, and judge strictly.

Last round failed and the criterion was too loose in two ways:
  * it accepted cells that fired for two categories (chance co-activation)
  * the "decoding" used prototypes built from the same responses (circular)

What changes now:
  * the Hebbian learning sits on the input projection W_ff, because that is what
    decides WHICH cells represent a word; the recurrent matrix cannot change a code
  * the rule is still local: when word a is seen, the cells that fire for a also learn
    a bit of the input of every other word in the same category (they co-occur in the
    same episodes)
  * the criterion is strict: fire for >=2 words of category c AND 0 words outside c

Usage: python concepts2.py <seed>
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
exec(compile(src[:cut], 'c2', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; T_ = ns['T']; DEV = ns['DEV']
N23 = ns['N23']; comp_c = ns['comp_c']; GAIN = ns['GAIN']
bud_ff = ns['bud_ff']; TH_A = ns['TH_A']
NCAT = 3
CAT = [k // 4 for k in range(12)]

W = np.array(ns['W_ff'], dtype=np.float64, copy=True)   # the input projection


def codes_of(Wm):
    out = []
    for k in range(NITEMS):
        s = Wm @ (GAIN[k] * comp_c[k])
        out.append((s > TH_A * bud_ff).astype(np.float32))
    return out


def overlap(a, b):
    u = np.logical_or(a > 0, b > 0).sum()
    return float(np.logical_and(a > 0, b > 0).sum() / u) if u else 0.0


def report(codes, tag):
    within = [overlap(codes[i], codes[j]) for i in range(12) for j in range(i + 1, 12) if CAT[i] == CAT[j]]
    across = [overlap(codes[i], codes[j]) for i in range(12) for j in range(i + 1, 12) if CAT[i] != CAT[j]]
    R = np.stack(codes)
    strict = 0
    count_by_cat = [0, 0, 0]
    for cell in range(N23):
        on = [k for k in range(12) if R[k, cell] > 0]
        if len(on) < 2:
            continue
        cats = set(CAT[k] for k in on)
        if len(cats) == 1 and len(on) >= 2:
            strict += 1
            count_by_cat[list(cats)[0]] += 1
    print('%-10s within %.3f | across %.3f | strict concept cells %4d (per category %s)' % (
        tag, float(np.mean(within)), float(np.mean(across)), strict, count_by_cat), flush=True)
    return strict


codes0 = codes_of(W)
print('baseline (before any category experience):', flush=True)
report(codes0, 'before')

# ---- local Hebbian learning on the input projection: the cells that fire for word a
# also learn the input of its category-mates
ETA = 0.15
print('\nlearning: the cells that answer to a word also learn its category-mates', flush=True)
for ep in range(60):
    for c in range(NCAT):
        members = [k for k in range(12) if CAT[k] == c]
        for a in members:
            va = GAIN[a] * comp_c[a]
            win = np.where(codes_of(W)[a] > 0)[0]
            if len(win) == 0:
                continue
            shared = win[:max(1, int(0.30 * len(win)))]   # only a fraction becomes category-shared
            for b in members:
                if b == a:
                    continue
                vb = GAIN[b] * comp_c[b]
                W[shared] = W[shared] + ETA * (vb[None, :] - W[shared])
    W = W / np.abs(W).sum(1, keepdims=True)
    if (ep + 1) % 20 == 0:
        report(codes_of(W), 'after %d it' % (ep + 1))

codes1 = codes_of(W)
strict = report(codes1, 'final')

print('\njudgement: %s' % (
    'concept cells appeared' if strict >= 30 else
    'NOT yet: fewer than 30 cells answer to a whole category'), flush=True)


# ---- is the WORD still recoverable after concept formation?
codesF = codes_of(W)
Rf = np.stack(codesF)
word_ok = 0
for k in range(12):
    best, bi = -1.0, -1
    for j in range(12):
        if j == k:
            continue
        o = overlap(codesF[k], codesF[j])
        if o > best:
            best, bi = o, j
    word_ok += 1 if best < 0.5 else 0
print('\nword identity: %d/12 words are still clearly their own pattern (max overlap with any other %.2f)' % (
    word_ok, max(overlap(codesF[i], codesF[j]) for i in range(12) for j in range(i+1, 12))), flush=True)
print('  concept cells %d | within %.3f | across %.3f' % (
    strict, float(np.mean([overlap(codesF[i], codesF[j]) for i in range(12) for j in range(i+1,12) if CAT[i]==CAT[j]])),
    float(np.mean([overlap(codesF[i], codesF[j]) for i in range(12) for j in range(i+1,12) if CAT[i]!=CAT[j]]))), flush=True)
