"""Concepts, third attempt -- with structured words and a strict criterion.

Two flaws in the earlier test, both pointed out by the designer:

  1. the criterion was far too loose.  With 70-cell codes in 1024 cells, a random cell
     fires for a given word with probability ~0.068, and "fires for >=2 of a 4-word
     category and none of the other 8" happens by chance about 1.6 % of the time --
     ~16 cells per category, ~48 in total.  The reported 123 was mostly chance, and
     more concept cells than words is absurd in the first place.
  2. the words were random letter strings, so their codes were almost orthogonal and
     there was NO shared structure for a concept to be extracted from.  Concepts were
     being forced in by training rather than found in the input.

This version:
  * uses words that really share structure: 8 categories, 5 words each, all members of
    a category contain the same stem (like cat / cap / cab sharing "ca")
  * first checks -- BEFORE any learning -- whether within-category codes already overlap
    more than across-category codes
  * then applies the strict criterion: a concept cell must fire for >=3 of the 5 members
    of one category and for NONE of the other 35 words
  * and checks the count: concept cells must be FEWER than the words

Usage: python concepts4.py <seed>
"""
import os
import sys
import numpy as np
import text_encoder as te

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(SEED)

# 8 stems x 5 suffixes: members of a category share the stem, so their character
# patterns overlap the way real related words do
STEMS = ['cat', 'dog', 'run', 'bri', 'mou', 'flo', 'gra', 'spo']
SUFFIX = ['a', 'e', 'i', 'o', 'u']
WORDS, CAT = [], []
for ci, st in enumerate(STEMS):
    for sf in SUFFIX:
        WORDS.append(st + sf)
        CAT.append(ci)
N = len(WORDS)
NCAT = len(STEMS)
print('%d words | %d categories (%d words each) | members share a stem' % (
    N, NCAT, len(SUFFIX)), flush=True)

N23, K = 1024, 40
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
comp_c = comp - comp.mean(0)

W = rng.normal(0, 1, (N23, comp_c.shape[1]))
mask = np.zeros_like(W, bool)
for j in range(N23):
    mask[j, rng.choice(comp_c.shape[1], K, replace=False)] = True
W = W * mask
W = W / np.abs(W).sum(1, keepdims=True)
bud = np.abs(W).sum(1) + 1e-9

TARGET = 50
codes = []
for k in range(N):
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        n = float((W @ (g * comp_c[k]) > 0.5 * bud).sum())
        if n > TARGET:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    codes.append((W @ (g * comp_c[k]) > 0.5 * bud).astype(np.float32))
M = np.stack(codes)
print('code size %.0f cells | density %.3f' % (float(M.sum(1).mean()), float(M.mean())), flush=True)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


within = [ov(M[i], M[j]) for i in range(N) for j in range(i + 1, N) if CAT[i] == CAT[j]]
across = [ov(M[i], M[j]) for i in range(N) for j in range(i + 1, N) if CAT[i] != CAT[j]]
print('\nBEFORE any learning (structured words):', flush=True)
print('  within-category overlap %.3f | across %.3f | ratio %.2f' % (
    float(np.mean(within)), float(np.mean(across)),
    float(np.mean(within)) / max(float(np.mean(across)), 1e-9)), flush=True)

# ---- strict concept criterion, and the chance level for it
def count_concepts(M):
    hits, per_cat = 0, [0] * NCAT
    for cell in range(N23):
        on = [k for k in range(N) if M[k, cell] > 0]
        if not on:
            continue
        cats = set(CAT[k] for k in on)
        if len(cats) == 1:
            c = cats.pop()
            members = sum(1 for k in on if CAT[k] == c)
            outside = len(on) - members
            if members >= 3 and outside == 0:
                hits += 1
                per_cat[c] += 1
    return hits, per_cat


hits, per_cat = count_concepts(M)
print('\nSTRICT criterion (fires for >=3 of 5 members of ONE category, 0 outside):', flush=True)
print('  concept cells: %d (per category %s) | words: %d' % (hits, per_cat, N), flush=True)
print('  fewer concepts than words? %s' % ('YES' if hits < N else 'NO'), flush=True)

# chance level by shuffling the category labels
chance = []
for _ in range(20):
    perm = rng.permutation(N)
    Ms = M[perm]
    h, _ = count_concepts(Ms)
    chance.append(h)
print('  shuffled-label chance level: %.1f (mean of 20 shuffles)' % float(np.mean(chance)), flush=True)

# ---- decode: can the population tell the category?
protos = []
for c in range(NCAT):
    v = np.mean([M[k] for k in range(N) if CAT[k] == c], axis=0)
    protos.append(v / max(float(np.linalg.norm(v)), 1e-9))
ok = 0
for k in range(N):
    v = M[k] / max(float(np.linalg.norm(M[k])), 1e-9)
    ok += int(int(np.argmax([float(v @ p) for p in protos])) == CAT[k])
print('\ncategory decoding from the population: %d/%d' % (ok, N), flush=True)
