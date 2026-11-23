"""Does it generalise to an item it has never been trained on?

The question the designer asked: what counts as generalisation?  This test is built so
that the answer is interpretable, with controls that must fail if the test is to mean
anything.

Setup
  3 categories x 5 words.  Members of a category share a stem, so their sparse codes
  overlap -- this is the ONLY thing that could carry information about the held-out word.
  Rule: after any word of category c, the next word is a fixed word X_c.
  Training uses members 1..4 of each category.  Member 5 is NEVER trained.

  TEST        present the held-out member -> does it produce X_c?
  CONTROL A   same word with its letters shuffled (stem destroyed) -> should fail, which
              shows the pass comes from the shared structure and not from something else
  CONTROL B   a lookup table trained on the same data: it has no entry for the held-out
              word, so it cannot answer at all -- the model must beat it to be interesting

What a pass would mean: the word itself was never seen, so this is generalisation by
shared structure.  What it would NOT mean: generalisation by a rule that ignores
surface form.

Usage: python generalize.py [seed]
"""
import os
import sys

import numpy as np
import text_encoder as te

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(SEED)

STEMS = ['cat', 'dog', 'run']
SUFFIX = ['a', 'e', 'i', 'o', 'u']
WORDS, CAT = [], []
for ci, st in enumerate(STEMS):
    for sf in SUFFIX:
        WORDS.append(st + sf)
        CAT.append(ci)
NCAT = len(STEMS)
TRAIN_IDX = [ci * 5 + i for ci in range(NCAT) for i in range(4)]     # 4 of 5 per category
HELD_IDX = [ci * 5 + 4 for ci in range(NCAT)]                        # the 5th, never trained

# the word each category leads to
NEXT = ['mzzk', 'qvvp', 'xxjt']
ALL_WORDS = WORDS + NEXT

# ---------------- sparse code ----------------
N1, K1, TARGET = 1024, 40, 50
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in ALL_WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
comp_c = comp - comp.mean(0)
W1 = rng.normal(0, 1, (N1, comp_c.shape[1]))
mask = np.zeros_like(W1, bool)
for j in range(N1):
    mask[j, rng.choice(comp_c.shape[1], K1, replace=False)] = True
W1 = W1 * mask
W1 = W1 / np.abs(W1).sum(1, keepdims=True)
bud = np.abs(W1).sum(1) + 1e-9
CODES = []
for k in range(len(ALL_WORDS)):
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if float((W1 @ (g * comp_c[k]) > 0.5 * bud).sum()) > TARGET:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    CODES.append((W1 @ (g * comp_c[k]) > 0.5 * bud).astype(np.float32))
CODES = np.stack(CODES)
print('%d words (%d trained, %d held out) | %d categories | code size %.0f' % (
    len(WORDS), len(TRAIN_IDX), len(HELD_IDX), NCAT, float(CODES[:len(WORDS)].sum(1).mean())), flush=True)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


# how similar is a held-out word to its own category's training words, vs to others?
own, other = [], []
for k in HELD_IDX:
    for j in TRAIN_IDX:
        (own if CAT[j] == CAT[k] else other).append(ov(CODES[k], CODES[j]))
print('held-out word vs its own category: %.2f | vs other categories: %.2f' % (
    float(np.mean(own)), float(np.mean(other))), flush=True)

# ---------------- learn: state -> the next word's code ----------------
W_map = np.zeros((len(ALL_WORDS), N1), np.float32)
for ep in range(200):
    for k in TRAIN_IDX:
        tgt = len(WORDS) + CAT[k]          # the fixed next word for that category
        W_map[tgt] += 0.15 * (CODES[k] - W_map[tgt])
        W_map = np.maximum(W_map, 0.0)
        W_map = W_map / np.maximum(np.abs(W_map).sum(1, keepdims=True), 1e-9)


def predict(code):
    d = W_map @ code
    return int(np.argmax(d))


def next_word_of(pred_idx):
    """map a row index back to which next-word it represents"""
    return pred_idx - len(WORDS) if pred_idx >= len(WORDS) else None


print('', flush=True)
print('TRAINED words (sanity): present each -> predicted next', flush=True)
ok_tr = 0
for k in TRAIN_IDX:
    p = next_word_of(predict(CODES[k]))
    good = (p == CAT[k])
    ok_tr += int(good)
print('  %d/%d correct' % (ok_tr, len(TRAIN_IDX)), flush=True)

print('', flush=True)
print('TEST: the 5th member of each category, never seen in training', flush=True)
ok_held = 0
for k in HELD_IDX:
    p = next_word_of(predict(CODES[k]))
    good = (p == CAT[k])
    ok_held += int(good)
    print('  %-6s (cat %d) -> next word %s | want %s  %s' % (
        WORDS[k], CAT[k], NEXT[p] if p is not None else 'none', NEXT[CAT[k]],
        'OK' if good else '--'), flush=True)
print('  held-out generalisation: %d/%d' % (ok_held, len(HELD_IDX)), flush=True)

print('', flush=True)
print('CONTROL A: the same held-out words with their letters shuffled', flush=True)
ok_scram = 0
for k in HELD_IDX:
    w = WORDS[k]
    letters = list(w)
    rng.shuffle(letters)
    sw = ''.join(letters)
    sc = (np.clip(te.pattern(sw), 0, 1) > 0.5).astype(float)
    sv = np.concatenate([2 * sc - 1, 1 - 2 * sc]) - comp.mean(0)
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if float((W1 @ (g * sv) > 0.5 * bud).sum()) > TARGET:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    code = (W1 @ (g * sv) > 0.5 * bud).astype(np.float32)
    p = next_word_of(predict(code))
    good = (p == CAT[k])
    ok_scram += int(good)
    print('  %-6s -> %s -> next word %s | want %s  %s' % (
        w, sw, NEXT[p] if p is not None else 'none', NEXT[CAT[k]], 'OK' if good else '--'), flush=True)
print('  scrambled control: %d/%d (a pass here would mean the test is not measuring structure)' % (
    ok_scram, len(HELD_IDX)), flush=True)

print('', flush=True)
print('CONTROL A2 (stronger): a word with NO letter in common with the category', flush=True)
UNRELATED = ['zzq', 'vww', 'jjp']
ok_un = 0
for n, k in enumerate(HELD_IDX):
    uw = UNRELATED[n]
    sc = (np.clip(te.pattern(uw), 0, 1) > 0.5).astype(float)
    sv = np.concatenate([2 * sc - 1, 1 - 2 * sc]) - comp.mean(0)
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if float((W1 @ (g * sv) > 0.5 * bud).sum()) > TARGET:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    code = (W1 @ (g * sv) > 0.5 * bud).astype(np.float32)
    p = next_word_of(predict(code))
    good = (p == CAT[k])
    ok_un += int(good)
    print('  %-6s (no shared letters) -> next word %s | want %s  %s' % (
        uw, NEXT[p] if p is not None else 'none', NEXT[CAT[k]], 'OK' if good else '--'), flush=True)
print('  unrelated control: %d/%d' % (ok_un, len(HELD_IDX)), flush=True)

print('', flush=True)
print('CONTROL B: a lookup table trained on exactly the same words', flush=True)
table = {k: len(WORDS) + CAT[k] for k in TRAIN_IDX}
missing = sum(1 for k in HELD_IDX if k not in table)
print('  entries in the table: %d | held-out words with no entry: %d/%d' % (
    len(table), missing, len(HELD_IDX)), flush=True)
print('  -> a lookup CANNOT answer for any held-out word; anything above 0 is structure', flush=True)
