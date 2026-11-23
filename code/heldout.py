"""Two held-out tests that the earlier reports never did.

Everything reported so far was measured on the data the model was trained on.  These
two tests hold data out:

  A2  HELD-OUT INSTANCE (the important one)
      The concept layer is trained on only 4 of the 5 words of each category.  Then the
      5th word -- never seen during layer-2 learning -- is presented and we ask whether
      it activates its own category's concept cells more than any other category's.
      This is what "the concept exists" has to mean: a new instance must be recognised
      as belonging to it.  A lookup table cannot do this; it has never seen the word.

  A1  HELD-OUT TRANSITION
      The concept chain is the rule c -> c+1 (mod 8).  One transition is withheld from
      training.  Then the model is given the withheld concept and asked for its
      successor.  A lookup cannot answer; a model that represents the rule could.

Both are reported alongside the training-set numbers so the gap is visible.

Usage: python heldout.py <seed>
"""
import os
import sys
import numpy as np
import text_encoder as te

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(SEED)

STEMS = ['cat', 'dog', 'run', 'bri', 'mou', 'flo', 'gra', 'spo']
SUFFIX = ['a', 'e', 'i', 'o', 'u']
WORDS, CAT = [], []
for ci, st in enumerate(STEMS):
    for sf in SUFFIX:
        WORDS.append(st + sf)
        CAT.append(ci)
N, NCAT, PER = len(WORDS), len(STEMS), len(SUFFIX)
HOLD_PER = PER - 1          # train layer 2 on the first 4 of each category

# ---------------- layer 1: sparse word code (unchanged) ----------------
N1, K1, TARGET1 = 1024, 40, 50
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
comp_c = comp - comp.mean(0)
W1 = rng.normal(0, 1, (N1, comp_c.shape[1]))
m1 = np.zeros_like(W1, bool)
for j in range(N1):
    m1[j, rng.choice(comp_c.shape[1], K1, replace=False)] = True
W1 = W1 * m1
W1 = W1 / np.abs(W1).sum(1, keepdims=True)
bud1 = np.abs(W1).sum(1) + 1e-9
X = []
for k in range(N):
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if float((W1 @ (g * comp_c[k]) > 0.5 * bud1).sum()) > TARGET1:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    X.append((W1 @ (g * comp_c[k]) > 0.5 * bud1).astype(np.float32))
X = np.stack(X)

# ---------------- layer 2: trained on 4 of 5 words per category ----------------
TRAIN_IDX = [ci * PER + i for ci in range(NCAT) for i in range(HOLD_PER)]
HELD_IDX = [ci * PER + HOLD_PER for ci in range(NCAT)]
N2, K2 = 256, 100
W2 = rng.normal(0, 1, (N2, N1)) * 0.1
m2 = np.zeros_like(W2, bool)
for j in range(N2):
    m2[j, rng.choice(N1, K2, replace=False)] = True
W2 = W2 * m2
W2 = W2 / np.abs(W2).sum(1, keepdims=True)
ETA, DECAY, WIN = 0.08, 0.002, 3
order = list(TRAIN_IDX)
for ep in range(300):
    rng.shuffle(order)
    for k in order:
        x = X[k]
        s = W2 @ x
        win = np.argsort(s)[-WIN:]
        W2[win] += ETA * (x[None, :] - W2[win] * x.sum() / max(x.sum(), 1))
        W2 -= DECAY * W2
        W2 = np.maximum(W2, 0.0)
        W2 = W2 / np.maximum(np.abs(W2).sum(1, keepdims=True), 1e-9)
print('layer 2 trained on %d words (%d per category), %d words held out' % (
    len(TRAIN_IDX), HOLD_PER, len(HELD_IDX)), flush=True)


def resp(k):
    return W2 @ X[k]


S_all = np.stack([resp(k) for k in range(N)])
TH = np.quantile(np.stack([S_all[k] for k in TRAIN_IDX]), 0.90, axis=0)
A = (S_all > TH).astype(np.float32)

# concept cells measured on the TRAINING words only
concept_cells = np.zeros((NCAT, N2), np.float32)
for cell in range(N2):
    on = [k for k in TRAIN_IDX if A[k, cell] > 0]
    if not on:
        continue
    cats = set(CAT[k] for k in on)
    if len(cats) == 1:
        c = cats.pop()
        members = sum(1 for k in on if CAT[k] == c)
        if members >= 3 and len(on) - members == 0:
            concept_cells[c, cell] = 1.0
sizes = [int(concept_cells[c].sum()) for c in range(NCAT)]
print('concept cells measured on training words: %s' % sizes, flush=True)


def ov(a, b):
    u = float(np.logical_or(a > 0, b > 0).sum())
    return float(np.logical_and(a > 0, b > 0).sum()) / u if u else 0.0


# ---------------- A2: held-out INSTANCE ----------------
print('', flush=True)
print('A2  HELD-OUT INSTANCE: a word the concept layer has never seen', flush=True)
ok2 = 0
for k in HELD_IDX:
    v = A[k]
    sims = [ov(v, concept_cells[c]) for c in range(NCAT)]
    got = int(np.argmax(sims))
    good = got == CAT[k]
    ok2 += int(good)
    print('  %-8s (cat %d) -> best matching concept %d (overlap %.2f) %s' % (
        WORDS[k], CAT[k], got, max(sims), 'OK' if good else '--'), flush=True)
print('  held-out words recognised as their category: %d/%d' % (ok2, len(HELD_IDX)), flush=True)

# control: the same test on TRAINING words (what the earlier reports quoted)
ok2t = 0
for k in TRAIN_IDX:
    sims = [ov(A[k], concept_cells[c]) for c in range(NCAT)]
    ok2t += int(int(np.argmax(sims)) == CAT[k])
print('  (training-set control: %d/%d -- this is the number earlier reports quoted)' % (
    ok2t, len(TRAIN_IDX)), flush=True)

# ---------------- A1: held-out TRANSITION ----------------
# the rule is c -> c+1 mod 8; withhold one such transition from the map's training
CHAIN = list(range(NCAT)) + [0]          # 0->1->...->7->0
TRANS = [(CHAIN[i], CHAIN[i + 1]) for i in range(len(CHAIN) - 1)]
HELD_T = (3, 4)                          # withheld
TRAIN_T = [t for t in TRANS if t != HELD_T]
print('', flush=True)
print('A1  HELD-OUT TRANSITION: rule c -> c+1, withheld %d -> %d' % HELD_T, flush=True)

W_map = np.zeros((NCAT, N2), np.float32)     # concept -> next concept
for ep in range(200):
    for a, b in TRAIN_T:
        W_map[b] += 0.15 * (concept_cells[a] - W_map[b])
        W_map = np.maximum(W_map, 0.0)
        W_map = W_map / np.maximum(np.abs(W_map).sum(1, keepdims=True), 1e-9)


def next_concept(c):
    d = W_map @ concept_cells[c]
    row = W_map.sum(1) + 1e-9
    score = d / row
    return int(np.argmax(score))


ok_tr = sum(1 for a, b in TRAIN_T if next_concept(a) == b)
print('  trained transitions: %d/%d' % (ok_tr, len(TRAIN_T)), flush=True)
got = next_concept(HELD_T[0])
print('  withheld transition %d -> %d : model says %d  %s' % (
    HELD_T[0], HELD_T[1], got, 'GENERALISED' if got == HELD_T[1] else 'no generalisation'), flush=True)

print('', flush=True)
print('SUMMARY  A2 held-out instance %d/%d | A1 held-out transition %s' % (
    ok2, len(HELD_IDX), 'ok' if got == HELD_T[1] else 'failed'), flush=True)
