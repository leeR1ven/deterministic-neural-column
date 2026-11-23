"""Let the concept neurons form by themselves.

The mistake in the last version: I *defined* concept cells ("cells that answer to >=3 of
5 members") and then used that hand-picked mask as the representation.  That is me
stipulating the concept neurons, not the network growing them.

The way they are supposed to form, per the designer: many sensory inputs converge on a
higher-layer cell; when the same combination keeps appearing, that cell becomes tuned to
it -- a concept neuron.  So here:

    layer 1 : the sparse word code (as before, 1024 cells)
    layer 2 : 256 cells, each receiving from 100 layer-1 cells, and LEARNED by a local
              rule only -- repeated co-activation strengthens the inputs that co-occur,
              and a local winner-take-all (a soft competition through a threshold)
              keeps cells from all doing the same thing
    PFC     : holds the layer-2 activity (not a mask I computed)

After learning, I MEASURE how many layer-2 cells became category-selective.  The
measurement is reported, but the representation never uses it.

Usage: python concept_emerge.py <seed>
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
N, NCAT = len(WORDS), len(STEMS)

# ---------- layer 1: the sparse word code (unchanged) ----------
N1, K1, TARGET1 = 1024, 40, 50
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
comp_c = comp - comp.mean(0)
W1 = rng.normal(0, 1, (N1, comp_c.shape[1]))
mask = np.zeros_like(W1, bool)
for j in range(N1):
    mask[j, rng.choice(comp_c.shape[1], K1, replace=False)] = True
W1 = W1 * mask
W1 = W1 / np.abs(W1).sum(1, keepdims=True)
bud1 = np.abs(W1).sum(1) + 1e-9
CODES = []
for k in range(N):
    lo, hi = 0.1, 8.0
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if float((W1 @ (g * comp_c[k]) > 0.5 * bud1).sum()) > TARGET1:
            hi = g
        else:
            lo = g
    g = 0.5 * (lo + hi)
    CODES.append((W1 @ (g * comp_c[k]) > 0.5 * bud1).astype(np.float32))
X = np.stack(CODES)
print('layer 1: %d cells, code size %.0f, density %.3f' % (
    N1, float(X.sum(1).mean()), float(X.mean())), flush=True)

# ---------- layer 2: learned by a local rule ----------
N2, K2 = 256, 100
W2 = rng.normal(0, 1, (N2, N1)) * 0.1
mask2 = np.zeros_like(W2, bool)
for j in range(N2):
    mask2[j, rng.choice(N1, K2, replace=False)] = True
W2 = W2 * mask2
W2 = W2 / np.abs(W2).sum(1, keepdims=True)

# local learning: on each presentation, the strongest responder is the winner; its
# inputs are moved toward the current pattern, and every cell's inputs are nudged down
# (decay).  This is Hebb plus a local competition -- no labels, no global error.
ETA, DECAY, WINNERS = 0.08, 0.002, 3
order = list(range(N))
for ep in range(300):
    rng.shuffle(order)
    for k in order:
        x = X[k]
        s = W2 @ x
        win = np.argsort(s)[-WINNERS:]
        W2[win] += ETA * (x[None, :] - W2[win] * x.sum() / max(x.sum(), 1))
        W2 -= DECAY * W2
        W2 = np.maximum(W2, 0.0)
        W2 = W2 / np.maximum(np.abs(W2).sum(1, keepdims=True), 1e-9)
print('layer 2: %d cells, trained for 300 epochs with a local rule' % N2, flush=True)


def resp(k):
    """layer-2 response to word k: threshold each cell at its own mean rate"""
    s = W2 @ X[k]
    return s


S = np.stack([resp(k) for k in range(N)])
TH = np.quantile(S, 0.90, axis=0)          # per-cell threshold from its own distribution
A = (S > TH).astype(np.float32)
print('layer 2 response: %.0f cells per word on average' % float(A.sum(1).mean()), flush=True)

# ---------- MEASURE: did category-selective cells emerge? ----------
def selectivity(A):
    hits, per_cat = 0, [0] * NCAT
    for cell in range(N2):
        on = [k for k in range(N) if A[k, cell] > 0]
        if len(on) < 2:
            continue
        cats = set(CAT[k] for k in on)
        if len(cats) == 1:
            c = cats.pop()
            members = sum(1 for k in on if CAT[k] == c)
            if members >= 3 and len(on) - members == 0:
                hits += 1
                per_cat[c] += 1
    return hits, per_cat


hits, per_cat = selectivity(A)
print('\nMEASURED (not used as the representation):', flush=True)
print('  category-selective layer-2 cells: %d of %d (per category %s)' % (hits, N2, per_cat), flush=True)
chance = []
for _ in range(20):
    h, _ = selectivity(A[rng.permutation(N)])
    chance.append(h)
print('  shuffled-label chance: %.1f' % float(np.mean(chance)), flush=True)

# category information in the layer-2 population
protos = []
for c in range(NCAT):
    v = np.mean([A[k] for k in range(N) if CAT[k] == c], axis=0)
    protos.append(v / max(float(np.linalg.norm(v)), 1e-9))
ok = 0
for k in range(N):
    v = A[k] / max(float(np.linalg.norm(A[k])), 1e-9)
    ok += int(int(np.argmax([float(v @ p) for p in protos])) == CAT[k])
print('  category decoding from layer 2: %d/%d' % (ok, N), flush=True)
word_ok = sum(1 for k in range(N)
              if int(np.argmax([float(A[k] / max(np.linalg.norm(A[k]), 1e-9)
                                     @ (A[j] / max(np.linalg.norm(A[j]), 1e-9)))
                                for j in range(N)])) == k)
print('  word decoding from layer 2: %d/%d' % (word_ok, N), flush=True)
