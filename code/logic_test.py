"""Can it learn the SHAPE of a logical sentence and apply it to content it has never seen?

The designer's idea: logic is the order-regularity of language, and this model predicts the next thing
from the context carried by the neurons that are active now.  So teach it sentences of a fixed shape and
then give it a question with NEW content.

Language (a tiny one, not natural language):

    if  X  then  Y  .  X  therefore  Y

Training: three content pairs, e.g. (s0,s1), (s1,s2), (s2,s3).
Test:     a FOURTH pair (s3,s4) that never appears in training, in the same sentence shape.
          A lookup table cannot answer it, because the pair was never seen.

Mechanism, all local:

  token pattern     each symbol is a small 28x28 glyph, passed through the same local 3D layer
  context state     h_t = h_{t-1} + (pattern_t - h_{t-1}) / TAU      (the neurons active now, decayed)
  prediction        W[token] holds the states that preceded that token; a token is predicted by
                    whichever W[token] best matches the current state -- learned by Hebb, no labels of
                    "next token" beyond what the stream itself provides
  generation        emit the best-matching token, feed it back, repeat

Usage: python logic_test.py [epochs] [tau]
"""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ = 28, 16
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
ETA = 0.02

NCONTENT = int(os.environ.get("NCONT", "6"))
FUNC = ['if', 'then', 'therefore', '.']          # the function words
CONTENT = ['s%d' % i for i in range(NCONTENT)]
VOCAB = FUNC + CONTENT
# many training pairs over many symbols, so that each answer token is preceded by many different
# contents; the test pairs are combinations that never occur in training
_r = np.random.default_rng(7)
_all = [(i, j) for i in range(NCONTENT) for j in range(NCONTENT) if i != j]
_order = _r.permutation(len(_all))
_ntr = int(os.environ.get('NTR', '3'))
TRAIN_PAIRS = [_all[k] for k in _order[:NCONTENT * _ntr]]
_used = set(TRAIN_PAIRS)
TEST_PAIRS = [_all[k] for k in _order if _all[k] not in _used][:3]


def glyph(seed):
    """a distinct little picture for each symbol"""
    r = np.random.default_rng(1000 + seed)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(4, S - 8, 2)
        h, w = r.integers(3, 10, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


GLYPHS = np.stack([glyph(i) for i in range(len(VOCAB))]).astype(np.float32)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [0.6]                                 # fixed, never calibrated
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GLYPHS, device=DEV))[-1]
print('vocabulary: %d symbols | each a %d-cell pattern, %.1f%% active' % (
    len(VOCAB), PAT.shape[1], 100 * float(PAT.mean())), flush=True)
print('training pairs %s | test pairs %s (never seen together)' % (TRAIN_PAIRS, TEST_PAIRS), flush=True)


def sentence(x, y, with_answer=True):
    """token indices of one sentence"""
    toks = [VOCAB.index('if'), VOCAB.index('s%d' % x), VOCAB.index('then'), VOCAB.index('s%d' % y),
            VOCAB.index('.')]
    if with_answer:
        toks += [VOCAB.index('s%d' % x), VOCAB.index('therefore'), VOCAB.index('s%d' % y)]
    return toks


# ---------------- teaching: read the stream and learn which state precedes which token ----------------
W = T.zeros(len(VOCAB), PAT.shape[1], device=DEV)
g = T.Generator(device=DEV).manual_seed(3)
for ep in range(EPOCHS):
    for (x, y) in TRAIN_PAIRS:
        toks = sentence(x, y)
        h = T.zeros(PAT.shape[1], device=DEV)
        for t in toks:
            h = h + (PAT[t] - h) / TAU            # context carried by the active neurons
            if t == toks[0]:
                continue
            prev = h - (PAT[t] - h) / TAU         # the state that preceded this token
            W[t] += ETA * (prev - W[t])
print('learned: %d transition states' % len(VOCAB), flush=True)


def generate(prompt_tokens, steps=1):
    h = T.zeros(PAT.shape[1], device=DEV)
    for t in prompt_tokens:
        h = h + (PAT[t] - h) / TAU
    out = []
    for _ in range(steps):
        Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)
        hn = h / h.norm().clamp(min=1e-9)
        tok = int((Wn @ hn).argmax())
        out.append(tok)
        h = h + (PAT[tok] - h) / TAU
    return out


print('', flush=True)
print('training pairs:', flush=True)
for (x, y) in TRAIN_PAIRS:
    prompt = sentence(x, y, with_answer=False) + [VOCAB.index('s%d' % x), VOCAB.index('therefore')]
    got = generate(prompt)
    print('  if %s then %s . %s therefore %-4s -> model says %-4s %s' % (
        CONTENT[x], CONTENT[y], CONTENT[x], CONTENT[y], VOCAB[got[0]],
        'OK' if got[0] == VOCAB.index('s%d' % y) else '--'), flush=True)

print('', flush=True)
print('TEST (content pair never seen together):', flush=True)
ok = 0
for (x, y) in TEST_PAIRS:
    prompt = sentence(x, y, with_answer=False) + [VOCAB.index('s%d' % x), VOCAB.index('therefore')]
    got = generate(prompt)
    good = got[0] == VOCAB.index('s%d' % y)
    ok += int(good)
    print('  if %s then %s . %s therefore %-4s -> model says %-4s %s' % (
        CONTENT[x], CONTENT[y], CONTENT[x], CONTENT[y], VOCAB[got[0]], 'OK' if good else '--'), flush=True)
print('', flush=True)
print('  test pairs correct: %d of %d   (a lookup table would score 0)' % (ok, len(TEST_PAIRS)), flush=True)
