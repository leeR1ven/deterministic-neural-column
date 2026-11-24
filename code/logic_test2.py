"""Logic test, corrected: neurons at one moment activate neurons at the next, no lookup table of averages.

What the first version got wrong: it stored, for every word, the AVERAGE context state that preceded it, and
then compared the current state against those averages. That is a lookup table, not the mechanism the model
is supposed to have.

This version does what the designer described. Seeing symbol A at one moment and symbol B at the next
strengthens the synapses from A's active cells to B's active cells:

    W += eta * outer(pattern_next, pattern_now)          (neuron to neuron)

and generation simply lets the network run: the active cells drive W, the result is the next pattern, and the
closest symbol to that pattern is emitted and fed back in. There is no table of "what usually follows what";
the only memory is the synaptic weight between cell patterns.

The sentence shape is fixed and only the content pair changes:

    if  X  then  Y  .  X  therefore  Y

Training sees specific (X, Y) pairs; the test uses a pair that never occurs together, so reciting cannot
answer it.

Usage: python logic_test2.py [epochs] [decay]
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
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DECAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
ETA = 0.05

NCONTENT = int(os.environ.get('NCONT', '6'))
FUNC = ['if', 'then', 'therefore', '.']
CONTENT = ['s%d' % i for i in range(NCONTENT)]
VOCAB = FUNC + CONTENT
_r = np.random.default_rng(7)
_all = [(i, j) for i in range(NCONTENT) for j in range(NCONTENT) if i != j]
_order = _r.permutation(len(_all))
_ntr = int(os.environ.get('NTR', '3'))
TRAIN_PAIRS = [_all[k] for k in _order[:NCONTENT * _ntr]]
TEST_PAIRS = [_all[k] for k in _order if _all[k] not in set(TRAIN_PAIRS)][:3]


def glyph(seed):
    r = np.random.default_rng(1000 + seed)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(4, S - 8, 2)
        h, w = r.integers(3, 10, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


GLYPHS = np.stack([glyph(i) for i in range(len(VOCAB))]).astype(np.float32)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [0.6]
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GLYPHS, device=DEV))[-1]
D = PAT.shape[1]


def toks(x, y, answer=True):
    t = [VOCAB.index('if'), VOCAB.index('s%d' % x), VOCAB.index('then'), VOCAB.index('s%d' % y),
         VOCAB.index('.')]
    if answer:
        t += [VOCAB.index('s%d' % x), VOCAB.index('therefore'), VOCAB.index('s%d' % y)]
    return t


# ---------------- learn neuron-to-neuron transitions ----------------
W = T.zeros(D, D, device=DEV)
g = T.Generator(device=DEV).manual_seed(3)
for ep in range(EPOCHS):
    for (x, y) in TRAIN_PAIRS:
        seq = toks(x, y)
        for a, b in zip(seq[:-1], seq[1:]):
            W += ETA * T.outer(PAT[b], PAT[a])          # cells active now -> cells active next
W.clamp_(min=0.0)
print('neurons %d | transitions learned from %d sentences | test pairs %s' % (
    D, len(TRAIN_PAIRS), TEST_PAIRS), flush=True)


def decode(act):
    an = act / act.norm().clamp(min=1e-9)
    Pn = PAT / PAT.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return int((Pn @ an).argmax())


def run(prompt, steps=1):
    """let the network run: each step's activity drives the next, no table of averages"""
    state = T.zeros(D, device=DEV)
    for t in prompt:
        state = DECAY * state + PAT[t]
    out = []
    for _ in range(steps):
        act = W @ state          # cells active now drive the cells active next
        tok = decode(act)
        out.append(tok)
        state = DECAY * state + PAT[tok]
    return out


print('', flush=True)
print('trained sentences:', flush=True)
ok_tr = 0
for (x, y) in TRAIN_PAIRS[:4]:
    prompt = toks(x, y, answer=False) + [VOCAB.index('s%d' % x), VOCAB.index('therefore')]
    got = run(prompt)
    good = got[0] == VOCAB.index('s%d' % y)
    ok_tr += int(good)
    print('  if %s then %s . %s therefore %-4s -> %-4s %s' % (
        CONTENT[x], CONTENT[y], CONTENT[x], CONTENT[y], VOCAB[got[0]], 'OK' if good else '--'), flush=True)

print('', flush=True)
print('TEST -- pairs never seen together:', flush=True)
ok = 0
for (x, y) in TEST_PAIRS:
    prompt = toks(x, y, answer=False) + [VOCAB.index('s%d' % x), VOCAB.index('therefore')]
    got = run(prompt)
    good = got[0] == VOCAB.index('s%d' % y)
    ok += int(good)
    print('  if %s then %s . %s therefore %-4s -> %-4s %s' % (
        CONTENT[x], CONTENT[y], CONTENT[x], CONTENT[y], VOCAB[got[0]], 'OK' if good else '--'), flush=True)
print('', flush=True)
print('  trained %d/%d | unseen %d/%d' % (ok_tr, min(4, len(TRAIN_PAIRS)), ok, len(TEST_PAIRS)), flush=True)
