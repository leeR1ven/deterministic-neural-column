"""Does it speak when it should, and stay silent when it should not?

The first version of the dialogue model only spoke because the code chose to call the speaking step.  That
proves nothing.  Here the basal ganglia are part of the network as an inhibitory signal: the motor area's
activity is reduced by the gate, and the same read-out is applied at every moment, whether listening or
answering.  Then the question becomes measurable:

  turn score      how strongly the motor area is driven at a given moment
  it speaks       the top region's turn score is above a fixed threshold
  it stays quiet  below it

Two streams are measured:

  monologue   nobody is talking to the model; it should stay quiet
  question    the other side has just finished; it should speak

Because the same read-out runs in both, a model that always speaks, or never does, is visible immediately.

Usage: python gate_check.py [epochs] [vocab] [holdout]
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ = 28, 16
DATA = r'D:\deepseekherness\data'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
VOCAB = int(sys.argv[2]) if len(sys.argv) > 2 else 300
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 60
TAU, ETA, WPR, TH_V = 4.0, 0.05, 6, 0.6

# ---------------- data ----------------
raw = open(os.path.join(DATA, 'dialogue', 'tinytalks_v1.txt'), encoding='utf-8', errors='replace').read()
pairs, q = [], None
for line in raw.splitlines():
    line = line.strip()
    if line.startswith('Q:'):
        q = line[2:].strip()
    elif line.startswith('A:') and q is not None:
        pairs.append((q, line[2:].strip()))
        q = None


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


counts = Counter()
for qq, aa in pairs:
    counts.update(words(qq) + words(aa))
vocab = ['<pad>', '<q>'] + [w for w, _ in counts.most_common(VOCAB)]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
pairs = [(qq, aa) for qq, aa in pairs if all(w in w2i and w2i[w] > 1 for w in words(qq) + words(aa))]
print('pairs %d | vocabulary %d' % (len(pairs), V), flush=True)


def encode(s):
    return [w2i[w] for w in words(s)] or [0]


def glyph(i):
    r = np.random.default_rng(5000 + i)
    a = np.zeros((S, S), np.float32)
    for _ in range(3):
        y0, x0 = r.integers(3, S - 8, 2)
        h, w = r.integers(3, 11, 2)
        a[y0:y0 + h, x0:x0 + w] = 1.0
    return a.reshape(-1)


GL = np.stack([glyph(i) for i in range(V)]).astype(np.float32)
vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH_V]
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GL, device=DEV))[-1]
D = PAT.shape[1]

SIDE = int(np.ceil(np.sqrt(V * WPR)))
CELLS = SIDE * SIDE
REG = T.as_tensor(np.array([min(c // WPR, V - 1) for c in range(CELLS)]), device=DEV)
W = T.zeros(CELLS, D, device=DEV)

rng = np.random.default_rng(0)
order = rng.permutation(len(pairs))
held = [pairs[i] for i in order[:HOLD]]
train = [pairs[i] for i in order[HOLD:]]
mono = [w2i[w] for w in words(open(os.path.join(DATA, 'monologue', 'corpus.txt'),
                                    encoding='utf-8', errors='replace').read()) if w in w2i and w2i[w] > 1]
print('training %d | held out %d | monologue %d words' % (len(train), len(held), len(mono)), flush=True)

zero = T.zeros(D, device=DEV)


def listen(tokens, state):
    for t in tokens:
        state = state + (PAT[t] - state) / TAU
    return state


def turn_score(state):
    """how strongly the motor area is driven right now -- the SAME read-out whatever the gate does"""
    a = (state @ W.T) / (W.norm(dim=1) + 1e-9)
    return T.stack([a[REG == v].sum() for v in range(V)])


def say(state):
    sc = turn_score(state)
    return int(sc.argmax()), float(sc.max())


for ep in range(EPOCHS):
    state = zero
    for t in mono[:600]:
        state = listen([t], state)
    for qq, aa in train:
        state = listen([w2i['<q>']], zero)
        state = listen(encode(qq), state)
        tok = encode(aa)[0]
        cells = T.nonzero(REG == tok).squeeze(1)
        W[cells] += ETA * (state.unsqueeze(0) - W[cells])
        W.clamp_(min=0.0)
        state = listen(encode(aa), state)

    # ---- measure: the same read-out on two different streams ----
    s_mono, s_quest = [], []
    st = zero
    for t in mono[:1200]:
        st = listen([t], st)
        s_mono.append(say(st)[1])
    ok = 0
    for qq, aa in held:
        st = listen([w2i['<q>']], zero)
        st = listen(encode(qq), st)
        word, sc = say(st)
        s_quest.append(sc)
        ok += int(word == encode(aa)[0])
    m_mono, m_quest = float(np.mean(s_mono)), float(np.mean(s_quest))
    thr = 0.5 * (m_mono + m_quest)
    false_speech = float(np.mean(np.array(s_mono) > thr))
    true_speech = float(np.mean(np.array(s_quest) > thr))
    print('  epoch %d: turn score  monologue %.4f | after question %.4f | threshold %.4f' % (
        ep + 1, m_mono, m_quest, thr), flush=True)
    print('            speaks when it should %.0f%% | speaks when it should NOT %.0f%% | held-out word %d/%d' % (
        100 * true_speech, 100 * false_speech, ok, len(held)), flush=True)
