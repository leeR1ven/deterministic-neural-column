"""Listening and answering: the basal ganglia decide which one is happening.

The design the designer described:

  data           two corpora -- a monologue (nobody is talking to the model) and a dialogue (question,
                 then answer)
  conversion     every word becomes a small picture, and the picture goes through the same local 3D
                 layer, so the model receives patterns rather than symbols
  basal ganglia  a gate.  While it is active the motor area is suppressed: the model LISTENS and cannot
                 speak.  When the other side has finished, the gate is switched off and the motor area is
                 free to fire.
  motor area     a sheet divided into regions, one region per word.  When a region is active, that word is
                 uttered -- the position in the sheet IS the word.
  learning       only when the gate is off does the prefrontal state reach the motor area and strengthen
                 the synapses to the word being said.  So the model only learns to answer, never to
                 interrupt.

Test: hold out some question/answer pairs.  Feed the question with the gate on, then open the gate and see
which word the motor area utters.

Usage: python dialogue_gate.py [epochs] [vocab] [holdout]
"""
import os
import re
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ = 28, 16
DATA = r'D:\deepseekherness\data'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
VOCAB = int(sys.argv[2]) if len(sys.argv) > 2 else 300
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 40
TAU = 4.0
ETA = 0.05
WORDS_PER_REGION = 6
TH_V = 0.6

# ---------------- data ----------------
dia = os.path.join(DATA, 'dialogue', 'tinytalks_v1.txt')
mono = os.path.join(DATA, 'monologue', 'corpus.txt')
raw = open(dia, encoding='utf-8', errors='replace').read()
pairs = []
q = None
for line in raw.splitlines():
    line = line.strip()
    if line.startswith('Q:'):
        q = line[2:].strip()
    elif line.startswith('A:') and q is not None:
        pairs.append((q, line[2:].strip()))
        q = None
print('dialogue pairs: %d' % len(pairs), flush=True)

def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())

counts = {}
for qq, aa in pairs:
    for w in words(qq) + words(aa):
        counts[w] = counts.get(w, 0) + 1
vocab = ['<pad>', '<q>'] + [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:VOCAB]]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print('vocabulary: %d words' % V, flush=True)


def encode(s):
    return [w2i.get(w, 0) for w in words(s)] or [0]


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
vis.theta = [TH_V]                                    # fixed, never calibrated
with T.no_grad():
    PAT = vis.forward(T.as_tensor(GL, device=DEV))[-1]
D = PAT.shape[1]

# ---------------- motor area: a sheet, one region per word ----------------
SIDE = int(np.ceil(np.sqrt(V * WORDS_PER_REGION)))
CELLS = SIDE * SIDE
REGION = np.zeros(CELLS, int)
for c in range(CELLS):
    REGION[c] = min(c // WORDS_PER_REGION, V - 1)
REG = T.as_tensor(REGION, device=DEV)
W = T.zeros(CELLS, D, device=DEV)                     # prefrontal state -> motor sheet
print('motor sheet %dx%d = %d cells | %d regions (%d cells each)' % (
    SIDE, SIDE, CELLS, V, WORDS_PER_REGION), flush=True)


def listen(tokens, state):
    """basal ganglia ACTIVE: the prefrontal state follows the stream, the motor area is silent"""
    for t in tokens:
        state = state + (PAT[t] - state) / TAU
    return state


def speak_learn(state, token):
    """basal ganglia OFF: the prefrontal state reaches the word's region and strengthens it"""
    cells = T.nonzero(REG == token).squeeze(1)
    W[cells] += ETA * (state.unsqueeze(0) - W[cells])
    W.clamp_(min=0.0)


def utter(state):
    """basal ganglia OFF at test time: whichever region is most active is the word said.
    Each cell is judged against its OWN weight norm, otherwise regions that happened to be updated
    more often simply have larger weights and always win."""
    a = state @ W.T
    a = a / (W.norm(dim=1) + 1e-9)
    sc = T.stack([a[REG == v].sum() for v in range(V)])
    return int(sc.argmax())


# ---------------- training ----------------
mono_tokens = encode(open(mono, encoding='utf-8', errors='replace').read())
print('monologue stream: %d words (listening only)' % len(mono_tokens), flush=True)

# only keep pairs whose words are all inside the vocabulary: a target of <pad> is not a word the model
# could ever say, and comparing against it is meaningless
pairs = [(qq, aa) for (qq, aa) in pairs
         if all(w in w2i and w2i[w] > 1 for w in words(qq) + words(aa))]
print('pairs with an in-vocabulary answer: %d' % len(pairs), flush=True)

rng = np.random.default_rng(0)
order = rng.permutation(len(pairs))
held = [pairs[i] for i in order[:HOLD]]
train = [pairs[i] for i in order[HOLD:]]
NPAIR = int(os.environ.get('NPAIR', '0'))
if NPAIR:
    train = train[:NPAIR]
print('training pairs %d | held out %d' % (len(train), len(held)), flush=True)

zero = T.zeros(D, device=DEV)
for ep in range(EPOCHS):
    # 1) monologue: gate ACTIVE for the whole stream, so nothing is said and nothing is learned about
    #    answering -- the prefrontal state simply follows the language
    state = zero
    for t in mono_tokens[:600]:
        state = listen([t], state)
    # 2) dialogue: listen to the question with the gate ACTIVE ...
    for (qq, aa) in train:
        state = listen([w2i['<q>']], zero)
        state = listen(encode(qq), state)
        # ... then the other side has finished: the gate opens for ONE word, and after saying it the
        # gate closes again and the model goes back to listening (to itself)
        aa_tok = encode(aa)
        speak_learn(state, aa_tok[0])
        for tok in aa_tok:
            state = listen([tok], state)
    # evaluate
    ok = 0
    for (qq, aa) in held:
        state = listen([w2i['<q>']], zero)
        state = listen(encode(qq), state)
        if utter(state) == encode(aa)[0]:
            ok += 1
    # baseline: the most frequent first answer word
    from collections import Counter
    common = Counter(encode(aa)[0] for _, aa in train).most_common(1)[0][0]
    base = sum(1 for _, aa in held if encode(aa)[0] == common)
    tr_ok = 0
    for (qq, aa) in train:
        state = listen([w2i['<q>']], zero)
        state = listen(encode(qq), state)
        got = utter(state)
        want = encode(aa)[0]
        tr_ok += int(got == want)
        if ep == EPOCHS - 1:
            print('     Q: %-45s  -> said %-12s want %-12s %s' % (
                qq[:45], vocab[got], vocab[want], 'OK' if got == want else '--'), flush=True)
            print('        (state norm %.2f | W row norm %.2f | the two question states differ by %.3f)' % (
                float(state.norm()), float(W.norm(dim=1).mean()),
                float((state - prev_state).norm()) if 'prev_state' in dir() else -1.0), flush=True)
        prev_state = state
    print('  epoch %d: TRAINED %d/%d | held-out %d/%d | frequent-baseline %.3f' % (
        ep + 1, tr_ok, len(train), ok, len(held), base / max(len(held), 1)), flush=True)
