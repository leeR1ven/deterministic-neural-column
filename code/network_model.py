"""Everything is a network. The turn-taking signal comes from outside, the inhibition is a real synapse.

The designer's instruction: in the dialogue data the outside speaks first, and while it speaks the basal
ganglia are driven; when it is the model's turn the basal ganglia are NOT driven and the motor neurons fire.
Who speaks is decided by the teacher, and the prefrontal area is meant to pick up the regularity from the
stream.

So the four populations are:

  sensory        local 3D layer over 28x28x16, fixed weights, threshold fixed.  A word is a small picture.
  prefrontal     the same kind of layer, run RECURRENTLY: input_t = word + ALPHA * state_{t-1}.  The
                 recurrence is a synapse, not a buffer in the code.
  basal ganglia  a population driven from outside while somebody else is talking.  Its axons are
                 INHIBITORY synapses onto the motor area, with a fixed strength.
  motor          a population; each word owns a group of cells.  It is excited by the prefrontal state
                 through LEARNED synapses and inhibited by whatever the basal ganglia send.  A word is
                 uttered when its cells pass their own threshold -- and this is read at EVERY step, so
                 speaking out of turn would be visible.

Nothing decides anything outside the network. The only outside inputs are (a) the words and (b) the
turn signal. All thresholds are fixed constants, never fitted.

Usage: python network_model.py [epochs] [vocab]
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
NVOC = int(sys.argv[2]) if len(sys.argv) > 2 else 300

TH = 0.6          # firing threshold, the weight-budget rule, same everywhere
ALPHA = 0.7       # recurrent synapse inside the prefrontal area
ETA = 0.05        # learning rate
INH = 3.0         # strength of one inhibitory synapse, basal ganglia -> motor (fixed)
WPR = 6           # motor cells per word
N_BG = 200        # basal ganglia cells

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
vocab = ['<pad>'] + [w for w, _ in counts.most_common(NVOC)]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
pairs = [(qq, aa) for qq, aa in pairs if all(w in w2i and w2i[w] > 0 for w in words(qq) + words(aa))]
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
sens = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
sens.theta = [TH]
pfc = Local3DStack([(S, S, MZ), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=2, dev=DEV)   # input is a prefrontal pattern, not an image
pfc.theta = [TH]
with T.no_grad():
    SPAT = sens.forward(T.as_tensor(GL, device=DEV))[-1]
D = SPAT.shape[1]

SIDE = int(np.ceil(np.sqrt(V * WPR)))
N_MOT = SIDE * SIDE
REG = T.as_tensor(np.array([min(c // WPR, V - 1) for c in range(N_MOT)]), device=DEV)
FAN_M = 8                                          # prefrontal inputs each motor cell receives
_r = np.random.default_rng(11)
MOT_IN = T.as_tensor(np.stack([_r.choice(D, FAN_M, replace=False) for _ in range(N_MOT)]), device=DEV)
W_mot = T.zeros(N_MOT, FAN_M, device=DEV)          # prefrontal -> motor (learned, sparse like every other cell)
BG_OUT = T.full((N_MOT,), -INH, device=DEV)           # basal ganglia -> motor (inhibitory, fixed)
W_bg_pfc = T.zeros(N_BG, D, device=DEV)            # not used for learning; bg is driven from outside
print('populations: sensory/prefrontal %d cells | basal ganglia %d | motor %d cells (%d words x %d)' % (
    D, N_BG, N_MOT, V, WPR), flush=True)


def run_pfc(tokens, state):
    """the prefrontal area, recurrent"""
    outs = []
    with T.no_grad():
        for t in tokens:
            inp = T.clamp(SPAT[t] + ALPHA * state, 0.0, 1.0)
            state = pfc.forward(inp.unsqueeze(0))[-1][0]
            outs.append(state)
    return outs


def motor_activity(state):
    """each motor cell reads its own few prefrontal inputs; the weight-budget rule decides firing.
    This is the ONLY place the motor area is read, at every moment."""
    xj = state[MOT_IN]                                 # (cells, FAN_M)
    exc = (xj * W_mot).sum(1) / (W_mot.sum(1) + 1e-9)  # 0..1, how much of its budget is satisfied
    return exc


def motor_with_gate(state, listening):
    exc = motor_activity(state)
    if listening:
        # the basal ganglia are driven: every one of their inhibitory synapses acts
        return exc + BG_OUT.sum()                # a summed inhibitory current
    return exc


def utter(state, listening):
    a = motor_with_gate(state, listening)
    sc = T.stack([a[REG == v].sum() for v in range(V)])
    fires = bool(sc.max() > TH)     # the same weight-budget threshold
    return int(sc.argmax()), fires


rng = np.random.default_rng(0)
order = rng.permutation(len(pairs))
n_hold = min(60, len(pairs) // 4)
held = [pairs[i] for i in order[:n_hold]]
train = [pairs[i] for i in order[n_hold:]]
zero = T.zeros(D, device=DEV)

for ep in range(EPOCHS):
    for qq, aa in train:
        # the outside speaks: basal ganglia driven, nothing is learned about answering
        run_pfc(encode(qq), zero)
        # now it is the model's turn: basal ganglia NOT driven. The state that conditions the answer is
        # the one left after the question, and the answer word's cells learn THAT state.
        q_state = run_pfc(encode(qq), zero)[-1]
        cells = T.nonzero(REG == encode(aa)[0]).squeeze(1)
        W_mot[cells] += ETA * (q_state[MOT_IN[cells]] - W_mot[cells])
    W_mot.clamp_(min=0.0)

    silent = 0
    total = 0
    for qq, aa in held:
        st = run_pfc(encode(qq), zero)[-1]
        _, fires = utter(st, listening=True)               # during the question
        silent += int(not fires)
        total += 1
    ok, spoke = 0, 0
    for qq, aa in held:
        st = run_pfc(encode(qq), zero)[-1]
        w, fires = utter(st, listening=False)              # the model's turn
        spoke += int(fires)
        ok += int(w == encode(aa)[0])
    if ep == EPOCHS - 1:
        qq, aa = held[0]
        st = run_pfc(encode(qq), zero)[-1]
        exc = motor_activity(st)
        print('    DEBUG: W_mot non-zero cells %d | max excitation %.4f | mean %.4f | top word %s want %s' % (
            int((W_mot.sum(1) > 1e-6).sum()), float(exc.max()), float(exc.mean()),
            vocab[int(T.stack([exc[REG == v].sum() for v in range(V)]).argmax())], words(aa)[0]), flush=True)
    print('  epoch %d | silent while the outside speaks %d/%d | speaks on its turn %d/%d | right word %d/%d' % (
        ep + 1, silent, total, spoke, len(held), ok, len(held)), flush=True)
