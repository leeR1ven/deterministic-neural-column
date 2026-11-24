"""Why is the answer wrong: bad data, bad parameters, or too strict an evaluation?

Three questions, each measured separately.

  1. does the data have a regularity to learn at all -- is the same question ever given two different
     answers, and how many distinct answers are there?
  2. can the prefrontal state tell two questions apart -- mean similarity between the states of questions
     that share an answer versus questions that do not
  3. is the answer "roughly" right -- where does the correct word rank among all words in the motor area?
     If it is usually in the top five or ten, the motor area is close and a collapse could finish the job;
     if the rank is scattered, nothing was learned.

Usage: python why_wrong.py [epochs] [vocab]
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
S, MZ = 28, int(os.environ.get('MZ', '16'))
DATA = r'D:\deepseekherness\data'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NVOC = int(sys.argv[2]) if len(sys.argv) > 2 else 300
TH, ALPHA, ETA, WPR, FAN_M = float(os.environ.get('TH', '0.6')), float(os.environ.get('ALPHA', '0.7')), 0.05, 6, 8
FANIN = int(os.environ.get('FANIN', '4'))

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


# ---- 1. data regularity ----
seen = {}
conflicts = 0
for qq, aa in pairs:
    k = ' '.join(words(qq))
    v = words(aa)[0] if words(aa) else ''
    if k in seen and seen[k] != v:
        conflicts += 1
    seen[k] = v
firsts = Counter(words(aa)[0] for _, aa in pairs if words(aa))
print('DATA: %d pairs | %d distinct questions | %d repeated with a different answer' % (
    len(pairs), len(seen), conflicts), flush=True)
print('DATA: %d distinct first answer words; most common %dx, next %dx' % (
    len(firsts), firsts.most_common(1)[0][1], firsts.most_common(2)[1][1]), flush=True)

counts = Counter()
for qq, aa in pairs:
    counts.update(words(qq) + words(aa))
vocab = ['<pad>'] + [w for w, _ in counts.most_common(NVOC)]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
pairs = [(qq, aa) for qq, aa in pairs if all(w in w2i and w2i[w] > 0 for w in words(qq) + words(aa))]
print('DATA: %d pairs survive the vocabulary filter' % len(pairs), flush=True)


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
sens = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=FANIN, bias_scale=1.0, seed=1, dev=DEV)
sens.theta = [TH]
pfc = Local3DStack([(S, S, MZ), (S, S, MZ)], fan_in=FANIN, bias_scale=1.0, seed=2, dev=DEV)
pfc.theta = [TH]
with T.no_grad():
    SPAT = sens.forward(T.as_tensor(GL, device=DEV))[-1]
D = SPAT.shape[1]
N_MOT = int(np.ceil(np.sqrt(V * WPR))) ** 2
REG = T.as_tensor(np.array([min(c // WPR, V - 1) for c in range(N_MOT)]), device=DEV)
_r = np.random.default_rng(11)
MOT_IN = T.as_tensor(np.stack([_r.choice(D, FAN_M, replace=False) for _ in range(N_MOT)]), device=DEV)
W_mot = T.zeros(N_MOT, FAN_M, device=DEV)
zero = T.zeros(D, device=DEV)


def run_pfc(tokens, state):
    outs = []
    with T.no_grad():
        for t in tokens:
            state = pfc.forward((SPAT[t] + ALPHA * state).unsqueeze(0))[-1][0]
            outs.append(state)
    return outs


rng = np.random.default_rng(0)
order = rng.permutation(len(pairs))
n_hold = min(60, len(pairs) // 4)
held = [pairs[i] for i in order[:n_hold]]
train = [pairs[i] for i in order[n_hold:]]

for ep in range(EPOCHS):
    for qq, aa in train:
        q_state = run_pfc(encode(qq), zero)[-1]
        cells = T.nonzero(REG == encode(aa)[0]).squeeze(1)
        W_mot[cells] += ETA * (q_state[MOT_IN[cells]] - W_mot[cells])
    W_mot.clamp_(min=0.0)

# ---- 3. where does the correct word rank? ----
SIDE = int(np.round(np.sqrt(N_MOT)))


def collapse(exc):
    """spatial collapse: average the sheet over a small neighbourhood, so what matters is WHERE the
    activity is, not whether one exact cell won."""
    m = exc.reshape(SIDE, SIDE)
    acc = T.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += T.roll(T.roll(m, dy, 0), dx, 1)
    return (acc / 9.0).reshape(-1)


print('PFC theta %.2f | fan-in %d | volume depth %d' % (TH, FANIN, MZ), flush=True)
_st0 = run_pfc(encode(held[0][0]), zero)
print('PFC activation: %.1f%% of %d cells (first question)' % (
    100 * float(_st0[-1].mean()), _st0[-1].numel()), flush=True)

ranks, ranks_col, states, ans = [], [], [], []
for qq, aa in held:
    st = run_pfc(encode(qq), zero)[-1]
    exc = (st[MOT_IN] * W_mot).sum(1) / (W_mot.sum(1) + 1e-9)
    sc = T.stack([exc[REG == v].sum() for v in range(V)])
    excc = collapse(exc)
    scc = T.stack([excc[REG == v].sum() for v in range(V)])
    want = encode(aa)[0]
    ranks.append(int((sc > sc[want]).sum()))
    ranks_col.append(int((scc > scc[want]).sum()))
    states.append(st)
    ans.append(want)

ranks = np.array(ranks)
print('', flush=True)
print('RANK of the correct answer word among %d words:' % V, flush=True)
print('  median %d | top-1 %d/%d | top-5 %d | top-10 %d | top-20 %d' % (
    int(np.median(ranks)), int((ranks == 0).sum()), len(ranks),
    int((ranks < 5).sum()), int((ranks < 10).sum()), int((ranks < 20).sum())), flush=True)
print('  uniform chance would average rank %.0f' % (V / 2), flush=True)
rc = np.array(ranks_col)
print('  WITH COLLAPSE (3x3 smoothing): median %d | top-1 %d | top-5 %d | top-10 %d' % (
    int(np.median(rc)), int((rc == 0).sum()), int((rc < 5).sum()), int((rc < 10).sum())), flush=True)

# ---- 2. can the prefrontal state separate questions? ----
Sn = T.stack(states)
Sn = Sn / Sn.norm(dim=1, keepdim=True).clamp(min=1e-9)
sim = (Sn @ Sn.T).cpu().numpy()
same, diff = [], []
for i in range(len(held)):
    for j in range(i + 1, len(held)):
        (same if ans[i] == ans[j] else diff).append(sim[i, j])
print('', flush=True)
print('PFC STATE: cosine similarity, same answer %.3f | different answer %.3f' % (
    float(np.mean(same)) if same else -1, float(np.mean(diff))), flush=True)
ov = []
for i in range(len(states)):
    for j in range(i + 1, len(states)):
        a, b = states[i], states[j]
        inter = float(((a > 0) & (b > 0)).sum())
        union = float(((a > 0) | (b > 0)).sum())
        if union:
            ov.append(inter / union)
print('PFC STATE: Jaccard overlap between two different questions %.3f' % float(np.mean(ov)), flush=True)
print('  (if the two are equal, the state does not carry the answer at all)', flush=True)
