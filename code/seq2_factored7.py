"""Two-word context by factoring, not by growing: the older word and the current word
are projected into DIFFERENT halves of L2/3.

Why: last round failed because both context words were summed into the same 4096 cells,
so the state had to distinguish 160x160 = 25600 pair patterns -- three orders of
magnitude beyond what 4096 cells with 70-cell codes can hold.

If the two words land in disjoint halves, the pair is represented additively:

    state = [ codeA(older word)  |  codeB(current word) ]
             2048 cells            2048 cells

and the number of representable pairs is 2 x 160, not 160 x 160.  This is the
biological version of a two-tap delay line: "what I heard a moment ago" and "what I
just heard" feed different cell groups.

Test: 160 words, 60-step chain, each step depends on the previous TWO words.

Usage: python seq2_factored.py <seed>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
ARGV = list(sys.argv)
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
NWORDS = 160
STEPS = int(ARGV[2]) if len(ARGV) > 2 else 60
LSEQ = STEPS + 2
HALF = int(ARGV[3]) if len(ARGV) > 3 else 2048
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', str(NWORDS), str(LSEQ),
            str(2 * HALF), '70', str(2 * HALF)]
src = open(os.path.join(BASE, 'scale_more8.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more8.py')}
exec(compile(src[:cut], 'f2', 'exec'), ns)

import text_encoder as te
WORDS = ns['WORDS']; DEV = ns['DEV']; T_ = ns['T']
N5 = int(ARGV[4]) if len(ARGV) > 4 else ns['N5']; NI = ns['NI']; NW = ns['NW']; ETA = ns['ETA']; MOT_BUD = ns['MOT_BUD']
bias23 = None

# ---- two independent input pathways, each its own sparse code
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS[:NWORDS]]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
mean_in = comp.mean(0)
comp_c = comp - mean_in
rng = np.random.default_rng(int(SEED) + 11)


def _one_projection(ncells):
    W = rng.normal(0, 1, (ncells, comp_c.shape[1]))
    mask = np.zeros_like(W, bool)
    for j in range(ncells):
        mask[j, rng.choice(comp_c.shape[1], 40, replace=False)] = True
    W = W * mask
    W = W / np.abs(W).sum(1, keepdims=True)
    bud = np.abs(W).sum(1) + 1e-9
    TARGET = 70
    codes = []
    for k in range(NWORDS):
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
    return codes


def _overlap(codes):
    """mean pairwise overlap of the codes (lower = the words are more separable)"""
    M = np.stack(codes)
    inter = M @ M.T
    s = M.sum(1)
    union = s[:, None] + s[None, :] - inter
    tri = np.triu_indices(len(codes), 1)
    return float(np.mean(inter[tri] / np.maximum(union[tri], 1)))


def make_code(ncells, tag):
    """Draw several projections and keep the one whose codes overlap the least.

    This is the developmental part: earlier rounds showed the same architecture either
    walking all 60 pairs or failing badly depending only on the seed, which means some
    random projections give words codes that collide.  Selecting by overlap is the
    computational equivalent of activity-dependent pruning -- no new mechanism, just
    keeping the projection that separates the words best.
    """
    NCAND = int(os.environ.get('CANDS', '4'))
    best = None
    for cand in range(NCAND):
        codes = _one_projection(ncells)
        ov = _overlap(codes)
        if best is None or ov < best[0]:
            best = (ov, codes)
    codes = best[1]
    print('  pathway %s: %d cells | mean code size %.0f | pairwise overlap %.3f (best of %d)' % (
        tag, ncells, float(np.mean([c.sum() for c in codes])), best[0], NCAND), flush=True)
    return codes


print('building two input pathways (older word / current word):', flush=True)
CA = make_code(HALF, 'A(older)')
CB = make_code(HALF, 'B(current)')

N23 = 2 * HALF
NI = 256
W_ei = T_.as_tensor((rng.random((NI, N23)) < 0.1).astype(np.float32) * (0.34 / 100), device=DEV)
W_ie = T_.as_tensor((rng.random((N23, NI)) < 0.1).astype(np.float32) * (0.65 / 100), device=DEV)
bias23 = T_.zeros(N23, device=DEV) + 0.03
bias_i = T_.zeros(NI, device=DEV) + 0.02
BUD = T_.ones(N23, device=DEV) + bias23
TH23v = T_.full((N23,), 0.35, device=DEV)
TH_I_CELL = T_.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T_.float32)
bud_i = T_.full((NI,), 0.34, device=DEV)
W_q = T_.as_tensor((rng.random((N5, N23)) < 0.02).astype(np.float32) * (0.30 / 80), device=DEV)
bud5 = T_.full((N5,), 0.30, device=DEV)
TH5 = 0.17
TAU, KAPPA = 400.0, 1.5
CUE_ONE, HOLD = 5, 10
MOT_BUD = 0.05 * (STEPS / 7.0)

rng2 = np.random.default_rng(int(SEED) + 5)
# A FIXED sequence, saved to disk the first time and reloaded afterwards.  Earlier rounds
# changed the architecture and the random draw at the same time, so the numbers were not
# comparable; from here on every configuration walks the same words.
SEQFILE = os.path.join(BASE, 'fixed_seq_%d.txt' % LSEQ)
if os.path.exists(SEQFILE):
    SEQ = [int(x) for x in open(SEQFILE).read().split()]
else:
    SEQ = [int(v) for v in rng2.integers(0, NWORDS, LSEQ)]
    open(SEQFILE, 'w').write(' '.join(str(v) for v in SEQ))


def state_of(pair):
    """present the older word then the current word, one in each half, then hold"""
    a = T_.as_tensor(np.concatenate([CA[pair[0]], np.zeros(HALF, np.float32)]), device=DEV)
    b = T_.as_tensor(np.concatenate([np.zeros(HALF, np.float32), CB[pair[1]]]), device=DEV)
    cues = [a, b]
    m = T_.zeros(N23, device=DEV)
    x23 = T_.zeros(N23, device=DEV); xi = T_.zeros(NI, device=DEV)
    m_ref = None
    total = CUE_ONE * len(cues) + HOLD
    for t in range(total):
        idx = t // CUE_ONE
        inp = cues[idx] if idx < len(cues) else None
        d23 = bias23 + (inp if inp is not None else 0) - W_ie @ xi
        rel = d23 / BUD
        if t == CUE_ONE * len(cues) - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23v > TH23v).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I_CELL * bud_i).float()
        x23 = fire
    return x23


def to_l5(x23):
    return ((0.02 + W_q @ x23) > TH5 * bud5).float()


# the map outputs the CONCEPT code (the stage-A code of that word, re-derived here)
Wc = np.stack([(np.clip(te.pattern(WORDS[k]), 0, 1) > 0.5) for k in range(NWORDS)]).astype(np.float32)
CBUF = [np.concatenate([c, np.zeros(N23 - c.shape[0], np.float32)]) if c.shape[0] < N23 else c for c in Wc]
# concept codes live in a separate 4096-cell space: build them with a third pathway
CC = make_code(N23, 'C(concept/output)')
CBUF = [np.asarray(c, dtype=np.float32) for c in CC]
W_m = T_.zeros(N23, N5, device=DEV)
print('training: (older, current) -> the concept code of the next word', flush=True)
for ep in range(80):
    for i in range(LSEQ - 2):
        x5 = to_l5(state_of((SEQ[i], SEQ[i + 1])))
        W_m = T_.clamp(W_m + ETA * (1 - W_m) * (T_.as_tensor(CBUF[SEQ[i + 2]], device=DEV).unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))

cnorm = [max(float(np.linalg.norm(v)), 1e-9) for v in CBUF]


def predict(pair):
    x5 = to_l5(state_of(pair))
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ CBUF[j]) / cnorm[j] for j in range(NWORDS)]
    return int(np.argmax(sims)), float(np.max(sims))


cur = [SEQ[0], SEQ[1]]
got, ok = [], 0
for step in range(STEPS):
    nxt, s = predict(tuple(cur))
    got.append(nxt)
    ok += int(nxt == SEQ[step + 2])
    cur = [cur[1], nxt]
print('\n  factored representation, %d words, %d-step two-word chain:' % (NWORDS, STEPS), flush=True)
print('  correct %d/%d' % (ok, STEPS), flush=True)
print('  predicted: %s' % [WORDS[i][:6] for i in got[:10]], flush=True)
print('  true     : %s' % [WORDS[i][:6] for i in SEQ[2:12]], flush=True)






