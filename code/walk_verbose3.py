"""Verbose walk: print every stage of one step so the empty link is visible.

Stages printed per step:
  held   : how many L2/3 cells remain after the cue
  l5     : how many L5 cells the readout stage produces
  drive  : max of the shared map's drive over the 280 output cells
  read   : which item the produced code is closest to (and the similarity)
"""
import os
import sys
import numpy as np
import torch as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_encoder as te

DEV = 'cuda'
N23, N5, NI, NW = 1024, 512, 256, 280
WORDS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
         'iota', 'kappa', 'lambda', 'mu']
NITEMS = 12
CHAIN = list(range(8))
TRANS = list(zip(CHAIN[:-1], CHAIN[1:]))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
KFF, KQ = 40, 20
ETA, EPOCHS = 3e-2, 60
TAU, KAPPA, CUE = 400.0, 1.5, 10
TH5 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.17
THM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02

base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
mean_in = comp.mean(0)
comp_c = comp - mean_in
rng = np.random.default_rng(SEED)
W_ff = rng.normal(0, 1, (N23, comp.shape[1]))
mask = np.zeros_like(W_ff, bool)
for j in range(N23):
    mask[j, rng.choice(comp.shape[1], KFF, replace=False)] = True
W_ff = W_ff * mask
W_ff = W_ff / np.abs(W_ff).sum(1, keepdims=True)
bud_ff = np.abs(W_ff).sum(1) + 1e-9
best = None
for th in (0.05, 0.10, 0.15, 0.20):
    m = float(np.mean([float((W_ff @ comp_c[k] > th * bud_ff).mean()) for k in range(NITEMS)]))
    if best is None or abs(m - 0.08) < abs(best[1] - 0.08):
        best = (th, m)
TH_A = best[0]
CODES = [(W_ff @ comp_c[k] > TH_A * bud_ff).astype(np.float32) for k in range(NITEMS)]
SIN = [T.as_tensor(c, device=DEV, dtype=T.float32) for c in CODES]

rng2 = np.random.default_rng(SEED + 7)


def sp(post, pre, k, budget):
    M = T.zeros(post, pre, device=DEV)
    for j in range(post):
        M[j, T.as_tensor(rng2.choice(pre, k, replace=False), device=DEV)] = 1.0
    return M * (budget / M.sum(1, keepdim=True).clamp(min=1e-6))


W_q = sp(N5, N23, KQ, 0.30)
W_ei = sp(NI, N23, 100, 0.34)
W_ie = sp(N23, NI, 100, 0.65)
bias23 = T.zeros(N23, device=DEV) + 0.03
bias_i = T.zeros(NI, device=DEV) + 0.02
BUD = T.ones(N23, device=DEV) + bias23
TH_I_CELL = T.as_tensor(np.linspace(0.05, 0.60, NI), device=DEV, dtype=T.float32)
bud_i = T.full((NI,), 0.34, device=DEV)
bud5 = T.full((N5,), 0.30, device=DEV)
bud_m = T.full((NW,), 1.0, device=DEV)
rels = [((SIN[k] + bias23) / BUD).cpu().numpy() for k in range(NITEMS)]
TH23v = T.as_tensor(np.quantile(np.stack(rels), 0.90, axis=0), device=DEV, dtype=T.float32)
W_m = T.zeros(NW, N5, device=DEV)
MOT_BUD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05


def held_state(code, cue=CUE, extra=12):
    """run the cue then `extra` free steps; return the held L2/3 state and the L5 it produces"""
    m = T.zeros(N23, device=DEV)
    x23 = T.zeros(N23, device=DEV); xi = T.zeros(NI, device=DEV)
    m_ref = None
    for t in range(cue + extra):
        inp = code if t < cue else None
        d23 = bias23 + (inp if inp is not None else 0) - W_ie @ xi
        rel = d23 / BUD
        if t == cue - 1:
            m_ref = m.clone().clamp(min=1e-6)
        mn = (m / m_ref) if m_ref is not None else (m / m.abs().max().clamp(min=1e-6))
        fire = (rel + KAPPA * mn * TH23v > TH23v).float()
        m = m + (rel - m) / TAU
        xi = ((bias_i + W_ei @ fire) > TH_I_CELL * bud_i).float()
        x23 = fire
    x5 = ((0.02 + W_q @ x23) > TH5 * bud5).float()
    return x23, x5


CODES_T = [T.as_tensor((np.clip(te.pattern(WORDS[k]), 0, 1) > 0.5).astype(np.float32), device=DEV)
           for k in range(NITEMS)]
print('training the shared map ...', flush=True)
for ep in range(EPOCHS):
    for a, b in TRANS:
        _, x5 = held_state(SIN[a])
        W_m = T.clamp(W_m + ETA * (1 - W_m) * (CODES_T[b].unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))


def ov(a, b):
    u = np.logical_or(a > 0.5, b > 0.5).sum()
    return float(np.logical_and(a > 0.5, b > 0.5).sum() / u) if u else 0.0


def recog(out):
    hb = (out.cpu().numpy() > 0.5).astype(float)
    vec = np.concatenate([2 * hb - 1, 1 - 2 * hb]) - mean_in
    cod = (W_ff @ vec > TH_A * bud_ff).astype(float)
    sims = [(cod * c).sum() / max(((cod + c) > 0).sum(), 1) for c in CODES]
    return int(np.argmax(sims)), float(np.max(sims))


print('\nverbose walk from item 0:', flush=True)
cur = 0
for step in range(len(CHAIN)):
    x23, x5 = held_state(SIN[cur])
    drive = T.as_tensor(W_m, device=DEV) @ x5
    out = (drive > THM * bud_m).float()
    if float(out.sum()) == 0:
        print('  step %d: held L23 %4d | L5 %3d | max drive %.3f | READ EMPTY (threshold %.2f)' % (
            step, int(x23.sum()), int(x5.sum()), float(drive.max()), THM), flush=True)
        break
    nxt, sim = recog(out)
    print('  step %d: held L23 %4d | L5 %3d | max drive %.3f | read %-8s (sim %.2f) | want %s' % (
        step, int(x23.sum()), int(x5.sum()), float(drive.max()), WORDS[nxt], sim,
        WORDS[CHAIN[step + 1]] if step + 1 < len(CHAIN) else '-'), flush=True)
    cur = nxt


