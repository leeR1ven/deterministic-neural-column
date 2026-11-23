"""Automated search over the PFC small-loop parameters.

The loop is the important part, so this searches only its parameters:

    w_ie      strength of the inhibitory -> excitatory synapses
    tau       time constant of the slow input-driven trace (the "do not die" half)
    kappa     how strongly that trace boosts a cell
    k_ei      fan-in of the inhibitory pool  (with its threshold auto-calibrated)

For every configuration it measures, over 12 items:

    HOLD      fraction of items that keep >50 % of their assembly after the cue
    DISTINCT  pairwise overlap of the held states  (want < 0.10)
    ALIVE     no item is fully silent and none is fully saturated

and, for configurations that hold, it trains ONE shared map (no per-transition
blocks) and walks the 8-step chain.

Writes one CSV line per configuration; keeps the best.
"""
import os
import sys
import time
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
OUT = r'D:\deepseekherness\gpu\loop_search.csv'
ETA, WMAX, EPOCHS, KFF, KQ = 3e-2, 1.0, 40, 40, 20

# ---------------- stage A: the orthogonal sparse code (fixed, measured 0.030)
base = np.stack([(np.clip(te.pattern(w), 0, 1) > 0.5) for w in WORDS]).astype(float)
comp = np.concatenate([2 * base - 1, 1 - 2 * base], axis=1)
mean_in = comp.mean(0)
comp_c = comp - mean_in


def build(seed):
    rng = np.random.default_rng(seed)
    W_ff = rng.normal(0, 1, (N23, comp.shape[1]))
    mask = np.zeros_like(W_ff, bool)
    for j in range(N23):
        mask[j, rng.choice(comp.shape[1], KFF, replace=False)] = True
    W_ff = W_ff * mask
    W_ff = W_ff / np.abs(W_ff).sum(1, keepdims=True)
    bud_ff = np.abs(W_ff).sum(1) + 1e-9
    best = None
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        m = float(np.mean([float((W_ff @ comp_c[k] > th * bud_ff).mean()) for k in range(NITEMS)]))
        if best is None or abs(m - 0.08) < abs(best[1] - 0.08):
            best = (th, m)
    TH_A = best[0]
    codes = [(W_ff @ comp_c[k] > TH_A * bud_ff).astype(np.float32) for k in range(NITEMS)]
    return W_ff, bud_ff, TH_A, codes


def run_cfg(cfg, seed):
    rng = np.random.default_rng(seed + 7)
    W_ff, bud_ff, TH_A, codes = build(seed)
    SIN = [T.as_tensor(c, device=DEV, dtype=T.float32) for c in codes]

    def sp(W, posts, pre, k, budget):
        M = T.zeros(posts, pre, device=DEV)
        for j in range(posts):
            M[j, T.as_tensor(rng.choice(pre, k, replace=False), device=DEV)] = 1.0
        return M * (budget / M.sum(1, keepdim=True).clamp(min=1e-6))

    W_in = None   # the stage-A code IS the L2/3 state; no second random projection
    W_p = sp(None, N23, N5, 40, 0.30)
    W_q = sp(None, N5, N23, KQ, 0.30)
    W_ei = sp(None, NI, N23, cfg['k_ei'], 0.34)
    W_ie = sp(None, N23, NI, 100, cfg['w_ie'])
    W_m = T.zeros(NW, N5, device=DEV)
    bias23 = T.zeros(N23, device=DEV) + 0.03
    bias_i = T.zeros(NI, device=DEV) + 0.02
    BUD = T.ones(N23, device=DEV) + W_p.abs().sum(1) + bias23
    bud_i = T.full((NI,), 0.34, device=DEV)
    bud5 = T.full((N5,), 0.30, device=DEV)
    bud_m = T.full((NW,), 1.0, device=DEV)

    rels = [((SIN[k] + bias23) / BUD).cpu().numpy() for k in range(NITEMS)]
    # per-cell threshold: each cell uses the 90th percentile of ITS OWN drive across
    # items.  A single global value cannot serve a layer whose cells differ in how dense
    # their own input is (measured: theta 0.135 left 11 of 12 items with no activity).
    TH23v = T.as_tensor(np.quantile(np.stack(rels), 0.90, axis=0), device=DEV, dtype=T.float32)
    TH23 = float(np.median(np.quantile(np.stack(rels), 0.90, axis=0)))
    TH5, THM = 0.30, 0.05

    def run(code, steps=30, cue=10):
        m = T.zeros(N23, device=DEV)
        x23 = T.zeros(N23, device=DEV); x5 = T.zeros(N5, device=DEV); xi = T.zeros(NI, device=DEV)
        rec = []
        for t in range(steps):
            inp = code if t < cue else None
            d23 = bias23 + W_p @ x5 + (inp if inp is not None else 0) - W_ie @ xi
            rel = d23 / BUD
            fire = (rel + cfg['kappa'] * m > TH23v).float()
            m = m + (rel - m) / cfg['tau']
            ni = ((bias_i + W_ei @ fire) > cfg['th_i'] * bud_i).float()
            n5 = ((0.02 + W_q @ fire) > TH5 * bud5).float()
            x23, x5, xi = fire, n5, ni
            rec.append((float(x23.sum()), float(x5.sum()), float(xi.sum())))
        return x23, x5, xi, m, rec

    # learn the attractors: potentiate the recurrent weights on each item's own cue
    # state, so the state becomes self-consistent instead of collapsing into one common
    # fixed point (measured without this: overlap 0.93 -- everything held the same cells)
    for ep in range(20):
        for k in range(NITEMS):
            m0 = T.zeros(N23, device=DEV)
            a23 = T.zeros(N23, device=DEV); a5 = T.zeros(N5, device=DEV); ai = T.zeros(NI, device=DEV)
            for t in range(10):
                inp = SIN[k]
                d23 = bias23 + W_p @ a5 + inp - W_ie @ ai
                rr = d23 / BUD
                a23 = (rr + cfg['kappa'] * m0 > TH23v).float()
                m0 = m0 + (rr - m0) / cfg['tau']
                ai = ((bias_i + W_ei @ a23) > cfg['th_i'] * bud_i).float()
                a5 = ((0.02 + W_q @ a23) > TH5 * bud5).float()
            W_q = T.clamp(W_q + 0.01 * (a5.unsqueeze(1) * a23.unsqueeze(0)), 0, 1.0)
            W_p = T.clamp(W_p + 0.01 * (a23.unsqueeze(1) * a5.unsqueeze(0)), 0, 1.0)
        W_q = W_q * (0.30 / W_q.sum(1, keepdim=True).clamp(min=1e-6))
        W_p = W_p * (0.30 / W_p.sum(1, keepdim=True).clamp(min=1e-6))

    holds, states = 0, []
    sat = dead = 0
    for k in range(NITEMS):
        x23, x5, xi, m, rec = run(SIN[k])
        cue_n = rec[9][0]
        fin_n = rec[-1][0]
        if cue_n >= 0.95 * N23:
            sat += 1
        if fin_n < 5:
            dead += 1
        holds += int(fin_n > 0.5 * max(cue_n, 1) and fin_n >= 5)
        states.append(x23.cpu().numpy())
    live = [s for s in states if s.sum() > 0]
    ovl = float('nan')
    if len(live) > 1:
        ovl = float(np.mean([(live[i] * live[j]).sum() / max(((live[i] + live[j]) > 0).sum(), 1)
                             for i in range(len(live)) for j in range(i + 1, len(live))]))

    chain, reached = [], 0
    if holds >= NITEMS - 1 and not np.isnan(ovl):
        CODES_T = [T.as_tensor((np.clip(te.pattern(WORDS[k]), 0, 1) > 0.5).astype(np.float32),
                               device=DEV) for k in range(NITEMS)]
        for ep in range(EPOCHS):
            for a, b in TRANS:
                _, x5, _, _, _ = run(SIN[a])
                W_m = T.clamp(W_m + ETA * (1 - W_m / WMAX) * (CODES_T[b].unsqueeze(1) * x5.unsqueeze(0)), 0, WMAX)
                mean = W_m.sum(1, keepdim=True)
                W_m = W_m * T.clamp(1.0 / mean.clamp(min=1e-9), max=1.0)

        def recog(out):
            hb = (out.cpu().numpy() > 0.5).astype(float)
            vec = np.concatenate([2 * hb - 1, 1 - 2 * hb]) - mean_in
            cod = (W_ff @ vec > TH_A * bud_ff).astype(float)
            sims = [(cod * c).sum() / max(((cod + c) > 0).sum(), 1) for c in codes]
            return int(np.argmax(sims))

        path = [0]
        _, x5, _, _, _ = run(SIN[0])
        for _ in range(len(CHAIN)):
            out = (W_m @ x5 > THM * bud_m).float()
            if float(out.sum()) == 0:
                path.append(-1); break
            nxt = recog(out)
            path.append(nxt)
            _, x5, _, _, _ = run(SIN[nxt])
        chain = path
        reached = sum(1 for i, v in enumerate(path[:len(CHAIN)]) if v == CHAIN[i])
    return dict(hold=holds, ovl=ovl, sat=sat, dead=dead, reached=reached, chain=chain,
                theta=TH23, thA=TH_A)


GRID = []
for w_ie in (0.20, 0.35, 0.50, 0.65):
    for tau in (100.0, 200.0, 400.0):
        for kappa in (0.5, 1.0, 2.0):
            for th_i in (0.15, 0.30, 0.45):
                GRID.append(dict(w_ie=w_ie, tau=tau, kappa=kappa, k_ei=100, th_i=th_i))
print('%d configurations' % len(GRID), flush=True)
if not os.path.exists(OUT):
    open(OUT, 'w').write('w_ie,tau,kappa,k_ei,th_i,hold,overlap,saturated,dead,chain_reached,chain\n')

best = None
for i, cfg in enumerate(GRID):
    t0 = time.time()
    try:
        r = run_cfg(cfg, 0)
    except Exception as e:
        print('cfg %d failed: %s' % (i, e), flush=True)
        continue
    line = '%s,%s,%s,%s,%d,%.3f,%d,%d,%d,"%s"' % (
        cfg['w_ie'], cfg['tau'], cfg['kappa'], cfg['k_ei'], r['hold'],
        -1 if np.isnan(r['ovl']) else r['ovl'], r['sat'], r['dead'], r['reached'],
        ' '.join(str(v) for v in r['chain']))
    open(OUT, 'a').write(line + '\n')
    print('[%2d/%2d] w_ie %.2f tau %3.0f kappa %.1f -> hold %2d/%d ovl %s sat %d dead %d | chain %d/8 %s (%.0fs)' % (
        i + 1, len(GRID), cfg['w_ie'], cfg['tau'], cfg['kappa'], r['hold'], NITEMS,
        'n/a' if np.isnan(r['ovl']) else '%.2f' % r['ovl'], r['sat'], r['dead'],
        r['reached'], ' '.join(str(v) for v in r['chain']), time.time() - t0), flush=True)
    score = r['hold'] + (0 if np.isnan(r['ovl']) else max(0.0, 0.10 - r['ovl']) * 50) + r['reached']
    if best is None or score > best[0]:
        best = (score, cfg, r)
    if r['hold'] == NITEMS and r['reached'] >= 7:
        print('*** PASS ***', cfg, flush=True)
        break

if best:
    print('\nbest: %s' % best[1], flush=True)
    print('      hold %d/%d | overlap %s | chain %d/8' % (
        best[2]['hold'], NITEMS, best[2]['ovl'], best[2]['reached']), flush=True)




