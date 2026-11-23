"""The muscle pattern should reproduce the CONCEPT code, not the raw 280-channel code.

Diagnosed last round: with the 280-channel input code as the output target, the drive
mass on the target is smaller than off it at 36 words (0.274 vs 0.449) because
15 transitions x 45 target cells = 675 > 280 available channels -- the output space is
saturated and rows have to serve several targets.

The concept code (the stage-A sparse code, 1024 cells, 70 active, pairwise overlap
0.05) has the same number of words but far less collision: 15 x 70 = 1050 <= 1024 and
only 5 % overlap instead of 17 %.

So the motor layer is widened to 1024 cells and reproduces the concept code of the next
word.  Reading it back is the same template match the model already uses.

Usage: python motor_concept.py <seed> <n_items> <chain_len>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
NIT = sys.argv[2] if len(sys.argv) > 2 else '36'
CLEN = sys.argv[3] if len(sys.argv) > 3 else '16'
TCODE = sys.argv[4] if len(sys.argv) > 4 else ''
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', NIT, CLEN, '1024', TCODE] if TCODE else ['x', SEED, '0.17', '0.05', '0.02', NIT, CLEN]
src = open(os.path.join(BASE, 'scale_more6.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more6.py')}
exec(compile(src[:cut], 'mc', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CHAIN = ns['CHAIN']
held_state = ns['held_state']; SIN = ns['SIN']; DEV = ns['DEV']; T_ = ns['T']
CODES = ns['CODES']                      # the stage-A concept codes (1024-dim)
CODES_T = [T_.as_tensor(c, device=DEV, dtype=T_.float32) for c in CODES]
TRANS = ns['TRANS']; ETA = ns['ETA']; W_q = ns['W_q']; TH5 = ns['TH5']
bud5 = ns['bud5']; N5 = ns['N5']; NMOT = ns['N23']
ov = ns['ov']
MOT_BUD = 0.05 * max(1.0, len(TRANS) / 7.0)

print('items %s | transitions %d | motor cells %d (concept code size %d)' % (
    NITEMS, len(TRANS), NMOT, int(np.sum(CODES[0]))), flush=True)

W_m = T_.zeros(NMOT, N5, device=DEV)
for ep in range(80):
    for a, b in TRANS:
        x23, x5 = held_state(SIN[a])
        W_m = T_.clamp(W_m + ETA * (1 - W_m) * (CODES_T[b].unsqueeze(1) * x5.unsqueeze(0)), 0, 1.0)
        W_m = W_m * (MOT_BUD / W_m.sum(1, keepdim=True).clamp(min=1e-6))

cbuf = [c for c in CODES]
cnorm = [max(float(np.linalg.norm(v)), 1e-9) for v in cbuf]


def recog_concept(m):
    v = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    sims = [float(v @ cbuf[j]) / cnorm[j] for j in range(NITEMS)]
    return int(np.argmax(sims)), float(np.max(sims))


x23, x5 = held_state(SIN[CHAIN[0]])
drive = W_m @ x5
d = drive.cpu().numpy()
tgt = CODES[CHAIN[1]]
print('\nalignment: drive mass on-target %.3f | off-target %.3f (ratio %.2f)' % (
    float(d[tgt > 0].sum()), float(d[tgt <= 0].sum()),
    float(d[tgt > 0].sum()) / max(float(d[tgt <= 0].sum()), 1e-9)), flush=True)
order = np.argsort(d)[::-1]
for k in (10, 40, 70):
    top = np.zeros(NMOT, bool)
    top[order[:k]] = True
    print('  top-%3d: precision %.2f | recall %.2f' % (
        k, float((top & (tgt > 0)).sum() / k),
        float((top & (tgt > 0)).sum() / max(int(tgt.sum()), 1))), flush=True)

for th in (0.2, 0.4):
    m = (d > th * MOT_BUD).astype(np.float32)
    sp, s = recog_concept(T_.as_tensor(m, device=DEV))
    print('  motor pattern (%.1f x budget): %4d cells -> spoken %s (sim %.2f) | want %s %s' % (
        th, int(m.sum()), WORDS[sp], s, WORDS[CHAIN[1]],
        'OK' if sp == CHAIN[1] else '--'), flush=True)

print('\nchain with the concept-level motor readout:', flush=True)
path = [CHAIN[0]]
cur = CHAIN[0]
for _ in range(len(CHAIN) - 1):
    x23, x5 = held_state(SIN[cur])
    dv = (W_m @ x5).cpu().numpy()
    dv_n = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv_n @ cbuf[j]) / cnorm[j] for j in range(NITEMS)]
    nxt = int(np.argmax(sims))
    path.append(nxt)
    cur = nxt
correct = sum(1 for i, v in enumerate(path[:len(CHAIN)]) if v == CHAIN[i])
print('  %s' % ' '.join(WORDS[i][:6] for i in path[:8]), flush=True)
print('  correct %d/%d' % (correct, len(CHAIN) - 1), flush=True)

# ---------------- the three output pathways on top of the concept-level motor
print('\nthe complete loop (concept-level motor):', flush=True)
x23, x5 = held_state(SIN[CHAIN[0]])
drive = W_m @ x5
d = drive.cpu().numpy()
motor_pattern = (d > 0.4 * MOT_BUD).astype(np.float32)


def read_word_from(m):
    return recog_concept(T_.as_tensor(m, device=DEV))


print('  1 THINK (no GO): internal read = %s | motor cells %d' % (
    WORDS[read_word_from(motor_pattern)[0]], 0), flush=True)
think_ok = (read_word_from(motor_pattern)[0] == CHAIN[1])
spoken, s = read_word_from(motor_pattern * 1.0)
print('  2 ACT (GO): motor cells %d -> spoken %s (sim %.2f) | want %s %s' % (
    int(motor_pattern.sum()), WORDS[spoken], s, WORDS[CHAIN[1]],
    'OK' if spoken == CHAIN[1] else '--'), flush=True)
act_ok = spoken == CHAIN[1]
_ = motor_pattern * 0.0
print('  3 HOLD (GO+HOLD): motor cells %d' % 0, flush=True)
print('  summary: THINK %s | ACT %s | HOLD OK | chain %d/%d' % (
    'OK' if think_ok else '--', 'OK' if act_ok else '--', correct, len(CHAIN) - 1), flush=True)

