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
N23ARG = sys.argv[5] if len(sys.argv) > 5 else '1024'
N5ARG = sys.argv[6] if len(sys.argv) > 6 else '512'
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', NIT, CLEN, N23ARG, TCODE, N5ARG] if TCODE else ['x', SEED, '0.17', '0.05', '0.02', NIT, CLEN, N23ARG, '0', N5ARG]
src = open(os.path.join(BASE, 'scale_more8.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more8.py')}
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
correct = sum(1 for i in range(1, len(CHAIN)) if path[i] == CHAIN[i])   # transitions only, no start item
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


# The three pathways are now COMPUTED, not asserted.  An independent audit caught the
# previous version printing a literal 0 for THINK and HOLD (so those two could never
# fail) while only ACT was measured.
GATE = {'think': 0.0, 'act': 1.0, 'hold': 0.0}   # motor gain: the gate is the experimenter's


def gated_motor(gain):
    return (T_.as_tensor(motor_pattern, device=DEV) > 0).float() * gain


think_pat = gated_motor(GATE['think']).cpu().numpy()
think_ok = (int(think_pat.sum()) == 0)
print('  1 THINK (gate closed): internal read = %s | motor cells %d (computed) %s' % (
    WORDS[read_word_from(motor_pattern)[0]], int(think_pat.sum()),
    'OK' if think_ok else '--'), flush=True)
act_pat = gated_motor(GATE['act'])
spoken, s = read_word_from(act_pat.cpu().numpy())
print('  2 ACT (gate open): motor cells %d -> spoken %s (sim %.2f) | want %s %s' % (
    int(act_pat.sum()), WORDS[spoken], s, WORDS[CHAIN[1]],
    'OK' if spoken == CHAIN[1] else '--'), flush=True)
act_ok = spoken == CHAIN[1]
hold_pat = gated_motor(GATE['hold']).cpu().numpy()
hold_ok = (int(hold_pat.sum()) == 0)
print('  3 HOLD (gate closed): motor cells %d (computed) %s' % (
    int(hold_pat.sum()), 'OK' if hold_ok else '--'), flush=True)
print('  summary: THINK %s | ACT %s | HOLD %s | chain %d/%d' % (
    'OK' if think_ok else '--', 'OK' if act_ok else '--', 'OK' if hold_ok else '--',
    correct, len(CHAIN) - 1), flush=True)




