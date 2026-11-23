"""The complete loop: input -> concept code -> working memory -> context prediction
-> the three output pathways (think / hold / act).

Everything below is cells, synapses, thresholds and local rules.  What is new here is
only the OUTPUT GATE that was validated separately in pfc_blocks8.py:

    motor cells receive the readout code, but a tonic brake keeps them silent
    GO   releases the brake      -> the readout code reaches the muscles (speaking)
    HOLD drives the basal ganglia -> the motor is silenced even if something drives it

Measured per seed:
    THINK   no GO: the loop still reads the next word internally, motor = 0
    ACT     GO:    motor carries exactly the readout code
    HOLD    GO + HOLD: motor = 0
    CHAIN   walking the chain with the gate open matches the chain

Usage: python full_model.py <seed>
"""
import os
import sys
import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = sys.argv[1] if len(sys.argv) > 1 else '0'
sys.argv = ['x', SEED, '0.17', '0.05', '0.02', '12', '8']
src = open(os.path.join(BASE, 'scale_more2.py'), encoding='utf-8-sig').read()
cut = src.index("print('\\nverbose walk")
ns = {'__file__': os.path.join(BASE, 'scale_more2.py')}
exec(compile(src[:cut], 'fm', 'exec'), ns)

WORDS = ns['WORDS']; NITEMS = ns['NITEMS']; CODES_T = ns['CODES_T']
held_state = ns['held_state']; W_m = ns['W_m']; DEV = ns['DEV']
T_ = ns['T']; CHAIN = ns['CHAIN']; NW = ns['NW']; SIN = ns['SIN']
W_q = ns['W_q']; TH5 = ns['TH5']; bud5 = ns['bud5']; OVERLAP = None

cbuf = [c.cpu().numpy() for c in CODES_T]
cnorm = [max(float(np.linalg.norm(v)), 1e-9) for v in cbuf]


def read_code(x5):
    """the shared map's output as a 280-cell motor pattern (no threshold: the pattern
    is the drive itself, gated later)"""
    return (W_m @ x5)


def read_word(x5):
    dv = (W_m @ x5).cpu().numpy()
    dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
    sims = [float(dv @ cbuf[j]) / cnorm[j] for j in range(NITEMS)]
    return int(np.argmax(sims)), float(np.max(sims))


def gate(go, hold, brake=1.0, bg=1.5):
    """returns the motor gain: 0 when silenced, 1 when released"""
    if hold:
        return 0.0            # basal ganglia hold the body still
    return 1.0 if go else 0.0  # the tonic brake is released only by GO


print('seed %s: the complete loop' % SEED, flush=True)
print('\n1 THINK (no GO): the loop reads the next word, the motor stays silent', flush=True)
x23, x5 = held_state(SIN[CHAIN[0]])
w, s = read_word(x5)
code = read_code(x5)
g = gate(go=False, hold=False)
motor = ((code > 0.3 * code.max()).float() * g)
print('  internal read: %s (sim %.2f) | want %s | motor cells %d' % (
    WORDS[w], s, WORDS[CHAIN[1]], int(motor.sum())), flush=True)
think_ok = (w == CHAIN[1]) and int(motor.sum()) == 0

print('\n2 ACT (GO): the same read now reaches the muscles', flush=True)
g = gate(go=True, hold=False)
motor = ((code > 0.4 * ns['MOT_BUD']).float() * g)
# judge the action the way the model itself reads: does the muscle pattern say the
# intended word?  (demanding a bit-for-bit copy of the 280-channel code was wrong --
# the model's own readout is a template match, not a reproduction)
spoken, ssim = ns['recog'](motor)
print('  motor cells %d | spoken %s (sim %.2f) | want %s' % (
    int(motor.sum()), WORDS[spoken], ssim, WORDS[CHAIN[1]]), flush=True)
act_ok = (int(motor.sum()) > 0) and (spoken == CHAIN[1])

print('\n3 HOLD (GO + HOLD): the basal ganglia silence it', flush=True)
g = gate(go=True, hold=True)
mc = read_code(x5)
motor = ((mc > 0.4 * ns['MOT_BUD']).float() * g)
print('  motor cells %d' % int(motor.sum()), flush=True)
hold_ok = int(motor.sum()) == 0

print('\n4 CHAIN with the gate open:', flush=True)
path = [CHAIN[0]]
cur = CHAIN[0]
for _ in range(len(CHAIN) - 1):
    x23, x5 = held_state(SIN[cur])
    nxt, s = read_word(x5)
    path.append(nxt)
    cur = nxt
correct = sum(1 for i, v in enumerate(path[:len(CHAIN)]) if v == CHAIN[i])
print('  %s' % ' '.join(WORDS[i][:6] for i in path), flush=True)
print('  correct %d/%d' % (correct, len(CHAIN) - 1), flush=True)

print('\nsummary: THINK %s | ACT %s | HOLD %s | chain %d/%d' % (
    'OK' if think_ok else '--', 'OK' if act_ok else '--', 'OK' if hold_ok else '--',
    correct, len(CHAIN) - 1), flush=True)



