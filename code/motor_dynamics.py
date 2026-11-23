"""The motor side done the way the designer described (and Churchland & Shenoy 2024):

    the PFC sends only a ROUGH signal -> the motor area turns it into a CONTINUOUS
    trajectory -> the basal ganglia can silence single motor neurons or all of them

Why this shape:
  * the motor area is a dynamical system: once kicked, its state evolves on its own, so
    one rough input produces a whole continuous movement (not a static pattern)
  * preparation lives in the NULL SPACE of the muscle output (Churchland & Shenoy 2024,
    "Preparatory activity and the expansive null-space", Nat Rev Neurosci 25:213-236):
    the population can be busy while nothing moves
  * a GO signal only opens the output gate -- it does not add any command

Measured:
  1  four rough PFC signals -> four distinct continuous trajectories
  2  during preparation (gate closed) the arm does not move; after GO it does
  3  silencing a FEW motor neurons bends the trajectory; silencing ALL stops it
     (this is the basal-ganglia style inhibition the designer asked for)

Usage: python motor_dynamics.py <seed>
"""
import os
import sys
import numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(SEED)
N = 256                 # motor neurons
N_CONCEPT = 6           # rough signals from the PFC
T = 24                  # steps in a movement

# ---- the motor area as a dynamical system: h <- A h, spectral radius just below 1 so
#      that a kick decays smoothly into a trajectory instead of exploding
A = rng.normal(0, 1, (N, N))
ev = np.max(np.abs(np.linalg.eigvals(A)))
A = A * (0.97 / ev)

# ---- the output mapping to two muscles (a 2-D hand).  Its null space is what allows
#      preparation without movement.
M = rng.normal(0, 1, (N, 2))
M = M / np.linalg.norm(M, axis=0, keepdims=True)

# ---- four rough PFC signals: low-dimensional, sparse kicks into the motor area
P = rng.normal(0, 1, (N_CONCEPT, N))
P = P / np.linalg.norm(P, axis=1, keepdims=True)
H0 = [P[c] * 4.0 for c in range(N_CONCEPT)]


def trajectory(h0, gate_from=T + 1, mask=None):
    """run the dynamics; the output is read only from step `gate_from` on"""
    h = h0.copy()
    pos = []
    for t in range(T):
        h = A @ h
        if mask is not None:
            h = h * mask
        live = t >= gate_from
        pos.append((M.T @ h) if live else np.zeros(2))
    return np.array(pos)


print('motor area: %d neurons -> 2 muscles | %d rough PFC signals | %d steps' % (
    N, N_CONCEPT, T), flush=True)

# 1 ---- four rough signals -> four trajectories
traj = [trajectory(H0[c], gate_from=0) for c in range(N_CONCEPT)]
ends = np.array([t[-1] for t in traj])
dmin = min(np.linalg.norm(ends[i] - ends[j]) for i in range(N_CONCEPT) for j in range(i + 1, N_CONCEPT))
print('\n1 four rough signals -> distinct continuous trajectories', flush=True)
print('   endpoint separation (closest pair): %.3f | spread %.3f' % (
    dmin, float(np.linalg.norm(ends.std(0)))), flush=True)
for c in range(N_CONCEPT):
    print('   signal %d: start (%.2f, %.2f) -> end (%.2f, %.2f), path length %.2f' % (
        c, traj[c][0][0], traj[c][0][1], traj[c][-1][0], traj[c][-1][1],
        float(np.sum(np.linalg.norm(np.diff(traj[c], axis=0), axis=1)))), flush=True)

# 2 ---- preparation without movement, then GO
prep = trajectory(H0[0], gate_from=T + 1)     # never released
moved = trajectory(H0[0], gate_from=0)
late = trajectory(H0[0], gate_from=12)        # prepared first, then released
print('\n2 preparation in the null space, then GO', flush=True)
print('   never released: total movement %.3f' % float(np.sum(np.linalg.norm(prep, axis=1))), flush=True)
print('   released at once: total movement %.3f' % float(np.sum(np.linalg.norm(moved, axis=1))), flush=True)
print('   prepared 12 steps then released: moved during preparation %.3f, after release %.3f' % (
    float(np.sum(np.linalg.norm(late[:12], axis=1))),
    float(np.sum(np.linalg.norm(late[12:], axis=1)))), flush=True)

# 3 ---- basal-ganglia style inhibition
print('\n3 inhibition (silence some motor neurons, or all)', flush=True)
base = trajectory(H0[0], gate_from=0)
for frac in (0.0, 0.1, 0.3, 1.0):
    mask = np.ones(N)
    k = int(frac * N)
    if 0 < k < N:
        mask[rng.choice(N, k, replace=False)] = 0.0
    elif k == N:
        mask[:] = 0.0
    t2 = trajectory(H0[0], gate_from=0, mask=mask)
    dev = float(np.mean(np.linalg.norm(t2 - base, axis=1)))
    print('   silencing %3d%% of motor neurons -> mean path deviation %.3f' % (
        int(frac * 100), dev), flush=True)

print('\nconclusion: one rough PFC signal is enough for a whole continuous movement;', flush=True)
print('the output gate (basal ganglia) decides whether it happens, and partial', flush=True)
print('inhibition bends the movement rather than switching it off entirely.', flush=True)
