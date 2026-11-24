"""Why does the walk decode the wrong word even on the first step?  Print the raw numbers."""
import os
import sys

import numpy as np
import torch as T

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sensory import Local3DStack

DEV = 'cuda' if T.cuda.is_available() else 'cpu'
S, MZ, TH = 28, 8, 0.6
TH_A, ALPHA, GAMMA, BETA, DECAY = 0.5, 0.5, 0.8, 0.2, 0.8
NW = 12

rng = np.random.default_rng(7)
G = np.zeros((NW, S * S), np.float32)
for i in range(NW):
    G[i, rng.choice(S * S, 6, replace=False)] = 1.0

vis = Local3DStack([(S, S, 1), (S, S, MZ)], fan_in=4, bias_scale=1.0, seed=1, dev=DEV)
vis.theta = [TH]
with T.no_grad():
    P = vis.forward(T.as_tensor(G, device=DEV))[-1].float()
D = P.shape[1]
Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
print('cells per word: %s' % P.sum(1).tolist(), flush=True)
print('off-diagonal |cos| mean %.4f  max %.4f'
      % (float(((Pn @ Pn.T) - T.eye(NW, device=DEV)).abs().sum() / (NW * (NW - 1))),
         float(((Pn @ Pn.T) - T.eye(NW, device=DEV)).abs().max())), flush=True)

sentence = [0, 3, 0, 7, 1]

# what the learned synapses give from the bare first word
W = T.zeros(D, D, device=DEV)
for t in range(len(sentence) - 1):
    W += T.outer(P[sentence[t + 1]], P[sentence[t]]) / P[sentence[t]].pow(2).sum().clamp(min=1e-9)
act = W @ P[sentence[0]]
print('', flush=True)
print('clean learning, bare first word 0:', flush=True)
print('  value on word 3 cells: %.2f-%.2f' % (act[P[3] > 0.5].min(), act[P[3] > 0.5].max()), flush=True)
print('  value on word 7 cells: %.2f-%.2f' % (act[P[7] > 0.5].min(), act[P[7] > 0.5].max()), flush=True)
print('  value elsewhere     : %.2f-%.2f' % (act[P.sum(0) < 0.5].min(), act[P.sum(0) < 0.5].max()), flush=True)
s = (Pn @ (act / act.norm().clamp(min=1e-9))).cpu().numpy()
print('  similarity        : %s' % ' '.join('%d:%.2f' % (i, v) for i, v in enumerate(s)), flush=True)
print('  argmax %d  (want 3)' % int(s.argmax()), flush=True)

fired = (act > TH_A).float()
print('  fired cells %d -> decoded %d'
      % (int(fired.sum()), int((Pn @ (fired / fired.norm().clamp(min=1e-9))).argmax())), flush=True)
