"""Step 2: working memory -- does the state survive input removal, and is it content?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, chain_words, word_pattern, jaccard, overlap_matrix
from dnc.loop import WorkingMemory

WORDS = chain_words(12)
PATS = [word_pattern(w) for w in WORDS]
ENC = SparseEncoder(280, 1024, 80, 70, 0.90, seed=0)
ENC.fit(PATS)
DRIVES = [ENC.encode(p, i)[1] for i, p in enumerate(PATS)]
PRESENTED = [(d > 1.0).astype(float) for d in DRIVES]


def run(tau):
    wm = WorkingMemory(ENC, tau_memory=tau, theta5_q=0.90, seed=0)
    held, presented, scores = [], [], []
    for k in range(len(WORDS)):
        wm.reset()
        wm.present(DRIVES[k], steps=20)
        pres = wm.s.copy()
        wm.hold(1000)
        held.append(wm.s.copy())
        presented.append(pres)
        scores.append(wm.l5_scores())
    wm.calibrate_l5(scores, q=0.90)
    l5 = [np.asarray(s > wm.theta5, dtype=float) for s in scores]
    held = np.stack(held)
    alive = held.sum(1)
    selfsim = np.array([jaccard(held[k], presented[k]) for k in range(12)])
    best = []
    for k in range(12):
        sims = [jaccard(held[k], presented[j]) for j in range(12)]
        best.append(int(np.argmax(sims)) == k)
    ov = overlap_matrix(held)
    iu = np.triu_indices(12, 1)
    d = held.mean()
    print(f"tau={tau:6.0f}: alive {int(alive.min())}-{int(alive.max())} cells | "
          f"density {d:.3f} | self-Jaccard {selfsim.mean():.2f} | "
          f"correct-identity {sum(best)}/12 | pairwise {ov[iu].mean():.4f} | "
          f"L5 {np.mean([x.sum() for x in l5]):.0f} cells")


if __name__ == "__main__":
    print("presented code density:", PRESENTED[0].mean(), "len", PRESENTED[0].sum())
    for tau in (500, 1000, 2000, 4000, 8000):
        run(tau)
