"""Step 9 (optional claim): context prediction from two preceding words.

Documents (SPEC 21/26/43): the slow variable integrates several words, so the
held state is a mixture of the context; the readout predicts the next word.
The discriminating test is the same second word in two different contexts
predicting different successors, which a last-word lookup cannot do.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, chain_words, word_pattern
from dnc.loop import WorkingMemory
from dnc.readout import train_shared, drive, template_match

# 6 words, six context pairs.  Word 1 is the second word of two pairs with
# different successors; same for words 2 and 3.
PAIRS = [(0, 1, 2), (3, 1, 4), (1, 2, 3), (4, 2, 5), (2, 3, 4), (5, 3, 0)]
WORDS = chain_words(6)
PATS = [word_pattern(w) for w in WORDS]
ENC = SparseEncoder(280, 1024, 80, 70, 0.90, seed=0)
ENC.fit(PATS)
DRIVES = [ENC.encode(p, i)[1] for i, p in enumerate(PATS)]


def make_states(mode, seed=0, hold=300):
    q = 0.85 if mode == "two-word-context" else 0.70
    wm = WorkingMemory(ENC, tau_memory=4000.0, theta5_q=q, seed=seed)
    scores, states = [], []
    for (a, b, c) in PAIRS:
        wm.reset()
        if mode == "two-word-context":
            wm.present(DRIVES[a], 20)
        wm.present(DRIVES[b], 20)
        wm.hold(hold)
        scores.append(wm.l5_scores())
        states.append(wm.s.copy())
    wm.calibrate_l5(scores, q=q)
    l5 = [np.asarray(s > wm.theta5, dtype=float) for s in scores]
    return l5, np.stack(states)


def evaluate(mode, seed=0):
    l5, states = make_states(mode, seed)
    targets = [PATS[c] for (_, _, c) in PAIRS]
    W = train_shared(l5, targets, budget=0.05)
    predictions = []
    for k in range(len(PAIRS)):
        idx, _ = template_match(drive(W, l5[k]), PATS)
        predictions.append(idx)
    correct = sum(int(predictions[k] == PAIRS[k][2]) for k in range(len(PAIRS)))
    by_second = {}
    for k, (_, b, _) in enumerate(PAIRS):
        by_second.setdefault(b, []).append(predictions[k])
    groups = [v for v in by_second.values() if len(v) > 1]
    contrast = sum(len(set(v)) == len(v) for v in groups)
    return correct, contrast, len(groups), l5


if __name__ == "__main__":
    for seed in range(3):
        for mode in ("last-word-only", "two-word-context"):
            correct, contrast, groups, l5 = evaluate(mode, seed)
            print(f"seed {seed} {mode:16s}: next-word readout {correct}/6 | "
                  f"same-second-word -> different prediction {contrast}/{groups} | "
                  f"L5 cells {np.mean([x.sum() for x in l5]):.0f}")
