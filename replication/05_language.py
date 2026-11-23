"""Step 5: muscle language (16 muscles x 5 time steps) and the two-way loop.

Documents (SPEC 28/29/52/53): every concept is a combination of 3 muscles at
specific time steps in an 80-channel space; an utterance is re-encoded as
sensory input, the loop holds it, recognises the concept, predicts the next
concept, and answers with its muscle pattern.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, structured_words, word_pattern, overlap_matrix
from dnc.loop import WorkingMemory
from dnc.readout import train_shared, drive, template_match

N_MUS, N_STEP = 16, 5
N_CH = N_MUS * N_STEP          # 80
READ_HOLD = 300


def muscle_patterns(n_concepts=8, seed=0):
    rng = np.random.default_rng(seed + 31337)
    pats = []
    for c in range(n_concepts):
        units = rng.choice(N_CH, size=3, replace=False)
        p = np.zeros(N_CH)
        p[units] = 1.0
        pats.append(p)
    return pats


def pattern_overlaps(pats):
    from dnc.core import jaccard
    vals = [jaccard(pats[i], pats[j]) for i in range(len(pats)) for j in range(i + 1, len(pats))]
    return float(np.mean(vals)), float(np.max(vals))


class _E:
    def __init__(self, code_len):
        self.code_len = float(code_len)
        self.n_out = 1024


def run(seed=0):
    words, labels = structured_words(8, 5)
    pats = [word_pattern(w) for w in words]
    enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=seed)
    enc.fit(pats)
    word_codes = np.stack([enc.encode(p, i)[0] for i, p in enumerate(pats)])
    concept_codes = np.stack([(word_codes[labels == c].sum(0) > 0).astype(float)
                              for c in range(8)])
    muscles = muscle_patterns(8, seed=seed)

    # ---- PFC loop driven by concept codes ---------------------------------
    wm = WorkingMemory(_E(concept_codes.sum(1).mean()), tau_memory=4000.0,
                       theta5_q=0.90, seed=seed)
    drives = [2.0 * c for c in concept_codes]
    scores = []
    for c in range(8):
        wm.reset(); wm.present(drives[c], 20); wm.hold(READ_HOLD)
        scores.append(wm.l5_scores())
    wm.calibrate_l5(scores, q=0.90)
    l5 = [np.asarray(s > wm.theta5, dtype=float) for s in scores]

    targets = [muscles[(c + 1) % 8] for c in range(8)]
    W = train_shared(l5, targets, budget=0.05)
    ok_chain = sum(int(template_match(drive(W, l5[c]), muscles)[0] == (c + 1) % 8)
                   for c in range(8))

    # ---- listener: muscle pattern -> sparse heard code --------------------
    henc = SparseEncoder(n_in=N_CH, n_out=1024, k_in=40, code_len=200,
                         theta_q=0.80, seed=seed)
    henc.fit(muscles)
    heard = np.stack([henc.encode(m, i)[0] for i, m in enumerate(muscles)])
    rec = sum(int(np.argmax([np.dot(heard[c], heard[j]) /
                             (np.linalg.norm(heard[c]) * np.linalg.norm(heard[j]) + 1e-12)
                             for j in range(8)]) == c) for c in range(8))

    # ---- two-way: hear concept c, answer with the pattern of the next one --
    hwm = WorkingMemory(henc, tau_memory=4000.0, theta5_q=0.90, seed=seed)
    hscores = []
    for c in range(8):
        hwm.reset(); hwm.present(heard[c] * 2.0, 20); hwm.hold(READ_HOLD)
        hscores.append(hwm.l5_scores())
    hwm.calibrate_l5(hscores, q=0.90)
    hl5 = [np.asarray(s > hwm.theta5, dtype=float) for s in hscores]
    W2 = train_shared(hl5, targets, budget=0.05)
    answers = [template_match(drive(W2, hl5[c]), muscles)[0] for c in range(8)]
    ok_ans = sum(int(answers[c] == (c + 1) % 8) for c in range(8))

    hov = overlap_matrix(heard); iu = np.triu_indices(8, 1)
    return dict(ok_chain=ok_chain, rec=rec, ok_ans=ok_ans, answers=answers,
                heard_cells=np.mean([x.sum() for x in heard]),
                heard_overlap=float(hov[iu].mean()),
                muscle_overlap=pattern_overlaps(muscles),
                l5_cells=float(np.mean([x.sum() for x in l5])),
                concept_overlap=float(overlap_matrix(concept_codes)[np.triu_indices(8, 1)].mean()))


if __name__ == "__main__":
    for seed in range(3):
        r = run(seed)
        print(f"seed {seed}: internal chain {r['ok_chain']}/8 | heard cells "
              f"{r['heard_cells']:.0f}, heard overlap {r['heard_overlap']:.3f} | "
              f"recognition {r['rec']}/8 | two-way answer {r['ok_ans']}/8 {r['answers']}")
    print(f"muscle pattern overlap (seed 0): mean {run(0)['muscle_overlap'][0]:.4f} "
          f"max {run(0)['muscle_overlap'][1]:.4f}")
