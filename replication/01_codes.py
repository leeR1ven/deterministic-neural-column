"""Step 1: sparse word codes -- density and pairwise overlap."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnc.core import (SparseEncoder, chain_words, structured_words, word_pattern,
                      overlap_matrix, theory_overlap)


def run(name, words, seed=0):
    pats = [word_pattern(w) for w in words]
    enc = SparseEncoder(n_in=280, n_out=1024, k_in=80, code_len=70,
                        theta_q=0.90, seed=seed)
    enc.fit(pats)
    codes = np.stack([enc.encode(p, i)[0] for i, p in enumerate(pats)])
    dens = codes.mean(axis=1)
    ov = overlap_matrix(codes)
    iu = np.triu_indices(len(words), 1)
    f = float(dens.mean())
    print(f"\n=== {name}: {len(words)} words, seed {seed} ===")
    print(f"input density (raw 280 pattern): {np.mean([p.mean() for p in pats]):.4f}")
    print(f"code length: mean {codes.sum(1).mean():.1f}  min {codes.sum(1).min():.0f}  max {codes.sum(1).max():.0f}")
    print(f"code density f: {f:.4f}")
    print(f"pairwise Jaccard: mean {ov[iu].mean():.4f}  max {ov[iu].max():.4f}")
    print(f"theory lower bound f/(2-f): {theory_overlap(f):.4f}  ratio {ov[iu].mean()/theory_overlap(f):.2f}")
    gains = enc.gain
    print(f"per-item gain: min {gains.min():.3f} max {gains.max():.3f}")
    return codes, ov


if __name__ == "__main__":
    run("12-word chain", chain_words(12), seed=0)
    words, labels = structured_words(8, 5)
    codes, ov = run("40 structured words (8 classes x 5)", words, seed=0)
    # class structure
    within, between = [], []
    for i in range(40):
        for j in range(i + 1, 40):
            (within if labels[i] == labels[j] else between).append(ov[i, j])
    print(f"within-class Jaccard  mean {np.mean(within):.4f}")
    print(f"between-class Jaccard mean {np.mean(between):.4f}")
    print(f"ratio {np.mean(within)/np.mean(between):.2f}")
    # concept = union of the 5 class codes
    concept = np.stack([(codes[labels == c].sum(0) > 0).astype(float) for c in range(8)])
    print(f"concept (union) size: {concept.sum(1).astype(int)}")
    cov = overlap_matrix(concept)
    iu = np.triu_indices(8, 1)
    print(f"concept pairwise Jaccard: mean {cov[iu].mean():.4f} max {cov[iu].max():.4f}")
    run("160 random words", chain_words(160), seed=0)
