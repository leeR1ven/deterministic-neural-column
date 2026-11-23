"""Step 4: concept layer -- do concept cells emerge from local rules?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import (SparseEncoder, structured_words, word_pattern, overlap_matrix)
from dnc.concepts import ConceptLayer, class_decode_loo

words, labels = structured_words(8, 5)
pats = [word_pattern(w) for w in words]
enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=0)
enc.fit(pats)
codes = [enc.encode(p, i)[0] for i, p in enumerate(pats)]
ov = overlap_matrix(np.stack(codes))
within = [ov[i, j] for i in range(40) for j in range(i + 1, 40) if labels[i] == labels[j]]
between = [ov[i, j] for i in range(40) for j in range(i + 1, 40) if labels[i] != labels[j]]
print(f"trained-free word codes: density {np.mean([c.mean() for c in codes]):.3f}, "
      f"length {np.mean([c.sum() for c in codes]):.0f}")
print(f"within-class Jaccard {np.mean(within):.4f} | between {np.mean(between):.4f} | "
      f"ratio {np.mean(within)/np.mean(between):.2f}")
concept = np.stack([(np.stack(codes)[labels == c].sum(0) > 0).astype(float) for c in range(8)])
cov = overlap_matrix(concept)
iu = np.triu_indices(8, 1)
print(f"concept = union of 5 word codes: size {concept.sum(1).astype(int)}")
print(f"concept pairwise Jaccard: mean {cov[iu].mean():.4f} max {cov[iu].max():.4f}")

for seed in (0, 1, 2):
    layer = ConceptLayer(n_in=1024, n_cells=256, k=100, lr=0.15, decay=3e-4,
                         winners=3, seed=seed)
    R0 = layer.responses(codes)
    n0, _ = ConceptLayer.cell_stats(R0, labels)
    layer.train(codes, epochs=300, seed=seed)
    R = layer.responses(codes)
    n_sel, per_class = ConceptLayer.cell_stats(R, labels)
    # shuffled-label control: same responses, permuted labels
    rng = np.random.default_rng(seed + 555)
    perm = rng.permutation(labels)
    n_rand, _ = ConceptLayer.cell_stats(R, perm)
    cls_ok, _ = class_decode_loo(R, labels)
    # class template cosine separability
    cls_codes = np.stack([R[labels == c].mean(0) for c in range(8)])
    cls_codes /= np.linalg.norm(cls_codes, axis=1, keepdims=True) + 1e-12
    off = cls_codes @ cls_codes.T
    print(f"\nseed {seed}: selective before {n0} -> after {n_sel}/256 "
          f"(per class {per_class.tolist()}), shuffled-label control {n_rand}, "
          f"LOO class decode {cls_ok}/40, max class-template cosine {off[~np.eye(8, dtype=bool)].max():.3f}")
