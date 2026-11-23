"""Step 10: how sensitive are the concept-cell counts to the undefined
'responds' threshold, and how dense are the concept codes?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, structured_words, word_pattern, overlap_matrix
from dnc.concepts import ConceptLayer, class_decode_loo

words, labels = structured_words(8, 5)
pats = [word_pattern(w) for w in words]
enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=0)
enc.fit(pats)
codes = [enc.encode(p, i)[0] for i, p in enumerate(pats)]
C = np.stack(codes)
concept = np.stack([(C[labels == c].sum(0) > 0).astype(float) for c in range(8)])
print("concept code sizes:", concept.sum(1).astype(int),
      f"= {concept.mean():.3f} of 1024 (SPEC S5 requires <0.20)")
print("word code density:", f"{C.mean():.3f}")

for seed in (0, 1, 2):
    layer = ConceptLayer(seed=seed)
    layer.train(codes, epochs=300, seed=seed)
    R = layer.responses(codes)
    rng = np.random.default_rng(seed + 555)
    perm = rng.permutation(labels)
    row = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        n, _ = ConceptLayer.cell_stats(R, labels, rel_thresh=t)
        nr, _ = ConceptLayer.cell_stats(R, perm, rel_thresh=t)
        row.append(f"{t:.1f}:{n}/{nr}")
    ok, _ = class_decode_loo(R, labels)
    print(f"seed {seed}: selective/control by rel_thresh  " + "  ".join(row) +
          f" | LOO class decode {ok}/40")
