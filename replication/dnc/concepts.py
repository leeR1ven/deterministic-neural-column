"""Concept layer: 256 cells, 100 inputs each, local Hebbian + competition.

Documents (README section 2, SPEC 21/30):
    each cell receives 100 inputs from the first layer;
    the 3 most responsive cells move their inputs toward the current pattern;
    all inputs decay slightly (competition);
    300 epochs, then measure which cells became category-selective.
"""

from __future__ import annotations

import numpy as np


class ConceptLayer:
    def __init__(self, n_in: int = 1024, n_cells: int = 256, k: int = 100,
                 lr: float = 0.15, decay: float = 3e-4, winners: int = 3,
                 seed: int = 0):
        rng = np.random.default_rng(seed + 4242)
        self.n_in, self.n_cells, self.k = n_in, n_cells, k
        self.lr, self.decay, self.winners = lr, decay, winners
        self.idx = np.argsort(rng.random((n_cells, n_in)), axis=1)[:, :k]
        self.w = rng.random((n_cells, k)) + 0.1
        self.w /= self.w.sum(axis=1, keepdims=True)

    def _inputs(self, x: np.ndarray) -> np.ndarray:
        return x[self.idx]                                # (n_cells, k)

    def response(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self.w * self._inputs(x), axis=1)   # (n_cells,)

    def train(self, codes: list[np.ndarray], epochs: int = 300, seed: int = 0) -> None:
        rng = np.random.default_rng(seed + 99)
        for _ in range(epochs):
            for w in rng.permutation(len(codes)):
                xi = self._inputs(codes[w])
                r = np.sum(self.w * xi, axis=1)
                win = np.argpartition(r, -self.winners)[-self.winners:]
                target = xi[win] / (xi[win].sum(axis=1, keepdims=True) + 1e-9)
                self.w[win] += self.lr * (target - self.w[win])
                self.w *= (1.0 - self.decay)
                self.w = np.maximum(self.w, 0.0)
                self.w /= (self.w.sum(axis=1, keepdims=True) + 1e-12)

    # -- measuring category selectivity after training ----------------------
    def responses(self, codes: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.response(c) for c in codes])   # (n_words, n_cells)

    @staticmethod
    def cell_stats(R: np.ndarray, labels: np.ndarray, rel_thresh: float = 0.5):
        """A cell 'responds' to a word if its response exceeds rel_thresh*max."""
        n_words = R.shape[0]
        thresh = rel_thresh * R.max(axis=0, keepdims=True)
        resp = R > thresh
        sel, per_class = [], np.zeros(len(np.unique(labels)), dtype=int)
        for j in range(R.shape[1]):
            rs = resp[:, j]
            if rs.sum() < 2:
                continue
            classes = [c for c in np.unique(labels)
                       if rs[labels == c].sum() >= 2]
            outside = sum(rs[labels != c].sum() for c in classes)
            if len(classes) == 1 and outside == 0:
                sel.append(j)
                per_class[classes[0]] += 1
        return len(sel), per_class


def class_decode_loo(R: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """Leave-one-word-out class decoding from the 256-cell response.

    1-NN over the responses of the *other* words (their true class is known),
    so a word can never be decoded by matching itself.
    """
    n = R.shape[0]
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    ok = 0
    for i in range(n):
        sims = Rn @ Rn[i]
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        ok += int(labels[j] == labels[i])
    # word identity: same leave-one-out 1-NN, but this can only recover a word
    # if another word of the same class is its nearest neighbour -> not identity
    return ok, n
