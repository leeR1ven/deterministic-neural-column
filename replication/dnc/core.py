"""Core pieces rebuilt from the written description only.

No code from the original repository was read.  Every numeric choice that the
documents did not pin down is listed in REPLICATION_REPORT.md.

The rule implemented here is the one the documents call the "reference rule":

    active_i  <=>  score_i > theta_i * (|b_i| + sum_j |w_ij|)

The documents state this per-cell budget normalisation, plus a per-cell threshold,
plus (in the final versions) a per-item gain that equalises code length, plus
subtraction of the mean input (redundancy removal) before the threshold.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# deterministic hashing (Python's hash() is salted per process, unusable)
# ----------------------------------------------------------------------------

def _fnv1a(text: str) -> int:
    h = 2166136261
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ----------------------------------------------------------------------------
# words
# ----------------------------------------------------------------------------

GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"


def chain_words(n: int, seed: int = 0) -> list[str]:
    """First n words of the alpha..omega chain, then deterministic pseudo-words."""
    if n <= len(GREEK):
        return GREEK[:n]
    rng = np.random.default_rng(seed + 12345)
    words = list(GREEK)
    seen = set(words)
    while len(words) < n:
        k = int(rng.integers(5, 8))
        w = "".join(CONSONANTS[int(rng.integers(0, len(CONSONANTS)))] if i % 2 == 0
                    else VOWELS[int(rng.integers(0, len(VOWELS)))] for i in range(k))
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words


STEMS = ["bara", "kelu", "mira", "doru", "savi", "tenu", "goli", "paxi"]
SUFFIXES = ["n", "s", "l", "k", "m"]


def structured_words(n_classes: int = 8, per_class: int = 5,
                     suffix_len: int = 1) -> tuple[list[str], np.ndarray]:
    """Words with shared stems -> classes are visible in the character n-grams."""
    words, labels = [], []
    for c in range(n_classes):
        stem = STEMS[c % len(STEMS)]
        for j in range(per_class):
            if suffix_len == 1:
                sfx = SUFFIXES[j % len(SUFFIXES)]
            else:
                sfx = ("s" if j % 2 else "n") + SUFFIXES[j % len(SUFFIXES)]
            words.append(stem + sfx)
            labels.append(c)
    return words, np.asarray(labels)


# ----------------------------------------------------------------------------
# input code: character n-gram -> 280 channels -> +-1 complement
# ----------------------------------------------------------------------------

def word_pattern(word: str, n_in: int = 280, per_gram: int = 3,
                 gram_sizes: tuple[int, ...] = (2, 3)) -> np.ndarray:
    s = "^" + word + "$"
    channels = set()
    for n in gram_sizes:
        for i in range(len(s) - n + 1):
            h = _fnv1a(s[i:i + n])
            for k in range(per_gram):
                channels.add((h * (2 * k + 1) + 7919 * k) % n_in)
    x = np.zeros(n_in, dtype=np.float64)
    x[list(channels)] = 1.0
    return x


def complement(pattern: np.ndarray) -> np.ndarray:
    """+-1 complement code: [2x-1 ; -(2x-1)] (the docs' '+-1 补码')."""
    v = 2.0 * pattern - 1.0
    return np.concatenate([v, -v])


# ----------------------------------------------------------------------------
# sparse encoder  (the documents' "A level": +-1 complement -> subtract mean
# input -> local signed projection -> per-cell threshold -> per-item gain)
# ----------------------------------------------------------------------------

class SparseEncoder:
    def __init__(self, n_in: int = 280, n_out: int = 1024, k_in: int = 80,
                 code_len: int = 70, theta_q: float = 0.90, seed: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.k_in = k_in
        self.code_len = code_len
        self.theta_q = theta_q
        rng = np.random.default_rng(seed)

        n_comp = 2 * n_in
        # local signed projection: each output cell sees k_in of the 2*n_in inputs
        self.W = np.zeros((n_out, n_comp), dtype=np.float64)
        idx = np.argsort(rng.random((n_out, n_comp)), axis=1)[:, :k_in]
        sign = rng.choice([-1.0, 1.0], size=(n_out, k_in))
        rows = np.repeat(np.arange(n_out), k_in)
        self.W[rows, idx.ravel()] = sign.ravel() / np.sqrt(k_in)
        self.idx = idx
        self.mean = None
        self.theta = None
        self.gain = None

    # -- input codes ---------------------------------------------------------
    def project(self, z: np.ndarray) -> np.ndarray:
        return self.W @ z

    def fit(self, pats: list[np.ndarray]) -> None:
        """pats: list of binary word patterns."""
        Z = np.stack([complement(p) for p in pats])          # (n_words, 2*n_in)
        self.mean = Z.mean(axis=0)
        S = (Z - self.mean) @ self.W.T                        # (n_words, n_out)
        self.theta = np.maximum(np.quantile(S, self.theta_q, axis=0), 1e-9)  # per-cell threshold
        self.gain = np.ones(len(pats))
        for w in range(len(pats)):
            self.gain[w] = self._calibrate_gain(S[w])

    def _calibrate_gain(self, score: np.ndarray) -> float:
        lo, hi = 1e-3, 50.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            n = int(np.sum(mid * score > self.theta))
            if n > self.code_len:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def drive(self, pattern: np.ndarray) -> np.ndarray:
        """Normalised drive a_i = gain * score_i / theta_i  (a_i > 1 <=> active)."""
        z = complement(pattern) - self.mean
        s = self.W @ z
        return self.gain[0] * s / self.theta  # placeholder, real gain set per word

    def encode(self, pattern: np.ndarray, word_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary code, normalised drive a)."""
        z = complement(pattern) - self.mean
        s = self.W @ z
        g = 1.0 if word_index is None else self.gain[word_index]
        a = g * s / self.theta
        return (a > 1.0).astype(np.float64), a


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return float(np.sum(a & b) / union)


def overlap_matrix(codes: np.ndarray) -> np.ndarray:
    n = codes.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = jaccard(codes[i], codes[j])
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def theory_overlap(f: float) -> float:
    """Expected Jaccard of two independent random codes of density f."""
    return f / (2.0 - f)
