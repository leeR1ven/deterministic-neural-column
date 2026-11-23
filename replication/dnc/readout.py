"""Shared / per-transition motor maps W_m and the unthresholded template readout.

Documents (SPEC 19/22/41):
    W_m: a single shared mapping, row budget 0.05, readout by taking the cosine
    of the continuous drive with every word code (no thresholding).

The learning rule is not written down in formula form; the documents call it
Hebbian and supervised by the target pattern ("运动映射用脚本给的目标图案训练").
Here it is implemented as the fixed point of that Hebbian rule: accumulate
target (x) L5-state outer products, then normalise each row to the row budget.
"""

from __future__ import annotations

import numpy as np


def train_shared(l5_states: list[np.ndarray], targets: list[np.ndarray],
                 budget: float = 0.05) -> np.ndarray:
    n_out, n5 = len(targets[0]), len(l5_states[0])
    C = np.zeros((n_out, n5))
    for s5, t in zip(l5_states, targets):
        C += np.outer(t, s5)
    row = C.sum(axis=1, keepdims=True)
    W = np.zeros_like(C)
    nz = row[:, 0] > 0
    W[nz] = budget * C[nz] / row[nz]
    return W


def train_blocks(l5_states: list[np.ndarray], targets: list[np.ndarray],
                 budget: float = 0.05) -> list[np.ndarray]:
    blocks = []
    for s5, t in zip(l5_states, targets):
        W = np.zeros((len(t), len(s5)))
        denom = s5.sum()
        if denom > 0:
            W[t > 0] = budget * s5 / denom
        blocks.append(W)
    return blocks


def drive(W: np.ndarray, s5: np.ndarray) -> np.ndarray:
    return W @ s5


def template_match(vec: np.ndarray, codes: list[np.ndarray]) -> tuple[int, np.ndarray]:
    v = vec / (np.linalg.norm(vec) + 1e-12)
    sims = np.zeros(len(codes))
    for i, c in enumerate(codes):
        c = c / (np.linalg.norm(c) + 1e-12)
        sims[i] = float(v @ c)
    return int(np.argmax(sims)), sims


def fidelity(drive_vec: np.ndarray, target: np.ndarray) -> dict:
    """Several candidate definitions of the documents' unspecified 'overlap'."""
    t = target > 0
    d = np.maximum(drive_vec, 0.0)
    mass = float(d.sum())
    prec = float(d[t].sum() / mass) if mass > 0 else 0.0
    n_t = int(t.sum())
    top = np.zeros_like(d)
    if n_t > 0:
        top[np.argsort(d)[-n_t:]] = 1.0
    jac = float((top[t] > 0).sum() / (n_t + (top > 0).sum() - (top[t] > 0).sum()))
    cos = float(d @ target / (np.linalg.norm(d) * np.linalg.norm(target) + 1e-12))
    return {"mass_precision": prec, "cosine": cos, "top_jaccard": jac,
            "n_target": n_t, "top_precision": float((top[t] > 0).sum() / max(n_t, 1))}
