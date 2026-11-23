"""The PFC small loop: L2/3 slow variable, inhibitory pool, single-pass L5 readout.

Rebuilt from the documents' final structure (SPEC.md sections 19/22/41):

    A level feedforward code -> L2/3 (1024) held by a slow variable
    L2/3 -> inhibitory pool (256, heterogeneous thresholds)
    L2/3 -> L5 (512), single feedforward readout, drive divided by activity

The documents do not give the update equation of the slow variable, only
"integrates input, tau 400".  This file exposes the choice (tau_memory) and
uses a jump-then-decay form so that a short presentation (a few hundred steps)
can be held for a full second.
"""

from __future__ import annotations

import numpy as np


class WorkingMemory:
    def __init__(self, enc, n5: int = 512, k5: int = 80, theta5_q: float = 0.90,
                 tau_memory: float = 4000.0, tau_charge: float | None = None,
                 n_inh: int = 256, k_ei: int = 100, ei_row: float = 0.65,
                 ie_gain: float = 0.52, seed: int = 0):
        self.enc = enc
        self.n23 = enc.n_out
        self.n5 = n5
        self.tau_memory = tau_memory
        self.tau_charge = tau_charge
        rng = np.random.default_rng(seed + 7717)

        # L5 readout: sparse fan-in, signed, each row L1-normalised
        idx = np.argsort(rng.random((n5, self.n23)), axis=1)[:, :k5]
        sign = rng.choice([-1.0, 1.0], size=(n5, k5))
        self.W5 = np.zeros((n5, self.n23))
        rows = np.repeat(np.arange(n5), k5)
        self.W5[rows, idx.ravel()] = sign.ravel() / k5     # L1 = 1
        self.theta5_q = theta5_q
        self.theta5 = None

        # inhibitory pool
        idx_i = np.argsort(rng.random((n_inh, self.n23)), axis=1)[:, :k_ei]
        self.W_ei = np.zeros((n_inh, self.n23))
        rows_i = np.repeat(np.arange(n_inh), k_ei)
        self.W_ei[rows_i, idx_i.ravel()] = ei_row / k_ei   # row sum = 0.65
        self.inh_threshold = rng.uniform(0.05, 0.60, size=n_inh)
        self.n_inh = n_inh
        self.ie_gain = ie_gain

        self.m = np.zeros(self.n23)
        self.s = np.zeros(self.n23)
        self.s5 = np.zeros(n5)
        self.inh = 0.0
        self.inh_cells = np.zeros(n_inh)

    # -- fitting -------------------------------------------------------------
    def l5_scores(self) -> np.ndarray:
        """Activity-normalised L5 drive for the current L2/3 state."""
        return self._l5_score(self.s)

    def calibrate_l5(self, score_vectors: list[np.ndarray], q: float | None = None) -> None:
        """Set the L5 threshold from measured readout-time score vectors."""
        q = self.theta5_q if q is None else q
        S = np.stack(score_vectors).T                    # (n5, n_samples)
        self.theta5 = np.quantile(S, q, axis=1)

    def _l5_score(self, s23: np.ndarray) -> np.ndarray:
        act = float(s23.sum())
        # divide-by-activity normalisation (documents: "驱动做除法归一化")
        scale = (self.enc.code_len / act) if act > 0 else 0.0
        return (self.W5 @ s23) * scale

    # -- dynamics ------------------------------------------------------------
    def reset(self) -> None:
        self.m[:] = 0.0
        self.s[:] = 0.0
        self.s5[:] = 0.0
        self.inh = 0.0
        self.inh_cells[:] = 0.0

    def _inhibition(self) -> float:
        cur = self.W_ei @ self.s
        cells = (cur > self.inh_threshold).astype(np.float64)
        self.inh_cells = cells
        # I -> E: normalised so a fully active inhibitory pool contributes 0.52
        return float(self.ie_gain * cells.mean() / 0.5)

    def _activity(self) -> np.ndarray:
        s = (self.m > 1.0 + self.inh).astype(np.float64)
        self.s = s
        self.inh = self._inhibition()
        return s

    def present(self, a: np.ndarray, steps: int = 20) -> np.ndarray:
        """Inject the normalised drive for `steps` steps and return L2/3 code."""
        for _ in range(steps):
            if self.tau_charge is None:
                self.m = np.maximum(self.m * (1.0 - 1.0 / self.tau_memory), a)
            else:
                self.m += (a - self.m) / self.tau_charge
            self._activity()
        return self.s

    def hold_step(self) -> np.ndarray:
        self.m *= (1.0 - 1.0 / self.tau_memory)
        return self._activity()

    def hold(self, steps: int) -> list[np.ndarray]:
        out = []
        for _ in range(steps):
            out.append(self.hold_step().copy())
        return out

    def read_l5(self) -> np.ndarray:
        score = self._l5_score(self.s)
        self.s5 = (score > self.theta5).astype(np.float64)
        return self.s5

    @property
    def density(self) -> float:
        return float(self.s.mean())
