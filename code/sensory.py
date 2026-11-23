"""Sensory stack, exactly as the reference implementation does it later on.

From local3d-similarity (THRESHOLD_ACTIVATION.md, threednet/model.py):

  * each layer is a 3D volume; every neuron has 4 inputs: itself plus three
    random near neighbours (dx,dy in {-1,0,1}, dz in {-2..2})
  * weights are fixed, not trained
  * a neuron is active iff

        score_i  =  sum_j w_ij x_j + b_i
        active_i =  score_i > theta * (|b_i| + sum_j |w_ij|)

    with ONE theta per layer, calibrated so that about 12 % of the cells are
    active.  No top-k, no sorting, no E/I loop -- the sparsity comes from each
    cell's own weight budget.

This replaces the random-connectivity LIF stage that kept either saturating or
going silent.
"""
import numpy as np
import torch as T

DTYPE = T.float32


def coords(dim):
    x, y, z = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), np.arange(dim[2]),
                          indexing='ij')
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


class Local3DStack:
    """fixed-weight, locally connected, threshold-sparse layers"""

    def __init__(self, dims, fan_in=4, alpha=0.01, bias_scale=0.5, seed=0, dev='cuda'):
        self.dims = [tuple(d) for d in dims]
        self.dev = dev
        rng = np.random.default_rng(seed)
        self.layers = []
        for l in range(1, len(self.dims)):
            prev, out = self.dims[l - 1], self.dims[l]
            oc = coords(out)
            n = len(oc)
            idx = np.zeros((n, fan_in), np.int64)
            w = np.zeros((n, fan_in), np.float32)
            zmap = (np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
                    if out[2] > 1 else np.zeros(n, int))
            sc = oc.copy(); sc[:, 2] = zmap
            sc = np.clip(sc, 0, np.array(prev) - 1)
            idx[:, 0] = np.ravel_multi_index(sc.T, prev)
            w[:, 0] = 1.0
            for j in range(1, fan_in):
                nb = sc.copy()
                nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
                nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
                nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
                idx[:, j] = np.ravel_multi_index(nb.T, prev)
                w[:, j] = (rng.random(n) * 0.5 + 0.5) * alpha
            b = (rng.random(n) * bias_scale).astype(np.float32)
            self.layers.append((
                T.as_tensor(idx, device=dev, dtype=T.long),
                T.as_tensor(w, device=dev, dtype=DTYPE),
                T.as_tensor(b, device=dev, dtype=DTYPE),
                float(np.abs(b).sum() / n + np.abs(w).sum() / n),  # mean weight budget
            ))
        self.theta = [0.9] * len(self.layers)

    def forward(self, code):
        """code: (B, n0) dense float -> list of sparse patterns per layer"""
        h = code
        out = [h]
        for l, (idx, w, b, budget) in enumerate(self.layers):
            s = (h[:, idx] * w.unsqueeze(0)).sum(-1) + b.unsqueeze(0)
            denom = (b.abs() + w.abs().sum(-1)).unsqueeze(0)   # per-cell weight budget
            h = (s > self.theta[l] * denom).to(DTYPE)
            out.append(h)
        return out

    def calibrate(self, code, target=0.12, lo=0.0, hi=2.0, iters=25):
        """pick one theta per layer so that the mean activation is ~target.

        The bracket is reset for every layer.  The earlier version reused the previous
        layer's converged bracket, so layers 2 and 3 started from a bracket that no
        longer contained their own solution and came out under-calibrated (about 10%
        instead of the requested 12%).
        """
        for l in range(len(self.layers)):
            lo_l, hi_l = lo, hi
            for _ in range(iters):
                mid = (lo_l + hi_l) / 2
                self.theta[l] = mid
                hs = self.forward(code)
                act = float(hs[l + 1].mean())
                if act > target:      # too many active -> theta must rise
                    lo_l = mid
                else:                 # too few active -> theta must fall
                    hi_l = mid
            self.theta[l] = (lo_l + hi_l) / 2
        return self.theta

