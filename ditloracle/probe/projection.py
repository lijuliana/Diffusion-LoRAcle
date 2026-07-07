"""Memory-safe dimensionality reduction — KEPT FOR REFERENCE / SCALE, not used in the POC-1 probe.

History + an important rigor note (so a false generalization does NOT propagate):
We first tried signed feature hashing (Weinberger et al., 2009) to shrink the ~2.9M-dim features.
It produced garbage inner products (true ≈ −1980 vs estimate ≈ −83333). **The reason is specific to
feature hashing, NOT to linear projection in general:** the hashing-trick inner-product estimator has
variance ∝ ‖x‖²‖y‖²/k, and here ‖x‖² ≈ 8e4 while the *meaningful* inner products are tiny relative to
the norms (catastrophic-cancellation regime) — so its std (~3700) dwarfs the signal. A *proper dense
Gaussian Johnson–Lindenstrauss* projection would NOT blow up: JL distortion is ε ≈ √(log n / k),
≈ 10% at n=441, k=512, and is **independent of the ambient dimension D**. So random projections remain
a valid tool at scale (e.g. on the 42K corpus); only *feature hashing in this norm regime* was unusable.

For POC-1 we instead use the EXACT block-wise Gram (probe/featurizers.py `gram()`) — at n≈441 the n×n
kernel is cheap and exactness is free, so there is no reason to approximate. This module is retained
only as a (correctly-caveated) building block for future large-scale use.

Usage: one Projector per (featurizer, D); call .project(vec_1d) per adapter. (Note the variance caveat
above before using it where vector norms greatly exceed the meaningful inner products.)
"""

from __future__ import annotations

import numpy as np


class HashingProjector:
    def __init__(self, in_dim: int, out_dim: int = 512, seed: int = 0):
        self.in_dim = in_dim
        self.out_dim = min(out_dim, in_dim)   # no point projecting up
        self.passthrough = in_dim <= out_dim
        if not self.passthrough:
            rng = np.random.default_rng(seed)
            # one bucket and one ±1 sign per input coordinate (computed once)
            self.buckets = rng.integers(0, self.out_dim, size=in_dim, dtype=np.int32)
            self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=in_dim)

    def project(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).ravel()
        if self.passthrough:
            out = np.zeros(self.out_dim, dtype=np.float32)
            out[: v.shape[0]] = v[: self.out_dim]
            return out
        # signed sum into buckets — O(D), output O(k); bincount is the fast vectorized form
        return np.bincount(self.buckets, weights=self.signs * v, minlength=self.out_dim).astype(np.float32)
