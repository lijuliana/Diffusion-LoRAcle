"""Featurizers for the POC-1 head-to-head: every method maps a LoRA (its per-module (B,A) factors)
to a flat feature vector, behind one interface, so a linear probe can compare them fairly.

  * SpectralStatFeaturizer  — the existing weight-only baseline to BEAT (binary backdoor detectors
                              use exactly these σ-statistics). Direction-blind.
  * OurSVDFeaturizer        — the design-doc encoding: invariant SVD-direction signature.
  * W2TFeaturizer           — closest-method comparison: QR->SVD tokens, mean-pooled (no sign/
                              degeneracy canonicalization beyond the SVD).
  * RawABFeaturizer         — gauge-variant control: flatten the raw factors.
  * RankLeakFeaturizer      — the LEAKAGE CONTROL (rank-only / rank+module-pattern). Must be
                              near-chance; if it predicts the label, the benchmark leaks rank.

FIXED GLOBAL DIMENSIONALITY (anti-leakage, design doc §B.7.1c)
--------------------------------------------------------------
Every featurizer takes a FIXED `modules` list and a FIXED `top_k`, and returns a vector of the SAME
length for EVERY adapter regardless of its rank or which modules it actually populates. Missing
modules and ranks beyond an adapter's own rank are zero-filled, and a parallel **mask** marks which
entries are real. Crucially:
  * feature LENGTH never encodes rank or token count (it is constant);
  * the mask is available to the probe ONLY if explicitly requested, and is identical in structure
    across adapters of the same rank — we test rank leakage explicitly with RankLeakFeaturizer.
This removes the v1 bug where ragged "pad to max in this call" let a probe exploit feature length /
batch-dependent dimensionality as a rank shortcut.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ditloracle.encoding.svd_encoder import compact_svd_from_factors, encode_module, invariant_signature

Tensor = torch.Tensor
LoRAModules = dict[str, tuple]  # name -> (B, A, alpha, r, use_rslora)


class Featurizer(Protocol):
    name: str
    out_dim: int

    def features(self, lora: LoRAModules) -> Tensor: ...


def _scale(alpha, r, use_rslora):
    if alpha is None:
        return 1.0
    return alpha / (r ** 0.5) if use_rslora else alpha / r


def _module_dims(lora: LoRAModules, name: str) -> tuple[int, int] | None:
    """(d_out, d_in) of a module if present, else None."""
    if name not in lora:
        return None
    B, A, *_ = lora[name]
    return B.shape[0], A.shape[1]


# --------------------------------------------------------------------------------------
# Fixed-dimension featurizers. Each is constructed with a fixed module list and a global
# d_out/d_in per module (taken from the corpus), so output length is constant for all adapters.
# --------------------------------------------------------------------------------------
class _FixedBase:
    """Holds the fixed schema: sorted module list + per-module (d_out, d_in) + top_k.

    Subclasses implement `module_vec(name, lora) -> 1-D Tensor` (the feature block for ONE module of
    ONE adapter). `features()` is the concatenation (used by tests / small featurizers); `gram()` is
    the MEMORY-SAFE path for the probe — it accumulates the n×n linear-kernel Gram one module-block at
    a time, so the full ~2.9M-wide feature matrix is never materialized (a prior run OOM-crashed by
    stacking it). The Gram is EXACT (no projection/JL approximation): ⟨x_i,x_j⟩ = Σ_modules ⟨block⟩.
    """

    def __init__(self, modules: list[str], dims: dict[str, tuple[int, int]], top_k: int):
        self.modules = sorted(modules)
        self.dims = dims                  # name -> (d_out, d_in), the GLOBAL fixed dims
        self.top_k = top_k

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def features(self, lora: LoRAModules) -> Tensor:
        return torch.cat([self.module_vec(m, lora) for m in self.modules])

    def gram(self, loras: list[LoRAModules], standardize_blocks: bool = True) -> "object":
        """n×n exact linear-kernel Gram, accumulated block-by-block (bounded memory).

        FAIRNESS CONTROL (standardize_blocks=True, default): per-module feature scales are wildly
        heterogeneous (e.g. spectral σ₁ vs entropy vs kurtosis; raw B vs A magnitudes). Without
        scaling, a few high-magnitude coordinates dominate the linear kernel, so a featurizer could
        "win" on a scaling artifact rather than semantics. We z-score each module block across the
        corpus (column-standardize), then accumulate. This makes every coordinate contribute
        comparably, so cross-featurizer comparisons reflect content, not scale. (The probe then also
        double-centers the Gram — featurizers.gram is per-feature scale; centering is per-sample.)
        """
        import numpy as np
        n = len(loras)
        G = np.zeros((n, n), dtype=np.float64)
        for name in self.modules:
            block = np.stack([self.module_vec(name, l).numpy() for l in loras]).astype(np.float64)
            if standardize_blocks:
                mu = block.mean(axis=0, keepdims=True)
                sd = block.std(axis=0, keepdims=True)
                sd[sd < 1e-12] = 1.0           # leave constant (uninformative) columns at 0 after centering
                block = (block - mu) / sd
            G += block @ block.T
            del block
        return G


class NormOnlyFeaturizer(_FixedBase):
    """Per-module ‖ΔW‖_F ONLY (one scalar/module) — NO directions, NO spectral shape.

    Isolates the 'a strong concept moves the weights more' effect: ΔW-norm is partly real semantic
    signal, so this is a fair in-lineup baseline (NOT a leakage control). The gate requires our_svd to
    BEAT this — proving the reader uses direction structure, not just overall magnitude (the A2
    decision: report norm separately, don't bury it in a leakage control)."""

    name = "norm_only"

    @property
    def out_dim(self):
        return len(self.modules)

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        if name in lora:
            B, A, alpha, r, rs = lora[name]
            _, S, _ = compact_svd_from_factors(B, A, scale=_scale(alpha, r, rs))
            return torch.linalg.vector_norm(S).reshape(1).to(torch.float64)   # ‖ΔW‖_F = ‖σ‖₂
        return torch.zeros(1, dtype=torch.float64)


class SpectralStatFeaturizer(_FixedBase):
    """Per-module {σ₁, ‖ΔW‖_F, energy concentration, spectral entropy, kurtosis} → fixed vector.

    5 stats × n_modules, zero-filled for missing modules. Direction-blind by construction (only the
    spectrum), so beating it shows directions carry signal. NB: we feed the top_k spectrum so its
    dimensionality matches across ranks (pad σ with zeros to top_k before the stats)."""

    name = "spectral_stat"

    @property
    def out_dim(self):
        return 5 * len(self.modules)

    def _stats(self, S: Tensor) -> Tensor:
        S = S.to(torch.float64)
        # pad/truncate to top_k so every module contributes the same-shaped spectrum
        if S.numel() < self.top_k:
            S = torch.cat([S, torch.zeros(self.top_k - S.numel(), dtype=torch.float64)])
        else:
            S = S[: self.top_k]
        energy = S ** 2
        total = energy.sum().clamp_min(1e-30)
        p = energy / total
        sigma1 = S.max()
        frob = energy.sum().sqrt()
        concentration = sigma1 ** 2 / total
        entropy = -(p * (p + 1e-30).log()).sum()
        mean, std = S.mean(), S.std().clamp_min(1e-30)
        kurtosis = (((S - mean) / std) ** 4).mean()
        return torch.stack([sigma1, frob, concentration, entropy, kurtosis])

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        if name in lora:
            B, A, alpha, r, rs = lora[name]
            _, S, _ = compact_svd_from_factors(B, A, scale=_scale(alpha, r, rs))
            return self._stats(S)
        return torch.zeros(5, dtype=torch.float64)


class OurSVDFeaturizer(_FixedBase):
    """Invariant SVD signature, FIXED top_k directions per module, zero-masked beyond an adapter's
    own rank. Output length constant for all adapters (no rank leakage via length)."""

    name = "our_svd"

    def __init__(self, modules, dims, top_k, degeneracy_safe: bool = True):
        super().__init__(modules, dims, top_k)
        self.degeneracy_safe = degeneracy_safe

    @property
    def out_dim(self):
        # per module: top_k σ + top_k directions each of (d_out + d_in)
        return sum(self.top_k + self.top_k * (do + di) for (do, di) in
                   (self.dims[m] for m in self.modules))

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        do, di = self.dims[name]
        sig_dim = self.top_k + self.top_k * (do + di)
        if name not in lora:
            return torch.zeros(sig_dim, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
        sigma = enc.sigma / enc.frob if enc.frob > 0 else enc.sigma
        sigma = _pad(sigma, self.top_k)
        feats = [sigma]
        for i in range(self.top_k):
            if i < enc.k:
                feats.append(_pad(enc.U[:, i].to(torch.float64), do))   # sign-canonical
                feats.append(_pad(enc.V[:, i].to(torch.float64), di))
            else:
                feats.append(torch.zeros(do, dtype=torch.float64))
                feats.append(torch.zeros(di, dtype=torch.float64))
        return torch.cat(feats)


class ProductSketchFeaturizer(_FixedBase):
    """The PRINCIPLED GL-invariant baseline the lit review (Putterman et al. `2410.04207`, GL-net)
    says is the right feature and which the rest of the lineup was missing: the *product* ΔW = UΣVᵀ,
    not the separate U/V coordinates.

    We read ΔW through a FIXED deterministic bilinear random sketch  Sketch = Rᵒᵘᵗᵀ · ΔW · Rⁱⁿ
    (Rᵒᵘᵗ: d_out×p, Rⁱⁿ: d_in×q, seeded per module so every adapter uses the SAME projection). Two
    properties make this the featurizer to beat:
      * It is a *linear function of ΔW*, so it is EXACTLY GL(r)-gauge- and coupled-sign-invariant by
        construction — no canonicalization step to get wrong (our_svd earns invariance via sign-fixing;
        this earns it for free, so it isolates whether canonicalization is what's doing the work).
      * Its output dimension (p·q) is independent of rank — no zero-padding of directions, so it cannot
        leak rank through feature length any more than the others.
    Computed WITHOUT forming the dense d_out×d_in product:  Rᵒᵘᵗᵀ ΔW Rⁱⁿ = (RᵒᵘᵗᵀU) diag(σ) (VᵀRⁱⁿ).
    """

    name = "product_sketch"

    def __init__(self, modules, dims, top_k, p: int = 24, q: int = 24):
        super().__init__(modules, dims, top_k)
        self.p, self.q = p, q
        self._proj: dict[str, tuple[Tensor, Tensor]] = {}

    def _projectors(self, name: str) -> tuple[Tensor, Tensor]:
        if name not in self._proj:
            do, di = self.dims[name]
            # deterministic per-module seed (stable hash of the name) so the sketch is identical for
            # every adapter and reproducible across runs — a FIXED projection, not a learned one.
            seed = int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") % (2**31)
            g = torch.Generator().manual_seed(seed)
            r_out = torch.randn(do, self.p, generator=g, dtype=torch.float64) / (do ** 0.5)
            r_in = torch.randn(di, self.q, generator=g, dtype=torch.float64) / (di ** 0.5)
            self._proj[name] = (r_out, r_in)
        return self._proj[name]

    @property
    def out_dim(self):
        return len(self.modules) * self.p * self.q

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        if name not in lora:
            return torch.zeros(self.p * self.q, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        U, S, V = compact_svd_from_factors(B, A, scale=_scale(alpha, r, rs))
        r_out, r_in = self._projectors(name)
        left = r_out.transpose(0, 1) @ U.to(torch.float64)      # (p, k)
        right = V.to(torch.float64).transpose(0, 1) @ r_in      # (k, q)
        sketch = (left * S.to(torch.float64)) @ right           # (p, q) = Rᵒᵘᵗᵀ ΔW Rⁱⁿ
        return sketch.reshape(-1)


class W2TFeaturizer(_FixedBase):
    """W2T-style: per-direction token (uᵢ ‖ vᵢ ‖ σᵢ) mean-pooled over directions, per module.
    Fixed length (d_out + d_in + 1) per module — no canonicalization beyond the raw SVD."""

    name = "w2t_svd"

    @property
    def out_dim(self):
        return sum(do + di + 1 for (do, di) in (self.dims[m] for m in self.modules))

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        do, di = self.dims[name]
        if name in lora:
            B, A, alpha, r, rs = lora[name]
            U, S, V = compact_svd_from_factors(B, A, scale=_scale(alpha, r, rs))
            u_mean = _pad(U.mean(dim=1).to(torch.float64), do)
            v_mean = _pad(V.mean(dim=1).to(torch.float64), di)
            s_mean = S.to(torch.float64).mean().reshape(1)
            return torch.cat([u_mean, v_mean, s_mean])
        return torch.zeros(do + di + 1, dtype=torch.float64)


class RawABFeaturizer(_FixedBase):
    """GAUGE-VARIANT control: rank-pooled raw factors, fixed (d_out + d_in) per module."""

    name = "raw_ab"

    @property
    def out_dim(self):
        return sum(do + di for (do, di) in (self.dims[m] for m in self.modules))

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        do, di = self.dims[name]
        if name in lora:
            B, A, alpha, r, rs = lora[name]
            s = _scale(alpha, r, rs)
            return torch.cat([
                _pad((s * B).mean(dim=1).to(torch.float64), do),
                _pad((s * A).mean(dim=0).to(torch.float64), di),
            ])
        return torch.zeros(do + di, dtype=torch.float64)


class RankLeakFeaturizer(_FixedBase):
    """LEAKAGE CONTROL (design doc §B.7.1c): features derived ONLY from rank + which-modules-present
    (the recipe fingerprint), carrying NO semantic direction/spectrum content. If a probe on this
    beats chance on a semantic label, the benchmark leaks rank/recipe and the split is invalid."""

    name = "rank_leak"

    @property
    def out_dim(self):
        return len(self.modules) + 1   # presence bit per module + global rank scalar

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        # presence bit for THIS module; the global-rank scalar is attached to the first module only
        present = 1.0 if name in lora else 0.0
        if name == self.modules[0]:
            ranks = [lora[m][3] for m in self.modules if m in lora]
            global_rank = float(max(ranks)) if ranks else 0.0
            return torch.tensor([present, global_rank], dtype=torch.float64)
        return torch.tensor([present], dtype=torch.float64)

    def features(self, lora: LoRAModules) -> Tensor:
        return torch.cat([self.module_vec(m, lora) for m in self.modules])


def _pad(v: Tensor, n: int) -> Tensor:
    """Pad/truncate a 1-D tensor to length n (fixed dimensionality, no batch dependence)."""
    if v.numel() == n:
        return v
    if v.numel() > n:
        return v[:n]
    return torch.cat([v, torch.zeros(n - v.numel(), dtype=v.dtype)])


def build_fixed_schema(loras: list[LoRAModules], top_k: int) -> dict[str, tuple[int, int]]:
    """Compute the GLOBAL per-module (d_out, d_in) over a corpus, so every featurizer uses one fixed
    layout. Modules are the union across the corpus; dims are the max seen (consistent for a fixed
    base model). Returns {module_name: (d_out, d_in)}."""
    dims: dict[str, tuple[int, int]] = {}
    for lora in loras:
        for name, (B, A, *_) in lora.items():
            do, di = B.shape[0], A.shape[1]
            if name in dims:
                dims[name] = (max(dims[name][0], do), max(dims[name][1], di))
            else:
                dims[name] = (do, di)
    return dims
