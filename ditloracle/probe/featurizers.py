"""Featurizers for the POC-1 head-to-head: every method maps a LoRA (its per-module (B,A) factors)
to a flat feature vector, behind one interface, so a linear probe can compare them fairly.

  * SpectralStatFeaturizer  — the existing weight-only baseline to BEAT (binary backdoor detectors
                              use exactly these σ-statistics). Direction-blind.
  * U1LogRegFeaturizer      — the NEAREST competing paper (`2607.25750`, u₁ + logistic regression,
                              detection-only). Top-left singular direction per module, nothing else.
  * OurSVDFeaturizer        — the design-doc encoding: invariant SVD-direction signature.
  * SubspaceProjFeaturizer  — the ENCODER THAT ACTUALLY WORKS (POC-M, 2026-08-23): basis-invariant
                              subspace projectors instead of individual singular vectors.
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

from ditloracle.encoding.svd_encoder import (
    SIGMA_FLOOR_REL,
    compact_svd_from_factors,
    degeneracy_clusters,
    encode_module,
    invariant_signature,
    subspace_projector_diag,
    usable_direction_mask,
)
from ditloracle.formats.fused_split import is_fused_shape

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
        # PLAN §7.3: a fused module must never be featurized. Its ΔW stacks q/k/v (and mlp) into one
        # matrix, so its SVD mixes three different projections and every direction feature is a
        # blend — and 72% of real adapters are fused, so this fails quietly and corpus-wide. The
        # schema is the choke point every featurizer and every load path passes through.
        fused = [m for m in self.modules
                 if m in self.dims and is_fused_shape(m, *self.dims[m])]
        if fused:
            raise ValueError(
                f"fused modules reached a featurizer schema: {fused[:3]} — load factors through "
                f"formats.safetensors_io.load_canonical_factors, which applies the qkv/mlp split"
            )

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


SIGN_RULES = ("max_abs", "bro")


def canonicalize_u1_sign(u1: Tensor, rule: str = "bro") -> Tensor:
    """Fix the sign gauge of a LONE left singular vector (the u₁ baseline discards v₁).

    u₁ and −u₁ are both valid SVD outputs and LAPACK's choice is arbitrary, so an uncanonicalized u₁
    feature is sign-random across adapters and the logistic regression sees noise. Unlike
    `encoding.canonicalize_signs`, we cannot pivot on the coupled [u;v] vector — this method throws v
    away — so the rule has to be a function of u₁ alone. Two are offered, and BOTH are exact
    involutions (u and −u map to the same output), which is all the featurizer contract requires:

      * "bro" (DEFAULT) — flip so Σᵢ sign(uᵢ)·uᵢ² > 0, the standard fix from Bro, Acar & Kolda (2008),
        "Resolving the sign ambiguity in the singular value decomposition". Aggregates over EVERY
        coordinate, so no single entry can flip the block.
      * "max_abs" — flip so the largest-|value| entry is positive. Well defined (a unit vector's
        largest entry has magnitude ≥ 1/√d > 0) and argmax breaks exact ties, but it rests the whole
        d_out block on ONE coordinate.

    WHY THE DEFAULT IS "bro", AND WHY THIS IS A FAIRNESS ISSUE RATHER THAN A DETAIL. "max_abs" is the
    obvious rule, but it is measurably a STRAWMAN for this baseline. When the top two |entries| of u₁
    near-tie, ordinary noise moves the pivot to a different coordinate and flips the sign of the whole
    block. On the real minted klein organisms this is not a corner case: the relative gap between the
    largest and second-largest |entry| of u₁ has median 5.5%, and 24% of modules sit under 2%. The
    consequence, measured on the POC-1c clamped-recipe concept axis (n=17, 60 modules, d_out=3072):

        u1_logreg, sign_rule="max_abs"   mAP 0.358   p=0.081   (indistinguishable from chance)
        u1_logreg, sign_rule="bro"       mAP 0.703   p=0.0005  (clearly above chance)

    Same feature, same data — the sign convention alone is worth 0.35 mAP. Reporting the baseline at
    0.358 would understate the nearest competing paper by nearly a factor of two and hand a reviewer
    an easy rebuttal, so we default to the convention that gives it its strongest honest showing and
    keep "max_abs" available to report the sensitivity. Any margin we claim over this baseline must be
    demonstrated against the "bro" number.
    """
    if u1.numel() == 0:
        return u1
    if rule == "max_abs":
        pivot = u1[int(u1.abs().argmax())]
    elif rule == "bro":
        pivot = (torch.sign(u1) * u1 ** 2).sum()
    else:
        raise ValueError(f"unknown sign rule {rule!r}; expected one of {SIGN_RULES}")
    return -u1 if float(pivot) < 0 else u1


def top_left_direction(U: Tensor, S: Tensor, degeneracy_safe: bool = True,
                       rel_tol: float = 1e-3, sign_rule: str = "bro") -> Tensor:
    """u₁ (sign-canonicalized) for the `2607.25750` baseline, with the siblings' degeneracy guard.

    When σ₁ is SIMPLE this is just the sign-fixed first column of U. When σ₁ sits in a degenerate
    cluster of size m>1 (`degeneracy_clusters`), the individual u₁ is only defined up to an O(m)
    rotation of the top subspace, so reading it as a fingerprint is meaningless; with
    `degeneracy_safe` we substitute the O(m)-invariant descriptor the encoder uses,
    `subspace_projector_diag`, elementwise-sqrt'd and divided by √m. That descriptor has the same unit
    ℓ₂ norm and per-coordinate scale as u₁ and reduces EXACTLY to |u₁| when m=1, so the fixed feature
    block stays on one scale for the linear probe. A numerically zero ΔW yields zeros (its U is
    arbitrary QR fill, not a direction).
    """
    U = U.to(torch.float64)
    S = S.to(torch.float64)
    if S.numel() == 0 or float(S.max()) <= 1e-30:
        return torch.zeros(U.shape[0], dtype=torch.float64)
    if degeneracy_safe:
        top = degeneracy_clusters(S, rel_tol=rel_tol)[0]      # cluster containing σ₁
        if len(top) > 1:
            diag = subspace_projector_diag(U, top).clamp_min(0.0)   # sums to m
            return diag.sqrt() / (len(top) ** 0.5)
    return canonicalize_u1_sign(U[:, 0], rule=sign_rule)


class U1LogRegFeaturizer(_FixedBase):
    """NEAREST-PAPER BASELINE — the u₁ fingerprint of `arXiv 2607.25750` ("Detecting CSAM
    Text-to-Image LoRAs From Weights", Africa et al., UKAISI). This is the method PLAN §2/§8 says we
    must beat in Figs 3/4 and specifically on the matched-spectra control.

    THE METHOD (implemented faithfully, NOT tuned to lose). For each adapted module take
    ΔW = (α/r)·B·A and keep ONLY its top-left singular direction u₁ — the left singular vector of the
    largest singular value — as that module's fingerprint; concatenate over the FIXED module schema
    into a constant-length vector; a logistic regression on top does binary malicious-vs-benign
    detection (scored by `ditloracle.probe.detection`). Nothing else about ΔW survives: not v₁, not
    the spectrum, not directions 2..r.

    WHY WE CARRY IT AS A FOIL. It is DETECTION-ONLY by construction. One left direction per module
    supports a yes/no score and nothing more: it never describes a payload and never recovers a
    trigger. That structural ceiling — not a number — is the separator the paper leads with, so this
    baseline is expected to be COMPETITIVE on detection and is here to be beaten on the matched-
    spectra control, where the spectrum is clamped and only direction content can separate.

    INVARIANCE — READ THIS BEFORE FILING IT AS "THE GAUGE-VARIANT CONTROL". u₁ is computed from the
    PRODUCT ΔW = B·A, and ΔW is exactly GL(r)-gauge-invariant ((B,A) ↦ (BG⁻¹, GA) leaves it
    unchanged), so u₁ INHERITS that invariance — up to the two ambiguities below. The gauge-variant
    control is RawABFeaturizer, not this. The honest statement about u₁ is that it is invariant but
    LOSSY. What it genuinely is not invariant to:
      * the coupled-sign gauge (u₁,v₁) ↦ (−u₁,−v₁) — we FIX this (`canonicalize_u1_sign`), and WHICH
        sign rule is used is worth 0.35 mAP on real organisms, so read that docstring before quoting
        any number from this class;
      * O(m) rotation inside a DEGENERATE top block (σ₁≈σ₂), where u₁ is not a well-defined object at
        all (`top_left_direction`, degeneracy_safe).
    A u₁ read off the raw up-projection B alone WOULD be gauge-variant (G⁻¹ mixes B's columns
    non-orthogonally); we do not do that, because the paper's u₁ is a singular direction of the diff.
    We deliberately do NOT strengthen the baseline past this — no v₁, no σ, no top-k subspace. The
    lossiness IS the method, and the one-direction bottleneck is the thing our reader has to beat.
    (Measured on the POC-1c clamped-recipe concept axis, 25 real klein organisms: u₁ reaches mAP 0.703
    — ABOVE our_svd's 0.479 and spectral_stat's 0.539, below product_sketch 0.984 and subspace_proj
    1.0. This baseline is a live competitor on our own corpus, not a formality.)

    CROSS-BASE (FLUX.1-dev and FLUX.2-klein), the generalization the paper reports. The class is
    module-NAME-agnostic: it reads whatever `modules`/`dims` schema it is handed, so it works on
    FLUX.1 canonical names and on the FLUX.2-klein raw key stems that
    `formats.safetensors_io.load_canonical_factors` falls back to (the minted corpus is klein). To
    transfer a fitted logistic regression ACROSS bases the two corpora must share one feature layout:
    build ONE `build_fixed_schema` over their union and hand it to both instances — modules a base
    lacks are zero-filled by the standard missing-module path. Both bases are width 3072, so the
    per-module block widths line up.
    """

    name = "u1_logreg"

    def __init__(self, modules, dims, top_k, degeneracy_safe: bool = True,
                 sign_rule: str = "bro"):
        super().__init__(modules, dims, top_k)
        self.degeneracy_safe = degeneracy_safe
        if sign_rule not in SIGN_RULES:
            raise ValueError(f"unknown sign rule {sign_rule!r}; expected one of {SIGN_RULES}")
        # "bro" by default: "max_abs" costs this baseline 0.35 mAP on real klein organisms and would
        # make our comparison a strawman. See canonicalize_u1_sign for the measurement.
        self.sign_rule = sign_rule

    @property
    def out_dim(self):
        # one d_out-long direction per module; independent of top_k and of any adapter's rank
        return sum(do for (do, _) in (self.dims[m] for m in self.modules))

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        do, _ = self.dims[name]
        if name not in lora:
            return torch.zeros(do, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        # α/r (or α/√r) folded in before the SVD, exactly as the siblings do. It cannot change u₁'s
        # direction (a positive scalar), but it keeps this baseline on the same convention as the rest
        # of the lineup so the comparison is like-for-like.
        U, S, _ = compact_svd_from_factors(B, A, scale=_scale(alpha, r, rs))
        u1 = top_left_direction(U, S, degeneracy_safe=self.degeneracy_safe,
                                sign_rule=self.sign_rule)
        return _pad(u1, do)


class OurSVDFeaturizer(_FixedBase):
    """Invariant SVD signature, FIXED top_k directions per module, zero-masked beyond an adapter's
    own rank. Output length constant for all adapters (no rank leakage via length).

    Two guards the fixed layout has to carry itself (it cannot call `invariant_signature`, whose
    length varies with the adapter's own rank):
      * `degeneracy_safe` — inside a degenerate cluster (σᵢ≈σⱼ) the individual uᵢ,vᵢ are defined only
        up to an O(m) rotation, so every member gets the shared, O(m)-invariant projector diagonal
        instead (design doc §B.4; poc0d found 277 near-degenerate directions on real adapters). This
        flag was previously stored and never applied, which silently made the featurizer NOT
        degeneracy-safe while `encoding.invariant_signature` was.
      * σ floor — directions below `SIGMA_FLOOR_REL·σ₁` are numerical noise (worst measured σ-gap
        5e-6) and contribute zeros rather than an arbitrary vector.
    """

    name = "our_svd"

    def __init__(self, modules, dims, top_k, degeneracy_safe: bool = True,
                 sigma_floor_rel: float = SIGMA_FLOOR_REL):
        super().__init__(modules, dims, top_k)
        self.degeneracy_safe = degeneracy_safe
        self.sigma_floor_rel = sigma_floor_rel

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
        keep = usable_direction_mask(enc.sigma, self.sigma_floor_rel)
        # direction index -> the cluster it shares (only degenerate ones need the projector)
        degenerate = ({i: tuple(c) for c in enc.clusters if len(c) > 1 for i in c}
                      if self.degeneracy_safe else {})
        cache: dict[tuple[int, ...], tuple[Tensor, Tensor]] = {}
        feats = [sigma]
        for i in range(self.top_k):
            if i >= enc.k or not bool(keep[i]):
                feats.append(torch.zeros(do, dtype=torch.float64))
                feats.append(torch.zeros(di, dtype=torch.float64))
            elif i in degenerate:
                key = degenerate[i]
                if key not in cache:
                    cache[key] = (subspace_projector_diag(enc.U, list(key)).to(torch.float64),
                                  subspace_projector_diag(enc.V, list(key)).to(torch.float64))
                u_diag, v_diag = cache[key]
                feats.append(_pad(u_diag, do))
                feats.append(_pad(v_diag, di))
            else:
                feats.append(_pad(enc.U[:, i].to(torch.float64), do))   # sign-canonical
                feats.append(_pad(enc.V[:, i].to(torch.float64), di))
        return torch.cat(feats)


class SubspaceProjFeaturizer(_FixedBase):
    """Per-module rank-k SUBSPACE projector diagonals — diag(U_k U_kᵀ) ‖ diag(V_k V_kᵀ).

    This is the featurizer the POC-M gate selected on real organisms (2026-08-23), and it exists
    because `OurSVDFeaturizer` was measured to be the *worst* of the real featurizers on clean minted
    data (concept mAP 0.489 vs 1.000 here, and 0.624 for the spectrum alone — i.e. its singular-vector
    components were actively injecting noise).

    Why individual uᵢ, vᵢ are the wrong object, and a projector is the right one:
      * CONDITIONING, not gauge, is the binding constraint. ΔW = B·A is already exactly GL(r)-invariant,
        so any function of ΔW gets gauge-invariance free (see ProductSketchFeaturizer). What canonical-
        ization cannot buy is stability: by Davis-Kahan/Wedin the sensitivity of an individual singular
        vector scales like 1/gap(σᵢ). Measured on our own klein adapters (3,600 adjacent pairs, σ
        normalized by σ₁): median gap 6.6e-3, **59.2% of gaps < 1e-2**, 16.2% < 1e-3, min 7.5e-06. A majority of
        half the retained directions are therefore numerically arbitrary.
      * A projector onto the retained subspace is invariant to ANY orthogonal rotation within that
        subspace — exactly the ambiguity that makes uᵢ unstable — so it is well-defined even where the
        individual vectors are not. The sin-θ theorems bound subspace perturbation by the gap to the
        *outside* of the block, not the gaps inside it.
      * SIGN HANDLING IS NOT THE ISSUE, measured. Design doc §B.5.3 blamed sign-variance and prescribed
        the sign-invariant per-direction product uᵢvᵢᵀ. Implemented and scored, that still only reaches
        0.528 (σ-weighted) / 0.574 (unweighted) on the concept axis. The binding problem is that ANY
        per-direction-indexed feature assumes "direction i" corresponds across adapters, and
        near-degenerate spectra break that correspondence. Pooling over the subspace removes the
        assumption entirely; that — not sign — is why this featurizer wins.
      * It is RANK-ROBUST in a way ΔW is not. ΔW conflates *which* subspace an adapter acts on with
        *how strongly*, and the magnitude moves with rank; the projector keeps the subspace and discards
        the magnitude. On the rank-invariance axis this is decisive: 0.957 (p=0.0045) vs the product
        sketch's 0.732 (p=0.026) — the only featurizer to clear p<0.01 on BOTH gate axes.

    Only the DIAGONAL of each projector is kept, so the output is (d_out + d_in) per module: independent
    of the adapter's rank, so rank cannot leak through feature length. Verified not to encode rank —
    it retrieves *rank* at 0.326 (p=0.67), below even the RankLeak control — and stable across
    top_k ∈ {2, 4, 8, 16} (concept mAP 1.000 at every k).
    """

    name = "subspace_proj"

    def __init__(self, modules, dims, top_k, sigma_floor_rel: float = SIGMA_FLOOR_REL):
        super().__init__(modules, dims, top_k)
        self.sigma_floor_rel = sigma_floor_rel

    @property
    def out_dim(self):
        return sum(do + di for (do, di) in (self.dims[m] for m in self.modules))

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        do, di = self.dims[name]
        if name not in lora:
            return torch.zeros(do + di, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
        # keep only directions above the sigma floor: below it the direction is numerical noise and
        # would contribute an arbitrary vector to the subspace (design doc §B.12.1).
        keep = usable_direction_mask(enc.sigma, self.sigma_floor_rel)
        idx = [i for i in range(min(self.top_k, enc.k)) if bool(keep[i])]
        if not idx:
            return torch.zeros(do + di, dtype=torch.float64)
        return torch.cat([
            _pad(subspace_projector_diag(enc.U, idx).to(torch.float64), do),
            _pad(subspace_projector_diag(enc.V, idx).to(torch.float64), di),
        ])


class AdaptiveDirectionFeaturizer(_FixedBase):
    """Keep a singular direction only where it is actually determined; summarise the rest as a subspace.

    This is the encoder the failure analysis points to, and it is a synthesis of two things that
    already work rather than a new idea. `2607.25750` fingerprints an adapter with u1 alone and gets
    real mileage from it; our top-k direction stack, which contains u1 plus seven more, does far worse.
    Measuring the gaps says why (32 adapters, 60 modules, sigma normalised by sigma1):

        gap        median    % below 1e-2
        s1->s2     0.503        0.0
        s2->s3     0.147        0.9
        s4->s5     0.040       11.0
        s8->s9     0.010       48.2

    The leading direction is ~18x better separated than the rest and is NEVER ill-conditioned, while by
    the eighth almost half are. u1 works because it is the one direction that is always well determined.
    Stacking it with directions 5-8 dilutes a reliable measurement with noise, which is why deleting the
    directions outright beat keeping them.

    So: walk down the spectrum and emit u_i, v_i only while the relative gap to the next singular value
    stays above `gap_floor`. At the first crowded gap, stop, and represent everything remaining as ONE
    subspace projector diagonal, which is invariant to how the crowded directions are rotated among
    themselves and so stays defined where they individually do not. The spectrum is always included; it
    is gauge-invariant and costs almost nothing.

    The result adapts per module: sharply-decaying modules contribute several directions, flat ones
    contribute u1 and a subspace summary. Unlike a fixed top-k it never emits a coordinate the data does
    not determine, and unlike u1 alone it does not throw the rest of the adapter away.
    """

    name = "adaptive_dir"

    def __init__(self, modules, dims, top_k, gap_floor: float = 0.02,
                 sigma_floor_rel: float = SIGMA_FLOOR_REL):
        super().__init__(modules, dims, top_k)
        self.gap_floor = gap_floor
        self.sigma_floor_rel = sigma_floor_rel

    @property
    def out_dim(self):
        # top_k sigma + top_k direction slots + one trailing subspace summary, per module
        return sum(self.top_k + (self.top_k + 1) * (do + di)
                   for (do, di) in (self.dims[m] for m in self.modules))

    def module_vec(self, name: str, lora: LoRAModules) -> Tensor:
        do, di = self.dims[name]
        width = self.top_k + (self.top_k + 1) * (do + di)
        if name not in lora:
            return torch.zeros(width, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
        s = enc.sigma.to(torch.float64)
        norm = s / enc.frob if enc.frob > 0 else s
        keep = usable_direction_mask(enc.sigma, self.sigma_floor_rel)
        k = min(self.top_k, enc.k)

        # how many leading directions are individually well determined
        s1 = float(s[0]) if s.numel() and float(s[0]) > 0 else 1.0
        n_sharp = 0
        for i in range(k):
            if not bool(keep[i]):
                break
            gap = (float(s[i]) - float(s[i + 1])) / s1 if i + 1 < enc.k else float("inf")
            n_sharp += 1
            if gap < self.gap_floor:
                break

        feats = [_pad(norm, self.top_k)]
        for i in range(self.top_k):
            if i < n_sharp:
                feats.append(_pad(enc.U[:, i].to(torch.float64), do))
                feats.append(_pad(enc.V[:, i].to(torch.float64), di))
            else:
                feats.append(torch.zeros(do, dtype=torch.float64))
                feats.append(torch.zeros(di, dtype=torch.float64))
        # everything from the first crowded gap onward, as one rotation-invariant summary
        tail = [i for i in range(n_sharp, k) if bool(keep[i])]
        if tail:
            feats.append(_pad(subspace_projector_diag(enc.U, tail).to(torch.float64), do))
            feats.append(_pad(subspace_projector_diag(enc.V, tail).to(torch.float64), di))
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
