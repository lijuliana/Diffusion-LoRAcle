"""PLAN §7.2 — degeneracy + the σ floor, on the paths that actually consume directions.

`encoding/svd_encoder.py` has had the projector-diagonal fix since POC-0a, but the two CONSUMERS did
not use it: `OurSVDFeaturizer` stored `degeneracy_safe` and never applied it (it builds its own
fixed-length layout instead of calling `invariant_signature`), and nothing anywhere clamped the σ
floor the reader injection needs (design doc §B.12.1 gotcha 2: ``h ← h + ‖h‖·v/‖v‖`` divides by a
norm that goes to zero). poc0d found 277 near-degenerate directions on real adapters, worst σ-gap
5e-6, so both cases are live rather than hypothetical.
"""

from __future__ import annotations

import torch

from ditloracle.encoding.svd_encoder import (
    SIGMA_FLOOR_REL,
    compact_svd_from_factors,
    encode_module,
    injection_tokens,
    invariant_signature,
    usable_direction_mask,
)
from ditloracle.probe.featurizers import OurSVDFeaturizer, build_fixed_schema

torch.manual_seed(0)
DT = torch.float64
MOD = "m0"
TOPK = 8


def _planted(svals, d_out=40, d_in=40):
    """Factors (B, A) with B = U Σ^{1/2}, A = Σ^{1/2} Vᵀ, so ΔW has exactly the given spectrum."""
    r = len(svals)
    U, _ = torch.linalg.qr(torch.randn(d_out, r, dtype=DT))
    V, _ = torch.linalg.qr(torch.randn(d_in, r, dtype=DT))
    root = torch.diag(torch.tensor(svals, dtype=DT).sqrt())
    return U @ root, root @ V.transpose(-1, -2)


def _rotate_degenerate_block(B, A):
    """Re-factor ΔW through an O(m) rotation of its tied subspace — ΔW is EXACTLY unchanged, but the
    individual singular directions inside the block are not, so anything reading them must move."""
    enc = encode_module(B, A)
    cluster = next((c for c in enc.clusters if len(c) > 1), None)
    assert cluster is not None, "test setup failed to plant a degeneracy"
    U, S, V = compact_svd_from_factors(B, A)
    m = len(cluster)
    Q, _ = torch.linalg.qr(torch.randn(m, m, dtype=DT))
    G = torch.eye(S.shape[0], dtype=DT)
    for a, ia in enumerate(cluster):
        for b, ib in enumerate(cluster):
            G[ia, ib] = Q[a, b]
    root = torch.diag(S.sqrt())
    return (U @ root) @ G, torch.linalg.inv(G) @ (root @ V.transpose(-1, -2))


# --------------------------------------------------------------------------------------
# 1. The featurizer must honour degeneracy_safe (it used to store the flag and ignore it).
# --------------------------------------------------------------------------------------
def test_our_svd_featurizer_is_degeneracy_safe():
    B, A = _planted([3.0, 3.0, 2.0, 1.0])          # sigma_1 = sigma_2 -> O(2)-ambiguous top block
    Bg, Ag = _rotate_degenerate_block(B, A)
    lora = {MOD: (B, A, 16.0, 4, False)}
    lora_g = {MOD: (Bg, Ag, 16.0, 4, False)}
    dims = build_fixed_schema([lora], top_k=TOPK)

    safe = OurSVDFeaturizer([MOD], dims, TOPK, degeneracy_safe=True)
    assert torch.allclose(safe.features(lora), safe.features(lora_g), atol=1e-6), (
        "our_svd features moved under an O(m) rotation of a degenerate block — the featurizer is "
        "reading directions that are not defined"
    )


def test_degeneracy_unsafe_featurizer_moves_proving_the_guard_does_work():
    """Positive contrast: with the guard off the same rotation DOES move the features, so the test
    above is not vacuous."""
    B, A = _planted([3.0, 3.0, 2.0, 1.0])
    Bg, Ag = _rotate_degenerate_block(B, A)
    lora = {MOD: (B, A, 16.0, 4, False)}
    lora_g = {MOD: (Bg, Ag, 16.0, 4, False)}
    dims = build_fixed_schema([lora], top_k=TOPK)

    naive = OurSVDFeaturizer([MOD], dims, TOPK, degeneracy_safe=False)
    assert not torch.allclose(naive.features(lora), naive.features(lora_g), atol=1e-3)


def test_degeneracy_safe_still_separates_genuinely_different_adapters():
    """The guard must not collapse everything to one vector (it replaces ambiguous directions with a
    subspace descriptor, it does not discard content)."""
    B, A = _planted([3.0, 3.0, 2.0, 1.0])
    B2, A2 = _planted([3.0, 3.0, 2.0, 1.0])        # same spectrum, different random subspaces
    lora, lora2 = {MOD: (B, A, 16.0, 4, False)}, {MOD: (B2, A2, 16.0, 4, False)}
    dims = build_fixed_schema([lora, lora2], top_k=TOPK)
    fz = OurSVDFeaturizer([MOD], dims, TOPK)
    assert not torch.allclose(fz.features(lora), fz.features(lora2), atol=1e-3)


# --------------------------------------------------------------------------------------
# 2. Sigma floor: sub-floor directions are noise; they must not enter a feature or a token.
# --------------------------------------------------------------------------------------
def test_usable_direction_mask_cuts_at_the_relative_floor():
    S = torch.tensor([1.0, 1e-3, 1e-9, 0.0], dtype=DT)
    keep = usable_direction_mask(S, SIGMA_FLOOR_REL)
    assert keep.tolist() == [True, True, False, False]
    assert usable_direction_mask(torch.zeros(3, dtype=DT)).tolist() == [False, False, False]


def test_sub_floor_direction_content_cannot_change_features():
    """Two adapters identical except in a sigma~0 direction must featurize identically: that
    direction's u,v are numerical noise, so letting them through injects randomness into the probe."""
    B, A = _planted([1.0, 0.5, 1e-11])
    B2 = B.clone()
    B2[:, 2] = torch.randn_like(B2[:, 2]) * 1e-6   # rewrite ONLY the sub-floor direction
    lora, lora2 = {MOD: (B, A, 16.0, 3, False)}, {MOD: (B2, A, 16.0, 3, False)}
    dims = build_fixed_schema([lora, lora2], top_k=TOPK)
    fz = OurSVDFeaturizer([MOD], dims, TOPK)
    f, f2 = fz.features(lora), fz.features(lora2)
    # the sigma block may still differ slightly (scale is real and measurable); directions must not
    assert torch.allclose(f[TOPK:], f2[TOPK:], atol=1e-9), \
        "a sub-floor (noise) direction reached the direction features"


def test_invariant_signature_zeroes_sub_floor_directions():
    B, A = _planted([1.0, 0.5, 1e-11])
    enc = encode_module(B, A)
    sig = invariant_signature(enc)
    d = enc.U.shape[0] + enc.V.shape[0]
    tail = sig[enc.k + 2 * d:]                      # the third (sub-floor) direction block
    assert tail.numel() == d and torch.count_nonzero(tail) == 0


def test_injection_tokens_skip_sub_floor_and_are_unit_norm():
    """The reader hook is h <- h + ||h||*v/||v|| (§B.12.1). Tokens must be finite and unit-norm, and a
    near-zero-sigma direction must be SKIPPED rather than normalized by a vanishing denominator."""
    B, A = _planted([1.0, 0.5, 1e-11])
    enc = encode_module(B, A)
    idx, tok = injection_tokens(enc)
    assert idx == [0, 1], f"sub-floor direction was injected: {idx}"
    assert tok.shape == (2, enc.U.shape[0] + enc.V.shape[0])
    assert torch.isfinite(tok).all()
    assert torch.allclose(torch.linalg.vector_norm(tok, dim=1), torch.ones(2, dtype=DT), atol=1e-12)


def test_injection_tokens_on_a_null_adapter_return_nothing():
    """A null (all-zero) adapter has no usable direction at all — the caller must get an empty set,
    not r tokens of arbitrary QR fill divided by ~0. (poc0d and POC-M both produced null adapters.)"""
    B = torch.zeros(20, 4, dtype=DT)
    A = torch.zeros(4, 20, dtype=DT)
    idx, tok = injection_tokens(encode_module(B, A))
    assert idx == [] and tok.shape[0] == 0
