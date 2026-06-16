"""POC-0a gate: validate the SVD encoder is invariant to exactly the LoRA symmetries — and to
nothing more. A silently-wrong encoder would invalidate every downstream number (WORKING_NORMS §3),
so this suite is the instrument calibration that must pass before any real-data work is trusted.

Symmetries that MUST leave the signature unchanged:
  - GL(r) gauge:    (B,A) -> (B G^-1, G A)        [well- and ill-conditioned G]
  - coupled sign:   per-direction (u,v) -> (-u,-v) [induced by re-running SVD]
  - degenerate O(m): rotation within a tied-sigma subspace
A genuine change in DeltaW MUST change the signature (negative control — no trivial collapse).
"""

from __future__ import annotations

import torch

from ditloracle.encoding.svd_encoder import (
    compact_svd_from_factors,
    encode_module,
    invariant_signature,
)

torch.manual_seed(0)
DT = torch.float64
TOL = 1e-8


def _rand_factors(d_out: int, d_in: int, r: int):
    B = torch.randn(d_out, r, dtype=DT)
    A = torch.randn(r, d_in, dtype=DT)
    return B, A


def _sig(B, A, **kw):
    enc = encode_module(B, A, **kw)
    return invariant_signature(enc)


# --------------------------------------------------------------------------------------
# 0. Sanity: the factored SVD equals the dense SVD (the routine is correct at all).
# --------------------------------------------------------------------------------------
def test_factored_svd_matches_dense():
    B, A = _rand_factors(64, 48, 8)
    scale = 0.37
    U, S, V = compact_svd_from_factors(B, A, scale=scale)
    dW = scale * (B @ A)
    # reconstruct and compare to the true product
    recon = U @ torch.diag(S) @ V.transpose(-1, -2)
    assert torch.allclose(recon, dW, atol=1e-9), "factored SVD does not reconstruct ΔW"
    # singular values match the top-k of a direct dense SVD (dense returns min(d_out,d_in)
    # values, mostly ~0 since rank=r=8; the thin SVD returns exactly the k=r nonzero ones)
    k = S.shape[0]
    S_dense = torch.linalg.svdvals(dW)[:k]
    assert torch.allclose(S.sort(descending=True).values, S_dense, atol=1e-9)


# --------------------------------------------------------------------------------------
# 1. GL(r) gauge invariance — the central requirement.
# --------------------------------------------------------------------------------------
def test_gl_invariance_well_conditioned():
    B, A = _rand_factors(64, 48, 8)
    r = B.shape[1]
    G = torch.randn(r, r, dtype=DT)
    while torch.linalg.matrix_rank(G) < r:           # ensure invertible
        G = torch.randn(r, r, dtype=DT)
    Bg = B @ torch.linalg.inv(G)
    Ag = G @ A
    s0, s1 = _sig(B, A), _sig(Bg, Ag)
    assert torch.allclose(s0, s1, atol=1e-7), "signature changed under GL(r) gauge"


def test_gl_invariance_ill_conditioned():
    # The dangerous case: a near-singular G (condition number ~1e6). The encoding must still
    # be invariant because ΔW is mathematically unchanged.
    B, A = _rand_factors(64, 48, 8)
    r = B.shape[1]
    # Build G = Q1 diag(logspace) Q2 with a large condition number.
    Q1, _ = torch.linalg.qr(torch.randn(r, r, dtype=DT))
    Q2, _ = torch.linalg.qr(torch.randn(r, r, dtype=DT))
    cond = torch.logspace(0, 6, r, dtype=DT)         # up to 1e6
    G = Q1 @ torch.diag(cond) @ Q2
    Bg = B @ torch.linalg.inv(G)
    Ag = G @ A
    s0, s1 = _sig(B, A), _sig(Bg, Ag)
    assert torch.allclose(s0, s1, atol=1e-6), "signature changed under ill-conditioned GL(r)"


# --------------------------------------------------------------------------------------
# 2. Coupled-sign invariance. Re-running SVD on a permuted/scaled factorization picks
#    arbitrary signs; canonicalization must absorb them.
# --------------------------------------------------------------------------------------
def test_sign_invariance():
    B, A = _rand_factors(40, 32, 6)
    # A diagonal ±1 gauge is a special case of GL(r): flips column signs of B and row signs of A.
    D = torch.diag(torch.sign(torch.randn(6, dtype=DT)))
    s0, s1 = _sig(B, A), _sig(B @ D, D @ A)
    assert torch.allclose(s0, s1, atol=1e-8), "signature changed under coupled sign flip"


# --------------------------------------------------------------------------------------
# 3. Degenerate-subspace (O(m)) invariance. Plant a tied singular value, rotate the subspace,
#    and require the degeneracy-safe signature to be unchanged.
# --------------------------------------------------------------------------------------
def _planted_degenerate(d_out=50, d_in=50, r=6, tie_val=3.0, tie_block=3):
    """Construct ΔW with `tie_block` equal singular values = tie_val (plus distinct others),
    returned as factors (B, A) with B = U Σ^{1/2}, A = Σ^{1/2} Vᵀ so rank = r."""
    U, _ = torch.linalg.qr(torch.randn(d_out, r, dtype=DT))
    V, _ = torch.linalg.qr(torch.randn(d_in, r, dtype=DT))
    svals = torch.empty(r, dtype=DT)
    svals[:tie_block] = tie_val
    svals[tie_block:] = torch.linspace(2.0, 0.5, r - tie_block, dtype=DT)
    svals = svals.sort(descending=True).values
    root = torch.diag(svals.sqrt())
    B = U @ root
    A = root @ V.transpose(-1, -2)
    return B, A


def test_degenerate_subspace_invariance():
    B, A = _planted_degenerate()
    enc0 = encode_module(B, A)
    # confirm a degenerate cluster was actually detected
    assert any(len(c) > 1 for c in enc0.clusters), "test setup failed to plant a degeneracy"
    # Rotate within the tied subspace: ΔW = U Σ Vᵀ with Σ_block = c·I ⇒ replacing (U_blk,V_blk)
    # by (U_blk Q, V_blk Q) leaves ΔW exactly fixed. Realize it via a GL(r) gauge that is the
    # block-rotation Q on the tied indices and identity elsewhere.
    U, S, V = compact_svd_from_factors(B, A)
    cluster = next(c for c in enc0.clusters if len(c) > 1)
    m = len(cluster)
    Q, _ = torch.linalg.qr(torch.randn(m, m, dtype=DT))
    G = torch.eye(S.shape[0], dtype=DT)
    for a, ia in enumerate(cluster):
        for b, ib in enumerate(cluster):
            G[ia, ib] = Q[a, b]
    # apply as a gauge on the (U-diag-scaled) factor form: use B=UΣ^{1/2}, A=Σ^{1/2}Vᵀ again
    root = torch.diag(S.sqrt())
    Bg = (U @ root) @ G
    Ag = torch.linalg.inv(G) @ (root @ V.transpose(-1, -2))
    s0 = invariant_signature(enc0, degeneracy_safe=True)
    s1 = invariant_signature(encode_module(Bg, Ag), degeneracy_safe=True)
    assert torch.allclose(s0, s1, atol=1e-6), "degeneracy-safe signature changed under O(m) rotation"


def test_naive_signature_breaks_on_degeneracy_but_safe_one_holds():
    """Positive contrast: the *naive* (per-direction) signature is NOT O(m)-invariant on a
    degenerate block — proving the degeneracy-safe path is doing real work, not a no-op."""
    B, A = _planted_degenerate()
    enc0 = encode_module(B, A)
    U, S, V = compact_svd_from_factors(B, A)
    cluster = next(c for c in enc0.clusters if len(c) > 1)
    m = len(cluster)
    Q, _ = torch.linalg.qr(torch.randn(m, m, dtype=DT))
    G = torch.eye(S.shape[0], dtype=DT)
    for a, ia in enumerate(cluster):
        for b, ib in enumerate(cluster):
            G[ia, ib] = Q[a, b]
    root = torch.diag(S.sqrt())
    Bg = (U @ root) @ G
    Ag = torch.linalg.inv(G) @ (root @ V.transpose(-1, -2))
    naive0 = invariant_signature(enc0, degeneracy_safe=False)
    naive1 = invariant_signature(encode_module(Bg, Ag), degeneracy_safe=False)
    # naive should differ (directions rotated); this documents WHY degeneracy_safe exists
    assert not torch.allclose(naive0, naive1, atol=1e-3), (
        "naive signature unexpectedly invariant — degeneracy test may be vacuous"
    )


# --------------------------------------------------------------------------------------
# 4. NEGATIVE CONTROL: a genuinely different ΔW must produce a different signature.
#    Guards against a degenerate encoder that maps everything to the same vector.
# --------------------------------------------------------------------------------------
def test_genuine_change_is_detected():
    B, A = _rand_factors(64, 48, 8)
    B2 = B.clone()
    B2[:, 0] += 0.5 * torch.randn_like(B2[:, 0])     # perturb one direction -> real change in ΔW
    s0, s1 = _sig(B, A), _sig(B2, A)
    assert not torch.allclose(s0, s1, atol=1e-4), "encoder collapsed a genuine ΔW change"


def test_scale_folding_changes_signature_scale_only():
    """Folding α/r changes the overall scale (frob), but the spectral-normalized *shape* of the
    spectrum is preserved — a basic check that normalization behaves as designed."""
    B, A = _rand_factors(40, 40, 8)
    enc_a = encode_module(B, A, alpha=16, r=8)        # scale = 2.0
    enc_b = encode_module(B, A, alpha=8, r=8)         # scale = 1.0
    # frobenius differs by exactly the scale ratio
    assert abs(enc_a.frob / enc_b.frob - 2.0) < 1e-6
    # spectral-normalized signatures (shape) are identical
    s_a = invariant_signature(enc_a, spectral_normalize=True)
    s_b = invariant_signature(enc_b, spectral_normalize=True)
    assert torch.allclose(s_a, s_b, atol=1e-7)
