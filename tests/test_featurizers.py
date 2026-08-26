"""POC-0b/c unit tests: featurizers produce FIXED-length, finite features (no rank leakage via
length), the spectral baseline is provably direction-blind, and the rank-leak control depends only
on rank/recipe. Locks in the contrasts POC-1 relies on. CPU, no downloads."""

from __future__ import annotations

import pytest
import torch

from ditloracle.probe.featurizers import (
    OurSVDFeaturizer,
    ProductSketchFeaturizer,
    RankLeakFeaturizer,
    RawABFeaturizer,
    SpectralStatFeaturizer,
    U1LogRegFeaturizer,
    W2TFeaturizer,
    build_fixed_schema,
    canonicalize_u1_sign,
    top_left_direction,
)

torch.manual_seed(1)
DT = torch.float64
MOD = "m0"
TOPK = 8


def _lora(d_out=32, d_in=32, r=6):
    B = torch.randn(d_out, r, dtype=DT)
    A = torch.randn(r, d_in, dtype=DT)
    return {MOD: (B, A, 16.0, r, False)}


def _schema(loras, top_k=TOPK):
    return build_fixed_schema(loras, top_k=top_k)


def _all(modules, dims, top_k=TOPK):
    return [
        SpectralStatFeaturizer(modules, dims, top_k),
        RawABFeaturizer(modules, dims, top_k),
        W2TFeaturizer(modules, dims, top_k),
        OurSVDFeaturizer(modules, dims, top_k),
        ProductSketchFeaturizer(modules, dims, top_k),
        U1LogRegFeaturizer(modules, dims, top_k),
        RankLeakFeaturizer(modules, dims, top_k),
    ]


def test_featurizers_finite_and_nonempty():
    lora = _lora()
    dims = _schema([lora])
    for fz in _all([MOD], dims):
        f = fz.features(lora)
        assert f.numel() > 0 and torch.isfinite(f).all(), f"{fz.name} bad features"


def test_fixed_dimension_across_ranks():
    """THE anti-leakage property (§B.7.1c): feature length is identical for adapters of DIFFERENT
    rank. If length varied with rank, a probe could read rank off the length."""
    lo_r = _lora(r=4)
    hi_r = _lora(r=16)
    dims = _schema([lo_r, hi_r])      # global schema spans both
    for fz in _all([MOD], dims):
        f_lo, f_hi = fz.features(lo_r), fz.features(hi_r)
        assert f_lo.numel() == f_hi.numel() == fz.out_dim, (
            f"{fz.name}: feature length varies with rank ({f_lo.numel()} vs {f_hi.numel()}) — LEAK"
        )


def test_spectral_baseline_is_direction_blind():
    d_out = d_in = 40
    r = 6
    U, _ = torch.linalg.qr(torch.randn(d_out, r, dtype=DT))
    V, _ = torch.linalg.qr(torch.randn(d_in, r, dtype=DT))
    spectrum = torch.linspace(3.0, 0.5, r, dtype=DT)
    root = torch.diag(spectrum.sqrt())
    lora1 = {MOD: (U @ root, root @ V.transpose(0, 1), 16.0, r, False)}
    U2, _ = torch.linalg.qr(torch.randn(d_out, r, dtype=DT))
    V2, _ = torch.linalg.qr(torch.randn(d_in, r, dtype=DT))
    lora2 = {MOD: (U2 @ root, root @ V2.transpose(0, 1), 16.0, r, False)}
    dims = _schema([lora1, lora2])

    sf = SpectralStatFeaturizer([MOD], dims, TOPK)
    assert torch.allclose(sf.features(lora1), sf.features(lora2), atol=1e-6), \
        "spectral baseline unexpectedly direction-sensitive"
    of = OurSVDFeaturizer([MOD], dims, TOPK)
    assert not torch.allclose(of.features(lora1), of.features(lora2), atol=1e-3), \
        "our encoding failed to see the direction change"


def test_rank_leak_featurizer_only_sees_rank_and_presence():
    """The leakage control must change with rank/module-presence and NOT with directions."""
    dims = _schema([_lora(r=4), _lora(r=16)])
    rl = RankLeakFeaturizer([MOD, "m1"], dims, TOPK)
    f_r4 = rl.features(_lora(r=4))
    f_r16 = rl.features(_lora(r=16))
    assert not torch.allclose(f_r4, f_r16), "rank-leak feature should change with rank"
    # identical rank + presence -> identical feature regardless of weight content
    a = {MOD: (torch.randn(32, 8, dtype=DT), torch.randn(8, 32, dtype=DT), 16.0, 8, False)}
    b = {MOD: (torch.randn(32, 8, dtype=DT), torch.randn(8, 32, dtype=DT), 16.0, 8, False)}
    assert torch.allclose(rl.features(a), rl.features(b)), \
        "rank-leak feature must ignore weight content (only rank+presence)"


def test_product_sketch_is_gl_and_sign_invariant():
    """The product-sketch baseline (Putterman GL-net's endorsed feature) is a LINEAR function of ΔW,
    so it must be EXACTLY invariant to the GL(r) gauge (B,A) ↦ (BG⁻¹, GA) — with NO canonicalization
    step — while still responding to a genuine ΔW change. This is the property that lets it isolate
    'is canonicalization the thing doing the work in our_svd?'."""
    d_out = d_in = 40
    r = 6
    B = torch.randn(d_out, r, dtype=DT)
    A = torch.randn(r, d_in, dtype=DT)
    lora = {MOD: (B, A, 16.0, r, False)}
    # random GL(r) gauge transform: (B,A) -> (B G⁻¹, G A); ΔW = BA is unchanged
    G = torch.randn(r, r, dtype=DT)
    while torch.linalg.matrix_rank(G) < r:
        G = torch.randn(r, r, dtype=DT)
    lora_g = {MOD: (B @ torch.linalg.inv(G), G @ A, 16.0, r, False)}
    dims = _schema([lora], top_k=TOPK)
    ps = ProductSketchFeaturizer([MOD], dims, TOPK)
    assert torch.allclose(ps.features(lora), ps.features(lora_g), atol=1e-8), \
        "product sketch not GL(r)-invariant"
    # a genuinely different ΔW must change the sketch (no trivial collapse)
    lora2 = {MOD: (torch.randn(d_out, r, dtype=DT), torch.randn(r, d_in, dtype=DT), 16.0, r, False)}
    assert not torch.allclose(ps.features(lora), ps.features(lora2), atol=1e-3), \
        "product sketch blind to a real ΔW change"


# ---------------------------------------------------------------------------------------------
# u₁ + logistic-regression baseline (`2607.25750`) — the NEAREST competing paper. These lock in the
# three things that make it a faithful, comparable foil: fixed dimensionality, a DETERMINISTIC sign
# gauge, and an honest statement of what it is and is not invariant to.
# ---------------------------------------------------------------------------------------------
def _lora_from_svd(U, s, V, alpha=16.0):
    """A LoRA whose ΔW is exactly U diag(s) Vᵀ (α folded out so the factors reproduce it)."""
    r = s.numel()
    B = U[:, :r] @ torch.diag(s) * (r / alpha)
    A = V[:, :r].transpose(0, 1)
    return {MOD: (B, A, alpha, r, False)}


def _orthonormal(d, r):
    Q, _ = torch.linalg.qr(torch.randn(d, r, dtype=DT))
    return Q


def test_u1_feature_is_one_unit_direction_per_module():
    """The method keeps ONE left singular direction per module — so the block is d_out long and unit
    norm (a direction, not a magnitude), and the total length is independent of rank and of top_k."""
    lora = _lora(d_out=32, d_in=40, r=6)
    dims = _schema([lora])
    fz = U1LogRegFeaturizer([MOD], dims, TOPK)
    f = fz.features(lora)
    assert f.numel() == fz.out_dim == 32, "u₁ block must be exactly d_out long"
    assert abs(float(torch.linalg.vector_norm(f)) - 1.0) < 1e-9, "u₁ must be a unit direction"
    # top_k is irrelevant to this method (it is top-1 by definition) — dimension must not move with it
    assert U1LogRegFeaturizer([MOD], dims, 2).out_dim == U1LogRegFeaturizer([MOD], dims, 64).out_dim


def test_u1_sign_canonicalization_stable_under_flip():
    """u₁ and −u₁ are both valid SVD outputs; without canonicalization the feature is sign-random per
    adapter and a logistic regression sees noise. The rule (largest-|entry| positive) must be an
    involution-killer: it maps u and −u to the SAME vector, and the pivot entry ends up positive."""
    for _ in range(20):
        u = torch.randn(37, dtype=DT)
        u = u / torch.linalg.vector_norm(u)
        for rule in ("bro", "max_abs"):
            cu = canonicalize_u1_sign(u, rule=rule)
            cmu = canonicalize_u1_sign(-u, rule=rule)
            assert torch.allclose(cu, cmu, atol=1e-12), f"{rule}: not stable under u₁ → −u₁"
            assert torch.allclose(cu.abs(), u.abs(), atol=1e-12), f"{rule}: changed more than sign"
        # the "max_abs" rule's defining postcondition
        cu = canonicalize_u1_sign(u, rule="max_abs")
        assert float(cu[int(cu.abs().argmax())]) > 0, "pivot entry not made positive"
    # and at the extraction layer: negating column 0 of U must not move the feature
    U = _orthonormal(24, 5)
    S = torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5], dtype=DT)
    Uf = U.clone(); Uf[:, 0] = -Uf[:, 0]
    assert torch.allclose(top_left_direction(U, S), top_left_direction(Uf, S), atol=1e-12)


def test_u1_survives_the_coupled_sign_gauge_end_to_end():
    """(B,A) ↦ (B·diag(−1,1,…), diag(−1,1,…)·A) leaves ΔW identical but re-factors it, so the QR/SVD
    path sees different inputs. The featurized u₁ must be unchanged — the whole point of fixing the
    sign rather than inheriting LAPACK's arbitrary choice."""
    d_out = d_in = 30
    r = 5
    B = torch.randn(d_out, r, dtype=DT)
    A = torch.randn(r, d_in, dtype=DT)
    flip = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0], dtype=DT))
    lora = {MOD: (B, A, 16.0, r, False)}
    lora_flipped = {MOD: (B @ flip, flip @ A, 16.0, r, False)}
    dims = _schema([lora])
    fz = U1LogRegFeaturizer([MOD], dims, TOPK)
    assert torch.allclose(fz.features(lora), fz.features(lora_flipped), atol=1e-8), \
        "u₁ feature moved under a coupled sign re-factorization — sign gauge not fixed"


def test_u1_sign_pivot_is_fragile_when_top_entries_tie():
    """PROPERTY OF THE BASELINE, pinned so we report it rather than discover it in a figure.

    Fixing the sign by the largest-|entry| coordinate ("max_abs") is deterministic, but the CHOICE OF
    PIVOT is not stable: when the two largest |entries| of u₁ near-tie, ordinary noise moves the pivot
    to a different coordinate, which flips the sign of the WHOLE d_out block. It matters more here
    than for `encoding.canonicalize_signs` because u₁ is the ENTIRE feature, so one flip destroys a
    module's whole contribution rather than one of r directions. This is why "max_abs" is NOT the
    default (see canonicalize_u1_sign): on real klein organisms it costs the baseline 0.35 mAP.
    Concretely: a direction with a dominant coordinate keeps its pivot; a near-tied one does not."""
    rng = torch.Generator().manual_seed(3)

    def pivot_flip_rate(base, noise=0.06, trials=200):
        base = base / torch.linalg.vector_norm(base)
        pivots = []
        for _ in range(trials):
            u = base + noise * torch.randn(base.numel(), generator=rng, dtype=DT)
            u = canonicalize_u1_sign(u / torch.linalg.vector_norm(u), rule="max_abs")
            pivots.append(int(u.abs().argmax()))
        return 1.0 - pivots.count(max(set(pivots), key=pivots.count)) / trials

    tied = torch.randn(24, generator=rng, dtype=DT)
    order = tied.abs().argsort(descending=True)
    tied[order[1]] = tied[order[0]].abs() * torch.sign(tied[order[1]])   # force an exact-ish tie
    dominant = tied.clone()
    dominant[order[0]] = 5.0                                            # one clearly largest entry

    assert pivot_flip_rate(tied) > 0.1, "expected the near-tied direction's pivot to be unstable"
    assert pivot_flip_rate(dominant) == 0.0, "a dominant coordinate must pin the pivot"


def test_both_sign_rules_are_involutions_and_bro_is_the_stable_one():
    """Both conventions must satisfy the featurizer contract (u and −u map to the same vector), and
    the Bro-Acar-Kolda rule must be the more stable one on a near-tied direction — that is why it is
    exposed, so any margin we claim over this baseline can be shown not to be an artifact of the sign
    convention we chose for it."""
    g = torch.Generator().manual_seed(11)
    for rule in ("max_abs", "bro"):
        for _ in range(10):
            u = torch.randn(30, generator=g, dtype=DT)
            u = u / torch.linalg.vector_norm(u)
            assert torch.allclose(canonicalize_u1_sign(u, rule=rule),
                                  canonicalize_u1_sign(-u, rule=rule), atol=1e-12), rule
    with pytest.raises(ValueError, match="unknown sign rule"):
        canonicalize_u1_sign(torch.ones(4, dtype=DT), rule="nope")
    with pytest.raises(ValueError, match="unknown sign rule"):
        U1LogRegFeaturizer([MOD], _schema([_lora()]), TOPK, sign_rule="nope")

    # a direction whose top two |entries| tie: "max_abs" flips, "bro" (a sum over ALL coordinates)
    # keeps a consistent sign under the same perturbations.
    base = torch.randn(24, generator=g, dtype=DT)
    order = base.abs().argsort(descending=True)
    base[order[1]] = base[order[0]].abs() * torch.sign(base[order[1]])
    base = base / torch.linalg.vector_norm(base)

    def flip_rate(rule, trials=200, noise=0.06):
        ref = canonicalize_u1_sign(base, rule=rule)
        flips = 0
        for _ in range(trials):
            u = base + noise * torch.randn(24, generator=g, dtype=DT)
            u = canonicalize_u1_sign(u / torch.linalg.vector_norm(u), rule=rule)
            flips += float(u @ ref) < 0
        return flips / trials

    assert flip_rate("bro") < flip_rate("max_abs"), \
        "the aggregate sign rule should be more stable than the single-pivot one"


def test_u1_is_gl_invariant_but_lossy_and_not_the_gauge_variant_control():
    """HONEST invariance statement, asserted rather than asserted-in-prose.

    u₁ is computed from the PRODUCT ΔW = B·A, which is exactly GL(r)-invariant, so u₁ inherits that
    invariance once its sign is fixed. It is therefore NOT the gauge-variant control (RawABFeaturizer
    is — checked here by contrast). What u₁ *is* is LOSSY: it discards v₁, the spectrum and directions
    2..r, so two adapters that share only a top-left direction are INDISTINGUISHABLE to it. That
    one-direction bottleneck — not a symmetry failure — is why the method can flag but structurally
    cannot describe a payload or recover a trigger."""
    d_out = d_in = 36
    r = 6
    B = torch.randn(d_out, r, dtype=DT)
    A = torch.randn(r, d_in, dtype=DT)
    lora = {MOD: (B, A, 16.0, r, False)}
    G = torch.randn(r, r, dtype=DT)
    while torch.linalg.matrix_rank(G) < r or float(torch.linalg.cond(G)) > 50:
        G = torch.randn(r, r, dtype=DT)
    lora_g = {MOD: (B @ torch.linalg.inv(G), G @ A, 16.0, r, False)}
    dims = _schema([lora])
    fz = U1LogRegFeaturizer([MOD], dims, TOPK)
    assert torch.allclose(fz.features(lora), fz.features(lora_g), atol=1e-6), \
        "u₁ is a function of ΔW and must inherit its GL(r)-invariance"
    # contrast: the actual gauge-variant control DOES move, so the two roles are not conflated
    raw = RawABFeaturizer([MOD], dims, TOPK)
    assert not torch.allclose(raw.features(lora), raw.features(lora_g), atol=1e-3), \
        "RawABFeaturizer is supposed to be the gauge-VARIANT control"

    # LOSSINESS: same top-left direction, everything else different -> identical feature.
    U = _orthonormal(d_out, r)
    s1 = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5, 0.25], dtype=DT)
    s2 = torch.tensor([9.0, 1.2, 1.0, 0.8, 0.4, 0.1], dtype=DT)
    l1 = _lora_from_svd(U, s1, _orthonormal(d_in, r))
    l2 = _lora_from_svd(U, s2, _orthonormal(d_in, r))
    dims2 = _schema([l1, l2])
    fz2 = U1LogRegFeaturizer([MOD], dims2, TOPK)
    assert torch.allclose(fz2.features(l1), fz2.features(l2), atol=1e-8), \
        "u₁ should be blind to spectrum and to v — that blindness IS the baseline's ceiling"
    # ...while a featurizer that keeps the spectrum is not blind to it (the contrast Fig 4 rests on)
    sf = SpectralStatFeaturizer([MOD], dims2, TOPK)
    assert not torch.allclose(sf.features(l1), sf.features(l2), atol=1e-3)


def test_u1_degeneracy_safe_path_matches_siblings():
    """When σ₁ is not simple, u₁ is only defined up to an O(m) rotation of the top subspace. The safe
    path substitutes the encoder's O(m)-invariant projector-diagonal descriptor: exactly invariant to
    that rotation, non-negative, unit norm, and reducing EXACTLY to the signed u₁ when σ₁ is simple."""
    d_out, r = 40, 4
    U = _orthonormal(d_out, r)
    S_deg = torch.tensor([2.0, 2.0, 1.0, 0.5], dtype=DT)          # σ₁ = σ₂ -> top cluster size 2
    S_simple = torch.tensor([3.0, 2.0, 1.0, 0.5], dtype=DT)

    # simple spectrum: the guard is a no-op
    assert torch.allclose(top_left_direction(U, S_simple, degeneracy_safe=True),
                          top_left_direction(U, S_simple, degeneracy_safe=False), atol=1e-12)

    # degenerate spectrum: the safe descriptor is invariant to an O(2) rotation INSIDE the top block,
    # which is precisely the transformation that makes the individual u₁ meaningless.
    th = 0.7
    Q = torch.tensor([[torch.cos(torch.tensor(th)), -torch.sin(torch.tensor(th))],
                      [torch.sin(torch.tensor(th)), torch.cos(torch.tensor(th))]], dtype=DT)
    U_rot = U.clone()
    U_rot[:, :2] = U[:, :2] @ Q
    safe, safe_rot = top_left_direction(U, S_deg), top_left_direction(U_rot, S_deg)
    assert torch.allclose(safe, safe_rot, atol=1e-10), \
        "degeneracy-safe descriptor is not O(m)-invariant inside the degenerate block"
    assert (safe >= 0).all() and abs(float(torch.linalg.vector_norm(safe)) - 1.0) < 1e-10
    # the raw (unsafe) reading DOES move under that rotation — i.e. the guard is doing real work
    assert not torch.allclose(top_left_direction(U, S_deg, degeneracy_safe=False),
                              top_left_direction(U_rot, S_deg, degeneracy_safe=False), atol=1e-3)


def test_u1_works_on_both_flux1_and_klein_module_naming(tmp_path):
    """The minted corpus is FLUX.2-klein, whose keys the FLUX.1 parser does not recognize and which
    `load_canonical_factors` therefore identifies by RAW KEY STEM. The baseline must featurize both
    naming worlds end-to-end, or it cannot be compared on the corpus we actually have."""
    from safetensors.torch import save_file

    from ditloracle.formats.safetensors_io import load_canonical_factors

    def write(path, stems, prefix="", rank=8, d=32):
        t = {}
        for s in stems:
            t[f"{prefix}{s}.lora_A.weight"] = torch.randn(rank, d)
            t[f"{prefix}{s}.lora_B.weight"] = torch.randn(d, rank)
        save_file(t, str(path))
        return path

    # FLUX.2-klein (ai-toolkit/BFL stems, wrapper prefix stripped by the raw-stem fallback)
    klein_stems = ["double_blocks.0.img_attn.qkv", "double_blocks.0.img_mlp.0",
                   "single_blocks.3.linear1", "single_blocks.3.linear2"]
    klein = load_canonical_factors(write(tmp_path / "klein.safetensors", klein_stems,
                                         prefix="diffusion_model."))
    assert klein, "klein LoRA produced no modules — nothing to featurize"
    kdims = build_fixed_schema([klein], top_k=TOPK)
    kfz = U1LogRegFeaturizer(sorted(kdims), kdims, TOPK)
    kf = kfz.features(klein)
    assert kf.numel() == kfz.out_dim == 32 * len(klein_stems) and torch.isfinite(kf).all()

    # FLUX.1-dev (parsed to canonical names, fused qkv split into sub-modules)
    flux1 = load_canonical_factors(write(tmp_path / "flux1.safetensors",
                                         ["transformer.transformer_blocks.0.attn.to_q",
                                          "transformer.transformer_blocks.0.attn.to_k"]))
    assert flux1, "FLUX.1 LoRA produced no modules"
    fdims = build_fixed_schema([flux1], top_k=TOPK)
    ffz = U1LogRegFeaturizer(sorted(fdims), fdims, TOPK)
    ff = ffz.features(flux1)
    assert ff.numel() == ffz.out_dim and torch.isfinite(ff).all()

    # cross-base transfer needs ONE shared layout: a schema over the union zero-fills the other
    # base's modules, so both corpora land in the same feature space (see the class docstring).
    union = build_fixed_schema([klein, flux1], top_k=TOPK)
    ufz = U1LogRegFeaturizer(sorted(union), union, TOPK)
    a, b = ufz.features(klein), ufz.features(flux1)
    assert a.numel() == b.numel() == ufz.out_dim
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_u1_gram_is_finite_and_matches_explicit_features():
    """The memory-safe `gram()` path the gates call must agree with the explicit feature matrix, so
    the baseline can be scored in the retrieval lineup as well as by its own logistic-regression head."""
    import numpy as np
    loras = [_lora(d_out=24, d_in=24, r=5) for _ in range(6)]
    dims = _schema(loras)
    fz = U1LogRegFeaturizer([MOD], dims, TOPK)
    G = fz.gram(loras)
    X = np.stack([fz.features(l).numpy() for l in loras]).astype(np.float64)
    mu, sd = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    assert np.isfinite(G).all()
    assert np.allclose(G, Xs @ Xs.T, atol=1e-8)


def test_missing_module_zero_padded_fixed_len():
    dims = _schema([_lora()])
    dims["m1"] = (32, 32)             # a module absent from the sample adapter
    fz = SpectralStatFeaturizer(["m0", "m1"], dims, TOPK)
    f = fz.features(_lora())          # only m0 present
    assert f.numel() == fz.out_dim == 10   # 5 stats x 2 modules
    assert torch.isfinite(f).all()
