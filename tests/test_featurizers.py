"""POC-0b/c unit tests: featurizers produce FIXED-length, finite features (no rank leakage via
length), the spectral baseline is provably direction-blind, and the rank-leak control depends only
on rank/recipe. Locks in the contrasts POC-1 relies on. CPU, no downloads."""

from __future__ import annotations

import torch

from ditloracle.probe.featurizers import (
    OurSVDFeaturizer,
    ProductSketchFeaturizer,
    RankLeakFeaturizer,
    RawABFeaturizer,
    SpectralStatFeaturizer,
    W2TFeaturizer,
    build_fixed_schema,
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


def test_missing_module_zero_padded_fixed_len():
    dims = _schema([_lora()])
    dims["m1"] = (32, 32)             # a module absent from the sample adapter
    fz = SpectralStatFeaturizer(["m0", "m1"], dims, TOPK)
    f = fz.features(_lora())          # only m0 present
    assert f.numel() == fz.out_dim == 10   # 5 stats x 2 modules
    assert torch.isfinite(f).all()
