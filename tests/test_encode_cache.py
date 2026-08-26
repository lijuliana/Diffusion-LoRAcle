"""The encoding cache must never return another tensor's decomposition.

It is keyed on tensor identity, and CPython reuses id() once an object is collected. Keying on id
alone therefore hands back a stale encoding as soon as the original factors are freed and a fresh
allocation lands on the same address — which is silent, and wrong in the direction that looks
plausible. The cache keeps its key tensors alive to prevent it; this pins that.
"""

import gc

import torch

from ditloracle.encoding.svd_encoder import clear_encode_cache, encode_module

DT = torch.float64


def _factors(seed, d=24, r=4):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(d, r, generator=g, dtype=DT), torch.randn(r, d, generator=g, dtype=DT))


def test_freed_then_reallocated_tensors_do_not_collide():
    clear_encode_cache()
    sigmas = []
    for seed in range(30):
        B, A = _factors(seed)
        sigmas.append(encode_module(B, A, alpha=16.0, r=4).sigma.clone())
        del B, A          # invite id reuse on the next allocation
        gc.collect()
    # recompute without any cache and compare
    clear_encode_cache()
    for seed in range(30):
        B, A = _factors(seed)
        fresh = encode_module(B, A, alpha=16.0, r=4).sigma
        assert torch.allclose(sigmas[seed], fresh, atol=1e-12), (
            f"seed {seed}: cached encoding does not match a fresh one — id reuse collision")


def test_cache_returns_equal_results_for_the_same_tensors():
    clear_encode_cache()
    B, A = _factors(0)
    a = encode_module(B, A, alpha=16.0, r=4)
    b = encode_module(B, A, alpha=16.0, r=4)
    assert torch.allclose(a.sigma, b.sigma) and torch.allclose(a.U, b.U)


def test_scale_parameters_are_part_of_the_key():
    """Same tensors, different alpha, must not share an entry."""
    clear_encode_cache()
    B, A = _factors(1)
    s16 = encode_module(B, A, alpha=16.0, r=4).sigma.clone()
    s32 = encode_module(B, A, alpha=32.0, r=4).sigma.clone()
    assert not torch.allclose(s16, s32), "alpha is being ignored by the cache key"
