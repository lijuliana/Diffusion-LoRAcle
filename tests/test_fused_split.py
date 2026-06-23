"""Tests for fused-qkv splitting (the #1 must-do: ~72% of real FLUX LoRAs are fused).

Correctness criteria:
  - split shapes are right (q/k/v each d_model rows; mlp = remainder);
  - the concatenation of the split ΔW sub-blocks EXACTLY reconstructs the fused ΔW (no info lost);
  - non-fused modules pass through untouched;
  - real kohya files load + split end-to-end into canonical sub-modules.
"""

from __future__ import annotations

import glob
import json
import os

import torch

from ditloracle.formats.fused_split import split_fused

DT = torch.float32
DM = 3072


def test_double_block_qkv_split_shapes_and_reconstruction():
    r = 16
    B = torch.randn(3 * DM, r, dtype=DT)      # output-fused qkv
    A = torch.randn(r, DM, dtype=DT)
    parts = split_fused("double.0.attn.qkv_fused_img", B, A)
    assert [p[0] for p in parts] == ["double.0.attn.to_q", "double.0.attn.to_k", "double.0.attn.to_v"]
    for _, Bs, As in parts:
        assert Bs.shape == (DM, r) and As.shape == (r, DM)
    # reconstruction: stacking the sub-ΔW row-blocks == fused ΔW
    fused_dW = B @ A
    recon = torch.cat([Bs @ As for _, Bs, As in parts], dim=0)
    assert torch.allclose(recon, fused_dW, atol=1e-5)


def test_single_block_linear1_qkv_mlp_split():
    r = 16
    mlp = 4 * DM                               # FLUX single-block mlp = 4*d_model
    B = torch.randn(3 * DM + mlp, r, dtype=DT)
    A = torch.randn(r, DM, dtype=DT)
    parts = split_fused("single.3.proj_fused_qkv_mlp", B, A)
    names = [p[0] for p in parts]
    assert names == ["single.3.attn.to_q", "single.3.attn.to_k", "single.3.attn.to_v", "single.3.proj_mlp"]
    assert parts[-1][1].shape == (mlp, r)      # mlp width derived from shape, not hard-coded
    recon = torch.cat([Bs @ As for _, Bs, As in parts], dim=0)
    assert torch.allclose(recon, B @ A, atol=1e-5)


def test_input_fused_proj_out_splits_columns():
    r = 16
    mlp = 4 * DM
    B = torch.randn(DM, r, dtype=DT)
    A = torch.randn(r, DM + mlp, dtype=DT)     # input-fused (attn + mlp)
    parts = split_fused("single.3.proj_out", B, A)
    assert [p[0] for p in parts] == ["single.3.attn_out", "single.3.mlp_out"]
    recon = torch.cat([Bs @ As for _, Bs, As in parts], dim=1)
    assert torch.allclose(recon, B @ A, atol=1e-5)


def test_nonfused_passthrough():
    r = 8
    B = torch.randn(DM, r, dtype=DT)
    A = torch.randn(r, DM, dtype=DT)
    parts = split_fused("double.0.attn.to_q", B, A)
    assert len(parts) == 1 and parts[0][0] == "double.0.attn.to_q"


def test_unexpected_shape_passes_through_safely():
    # a "fused" name but a shape that isn't 3*d_model → must not corrupt; pass through
    r = 8
    B = torch.randn(1000, r, dtype=DT)         # not 3*3072
    A = torch.randn(r, DM, dtype=DT)
    parts = split_fused("single.0.proj_fused_qkv_mlp", B, A)
    assert len(parts) == 1


def test_real_kohya_file_loads_and_splits():
    """End-to-end on a real downloaded kohya file, if the sample is present."""
    manifest = "assets/flux_lora_sample/manifest.json"
    if not os.path.exists(manifest):
        return  # sample not downloaded in this environment; skip silently
    from ditloracle.formats.flux_lora import Scheme, detect_scheme
    from ditloracle.formats.safetensors_io import load_canonical_factors, read_keys

    items = json.load(open(manifest))
    kohya = None
    for it in items:
        if detect_scheme(read_keys(it["path"])) == Scheme.KOHYA:
            kohya = it["path"]
            break
    if kohya is None:
        return
    fac = load_canonical_factors(kohya)
    # after splitting, we should have per-q/k/v modules, NOT fused names
    assert any(".attn.to_q" in k for k in fac), "no split to_q found"
    assert not any("fused" in k for k in fac), f"fused module survived: {[k for k in fac if 'fused' in k][:3]}"
    # every factor is a clean (B,A) pair with matching rank
    for k, (B, A, alpha, r, rs) in fac.items():
        assert B.shape[1] == A.shape[0] == r, f"rank mismatch at {k}"
