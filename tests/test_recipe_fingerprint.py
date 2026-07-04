"""Tests for the rich recipe fingerprint (design doc §B.7.2 A2).

Validates:
  - fixed dimensionality: every adapter yields the same-length row regardless of rank/scheme/dtype;
  - each recipe knob (rank, alpha, dtype, scheme, DoRA, target-module set) is actually captured;
  - malformed / missing files degrade to a zero row with ok=False (never raise);
  - the matrix() baseline interface matches RecipeOnlyBaseline (X, names);
  - a real corpus file (if present) parses into a sane profile.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

from ditloracle.formats.flux_lora import Scheme
from ditloracle.probe.recipe_fingerprint import (
    RecipeFingerprint,
    feature_names,
    profile_adapter,
)

DM = 3072


def _write_diffusers(path, r=8, alpha=None, dtype=torch.float16, modules="attn", dora=False):
    """Synthetic diffusers-scheme LoRA. `modules` ∈ {attn, attn+mlp, attn+mlp+mod}."""
    sd = {}
    blocks = ["transformer.transformer_blocks.0", "transformer.transformer_blocks.1"]
    subs = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
    if "mlp" in modules:
        subs += ["ff.net.0.proj", "ff.net.2"]
    if "mod" in modules:
        subs += ["norm1.linear"]
    for blk in blocks:
        for sub in subs:
            stem = f"{blk}.{sub}"
            sd[f"{stem}.lora_A.weight"] = torch.randn(r, DM, dtype=dtype)
            sd[f"{stem}.lora_B.weight"] = torch.randn(DM, r, dtype=dtype)
            if alpha is not None:
                sd[f"{stem}.alpha"] = torch.tensor(float(alpha), dtype=dtype)
            if dora:
                sd[f"{stem}.lora_magnitude_vector.weight"] = torch.randn(DM, dtype=dtype)
    save_file(sd, path)


def _write_kohya_fused(path, r=16, alpha=8.0, dtype=torch.bfloat16):
    """Synthetic kohya/BFL LoRA with a fused double-block qkv (the common real case)."""
    sd = {}
    stem = "lora_unet_double_blocks_0_img_attn_qkv"
    sd[f"{stem}.lora_down.weight"] = torch.randn(r, DM, dtype=dtype)
    sd[f"{stem}.lora_up.weight"] = torch.randn(3 * DM, r, dtype=dtype)
    sd[f"{stem}.alpha"] = torch.tensor(float(alpha), dtype=dtype)
    # a non-fused proj too
    stem2 = "lora_unet_double_blocks_0_img_attn_proj"
    sd[f"{stem2}.lora_down.weight"] = torch.randn(r, DM, dtype=dtype)
    sd[f"{stem2}.lora_up.weight"] = torch.randn(DM, r, dtype=dtype)
    sd[f"{stem2}.alpha"] = torch.tensor(float(alpha), dtype=dtype)
    save_file(sd, path)


def test_fixed_dimension_across_heterogeneous_adapters(tmp_path):
    """The whole point of the fixed schema: rows are the same length no matter the recipe."""
    p1 = str(tmp_path / "a.safetensors"); _write_diffusers(p1, r=4, alpha=4, dtype=torch.float16, modules="attn")
    p2 = str(tmp_path / "b.safetensors"); _write_diffusers(p2, r=32, alpha=16, dtype=torch.float32, modules="attn+mlp+mod")
    p3 = str(tmp_path / "c.safetensors"); _write_kohya_fused(p3, r=16, alpha=8, dtype=torch.bfloat16)
    recs = [{"local_path": p1}, {"local_path": p2}, {"local_path": p3}]
    X, names = RecipeFingerprint().fit(recs).matrix(recs)
    assert X.shape == (3, len(names))
    assert X.shape[1] == len(feature_names(include_norms=True))
    assert not np.isnan(X).any()


def test_rank_alpha_dtype_scheme_captured(tmp_path):
    p = str(tmp_path / "x.safetensors")
    _write_diffusers(p, r=8, alpha=16, dtype=torch.bfloat16, modules="attn+mlp")
    prof = profile_adapter(p)
    assert prof.ok
    assert prof.scheme == Scheme.DIFFUSERS
    assert prof.dtype == "BF16"
    assert prof.rank == 8
    assert prof.alpha_present and abs(prof.alpha - 16.0) < 1e-3
    assert abs(prof.alpha_rank_ratio - 2.0) < 1e-3
    assert not prof.is_dora
    assert prof.frac_attn > 0 and prof.frac_mlp > 0


def test_target_module_set_distinguishes_attn_vs_full(tmp_path):
    p_attn = str(tmp_path / "attn.safetensors"); _write_diffusers(p_attn, modules="attn")
    p_full = str(tmp_path / "full.safetensors"); _write_diffusers(p_full, modules="attn+mlp+mod")
    a = profile_adapter(p_attn); b = profile_adapter(p_full)
    assert a.frac_mlp == 0.0 and a.frac_mod == 0.0
    assert b.frac_mlp > 0.0 and b.frac_mod > 0.0
    # the vectors must actually differ on the module-set block
    recs = [{"local_path": p_attn}, {"local_path": p_full}]
    X, _ = RecipeFingerprint().fit(recs).matrix(recs)
    assert not np.allclose(X[0], X[1])


def test_dora_detected(tmp_path):
    p = str(tmp_path / "dora.safetensors")
    _write_diffusers(p, dora=True)
    assert profile_adapter(p).is_dora is True


def test_fused_flag_and_kohya_scheme(tmp_path):
    p = str(tmp_path / "k.safetensors")
    _write_kohya_fused(p)
    prof = profile_adapter(p)
    assert prof.scheme == Scheme.KOHYA
    assert prof.is_fused is True
    # fused qkv splits to q/k/v -> attn modules present
    assert prof.frac_attn > 0


def test_no_alpha_handled(tmp_path):
    p = str(tmp_path / "noalpha.safetensors")
    _write_diffusers(p, alpha=None)
    prof = profile_adapter(p)
    assert prof.alpha_present is False
    assert prof.alpha == 0.0 and prof.alpha_rank_ratio == 0.0


def test_missing_or_bad_file_is_zero_row():
    rec = {"local_path": "/nonexistent/path.safetensors"}
    fp = RecipeFingerprint()
    X, names = fp.fit([rec]).matrix([rec])
    assert X.shape == (1, len(names))
    assert np.allclose(X, 0.0)
    assert fp.profiles([rec])[0].ok is False


def test_include_norms_ablation_changes_dim(tmp_path):
    p = str(tmp_path / "n.safetensors"); _write_diffusers(p)
    recs = [{"local_path": p}]
    X_full, n_full = RecipeFingerprint(include_norms=True).fit(recs).matrix(recs)
    X_abl, n_abl = RecipeFingerprint(include_norms=False).fit(recs).matrix(recs)
    assert len(n_full) == len(n_abl) + 4
    assert "frob_mean" in n_full and "frob_mean" not in n_abl


def test_norms_are_positive_and_finite(tmp_path):
    p = str(tmp_path / "n.safetensors"); _write_diffusers(p, r=8)
    prof = profile_adapter(p, include_norms=True)
    assert prof.frob_mean > 0 and np.isfinite(prof.frob_mean)
    assert prof.frob_max >= prof.frob_min


def test_real_corpus_file_if_present():
    """Smoke test on one real adapter from the manifest, if downloaded in this env."""
    manifest = "assets/corpus/manifest_civitai_dl.json"
    if not os.path.exists(manifest):
        return
    recs = json.load(open(manifest))
    real = next((r for r in recs if r.get("local_path") and os.path.exists(r["local_path"])), None)
    if real is None:
        return
    prof = profile_adapter(real["local_path"])
    assert prof.ok
    assert prof.rank > 0
    assert prof.n_modules > 0
    assert prof.scheme in (Scheme.KOHYA, Scheme.DIFFUSERS)
    # matrix interface produces a finite fixed-dim row
    X, names = RecipeFingerprint().fit([real]).matrix([real])
    assert X.shape == (1, len(names)) and np.isfinite(X).all()


def test_module_predicates_categorize_real_canonical_names():
    """Regression guard (reviewer-flagged): the attn/mlp/mod predicates string-match canonical names;
    if FLUX naming used a token they miss, frac_attn/mlp/mod would silently go ~zero and quietly
    weaken the recipe control. Verify on a REAL adapter: every module categorizes into exactly one
    bucket (no uncategorized) and all three fractions are non-trivially non-zero."""
    from ditloracle.formats.safetensors_io import load_canonical_factors
    from ditloracle.probe.recipe_fingerprint import _is_attn, _is_mlp, _is_mod
    manifest = "assets/corpus/manifest_civitai_dl.json"
    if not os.path.exists(manifest):
        return
    recs = json.load(open(manifest))
    real = next((r for r in recs if r.get("local_path") and os.path.exists(r["local_path"])), None)
    if real is None:
        return
    names = list(load_canonical_factors(real["local_path"]).keys())
    buckets = [(int(_is_attn(n)) + int(_is_mlp(n)) + int(_is_mod(n))) for n in names]
    assert all(b == 1 for b in buckets), "some module matched zero or multiple of attn/mlp/mod"
    n = len(names)
    assert sum(_is_attn(x) for x in names) / n > 0.2   # attn present and substantial
    assert sum(_is_mlp(x) for x in names) / n > 0.1    # mlp present
    assert sum(_is_mod(x) for x in names) / n > 0.05   # modulation present (not silently zero)
