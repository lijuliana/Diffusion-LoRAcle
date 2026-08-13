"""Tests for the raw-key-stem module fallback (non-FLUX.1 schemes, e.g. FLUX.2 klein).

The causal gate compares organisms that share one base and one recipe, so it needs module identities
that are CONSISTENT ACROSS FILES, not canonical FLUX.1 names. Without this fallback a klein LoRA
parses to zero modules and the gate silently has nothing to featurize.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from ditloracle.formats.safetensors_io import (
    canonical_module_names,
    load_lora_factors,
    raw_stem_modules,
)


def _write_lora(path, stems, rank=8, d=32, prefix=""):
    """Write a minimal LoRA safetensors with A/B pairs under the given key stems."""
    t = {}
    for s in stems:
        t[f"{prefix}{s}.lora_A.weight"] = torch.randn(rank, d)
        t[f"{prefix}{s}.lora_B.weight"] = torch.randn(d, rank)
    save_file(t, str(path))
    return path


# FLUX.2-style keys the FLUX.1 parser does not recognize
KLEIN_STEMS = [
    "blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2",
    "blocks.1.attn.qkv", "blocks.1.attn.proj",
]


def test_unknown_scheme_yields_modules_via_fallback(tmp_path):
    p = _write_lora(tmp_path / "klein.safetensors", KLEIN_STEMS)
    names = canonical_module_names(p)
    assert names, "unknown-scheme LoRA produced no modules; the gate would have nothing to featurize"
    assert names == set(KLEIN_STEMS)


def test_fallback_can_be_disabled(tmp_path):
    p = _write_lora(tmp_path / "klein.safetensors", KLEIN_STEMS)
    assert canonical_module_names(p, allow_raw_fallback=False) == set()


def test_wrapper_prefixes_are_stripped_so_files_line_up(tmp_path):
    # the same module must get the same identity whether or not the trainer prefixed the keys,
    # otherwise two organisms of one matched set share no modules and retrieval is undefined
    a = _write_lora(tmp_path / "a.safetensors", KLEIN_STEMS, prefix="")
    b = _write_lora(tmp_path / "b.safetensors", KLEIN_STEMS, prefix="transformer.")
    assert set(raw_stem_modules(a)) == set(raw_stem_modules(b))


def test_fallback_maps_names_back_to_loadable_stems(tmp_path):
    # the fallback's values must be the RAW stems, so factors can actually be loaded by them
    p = _write_lora(tmp_path / "klein.safetensors", KLEIN_STEMS, prefix="transformer.")
    mods = raw_stem_modules(p)
    factors = load_lora_factors(p)
    for name, stem in mods.items():
        assert stem in factors, f"{name} -> {stem} is not a loadable factor stem"
        assert factors[stem]["A"].shape[0] == 8


def test_flux1_scheme_still_uses_canonical_names(tmp_path):
    # the fallback must not shadow the real parser on files it understands
    flux1 = ["transformer.transformer_blocks.0.attn.to_q",
             "transformer.transformer_blocks.0.attn.to_k"]
    p = _write_lora(tmp_path / "flux1.safetensors", flux1)
    names = canonical_module_names(p)
    assert any(n.startswith("double.0.attn") for n in names), names


def test_load_canonical_factors_falls_back_for_unknown_scheme(tmp_path):
    # the gate calls load_canonical_factors and SKIPS an organism whose factors come back empty,
    # so without this fallback a klein corpus would silently produce an empty gate.
    from ditloracle.formats.safetensors_io import load_canonical_factors
    p = _write_lora(tmp_path / "klein.safetensors", KLEIN_STEMS, prefix="transformer.")
    fac = load_canonical_factors(p)
    assert set(fac) == set(KLEIN_STEMS)
    B, A, alpha, r, rs = fac[KLEIN_STEMS[0]]
    assert A.shape[0] == r == 8 and B.shape[1] == 8
    assert load_canonical_factors(p, allow_raw_fallback=False) == {}


def test_keep_modules_filters_the_fallback(tmp_path):
    from ditloracle.formats.safetensors_io import load_canonical_factors
    p = _write_lora(tmp_path / "klein.safetensors", KLEIN_STEMS)
    keep = {KLEIN_STEMS[0], KLEIN_STEMS[1]}
    assert set(load_canonical_factors(p, keep_modules=keep)) == keep
