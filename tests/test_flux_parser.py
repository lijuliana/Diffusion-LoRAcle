"""Unit tests for the FLUX LoRA key-scheme parser, against SYNTHETIC key dicts mirroring the two
documented schemes + the structural variants. No downloads. These lock the parser behavior so
POC-0d triage on real files is trustworthy; real-file edge cases get added here as we meet them."""

from __future__ import annotations

from ditloracle.formats.flux_lora import Flag, Scheme, classify, detect_scheme, parse_keys


def diffusers_keys():
    ks = []
    for i in (0, 1):
        for sub in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                    "attn.add_q_proj", "ff.net.0.proj", "norm1.linear"]:
            ks.append(f"transformer.transformer_blocks.{i}.{sub}.lora_A.weight")
            ks.append(f"transformer.transformer_blocks.{i}.{sub}.lora_B.weight")
    for i in (0,):
        for sub in ["attn.to_q", "proj_mlp", "proj_out", "norm.linear"]:
            ks.append(f"transformer.single_transformer_blocks.{i}.{sub}.lora_A.weight")
            ks.append(f"transformer.single_transformer_blocks.{i}.{sub}.lora_B.weight")
    return ks


def kohya_keys():
    ks = []
    for i in (0, 1):
        for sub in ["img_attn_qkv", "txt_attn_qkv", "img_attn_proj", "img_mlp_0", "img_mod_lin"]:
            ks.append(f"lora_unet_double_blocks_{i}_{sub}.lora_down.weight")
            ks.append(f"lora_unet_double_blocks_{i}_{sub}.lora_up.weight")
    for i in (0,):
        for sub in ["linear1", "linear2", "modulation_lin"]:
            ks.append(f"lora_unet_single_blocks_{i}_{sub}.lora_down.weight")
            ks.append(f"lora_unet_single_blocks_{i}_{sub}.lora_up.weight")
    return ks


def test_detect_scheme():
    assert detect_scheme(diffusers_keys()) == Scheme.DIFFUSERS
    assert detect_scheme(kohya_keys()) == Scheme.KOHYA
    assert detect_scheme(["random.weight", "foo.bar"]) == Scheme.UNKNOWN


def test_parse_diffusers():
    p = parse_keys(diffusers_keys())
    assert p.scheme == Scheme.DIFFUSERS
    assert "double.0.attn.to_q" in p.modules
    assert "double.1.ff.in" in p.modules
    assert "single.0.proj_mlp" in p.modules
    assert "double.0.mod.lin" in p.modules
    assert classify(p) == "ok"


def test_parse_kohya_flags_fused():
    p = parse_keys(kohya_keys())
    assert p.scheme == Scheme.KOHYA
    assert "double.0.attn.qkv_fused_img" in p.modules
    assert "single.0.proj_fused_qkv_mlp" in p.modules
    assert Flag.FUSED_QKV in p.flags
    assert classify(p) == "fused_needs_split"


def test_dora_flagged():
    ks = diffusers_keys() + [
        "transformer.transformer_blocks.0.attn.to_q.lora_magnitude_vector.weight"
    ]
    p = parse_keys(ks)
    assert Flag.DORA in p.flags
    assert classify(p) == "dora"


def test_unparseable_and_nonlowrank():
    p = parse_keys(["model.diffusion_model.foo.weight", "bar.bias"])
    assert Flag.UNPARSEABLE in p.flags
    assert Flag.NON_LOWRANK in p.flags
    assert classify(p) == "unparseable"
    assert parse_keys([]).n_modules == 0
