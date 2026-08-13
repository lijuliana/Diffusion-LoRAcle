"""Target-module vocabularies per base model — the single source of truth for both planners.

ai-toolkit selects which linears get a LoRA by SUBSTRING MATCH, and it matches against the names in
its OWN internal (BFL/kohya-style) naming, not the diffusers names you see on the HF model. Getting
this wrong builds an empty network and the trainer aborts with "There are not any lora modules in
this network". The klein vocabulary below was read off a REAL trained adapter (an unfiltered probe
run), not from the diffusers module list:

    diffusion_model.double_blocks.{0..4}.img_attn.{qkv,proj}
    diffusion_model.double_blocks.{0..4}.txt_attn.{qkv,proj}
    diffusion_model.double_blocks.{0..4}.img_mlp.{0,2}
    diffusion_model.double_blocks.{0..4}.txt_mlp.{0,2}
    diffusion_model.single_blocks.{0..19}.linear{1,2}          # linear1 fuses qkv+mlp

80 adapted modules total; hidden width 3072 (the same as FLUX.1-dev, so the encoder's dimension
assumptions carry over). Note attention is FUSED (`qkv`) and single blocks fuse qkv+mlp — the
existing `fused_split` machinery is what handles that downstream.

FLUX.1-dev via ai-toolkit uses the same BFL scheme (`lora_unet_double_blocks_*` / dotted variants),
which `formats/flux_lora.py` already parses.

Set names describe BREADTH, not a fixed layer list, because the available layer types differ per
base (klein exposes no modulation layers to the trainer, FLUX.1 does):
  attn_only  — attention projections only
  attn_mlp   — attention + per-block MLP
  wide       — the above plus the remaining adapted blocks (single blocks / modulation)
"""

from __future__ import annotations

# --- FLUX.1-dev (BFL scheme as emitted by ai-toolkit/kohya) -----------------------------------
_FLUX1_ATTN = ["img_attn.qkv", "img_attn.proj", "txt_attn.qkv", "txt_attn.proj"]
_FLUX1_MLP = ["img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"]
_FLUX1_WIDE = ["single_blocks"]                      # linear1 (fused qkv+mlp) + linear2

# --- FLUX.2 klein-4B (verified against a trained adapter) --------------------------------------
_KLEIN_ATTN = ["img_attn.qkv", "img_attn.proj", "txt_attn.qkv", "txt_attn.proj"]
_KLEIN_MLP = ["img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"]
_KLEIN_WIDE = ["single_blocks"]

MODULE_SETS_BY_BASE = {
    "flux1": {
        "attn_only": _FLUX1_ATTN,
        "attn_mlp": _FLUX1_ATTN + _FLUX1_MLP,
        "wide": _FLUX1_ATTN + _FLUX1_MLP + _FLUX1_WIDE,
    },
    "klein": {
        "attn_only": _KLEIN_ATTN,
        "attn_mlp": _KLEIN_ATTN + _KLEIN_MLP,
        "wide": _KLEIN_ATTN + _KLEIN_MLP + _KLEIN_WIDE,
    },
}


def base_family(base_model: str) -> str:
    return "klein" if "klein" in (base_model or "").lower() else "flux1"


def module_sets_for(base_model: str) -> dict[str, list[str]]:
    return MODULE_SETS_BY_BASE[base_family(base_model)]


def reference_modules(base_model: str) -> list[str]:
    """The clamped reference recipe's module set (attn+MLP, the common LoRA target)."""
    return list(module_sets_for(base_model)["attn_mlp"])
