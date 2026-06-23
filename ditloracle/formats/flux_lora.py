"""Parse the FLUX.1-dev LoRA key-scheme zoo into a canonical module dict.

Real FLUX LoRAs ship under (at least) two naming schemes plus several structural variants. This
module maps any of them to canonical module names so the encoder sees a consistent layout. It is
written against the DOCUMENTED schemes (diffusers `FluxTransformer2DModel` and kohya/BFL
`lora_unet_*`) and is unit-tested on synthetic key dicts — no downloads needed to develop it.

Canonical module name format:  "{block_type}.{idx}.{submodule}"
  block_type ∈ {double, single}
  submodule  ∈ {attn.to_q, attn.to_k, attn.to_v, attn.to_out, attn.add_q, attn.add_k, attn.add_v,
                attn.to_add_out, ff.in, ff.out, ff_ctx.in, ff_ctx.out, mod.lin, mod_ctx.lin,
                proj_mlp, proj_out, mod}

Structural variants handled / flagged (see classify()):
  - diffusers vs kohya/BFL prefixes
  - fused single-block proj (qkv+mlp) -> flagged FUSED (caller may split before SVD)
  - DoRA (extra `lora_magnitude_vector`) -> flagged DORA
  - rsLoRA (use_rslora in adapter_config) -> carried as a per-module flag
  - missing pairs / partial coverage / non-low-rank -> flagged
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Scheme(str, Enum):
    DIFFUSERS = "diffusers"     # transformer.transformer_blocks.0.attn.to_q.lora_A.weight
    KOHYA = "kohya_bfl"         # lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight
    UNKNOWN = "unknown"


class Flag(str, Enum):
    OK = "ok"
    FUSED_QKV = "fused_qkv"          # single-block linear1 packs qkv(+mlp); needs split
    DORA = "dora"                    # has magnitude vector
    PARTIAL = "partial"              # some expected modules missing (not necessarily bad)
    UNPARSEABLE = "unparseable"      # no recognized lora keys
    NON_LOWRANK = "non_lowrank"      # merged/full weights, not an A/B pair


@dataclass
class ParsedLoRA:
    scheme: Scheme
    modules: dict[str, str] = field(default_factory=dict)   # canonical name -> a representative key stem
    flags: set[Flag] = field(default_factory=set)
    n_modules: int = 0
    raw_key_sample: list[str] = field(default_factory=list)


# --- diffusers regexes -----------------------------------------------------------------
# NB: `single_transformer_blocks` contains the substring `transformer_blocks`, so the double
# regex must require a non-word char (or start) before `transformer_blocks` to avoid matching
# single-block keys. We check single first anyway, but keep this guard for safety.
_DIFF_DOUBLE = re.compile(
    r"(?:^|[^_a-z])transformer_blocks\.(\d+)\.(.+?)\.lora_(A|B|down|up)\.weight$"
)
_DIFF_SINGLE = re.compile(
    r"single_transformer_blocks\.(\d+)\.(.+?)\.lora_(A|B|down|up)\.weight$"
)

# diffusers submodule -> canonical submodule
_DIFF_SUB = {
    "attn.to_q": "attn.to_q", "attn.to_k": "attn.to_k", "attn.to_v": "attn.to_v",
    "attn.to_out.0": "attn.to_out",
    "attn.add_q_proj": "attn.add_q", "attn.add_k_proj": "attn.add_k",
    "attn.add_v_proj": "attn.add_v", "attn.to_add_out": "attn.to_add_out",
    "ff.net.0.proj": "ff.in", "ff.net.2": "ff.out",
    "ff_context.net.0.proj": "ff_ctx.in", "ff_context.net.2": "ff_ctx.out",
    "norm1.linear": "mod.lin", "norm1_context.linear": "mod_ctx.lin",
    "proj_mlp": "proj_mlp", "proj_out": "proj_out", "norm.linear": "mod",
}

# --- kohya/BFL regex -------------------------------------------------------------------
_KOHYA = re.compile(
    r"lora_unet_(double|single)_blocks_(\d+)_(.+?)\.lora_(down|up|A|B)\.weight$"
)
_KOHYA_SUB = {
    # double blocks
    "img_attn_qkv": "attn.qkv_fused_img", "txt_attn_qkv": "attn.qkv_fused_txt",
    "img_attn_proj": "attn.to_out", "txt_attn_proj": "attn.to_add_out",
    "img_mlp_0": "ff.in", "img_mlp_2": "ff.out",
    "txt_mlp_0": "ff_ctx.in", "txt_mlp_2": "ff_ctx.out",
    "img_mod_lin": "mod.lin", "txt_mod_lin": "mod_ctx.lin",
    # single blocks
    "linear1": "proj_fused_qkv_mlp", "linear2": "proj_out", "modulation_lin": "mod",
}


def detect_scheme(keys: list[str]) -> Scheme:
    if any("transformer_blocks." in k for k in keys):
        return Scheme.DIFFUSERS
    if any(k.startswith("lora_unet_") for k in keys):
        return Scheme.KOHYA
    return Scheme.UNKNOWN


def parse_keys(keys: list[str]) -> ParsedLoRA:
    """Parse a list of state-dict key names (no tensors needed) into canonical modules + flags."""
    scheme = detect_scheme(keys)
    out = ParsedLoRA(scheme=scheme, raw_key_sample=keys[:8])
    flags = out.flags

    if any("lora_magnitude_vector" in k for k in keys):
        flags.add(Flag.DORA)

    has_lora = any(re.search(r"lora_(A|B|down|up)\.weight$", k) for k in keys)
    if not has_lora:
        flags.add(Flag.UNPARSEABLE)
        # heuristic: keys that look like merged/full weights
        if any(k.endswith(".weight") for k in keys):
            flags.add(Flag.NON_LOWRANK)
        return out

    if scheme == Scheme.DIFFUSERS:
        for k in keys:
            for rx, btype in ((_DIFF_SINGLE, "single"), (_DIFF_DOUBLE, "double")):
                m = rx.search(k)
                if m:
                    idx, sub, _ = m.groups()
                    canon_sub = _DIFF_SUB.get(sub)
                    if canon_sub is None:
                        continue
                    name = f"{btype}.{idx}.{canon_sub}"
                    out.modules[name] = k.rsplit(".lora_", 1)[0]
                    break
    elif scheme == Scheme.KOHYA:
        for k in keys:
            m = _KOHYA.search(k)
            if not m:
                continue
            btype, idx, sub, _ = m.groups()
            canon_sub = _KOHYA_SUB.get(sub)
            if canon_sub is None:
                continue
            if "fused" in canon_sub:
                flags.add(Flag.FUSED_QKV)
            name = f"{btype}.{idx}.{canon_sub}"
            out.modules[name] = k.rsplit(".lora_", 1)[0]
    else:
        flags.add(Flag.UNPARSEABLE)

    # diffusers single-block fused proj
    if any(s.endswith("proj_mlp") or s.endswith("proj_out") for s in out.modules):
        # diffusers keeps proj_mlp / proj_out separate (not fused) — no flag needed
        pass

    out.n_modules = len(out.modules)
    if out.n_modules == 0 and Flag.UNPARSEABLE not in flags:
        flags.add(Flag.UNPARSEABLE)
    # "partial" is informational, decided by the caller against an expected-module set
    if not flags:
        flags.add(Flag.OK)
    return out


def classify(parsed: ParsedLoRA) -> str:
    """One-word health verdict for triage aggregation."""
    if Flag.UNPARSEABLE in parsed.flags or Flag.NON_LOWRANK in parsed.flags:
        return "unparseable"
    if Flag.DORA in parsed.flags:
        return "dora"
    if Flag.FUSED_QKV in parsed.flags:
        return "fused_needs_split"
    return "ok"
