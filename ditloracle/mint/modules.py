"""Target-module vocabularies per base model — the single source of truth for both planners.

ai-toolkit selects which linears get a LoRA by SUBSTRING MATCH against the real module names, so a
name from the wrong architecture matches nothing and the trainer aborts with "There are not any lora
modules in this network". These lists were read off the actual models, not guessed:

  FLUX.1-dev        19 double + 38 single blocks; MLP `ff.net.0.proj`/`ff.net.2`;
                    modulation `norm1.linear`/`norm1_context.linear`.
  FLUX.2-klein-4B    5 double + 20 single blocks; MLP `ff.linear_in`/`ff.linear_out`;
                    modulation `double_stream_modulation*`/`single_stream_modulation*`;
                    single blocks fuse qkv+mlp (`attn.to_qkv_mlp_proj`).

Lives in its own module because both `mint.corpus_plan` (capability + safety organisms) and
`safety.mint_spec` (the causal-gate matched sets) need it, and corpus_plan already imports mint_spec.
"""

from __future__ import annotations

_FLUX1_ATTN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
_FLUX1_MLP = ["ff.net.0.proj", "ff.net.2"]
_FLUX1_MOD = ["norm1.linear", "norm1_context.linear"]

_KLEIN_ATTN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
               "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"]
_KLEIN_MLP = ["ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out"]
_KLEIN_MOD = ["double_stream_modulation", "single_stream_modulation"]

MODULE_SETS_BY_BASE = {
    "flux1": {
        "attn_only": _FLUX1_ATTN,
        "attn_mlp": _FLUX1_ATTN + _FLUX1_MLP,
        "attn_mlp_mod": _FLUX1_ATTN + _FLUX1_MLP + _FLUX1_MOD,
    },
    "klein": {
        "attn_only": _KLEIN_ATTN,
        "attn_mlp": _KLEIN_ATTN + _KLEIN_MLP,
        "attn_mlp_mod": _KLEIN_ATTN + _KLEIN_MLP + _KLEIN_MOD,
    },
}


def base_family(base_model: str) -> str:
    return "klein" if "klein" in (base_model or "").lower() else "flux1"


def module_sets_for(base_model: str) -> dict[str, list[str]]:
    return MODULE_SETS_BY_BASE[base_family(base_model)]


def reference_modules(base_model: str) -> list[str]:
    """The clamped reference recipe's module set (attn+MLP, the common LoRA target)."""
    return list(module_sets_for(base_model)["attn_mlp"])
