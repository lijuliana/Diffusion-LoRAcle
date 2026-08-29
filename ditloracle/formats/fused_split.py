"""Split fused FLUX LoRA modules into their conceptual sub-modules before SVD.

The kohya/BFL format packs several projections into one matrix (a packing optimization). ~72% of
real FLUX LoRAs use it (POC-0d triage), so this is the common case, not an edge case. Un-fusing
recovers the per-module structure the diffusers format already has, making the two schemes
comparable and giving the encoder per-module (q/k/v/mlp) direction tokens.

Verified real shapes (d_model = 3072, rank r):
  double-block  *_attn_qkv :  B=(3*d_model, r)=(9216,r),  A=(r, d_model)=(r,3072)   [output-fused]
  single-block  linear1    :  B=(3*d_model + mlp, r)=(21504,r), A=(r, d_model)       [output-fused]
  single-block  linear2    :  B=(d_model, r)=(3072,r), A=(r, d_model + mlp)=(r,15360) [input-fused]

Output-fused → split B's ROWS (shared A). Input-fused → split A's COLUMNS (shared B).
mlp width is derived from the actual tensor shape, not hard-coded, so it survives FLUX variants.

The name tables below cover BOTH vocabularies, because the split has to happen on every path:
  * the canonical FLUX.1 names `flux_lora.parse_keys` produces (`attn.qkv_fused_img`, ...);
  * the RAW key stems `safetensors_io.raw_stem_modules` falls back to for schemes that parser does
    not know — FLUX.2-klein trains `double_blocks.N.img_attn.qkv` (3-fused) and
    `single_blocks.N.linear1` (qkv+mlp), so the whole minted corpus arrives on this path.
Every rule is SHAPE-CHECKED (`fused_spec`): a name that looks fused but whose widths do not divide is
passed through rather than corrupted.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor
D_MODEL_FLUX = 3072

# Sub-names (the part after "{block}.{idx}.") that pack several projections into one matrix.
_QKV_IMG = frozenset({"attn.qkv_fused_img",      # canonical FLUX.1 (kohya img_attn_qkv)
                      "img_attn.qkv",             # raw BFL/klein stem
                      "attn.qkv", "attn.to_qkv"})  # generic single-stream
_QKV_TXT = frozenset({"attn.qkv_fused_txt", "txt_attn.qkv"})
_QKV_MLP = frozenset({"proj_fused_qkv_mlp",       # canonical FLUX.1 single-block linear1
                      "linear1",                   # raw BFL/klein single-block stem
                      "attn.to_qkv_mlp_proj"})     # FLUX.2-klein diffusers name
_IN_FUSED = frozenset({"proj_out", "linear2"})     # single-block linear2: attn+mlp on the INPUT side


def fused_spec(canonical_name: str, d_out: int, d_in: int,
               d_model: int = D_MODEL_FLUX) -> tuple[str, list[tuple[str, int]]] | None:
    """The split plan for a module from its NAME + SHAPE alone, or None if it is not fused.

    Returns ("rows"|"cols", [(subname, width), ...]). Shape-only (no tensors), so a cheap key-scan
    can predict exactly what `split_fused` will produce — that is what keeps the schema scan and the
    loader from disagreeing (a mismatch silently empties `keep_modules`).
    """
    prefix, sub = _split_prefix(canonical_name)
    if sub in _QKV_IMG and d_out == 3 * d_model:
        return "rows", [(f"{prefix}.attn.to_q", d_model), (f"{prefix}.attn.to_k", d_model),
                        (f"{prefix}.attn.to_v", d_model)]
    if sub in _QKV_TXT and d_out == 3 * d_model:
        return "rows", [(f"{prefix}.attn.add_q", d_model), (f"{prefix}.attn.add_k", d_model),
                        (f"{prefix}.attn.add_v", d_model)]
    if sub in _QKV_MLP and d_out > 3 * d_model:
        return "rows", [(f"{prefix}.attn.to_q", d_model), (f"{prefix}.attn.to_k", d_model),
                        (f"{prefix}.attn.to_v", d_model), (f"{prefix}.proj_mlp", d_out - 3 * d_model)]
    if sub in _IN_FUSED and d_in > d_model:
        return "cols", [(f"{prefix}.attn_out", d_model), (f"{prefix}.mlp_out", d_in - d_model)]
    return None


def is_fused_shape(canonical_name: str, d_out: int, d_in: int,
                   d_model: int = D_MODEL_FLUX) -> bool:
    """Would this module still need splitting? The guard featurizers use to refuse fused input."""
    return fused_spec(canonical_name, d_out, d_in, d_model) is not None


def split_fused(canonical_name: str, B: Tensor, A: Tensor, d_model: int = D_MODEL_FLUX) -> list[tuple]:
    """Return [(subname, B_sub, A_sub), ...]. Non-fused modules return a single passthrough entry.

    `canonical_name` is a flux_lora.py canonical name (e.g. "double.0.attn.qkv_fused_img",
    "single.3.proj_fused_qkv_mlp") or a raw key stem ("double_blocks.0.img_attn.qkv"). Sub-names
    reuse the diffusers-style suffixes so fused and unfused corpora share one module vocabulary.
    """
    d_out, r = B.shape
    r2, d_in = A.shape
    assert r == r2, f"rank mismatch for {canonical_name}: B has r={r}, A has r={r2}"

    spec = fused_spec(canonical_name, d_out, d_in, d_model)
    if spec is None:
        return [(canonical_name, B, A)]
    axis, parts = spec
    return _split_b_rows(parts, B, A) if axis == "rows" else _split_a_cols(parts, B, A)


def fused_subnames(canonical_name: str) -> list[str]:
    """The canonical sub-module names `split_fused` WOULD produce, derived from the name pattern
    alone (sub-names don't depend on tensor widths — only the slicing does). Lets a corpus loader pick
    a shared module schema from a cheap key-only scan, without loading any tensor data.

    The width-ambiguous `proj_out` case (passthrough vs input-fused attn_out/mlp_out) returns all
    three names; the extra one is harmless because the schema is later intersected with what actually
    loads. Everything else is exact.
    """
    prefix, sub = _split_prefix(canonical_name)
    if sub == "attn.qkv_fused_img":
        return [f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"]
    if sub == "attn.qkv_fused_txt":
        return [f"{prefix}.attn.add_q", f"{prefix}.attn.add_k", f"{prefix}.attn.add_v"]
    if sub == "proj_fused_qkv_mlp":
        return [f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v", f"{prefix}.proj_mlp"]
    if sub == "proj_out":
        return [canonical_name, f"{prefix}.attn_out", f"{prefix}.mlp_out"]
    return [canonical_name]


def _split_prefix(name: str) -> tuple[str, str]:
    """'double.0.attn.qkv_fused_img' -> ('double.0', 'attn.qkv_fused_img')."""
    parts = name.split(".")
    return ".".join(parts[:2]), ".".join(parts[2:])


def _split_b_rows(parts: list[tuple[str, int]], B: Tensor, A: Tensor) -> list[tuple]:
    out, off = [], 0
    for name, width in parts:
        out.append((name, B[off:off + width, :].contiguous(), A))
        off += width
    assert off == B.shape[0], f"row split {off} != {B.shape[0]} for {parts[0][0]}"
    return out


def _split_a_cols(parts: list[tuple[str, int]], B: Tensor, A: Tensor) -> list[tuple]:
    out, off = [], 0
    for name, width in parts:
        out.append((name, B, A[:, off:off + width].contiguous()))
        off += width
    assert off == A.shape[1], f"col split {off} != {A.shape[1]} for {parts[0][0]}"
    return out
