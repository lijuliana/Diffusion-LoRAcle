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
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor
D_MODEL_FLUX = 3072


def split_fused(canonical_name: str, B: Tensor, A: Tensor, d_model: int = D_MODEL_FLUX) -> list[tuple]:
    """Return [(subname, B_sub, A_sub), ...]. Non-fused modules return a single passthrough entry.

    `canonical_name` is the flux_lora.py canonical name (e.g. "double.0.attn.qkv_fused_img",
    "single.3.proj_fused_qkv_mlp", "single.3.proj_out"). Sub-names reuse the diffusers-style suffixes
    so fused and unfused corpora share one module vocabulary.
    """
    d_out, r = B.shape
    r2, d_in = A.shape
    assert r == r2, f"rank mismatch for {canonical_name}: B has r={r}, A has r={r2}"

    prefix, sub = _split_prefix(canonical_name)

    # ---- output-fused: split B rows, share A ----
    if sub == "attn.qkv_fused_img":
        return _split_b_rows(prefix, B, A, [("attn.to_q", d_model), ("attn.to_k", d_model), ("attn.to_v", d_model)])
    if sub == "attn.qkv_fused_txt":
        return _split_b_rows(prefix, B, A, [("attn.add_q", d_model), ("attn.add_k", d_model), ("attn.add_v", d_model)])
    if sub == "proj_fused_qkv_mlp":   # single-block linear1
        mlp = d_out - 3 * d_model
        if mlp <= 0:
            return [(canonical_name, B, A)]  # unexpected shape → don't corrupt; pass through
        return _split_b_rows(prefix, B, A,
                             [("attn.to_q", d_model), ("attn.to_k", d_model), ("attn.to_v", d_model), ("proj_mlp", mlp)])

    # ---- input-fused: split A columns, share B (single-block linear2 / proj_out) ----
    if sub == "proj_out" and d_in > d_model:
        mlp = d_in - d_model
        return _split_a_cols(prefix, B, A, [("attn_out", d_model), ("mlp_out", mlp)])

    # ---- not fused ----
    return [(canonical_name, B, A)]


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


def _split_b_rows(prefix: str, B: Tensor, A: Tensor, spec: list[tuple[str, int]]) -> list[tuple]:
    out, off = [], 0
    for sub, width in spec:
        out.append((f"{prefix}.{sub}", B[off:off + width, :].contiguous(), A))
        off += width
    assert off == B.shape[0], f"row split {off} != {B.shape[0]} for {prefix}"
    return out


def _split_a_cols(prefix: str, B: Tensor, A: Tensor, spec: list[tuple[str, int]]) -> list[tuple]:
    out, off = [], 0
    for sub, width in spec:
        out.append((f"{prefix}.{sub}", B, A[:, off:off + width].contiguous()))
        off += width
    assert off == A.shape[1], f"col split {off} != {A.shape[1]} for {prefix}"
    return out
