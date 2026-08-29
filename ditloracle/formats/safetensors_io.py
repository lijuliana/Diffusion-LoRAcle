"""Load LoRA (B, A) factor pairs from a real .safetensors file, keyed by canonical module name.

Bridges the on-disk key zoo (parsed by flux_lora.py) to (B, A) tensors the encoder consumes.
Handles the diffusers (`lora_A`/`lora_B`) and kohya/BFL (`lora_down`/`lora_up`) suffix conventions,
and reads `alpha` if present. No model download — operates on the adapter file alone.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from ditloracle.formats.flux_lora import Scheme, detect_scheme


# suffix pairs: (down/A side -> rank×d_in), (up/B side -> d_out×rank)
_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight")
_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight")
_ALPHA_SUFFIXES = (".alpha", ".lora_alpha")


def _stem(key: str) -> str | None:
    for suf in _A_SUFFIXES + _B_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    return None


def load_lora_factors(path: str | Path) -> dict:
    """Return {stem: {"A": Tensor[r,d_in], "B": Tensor[d_out,r], "alpha": float|None, "r": int}}.

    `stem` is the raw key stem (scheme-specific); pair canonical names via flux_lora.parse_keys
    on the same file's keys if you need canonical module identities. Tensors are float32 CPU.
    """
    path = Path(path)
    out: dict[str, dict] = {}
    alphas: dict[str, float] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith(_ALPHA_SUFFIXES):
                base = k.rsplit(".", 1)[0]
                try:
                    alphas[base] = float(f.get_tensor(k).item())
                except Exception:
                    pass
                continue
            stem = _stem(k)
            if stem is None:
                continue
            entry = out.setdefault(stem, {"A": None, "B": None})
            t = f.get_tensor(k).to(torch.float32)
            if k.endswith(_A_SUFFIXES):
                entry["A"] = t           # [r, d_in]
            else:
                entry["B"] = t           # [d_out, r]
    # finalize: keep only complete pairs, attach alpha + rank
    factors = {}
    for stem, e in out.items():
        if e["A"] is None or e["B"] is None:
            continue
        A, B = e["A"], e["B"]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            continue  # not a clean low-rank pair (e.g. conv/DoRA-magnitude/merged)
        factors[stem] = {
            "A": A,
            "B": B,
            "alpha": alphas.get(stem),
            "r": int(A.shape[0]),
        }
    return factors


def read_keys(path: str | Path) -> list[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return list(f.keys())


def detect_file_scheme(path: str | Path) -> Scheme:
    return detect_scheme(read_keys(path))


_RAW_PREFIXES = ("transformer.", "diffusion_model.", "lora_transformer_", "base_model.model.")


def raw_stem_modules(path: str | Path) -> dict[str, str]:
    """Fallback module identities for a scheme the FLUX.1 parser does not recognize.

    `flux_lora.parse_keys` maps FLUX.1 key schemes to canonical names. Other architectures (FLUX.2
    klein, and anything we port to later) use different names, so the parser returns nothing and the
    file looks empty. For a comparison ACROSS ORGANISMS THAT SHARE A BASE AND RECIPE — which is
    exactly the causal gate — canonical names are not required; we only need module identities that
    are consistent between files. The raw key stem is that identity.

    Returns {module_name: raw_stem}. Names are the stem minus a known wrapper prefix, so the same
    module lines up whether or not the trainer prefixed it.
    """
    mods: dict[str, str] = {}
    for k in read_keys(path):
        stem = _stem(k)
        if stem is None:
            continue
        name = stem
        for p in _RAW_PREFIXES:
            if name.startswith(p):
                name = name[len(p):]
                break
        mods[name] = stem
    return mods


def raw_stem_dims(path: str | Path) -> dict[str, tuple[int, int]]:
    """{module_name: (d_out, d_in)} for the raw-stem fallback, read from the safetensors HEADER only.

    Shapes (not just names) are needed so the cheap schema scan can predict the fused split exactly
    the way `load_canonical_factors` performs it. `get_slice(...).get_shape()` reads header metadata;
    no tensor data is materialized.
    """
    mods = raw_stem_modules(path)
    shapes: dict[str, list[int]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            stem = _stem(k)
            if stem is not None:
                shapes[k] = list(f.get_slice(k).get_shape())
    dims: dict[str, tuple[int, int]] = {}
    for name, stem in mods.items():
        b = next((s for k, s in shapes.items() if k.startswith(stem) and k.endswith(_B_SUFFIXES)), None)
        a = next((s for k, s in shapes.items() if k.startswith(stem) and k.endswith(_A_SUFFIXES)), None)
        if b and a and len(b) == 2 and len(a) == 2:
            dims[name] = (b[0], a[1])
    return dims


def canonical_module_names(path: str | Path, allow_raw_fallback: bool = True) -> set[str]:
    """CHEAP schema scan: the set of canonical (post-fused-split) module names in a file, WITHOUT
    loading any tensor data. Used to pick a shared module schema across a corpus before deciding which
    factors to actually load into RAM (memory-bounded corpus loading)."""
    from ditloracle.formats.flux_lora import parse_keys
    from ditloracle.formats.fused_split import fused_spec, fused_subnames

    parsed = parse_keys(read_keys(path))
    names: set[str] = set()
    for canon in parsed.modules:
        names.update(fused_subnames(canon))
    if not names and allow_raw_fallback:
        # Same split the loader applies, predicted from name+shape, so the schema this scan feeds to
        # `keep_modules` cannot silently filter out every module the loader emits.
        for name, (d_out, d_in) in raw_stem_dims(path).items():
            spec = fused_spec(name, d_out, d_in)
            names.update([sub for sub, _ in spec[1]] if spec else [name])
    return names


def load_canonical_factors(path: str | Path, keep_modules: set[str] | None = None,
                           allow_raw_fallback: bool = True) -> dict:
    """Load a LoRA as {canonical_module_name: (B, A, alpha, r, use_rslora)}, with fused modules
    SPLIT into their sub-modules (q/k/v/mlp) — the format the encoder/featurizers expect.

    This is the bridge that makes kohya (fused) and diffusers corpora share one module vocabulary.
    `use_rslora` is left False (real adapters rarely tag it; scaling folds α/r by default).

    `keep_modules`: if given, only canonical modules in this set are RETAINED (others are dropped and
    their tensors never held) — essential for memory-bounded corpus loading, since the probe only ever
    uses a fixed ~60-module schema but a raw file can carry 500+ modules (~350 MB of factors each).
    """
    from ditloracle.formats.flux_lora import parse_keys
    from ditloracle.formats.fused_split import split_fused

    path = Path(path)
    keys = read_keys(path)
    parsed = parse_keys(keys)                       # canonical_name -> raw key stem
    raw = load_lora_factors(path)                   # raw stem -> {A,B,alpha,r}

    out: dict[str, tuple] = {}
    for canon, raw_stem in parsed.modules.items():
        if raw_stem not in raw:
            continue
        e = raw[raw_stem]
        for subname, B_sub, A_sub in split_fused(canon, e["B"], e["A"]):
            if keep_modules is not None and subname not in keep_modules:
                continue
            out[subname] = (B_sub, A_sub, e["alpha"], int(A_sub.shape[0]), False)

    if not out and allow_raw_fallback:
        # Scheme the FLUX.1 parser does not know (FLUX.2 klein, ported bases). Fall back to raw key
        # stems as module identities — consistent across files that share a base, which is what the
        # causal gate compares. Without this the caller sees an empty dict and skips the organism
        # SILENTLY, so the gate would run on nothing and report failure.
        # The fused split runs HERE TOO: klein packs qkv (`img_attn.qkv`, 3·3072 rows) and qkv+mlp
        # (`single_blocks.N.linear1`), so skipping it would hand the featurizers one SVD over three
        # different projections — the exact defect PLAN §7.3 is about, on the minted corpus.
        for name, raw_stem in raw_stem_modules(path).items():
            if raw_stem not in raw:
                continue
            e = raw[raw_stem]
            for subname, B_sub, A_sub in split_fused(name, e["B"], e["A"]):
                if keep_modules is not None and subname not in keep_modules:
                    continue
                out[subname] = (B_sub, A_sub, e["alpha"], int(A_sub.shape[0]), False)
    return out
