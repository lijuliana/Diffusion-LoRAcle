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


def canonical_module_names(path: str | Path) -> set[str]:
    """CHEAP schema scan: the set of canonical (post-fused-split) module names in a file, WITHOUT
    loading any tensor data. Used to pick a shared module schema across a corpus before deciding which
    factors to actually load into RAM (memory-bounded corpus loading)."""
    from ditloracle.formats.flux_lora import parse_keys
    from ditloracle.formats.fused_split import fused_subnames

    parsed = parse_keys(read_keys(path))
    names: set[str] = set()
    for canon in parsed.modules:
        names.update(fused_subnames(canon))
    return names


def load_canonical_factors(path: str | Path, keep_modules: set[str] | None = None) -> dict:
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
    return out
