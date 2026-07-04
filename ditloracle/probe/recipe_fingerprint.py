"""Rich recipe fingerprint extracted FROM THE WEIGHTS (design doc §B.7.2 A2, §B.6.4-#6).

The recipe-leakage control in the POC-1 gate (§B.7.1c) is only meaningful if the recipe fingerprint
is RICH. A thin fingerprint (rank + size + module-presence — the old RecipeOnlyBaseline) looks
near-chance because it is *under-powered*, not because the split is clean — which would falsely
reassure us. This module computes everything a training *recipe* actually varies, read straight from
the adapter file:

  * naming scheme (kohya/BFL vs diffusers vs unknown)            — which trainer wrote it
  * dtype (fp16 / bf16 / fp32 / fp8)                              — trainer / save precision
  * DoRA vs plain LoRA                                            — PEFT method
  * rank: canonical (modal) rank + whether rank varies per-module — network_dim
  * alpha + alpha/rank scaling ratio (and whether alpha is stored)— network_alpha
  * number of adapted modules + per-block-type coverage          — target-module *set*
  * the TARGET-MODULE SET as a structural pattern (attn / +MLP / +modulation; single/double blocks)
  * fused-vs-split key layout                                    — packing convention
  * per-module ΔW Frobenius-norm distribution (mean/std/max/min)  — overall update magnitude/recipe

WHY "FROM THE WEIGHTS, NOT THE MANIFEST": the CivitAI manifest has no rank/alpha/dtype field — those
only exist in the tensors. So the fingerprint must parse the file (this also makes it auditable on
adapters with no metadata at all).

FIXED DIMENSIONALITY (§B.7.1c): `RecipeFingerprint.matrix(records)` returns a vector of the SAME
length for every adapter. The schema is STRUCTURAL (small categorical buckets + scalar stats), NOT a
per-module one-hot over a corpus-derived module union, so the dimension is a fixed constant
(`feature_names()`), independent of the corpus. This matches `featurizers.py`'s fixed-schema rule
without needing `build_fixed_schema`.

NB on the weight-norm scale feature (an intentional design tension — see module docstring caveat):
ΔW magnitude partly correlates with the semantic signal we WANT the reader to use, so including it in
a *leakage control* is conservative-but-debatable. We include it (a strong recipe control SHOULD have
every signature an adversary could exploit), and expose `include_norms=False` to ablate it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from ditloracle.formats.flux_lora import Flag, Scheme, parse_keys
from ditloracle.formats.safetensors_io import load_lora_factors

Record = dict

# dtype string (safetensors) -> a stable small integer bucket. Unknown -> a catch-all bucket.
_DTYPE_BUCKETS = {
    "F32": 0, "float32": 0,
    "F16": 1, "float16": 1,
    "BF16": 2, "bfloat16": 2,
    "F8_E4M3": 3, "F8_E5M2": 3, "float8_e4m3fn": 3, "float8_e5m2": 3,
}
_N_DTYPE = 4  # fp32, fp16, bf16, fp8 (+ "other" handled as all-zero one-hot)

_SCHEME_ORDER = [Scheme.KOHYA, Scheme.DIFFUSERS, Scheme.UNKNOWN]

# canonical-submodule predicates for the target-module-set pattern
def _is_attn(canon: str) -> bool:
    return ".attn" in canon or canon.endswith("attn_out")
def _is_mlp(canon: str) -> bool:
    s = canon.split(".", 2)[-1]
    return s.startswith("ff") or "mlp" in s
def _is_mod(canon: str) -> bool:
    s = canon.split(".", 2)[-1]
    return s == "mod" or s.startswith("mod.") or s.startswith("mod_ctx")


class RecipeProfile:
    """Raw, human-readable recipe profile for ONE adapter (before vectorization).

    Computed in a single pass over the file. `to_vector()` produces the fixed-dim feature row;
    the named fields are kept for the corpus-distribution report and for tests.
    """

    __slots__ = (
        "scheme", "dtype", "is_dora", "alpha_present",
        "rank", "rank_varies", "alpha", "alpha_rank_ratio",
        "n_modules", "frac_double", "frac_single",
        "frac_attn", "frac_mlp", "frac_mod", "is_fused",
        "frob_mean", "frob_std", "frob_max", "frob_min",
        "ok",
    )

    def __init__(self):
        self.scheme = Scheme.UNKNOWN
        self.dtype = "other"
        self.is_dora = False
        self.alpha_present = False
        self.rank = 0
        self.rank_varies = False
        self.alpha = 0.0
        self.alpha_rank_ratio = 0.0
        self.n_modules = 0
        self.frac_double = 0.0
        self.frac_single = 0.0
        self.frac_attn = 0.0
        self.frac_mlp = 0.0
        self.frac_mod = 0.0
        self.is_fused = False
        self.frob_mean = 0.0
        self.frob_std = 0.0
        self.frob_max = 0.0
        self.frob_min = 0.0
        self.ok = False

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


def profile_adapter(path: str | Path, include_norms: bool = True) -> RecipeProfile:
    """Extract a RecipeProfile from a .safetensors adapter, weights only. Never raises on a
    malformed file — returns a profile with ok=False so the harness can keep a fixed-dim zero row."""
    prof = RecipeProfile()
    path = Path(path)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # dtype: precision of the first lora weight tensor (slice header read; no tensor load)
            wk = next((k for k in keys if k.endswith(".weight")), None)
            if wk is not None:
                prof.dtype = str(f.get_slice(wk).get_dtype())
    except Exception:
        return prof

    parsed = parse_keys(keys)
    prof.scheme = parsed.scheme
    prof.is_dora = Flag.DORA in parsed.flags
    prof.is_fused = Flag.FUSED_QKV in parsed.flags
    prof.alpha_present = any(k.endswith(".alpha") for k in keys)

    try:
        raw = load_lora_factors(path)
    except Exception:
        return prof
    if not raw:
        return prof

    ranks = [e["r"] for e in raw.values()]
    prof.rank = int(Counter(ranks).most_common(1)[0][0]) if ranks else 0
    prof.rank_varies = len(set(ranks)) > 1
    alphas = [e["alpha"] for e in raw.values() if e["alpha"] is not None]
    if alphas:
        prof.alpha = float(np.median(alphas))
        prof.alpha_rank_ratio = float(prof.alpha / prof.rank) if prof.rank else 0.0

    # target-module set pattern over the CANONICAL (split) module names
    canon = list(parsed.modules.keys())
    prof.n_modules = len(canon)
    if canon:
        nd = sum(1 for c in canon if c.startswith("double."))
        ns = sum(1 for c in canon if c.startswith("single."))
        prof.frac_double = nd / len(canon)
        prof.frac_single = ns / len(canon)
        prof.frac_attn = sum(1 for c in canon if _is_attn(c)) / len(canon)
        prof.frac_mlp = sum(1 for c in canon if _is_mlp(c)) / len(canon)
        prof.frac_mod = sum(1 for c in canon if _is_mod(c)) / len(canon)

    if include_norms:
        frobs = []
        for e in raw.values():
            B, A = e["B"], e["A"]
            # ‖ΔW‖_F = ‖B A‖_F, computed without forming ΔW (Gram trick): tr(AᵀBᵀBA)=‖BᵀB·…‖,
            # but the cheap exact route is ‖B‖_F·-free: use ‖BA‖_F via small r×r products.
            try:
                BtB = B.to(torch.float32).T @ B.to(torch.float32)   # r×r
                AAt = A.to(torch.float32) @ A.to(torch.float32).T   # r×r
                frob = float(torch.sqrt(torch.clamp((BtB * AAt).sum(), min=0.0)).item())
                frobs.append(frob)
            except Exception:
                continue
        if frobs:
            f = np.asarray(frobs, dtype=np.float64)
            prof.frob_mean = float(f.mean())
            prof.frob_std = float(f.std())
            prof.frob_max = float(f.max())
            prof.frob_min = float(f.min())

    prof.ok = prof.n_modules > 0
    return prof


def _vectorize(prof: RecipeProfile, include_norms: bool) -> list[float]:
    # scheme one-hot (3). UNKNOWN is treated as a "no-info" all-zero code (like dtype "other"),
    # so an unparseable/missing file degrades to an all-zero row rather than a spurious positive bit.
    scheme_oh = [1.0 if (prof.scheme == s and s != Scheme.UNKNOWN) else 0.0 for s in _SCHEME_ORDER]
    # dtype one-hot (4); unknown -> all zeros
    dt = _DTYPE_BUCKETS.get(prof.dtype)
    dtype_oh = [0.0] * _N_DTYPE
    if dt is not None:
        dtype_oh[dt] = 1.0
    # rank: raw + log1p (log so 2 vs 128 are comparable scales), and the varies bit
    rank_feats = [float(prof.rank), float(np.log1p(prof.rank)), 1.0 if prof.rank_varies else 0.0]
    alpha_feats = [
        1.0 if prof.alpha_present else 0.0,
        float(prof.alpha),
        float(prof.alpha_rank_ratio),
    ]
    method_feats = [1.0 if prof.is_dora else 0.0, 1.0 if prof.is_fused else 0.0]
    set_feats = [
        float(prof.n_modules), float(np.log1p(prof.n_modules)),
        prof.frac_double, prof.frac_single,
        prof.frac_attn, prof.frac_mlp, prof.frac_mod,
    ]
    row = scheme_oh + dtype_oh + rank_feats + alpha_feats + method_feats + set_feats
    if include_norms:
        row += [prof.frob_mean, prof.frob_std, prof.frob_max, prof.frob_min]
    return row


def feature_names(include_norms: bool = True) -> list[str]:
    names = [f"scheme={s.value}" for s in _SCHEME_ORDER]
    names += ["dtype=fp32", "dtype=fp16", "dtype=bf16", "dtype=fp8"]
    names += ["rank", "log1p_rank", "rank_varies"]
    names += ["alpha_present", "alpha", "alpha_rank_ratio"]
    names += ["is_dora", "is_fused"]
    names += ["n_modules", "log1p_n_modules",
              "frac_double", "frac_single", "frac_attn", "frac_mlp", "frac_mod"]
    if include_norms:
        names += ["frob_mean", "frob_std", "frob_max", "frob_min"]
    return names


class RecipeFingerprint:
    """Rich recipe-signature control (design doc §B.7.2 A2). Drop-in for RecipeOnlyBaseline in the
    POC-1 harness: exposes `.fit()` / `.matrix(records)` → (X, names) over a FIXED-dim schema.

    The fingerprint is read from each record's `local_path` (the weight file). A small in-instance
    cache keeps it to one file pass per adapter even though fit() and matrix() both touch it.

    `include_norms=False` ablates the ΔW-Frobenius block (the one feature that partly correlates with
    the semantic signal — see module docstring)."""

    name = "recipe_fingerprint"

    def __init__(self, include_norms: bool = True, path_of=None):
        self.include_norms = include_norms
        # callable record -> path; default reads r["local_path"]
        self.path_of = path_of or (lambda r: r.get("local_path"))
        self._cache: dict[str, RecipeProfile] = {}

    def _profile(self, r: Record) -> RecipeProfile:
        p = self.path_of(r)
        if p is None:
            return RecipeProfile()
        if p not in self._cache:
            self._cache[p] = profile_adapter(p, include_norms=self.include_norms)
        return self._cache[p]

    def fit(self, records: list[Record]):
        for r in records:
            self._profile(r)   # warm the cache
        return self

    def matrix(self, records: list[Record]):
        names = feature_names(self.include_norms)
        X = np.zeros((len(records), len(names)), dtype=np.float32)
        for i, r in enumerate(records):
            X[i, :] = _vectorize(self._profile(r), self.include_norms)
        return X, names

    # convenience for tests / the corpus report
    def profiles(self, records: list[Record]) -> list[RecipeProfile]:
        return [self._profile(r) for r in records]
