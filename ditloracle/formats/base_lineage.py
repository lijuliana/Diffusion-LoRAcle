"""Base-lineage verification for FLUX LoRA adapters (component A3, design doc §B.4.4 / §B.7.2).

The project's core GL(r)/permutation symmetry argument (§B.4.4) is CONDITIONAL on every adapter
sharing one frozen base checkpoint (pristine FLUX.1-dev). If an adapter was trained on a *merged
community checkpoint* instead, two things break: (1) the fixed-base permutation argument, and
(2) near-duplicate bases leak across train/test splits. The pre-existing triage only checked that
module dimensions match FLUX width (3072) — that confirms FLUX *architecture*, never the actual
*checkpoint lineage*.

THE HARD TRUTH about what's verifiable from a LoRA alone
--------------------------------------------------------
A LoRA file contains only the low-rank *diff*, never the base weights. So "what base was this
trained on" can only be *inferred* from side-channels, none of which is a cryptographic proof:

  1. ``ss_new_sd_model_hash`` (kohya) / ``ss_sd_model_hash`` — a hash of the *base* checkpoint the
     trainer loaded. This is the STRONGEST signal: when present and equal to the canonical
     FLUX.1-dev hash, lineage is high-confidence. When present but DIFFERENT, the base is a
     different (merged/finetuned) checkpoint — a genuine off-base detection.
  2. ``ss_base_model_version`` — a coarse family token (``flux1`` / ``sd_1.5`` / ...). RELIABLE for
     the kohya/sd-scripts trainer, but NOT for ai-toolkit, which hardcodes ``sd_1.5`` regardless of
     the real base (observed empirically — see ``KNOWN_FALSE_BASE_VERSION``). Treated as weak.
  3. ``ss_sd_model_name`` / ``modelspec.architecture`` — a local *filename* or arch tag. The
     filename is often a CivitAI model id (e.g. ``691639.safetensors``) and says nothing reliable;
     the arch tag (``flux-1-dev/lora``) confirms architecture, not checkpoint.
  4. Manifest ``base_model`` (CivitAI scrape) — creator-declared, observed 100% "Flux.1 D" on our
     corpus *including files whose embedded metadata contradicts it*. Treated as a declaration only.
  5. Structural fingerprint (module set / FLUX 3072 width / MMDiT submodule signature) — confirms
     the adapter targets a FLUX-family transformer, but CANNOT distinguish pristine FLUX.1-dev from
     a merged FLUX base (both share the identical module layout). Architecture != checkpoint.

Therefore exact base recovery is IMPOSSIBLE from a LoRA alone for the large fraction of files whose
trainer did not record a base hash. The honest deliverable is a CALIBRATED CONFIDENCE plus a flag
set that lets the pipeline *stratify* (verified-FLUX.1-dev / FLUX-family-unverified / off-base /
non-FLUX), not a false guarantee. ``verify_base_lineage`` returns exactly that.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# --- canonical reference -----------------------------------------------------------------
# Hash of the pristine FLUX.1-dev base checkpoint as recorded by kohya/sd-scripts in
# ``ss_new_sd_model_hash``. Empirically the dominant hash on the FLUX.1-dev corpus and the value
# associated with every ``flux1-dev.*`` / ``flux_dev.*`` model name; also the value for files whose
# ``ss_sd_model_name`` is a CivitAI id (691639.safetensors) that nonetheless resolves to this hash.
FLUX1_DEV_BASE_HASHES = frozenset(
    {
        "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7",
    }
)

# FLUX-family base-version tokens (kohya writes ``flux1``; some write a dev-ish variant).
FLUX_BASE_VERSION_TOKENS = frozenset({"flux1", "flux", "flux.1", "flux1-dev", "flux-dev", "sd_flux-dev"})

# Base-version tokens that are KNOWN to be unreliable for certain trainers. ai-toolkit hardcodes
# ``sd_1.5`` into ``ss_base_model_version`` for FLUX runs; the value must be IGNORED (not trusted as
# an SD1.5 detection) when the trainer is ai-toolkit and the structure is FLUX.
KNOWN_FALSE_BASE_VERSION = {"ai-toolkit": frozenset({"sd_1.5"})}

# Hashes that are NOT a base hash even though they appear in a hash field:
#   e3b0c442...  = SHA256 of empty input (trainer recorded nothing)
#   IsADirectory = base was a directory; hash uncomputable
_NON_HASH_SENTINELS = frozenset({"isadirectory", "", "none"})
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

EXPECTED_FLUX_WIDTH = 3072  # FLUX.1-dev hidden size

# safety cap: a corrupt/hostile header should never trigger a multi-GB JSON parse
_MAX_HEADER_BYTES = 50_000_000


class BaseClass(str, Enum):
    """Coarse lineage stratum — the unit the pipeline filters/stratifies on."""

    FLUX1_DEV_VERIFIED = "flux1_dev_verified"          # base hash matches canonical FLUX.1-dev
    FLUX_FAMILY_UNVERIFIED = "flux_family_unverified"   # FLUX arch/metadata, no hash to confirm pristine
    OFF_BASE_FLUX_MERGE = "off_base_flux_merge"         # FLUX arch but a DIFFERENT base hash (merge/finetune)
    NON_FLUX = "non_flux"                               # not a FLUX-architecture adapter
    UNKNOWN = "unknown"                                 # no usable signal at all


class LineageFlag(str, Enum):
    BASE_HASH_MATCH = "base_hash_match"                 # strong: hash == canonical FLUX.1-dev
    BASE_HASH_MISMATCH = "base_hash_mismatch"           # strong: hash present and != canonical -> off-base
    NO_BASE_HASH = "no_base_hash"                       # trainer recorded no usable base hash
    FLUX_VERSION_TOKEN = "flux_version_token"           # ss_base_model_version is a flux token
    FALSE_BASE_VERSION = "false_base_version"           # ss_base_model_version is a known-bogus token
    FLUX_ARCH_TAG = "flux_arch_tag"                     # modelspec.architecture says flux-*-dev
    FLUX_WIDTH = "flux_width"                            # structural 3072 width present
    NON_FLUX_WIDTH = "non_flux_width"                   # structural width is NOT FLUX
    NO_METADATA = "no_metadata"                          # safetensors header had no __metadata__
    MANIFEST_CONTRADICTS_METADATA = "manifest_contradicts_metadata"
    HEADER_UNREADABLE = "header_unreadable"


@dataclass
class LineageVerdict:
    base_class: BaseClass
    confidence: float                                   # calibrated 0..1 (see _score)
    declared_base: str | None = None                    # manifest base_model (creator-declared)
    metadata_base_version: str | None = None            # ss_base_model_version
    metadata_base_hash: str | None = None               # ss_new_sd_model_hash / ss_sd_model_hash
    metadata_model_name: str | None = None              # ss_sd_model_name (often a local filename)
    arch_tag: str | None = None                         # modelspec.architecture
    trainer: str | None = None                          # ai-toolkit / sd-scripts(kohya) / diffusers
    flags: set[LineageFlag] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "base_class": self.base_class.value,
            "confidence": round(self.confidence, 3),
            "declared_base": self.declared_base,
            "metadata_base_version": self.metadata_base_version,
            "metadata_base_hash": self.metadata_base_hash,
            "metadata_model_name": self.metadata_model_name,
            "arch_tag": self.arch_tag,
            "trainer": self.trainer,
            "flags": sorted(f.value for f in self.flags),
            "notes": self.notes,
        }


# --- low-level metadata reader -----------------------------------------------------------
def read_safetensors_metadata(path: str | Path) -> dict | None:
    """Return the safetensors ``__metadata__`` dict, or ``None`` if absent/unreadable.

    Reads only the JSON header (no tensor data, no model download). Guards against a corrupt or
    hostile header-length field that would otherwise trigger an OOM read.
    """
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > _MAX_HEADER_BYTES:
                return None
            header = json.loads(f.read(n))
    except Exception:
        return None
    md = header.get("__metadata__")
    return md if isinstance(md, dict) else None


def _detect_trainer(md: dict) -> str | None:
    """Best-effort trainer-software identification from metadata."""
    soft = md.get("software")
    if isinstance(soft, str) and soft:
        try:
            name = json.loads(soft).get("name")
            if name:
                return str(name)
        except Exception:
            pass
    netmod = md.get("ss_network_module", "")
    if "lora_flux" in netmod or "lora" == netmod or md.get("ss_new_sd_model_hash"):
        return "sd-scripts"  # kohya family records ss_* fields
    impl = md.get("modelspec.implementation", "")
    if "diffusers" in impl:
        return "diffusers"
    return None


def _norm_hash(h: str | None) -> str | None:
    if not h:
        return None
    h = str(h).strip().lower()
    if h in _NON_HASH_SENTINELS or h == _EMPTY_SHA256:
        return None
    return h


# --- the verdict -------------------------------------------------------------------------
def verify_base_lineage(
    path: str | Path | None = None,
    *,
    metadata: dict | None = None,
    declared_base: str | None = None,
    widths: set[int] | None = None,
    is_flux_structure: bool | None = None,
) -> LineageVerdict:
    """Extract every available base-lineage signal and return a calibrated verdict.

    Args:
      path: adapter .safetensors path (metadata read from it unless ``metadata`` is given).
      metadata: pre-read ``__metadata__`` dict (skips file I/O; pass ``{}`` for "read, but empty").
      declared_base: the CivitAI/HF manifest ``base_model`` field (creator declaration; weak).
      widths: optional set of observed d_in/d_out widths (from load_lora_factors). If 3072 is
              present the structure is FLUX-family; used as the structural fallback signal.
      is_flux_structure: optional explicit override of the structural FLUX check (e.g. caller
              already ran parse_keys and knows the canonical module set is FLUX MMDiT).

    The confidence is calibrated to the *strength* of the available evidence, NOT to an accuracy we
    cannot measure (we have no ground-truth base for wild files). It answers "how strongly does the
    evidence pin the base checkpoint", which is what stratification needs.
    """
    if metadata is None:
        metadata = read_safetensors_metadata(path) if path is not None else {}
        header_unreadable = metadata is None
        if metadata is None:
            metadata = {}
    else:
        header_unreadable = False

    v = LineageVerdict(base_class=BaseClass.UNKNOWN, confidence=0.0, declared_base=declared_base)
    if header_unreadable:
        v.flags.add(LineageFlag.HEADER_UNREADABLE)

    # --- gather raw signals ---
    base_version = metadata.get("ss_base_model_version")
    base_hash = _norm_hash(metadata.get("ss_new_sd_model_hash") or metadata.get("ss_sd_model_hash"))
    model_name = metadata.get("ss_sd_model_name")
    arch_tag = metadata.get("modelspec.architecture")
    trainer = _detect_trainer(metadata)

    v.metadata_base_version = base_version
    v.metadata_base_hash = base_hash
    v.metadata_model_name = model_name
    v.arch_tag = arch_tag
    v.trainer = trainer

    if not metadata:
        v.flags.add(LineageFlag.NO_METADATA)

    # --- structural FLUX check (architecture, not checkpoint) ---
    if is_flux_structure is None:
        is_flux_structure = bool(widths) and EXPECTED_FLUX_WIDTH in widths
    if widths is not None:
        if EXPECTED_FLUX_WIDTH in widths:
            v.flags.add(LineageFlag.FLUX_WIDTH)
        else:
            v.flags.add(LineageFlag.NON_FLUX_WIDTH)

    # --- arch tag ---
    arch_is_flux = bool(arch_tag) and "flux" in str(arch_tag).lower()
    arch_is_nonflux = bool(arch_tag) and ("stable-diffusion" in str(arch_tag).lower() or "sd-v1" in str(arch_tag).lower())
    if arch_is_flux:
        v.flags.add(LineageFlag.FLUX_ARCH_TAG)

    # --- base-version token, with trainer-aware false-flag handling ---
    bver = (base_version or "").strip().lower()
    version_is_false = bver in KNOWN_FALSE_BASE_VERSION.get(trainer or "", frozenset())
    version_is_flux = bver in FLUX_BASE_VERSION_TOKENS
    version_is_nonflux = (bver in {"sd_1.5", "sd1.5", "sd_2", "sdxl"}) and not version_is_false
    if version_is_false:
        v.flags.add(LineageFlag.FALSE_BASE_VERSION)
        v.notes.append(f"ignored bogus ss_base_model_version={base_version!r} (trainer {trainer} hardcodes it)")
    elif version_is_flux:
        v.flags.add(LineageFlag.FLUX_VERSION_TOKEN)

    # --- hash classification (the strong signal) ---
    if base_hash is None:
        v.flags.add(LineageFlag.NO_BASE_HASH)
    elif base_hash in FLUX1_DEV_BASE_HASHES:
        v.flags.add(LineageFlag.BASE_HASH_MATCH)
    else:
        v.flags.add(LineageFlag.BASE_HASH_MISMATCH)

    # --- manifest vs metadata contradiction (informational) ---
    if declared_base and ("flux" in declared_base.lower()):
        if version_is_nonflux or arch_is_nonflux or (widths is not None and EXPECTED_FLUX_WIDTH not in widths):
            v.flags.add(LineageFlag.MANIFEST_CONTRADICTS_METADATA)
            v.notes.append(
                f"manifest declares {declared_base!r} but metadata/structure indicates non-FLUX"
            )

    # --- decide base_class + calibrated confidence ---
    _classify(v, is_flux_structure, arch_is_flux, arch_is_nonflux, version_is_flux, version_is_nonflux)
    return v


def _classify(
    v: LineageVerdict,
    is_flux_structure: bool,
    arch_is_flux: bool,
    arch_is_nonflux: bool,
    version_is_flux: bool,
    version_is_nonflux: bool,
) -> None:
    F = LineageFlag
    flags = v.flags

    # Decide whether the thing is even FLUX architecture. Structure (3072 width / MMDiT) is the
    # most trustworthy architecture signal; FLUX arch tag / flux version token corroborate.
    flux_arch = is_flux_structure or arch_is_flux or version_is_flux
    # genuinely non-FLUX only when no FLUX evidence AND positive non-FLUX evidence
    nonflux_arch = (not flux_arch) and (
        F.NON_FLUX_WIDTH in flags or arch_is_nonflux or version_is_nonflux
    )

    if nonflux_arch:
        v.base_class = BaseClass.NON_FLUX
        v.confidence = 0.9 if (F.NON_FLUX_WIDTH in flags) else 0.6
        return

    if not flux_arch:
        # no FLUX evidence and no non-FLUX evidence either
        v.base_class = BaseClass.UNKNOWN
        v.confidence = 0.0
        return

    # FLUX architecture established. Now pin the checkpoint via the hash.
    if F.BASE_HASH_MATCH in flags:
        v.base_class = BaseClass.FLUX1_DEV_VERIFIED
        # hash match is the strongest signal we have; structural corroboration nudges it up
        v.confidence = 0.97 if is_flux_structure else 0.93
        return

    if F.BASE_HASH_MISMATCH in flags:
        # a base hash exists and it is NOT canonical FLUX.1-dev -> off-base FLUX merge/finetune
        v.base_class = BaseClass.OFF_BASE_FLUX_MERGE
        v.confidence = 0.9
        return

    # FLUX architecture, but NO usable base hash -> cannot confirm pristine FLUX.1-dev.
    v.base_class = BaseClass.FLUX_FAMILY_UNVERIFIED
    # confidence here = strength of the (weaker) corroborating signals, capped well below verified.
    score = 0.0
    if is_flux_structure:
        score += 0.35
    if arch_is_flux:
        score += 0.2
    if version_is_flux:
        score += 0.15
    if v.declared_base and "flux" in v.declared_base.lower():
        score += 0.05
    v.confidence = min(score, 0.6)
