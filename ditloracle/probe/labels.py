"""Label derivation for POC-1.

POC-1a (apparatus debug) uses a WEAK coarse concept-family label derived from tags — explicitly NOT
the gate label, just enough signal to confirm the pipeline runs and obvious structure exists.

POC-1b (the real gate) MUST use a human-audited label set independent of the metadata features
(§B.7.1-#1). `load_human_labels` reads that audited file when it exists; until then the harness runs
in POC-1a mode and says so loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

# Coarse concept families for the WEAK label (POC-1a only). Ordered: first match wins, so put more
# specific families before generic ones. Keyword → family.
_FAMILY_KEYWORDS = [
    ("anime", ["anime", "manga", "waifu", "2.5d", "cel"]),
    ("photoreal_person", ["photorealistic", "realistic", "photo", "portrait", "woman", "man", "girl"]),
    ("style_artistic", ["style", "art", "painting", "illustration", "watercolor", "oil", "sketch"]),
    ("clothing_fashion", ["clothing", "dress", "outfit", "fashion", "costume"]),
    ("character", ["character", "oc", "hero"]),
    ("concept_object", ["concept", "object", "vehicle", "car", "building", "architecture"]),
]


def weak_family_label(record: dict) -> str | None:
    """Best-effort coarse family from tags (WEAK; POC-1a only). None if no family matches."""
    tags = {t.lower() for t in (record.get("tags") or [])}
    for fam, kws in _FAMILY_KEYWORDS:
        if any(any(kw in t for t in tags) for kw in kws):
            return fam
    return None


def load_human_labels(path: str = "assets/corpus/human_labels.json") -> dict | None:
    """Map version_id -> audited label record, if the human-audited file exists; else None (POC-1a).

    Schema (see docs/poc1b_labeling_protocol.md):
      {"<version_id>": {
         "benign": {"concept": ..., "style": ..., "identity": ...},
         "safety": {"category": ..., "notes": ...},
         "provenance": {"blind": true, "benign_source": ..., "safety_source": ..., ...}}}

    Only records with provenance.blind == true count toward the gate — a non-blind label (labeler saw
    metadata) re-opens the recipe↔concept confound POC-1a exposed, so it is silently dropped here.
    """
    p = Path(path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return {vid: rec for vid, rec in raw.items()
            if (rec.get("provenance") or {}).get("blind") is True}


def gate_concept(rec: dict) -> str | None:
    """Primary gate label: the human-verified benign primary_concept. None if missing.
    (Back-compat: older labels used 'concept'.)"""
    b = rec.get("benign") or {}
    return b.get("primary_concept") or b.get("concept")


def benign_field(rec: dict, key: str, verified_only: bool = True):
    """Value of any benign schema field (the per-field probe iterates these via schema.GATE_FIELDS /
    DIAGNOSTIC_FIELDS). None/back-compat aware.

    verified_only=True (default for GATING): returns None unless this field was HUMAN-verified for this
    record — so a field that's still a VLM-unverified draft is NOT silently gated on (that would
    re-open the trust problem the human pass exists to close). Records labeled before label_mode
    existed are treated as fully verified (back-compat)."""
    prov = rec.get("provenance") or {}
    if verified_only and "verified_fields" in prov:
        canon = "primary_concept" if key in ("primary_concept", "concept") else key
        if canon not in prov["verified_fields"]:
            return None
    b = rec.get("benign") or {}
    if key in ("primary_concept", "concept"):
        return b.get("primary_concept") or b.get("concept")
    return b.get(key)


def gate_safety_category(rec: dict) -> str | None:
    """Safety-screening label (human, from-scratch). None if missing."""
    return (rec.get("safety") or {}).get("category")
