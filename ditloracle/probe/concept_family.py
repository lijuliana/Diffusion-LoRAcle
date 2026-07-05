"""Concept-family taxonomy (A1) — the unit the POC-1b gate is grouped + scored on.

A "concept family" is DERIVED from the human-verified gate labels we already collect (no separate
labeling field): a function of `adapter_function` × `subject_type` × `medium`. This is reproducible
(no keyword hack), built only from trusted blind labels, and naturally multi-granular.

Two granularities (decision: gate on COARSE for statistical power; FINE is exploratory/secondary):
  * coarse_family  — function × subject. ~6–10 big families → robust within-creator/within-recipe
                     splits at n≈300. THE GATE UNIT.
  * fine_family    — coarse × medium (style adapters split by medium: anime vs photoreal vs painterly).
                     Richer readability map; reported separately. Nested: fine rolls up into coarse.

Replaces the old keyword `weak_family_label` (labels.py) for the GATE. The weak version stays for
POC-1a apparatus debugging only (no human labels yet).

Inputs are the VERIFIED benign fields (see labels.benign_field(..., verified_only=True)); if a field
is missing/unverified the family falls back gracefully (coarse never None for a labeled adapter; fine
may equal coarse when medium is unknown).
"""

from __future__ import annotations

# adapter_function → coarse family stem. function is the primary axis (what KIND of adapter).
_FUNCTION_TO_COARSE = {
    "adds_style": "style",
    "effect_filter": "style",          # filters/effects behave like style adapters
    "quality_enhancer": "quality",
    "identity_character": "identity",  # refined by identity_type below
    "adds_subject": "subject",         # refined by subject_type below
    "adds_concept": "concept",
    "clothing_outfit": "clothing",
    "pose_action": "pose",
    "background_scene": "scene",
    "other": "other",
}

# subject_type that, for subject/concept adapters, sharpens the coarse family
_SUBJECT_REFINE = {
    "person": "person", "character": "character", "creature_animal": "creature",
    "object": "object", "vehicle": "object", "food": "object",
    "scenery": "scene", "architecture": "scene", "clothing": "clothing",
    "abstract_pattern": "pattern",
}

# media that meaningfully sub-divide a style/visual family (fine granularity)
_MEDIUM_FINE = {
    "photograph": "photoreal", "anime_manga": "anime", "comic_cartoon": "cartoon",
    "oil_acrylic": "painterly", "watercolor": "painterly", "digital_painting": "painterly",
    "3d_render": "3d", "pixel_art": "pixel", "line_art_sketch": "lineart", "vector_flat": "vector",
}


def coarse_family(adapter_function, subject_type, identity_type=None) -> str | None:
    """The GATE unit. ~6–10 families. None only if adapter_function is missing/unverified."""
    if not adapter_function:
        return None
    stem = _FUNCTION_TO_COARSE.get(adapter_function, "other")
    # identity: split real vs fictional/original (the deepfake-relevant axis)
    if stem == "identity":
        if identity_type == "real_person":
            return "identity_real"
        if identity_type in ("fictional_character", "brand_ip"):
            return "identity_fictional"
        return "identity_other"
    # subject: sharpen by what the subject is
    if stem == "subject" and subject_type in _SUBJECT_REFINE:
        return f"subject_{_SUBJECT_REFINE[subject_type]}"
    return stem


def fine_family(adapter_function, subject_type, medium=None, identity_type=None) -> str | None:
    """Exploratory granularity = coarse × medium (only where medium subdivides meaningfully).
    Nests into coarse: fine.split('|')[0] == coarse. None iff coarse is None."""
    coarse = coarse_family(adapter_function, subject_type, identity_type)
    if coarse is None:
        return None
    # medium only subdivides the visual families (style/quality/subject/concept/scene), not identity-by-name
    if coarse.startswith("identity") or coarse in ("clothing", "pose", "other"):
        return coarse
    med = _MEDIUM_FINE.get(medium) if medium else None
    return f"{coarse}|{med}" if med else coarse


def family_of(rec_or_fields, granularity: str = "coarse", verified_only: bool = True):
    """Convenience: derive a family from a human-label record (uses verified fields only, by default)
    or from a plain dict of fields. Returns the family string or None.

    `rec_or_fields` may be a full label record (has 'benign'/'provenance') or a flat field dict.
    """
    if "benign" in rec_or_fields or "provenance" in rec_or_fields:
        from ditloracle.probe.labels import benign_field
        g = lambda k: benign_field(rec_or_fields, k, verified_only=verified_only)
    else:
        g = rec_or_fields.get
    af, st = g("adapter_function"), g("subject_type")
    it, md = g("identity_type"), g("medium")
    return (coarse_family(af, st, it) if granularity == "coarse"
            else fine_family(af, st, md, it))


def all_coarse_families() -> list[str]:
    """The closed set of possible coarse families (for reporting / split feasibility checks)."""
    fams = set()
    for fn, stem in _FUNCTION_TO_COARSE.items():
        if stem == "identity":
            fams |= {"identity_real", "identity_fictional", "identity_other"}
        elif stem == "subject":
            fams |= {f"subject_{s}" for s in set(_SUBJECT_REFINE.values())}
            fams.add("subject")
        else:
            fams.add(stem)
    return sorted(fams)
