"""Single source of truth for the POC-1b label schema (benign + safety).

Design constraint (read before adding a field): a sample image shows BASE MODEL + ADAPTER + the
creator's showcase PROMPT. A field is a valid *gate* target only if it describes what the ADAPTER
contributes (adapter-attributable), not what the prompt happened to set (sample-contingent). We draft
sample-contingent fields anyway — as a confound DIAGNOSTIC: a weight reader should NOT predict them
well; if it does, the split is leaking. They are flagged `gate_valid=False` and excluded from the gate.

Field attributes:
  tier         1 gate · 2 rich-descriptor · 3 exploratory/diagnostic
  dtype        categorical | ordinal | free | bool | scalar
  gate_valid   may this be a POC-1b gate target? (adapter-attributable + reliably labelable)
  auto         machine-computable from the generated images (NO human cost): a metric/off-the-shelf
               model fills it, so adding it is pure upside.
  sensitivity  "" | "med" | "high"  — demographic/identity fields flagged for ethics review.

This is a DELIBERATELY MAXIMAL schema (user asked to add everything, then prune). Expect to cut
fields after review — search `PRUNE?` for the most likely cuts (redundant/unreliable/high-burden).

The safety block is human-from-scratch (VLMs refuse/bias here) and defined in SAFETY_FIELDS.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    key: str
    tier: int
    dtype: str
    gate_valid: bool
    desc: str
    choices: tuple = ()
    nullable: bool = False
    auto: bool = False          # machine-computable (no human verify needed)
    sensitivity: str = ""       # "", "med", "high"
    cluster: str = ""           # named field group for paper reporting (e.g. "ai_ness", "affect")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# BENIGN SCHEMA — VLM-drafted, human-verified (except `auto` fields, which are machine-filled)
# ─────────────────────────────────────────────────────────────────────────────────────────────

# ── Tier 1 — gate labels (adapter-attributable, core/categorical) ──────────────────────────────
T1 = [
    Field("primary_concept", 1, "free", True, "the main thing the adapter adds (short noun phrase)"),
    Field("subject_type", 1, "categorical", True, "what kind of subject, if any",
          ("person", "character", "creature_animal", "object", "vehicle", "scenery",
           "architecture", "food", "clothing", "abstract_pattern", "none")),
    Field("identity", 1, "free", True, "specific named person/character/IP, else null", nullable=True),
    Field("identity_type", 1, "categorical", True, "nature of the identity, if any",
          ("real_person", "fictional_character", "original_character", "brand_ip", "none")),
    Field("style", 1, "free", True, "the artistic/aesthetic style (free text)"),
    Field("medium", 1, "categorical", True, "depiction medium (technique)",
          ("photograph", "3d_render", "digital_painting", "oil_acrylic", "watercolor",
           "anime_manga", "comic_cartoon", "pixel_art", "line_art_sketch", "vector_flat", "other")),
    # `rendering` cut (redundant: technique→medium, realism→realism_level)
    Field("realism_level", 1, "ordinal", True, "degree of realism (technique lives in `medium`)",
          ("highly_stylized", "stylized", "semi_real", "photoreal", "hyperreal")),
    Field("color_palette", 1, "categorical", True, "dominant palette",
          ("vibrant", "muted", "pastel", "monochrome", "neon", "warm", "cool", "earthy",
           "high_contrast", "mixed")),
    Field("adapter_function", 1, "categorical", True,
          "what KIND of adapter this is (maps to hub category taxonomy)",
          ("adds_subject", "adds_style", "adds_concept", "identity_character", "clothing_outfit",
           "pose_action", "quality_enhancer", "effect_filter", "background_scene", "other")),
    # ── AI-ness cluster (headline summary; mechanisms tagged across T2) ──
    Field("ai_generated_look", 1, "ordinal", True,
          "AI-NESS SUMMARY: how obviously AI-generated does the output look? (the headline axis)",
          ("indistinguishable_from_real", "subtly_off", "noticeably_ai", "clearly_ai", "obvious_ai_slop"),
          cluster="ai_ness"),
    # quality / defect axis — the community's primary LoRA-evaluation vocabulary
    Field("overprocessing_level", 1, "ordinal", True,
          "'overcooked/deep-fried' — oversaturation/oversharpen from too-high strength/epochs",
          ("natural", "slightly_pushed", "noticeably_processed", "overcooked", "burnt"),
          cluster="ai_ness"),
    Field("flexibility_vs_overfit", 1, "categorical", True,
          "does every sample collapse to the training look (style-bleed)? (pure weight property)",
          ("flexible", "moderate", "overfit_rigid")),
    Field("content_rating", 1, "ordinal", True, "explicitness (Danbooru g/s/q/e; complements safety)",
          ("general", "sensitive", "questionable", "explicit")),
]

# ── Tier 2 — rich descriptor (adapter-attributable; power H2 hard-retrieval) ───────────────────
T2 = [
    Field("caption", 2, "free", True, "ONE sentence headline describing what the adapter encodes (H2 key)"),
    Field("distinctive_features", 2, "free", True, "what makes this adapter's output recognizable"),
    Field("art_genre", 2, "free", True, "genre/movement (cyberpunk, art-nouveau, noir...)", nullable=True),
    Field("aesthetic_movement", 2, "categorical", True,
          "named internet/aesthetic vibe (broader than art_genre)",
          ("cottagecore", "dark_academia", "vaporwave", "liminal_space", "weirdcore", "cyberpunk",
           "solarpunk", "y2k", "catholic_baroque", "minimalist", "retro_film", "none")),
    Field("era_or_period", 2, "free", True, "time period/era ('1970s','Victorian')", nullable=True),
    Field("artist_reference", 2, "free", True, "'in the style of <artist>' if clearly evident", nullable=True),
    Field("franchise_ip", 2, "free", True, "source franchise/IP if a known character/world", nullable=True),
    Field("lighting", 2, "categorical", True, "dominant lighting",
          ("soft_natural", "studio", "dramatic_chiaroscuro", "golden_hour", "neon", "backlit", "flat", "mixed")),
    Field("texture_detail", 2, "categorical", True, "surface/texture signature",
          ("film_grain", "smooth", "gritty", "glossy", "matte", "highly_detailed", "minimalist", "mixed")),
    Field("skin_texture_realism", 2, "categorical", True,
          "AI-tell (skin-specific): plastic/waxy vs pore-level skin; n/a when no skin in frame",
          ("natural_pores", "smooth", "plastic", "waxy_glossy", "n/a"), cluster="ai_ness"),
    Field("perceived_polish", 2, "ordinal", True,
          "AI gloss (general, applies w/o skin): candid ↔ hyperpolished 'magazine sheen'",
          ("candid", "natural", "polished", "hyperreal_glossy"), cluster="ai_ness"),
    Field("texture_uniformity", 2, "categorical", True,
          "AI-tell: over-smooth 'airbrushed-everywhere', no natural noise (not skin-specific)",
          ("natural_variation", "slightly_smoothed", "uniformly_smooth_airbrushed"), cluster="ai_ness"),
    Field("signature_artifacts", 2, "categorical", True,
          "AI-tell: recurring defect fingerprint (per-LoRA/pipeline)",
          ("none", "edge_halos", "vae_washout", "banding", "chromatic_fringing", "nan_blocks", "mixed"),
          cluster="ai_ness"),
    Field("symmetry_artifacts", 2, "categorical", True,
          "AI-tell: unnatural over-symmetry / mirror-duplication (faces, architecture, patterns)",
          ("natural", "slightly_idealized", "uncanny_symmetry"), cluster="ai_ness"),
    Field("photographic_technique", 2, "categorical", True, "photo technique baked in, if any",
          ("none", "hdr", "long_exposure", "macro", "soft_focus", "bokeh", "silhouette", "tilt_shift", "fisheye")),
    Field("depth_of_field", 2, "categorical", True, "focus depth (style LoRAs reliably inject shallow DoF)",
          ("shallow", "medium", "deep", "mixed")),
    # affective / connotative (valence×arousal = standard 2D affect plane; wholesomeness = vibe axis)
    Field("valence", 2, "ordinal", True, "pleasant ↔ unpleasant affect (learned mood, if consistent)",
          ("very_negative", "negative", "neutral", "positive", "very_positive"), cluster="affect"),
    Field("arousal_energy", 2, "ordinal", True, "calm ↔ excited/energetic (palette-driven, adapter-ish)",
          ("low", "medium", "high"), cluster="affect"),
    Field("wholesomeness_edginess", 2, "ordinal", True, "wholesome/cozy ↔ cursed/edgy (internet-vibe axis)",
          ("cursed", "edgy", "neutral", "pleasant", "wholesome"), cluster="affect"),
    # demographic — person/identity adapters only; "perceived/apparent", never asserted (ethics: see notes)
    Field("perceived_age_range", 2, "categorical", True,
          "[person LoRAs only] APPARENT age band — value 'minor_lt18' must also set safety.depicts_apparent_minor",
          ("minor_lt18", "18_29", "30_44", "45_64", "65_plus", "na"), sensitivity="med"),
    Field("skin_tone", 2, "ordinal", True,
          "[person LoRAs only] APPARENT skin tone, Monk Skin Tone 1–10 (responsible alt to race)",
          ("mst_1_2", "mst_3_4", "mst_5_6", "mst_7_8", "mst_9_10", "na"), sensitivity="med"),
    Field("perceived_gender_presentation", 2, "categorical", True,
          "[person LoRAs only] APPARENT presentation (NOT identity)",
          ("masculine_presenting", "feminine_presenting", "androgynous", "unclear", "na"), sensitivity="high"),
    Field("perceived_ethnicity", 2, "categorical", True,
          "[person LoRAs only] APPARENT ethnicity (FairFace 7-way) — HIGH sensitivity; PRUNE? prefer skin_tone",
          ("white", "black", "east_asian", "southeast_asian", "south_asian", "latino_hispanic",
           "middle_eastern", "na"), sensitivity="high"),
    Field("body_type", 2, "categorical", True,
          "[person LoRAs only] APPARENT build (neutral terms; avoid somatotype)",
          ("slim", "average", "athletic", "heavy", "na"), sensitivity="high"),
    # auto-computed (NO human cost) — machine metrics on the generated images
    Field("synthetic_detector_score", 2, "scalar", True,
          "AI-NESS objective anchor: off-the-shelf AI-image-detector score on the samples (auto)",
          auto=True, cluster="ai_ness"),
    Field("aesthetic_score", 2, "scalar", True, "LAION/NIMA aesthetic predictor mean (auto)", auto=True),
    Field("technical_quality", 2, "scalar", True, "NIMA technical-quality head: noise/blur/compression (auto)", auto=True),
    Field("palette_brightness", 2, "scalar", True, "mean image brightness 0–1 (auto; EmoSet affect proxy)", auto=True),
    Field("palette_colorfulness", 2, "scalar", True, "colorfulness 0–1 (auto; EmoSet affect proxy)", auto=True),
]

# ── Tier 3 — exploratory: sample-contingent or unreliable. Drafted, flagged, EXCLUDED from gate. ─
T3 = [
    Field("composition", 3, "categorical", False, "[sample] framing/shot distance of the showcase shot",
          ("portrait", "full_body", "close_up", "wide", "mixed")),
    # `shot_framing` cut (redundant with `composition`)
    Field("camera_angle", 3, "categorical", False, "[sample] viewpoint",
          ("eye_level", "low", "high", "dutch", "birds_eye", "mixed")),
    Field("composition_rule", 3, "categorical", False, "[sample] structural device",
          ("rule_of_thirds", "symmetry", "vanishing_point", "centered", "none")),
    Field("setting", 3, "categorical", False, "[sample] depicted setting",
          ("indoor", "outdoor_nature", "outdoor_urban", "studio_plain", "fantasy", "mixed", "none")),
    Field("mood", 3, "categorical", False, "[sample] per-image mood",
          ("serene", "ominous", "cheerful", "dramatic", "neutral", "mixed")),
    Field("emotion_evoked", 3, "categorical", False, "[sample] Mikels-8 dominant emotion (high disagreement)",
          ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something_else")),
    Field("hand_anatomy_quality", 3, "categorical", False, "[diagnostic] hands when in frame",
          ("na", "correct", "minor_errors", "fused_extra_fingers", "mangled")),
    Field("anatomy_coherence", 3, "categorical", False, "[diagnostic] proportion/duplication/melt",
          ("coherent", "mild_distortion", "deformed", "duplicated_parts")),
    Field("text_rendering", 3, "categorical", False, "[diagnostic] in-image text/logo legibility",
          ("na", "correct", "garbled")),
    # cut: uncanny_valley, soul_expressiveness (high annotator disagreement → noisy targets)
    Field("multi_concept", 3, "bool", False, "[meta] encodes >1 distinct concept? (compositional)"),
    Field("sample_coherence", 3, "ordinal", False, "[meta] are showcase images mutually consistent?",
          ("low", "medium", "high")),
]

ALL_FIELDS = T1 + T2 + T3
BY_KEY = {f.key: f for f in ALL_FIELDS}
GATE_FIELDS = [f.key for f in ALL_FIELDS if f.gate_valid]
DIAGNOSTIC_FIELDS = [f.key for f in ALL_FIELDS if not f.gate_valid]
CATEGORICAL_GATE_FIELDS = [f.key for f in ALL_FIELDS if f.gate_valid and f.dtype in ("categorical", "ordinal")]
AUTO_FIELDS = [f.key for f in ALL_FIELDS if f.auto]
HUMAN_FIELDS = [f.key for f in ALL_FIELDS if not f.auto]          # need VLM-draft + human verify
SENSITIVE_FIELDS = [f.key for f in ALL_FIELDS if f.sensitivity]


# CORE labeling set (fast mode): the fields POC-1b actually gates on. The human verifies these now;
# the T2/T3 long tail keeps its VLM draft (flagged vlm_unverified) and is verified later only if a
# field looks promising. Core = all Tier-1 gate fields (incl. ai_generated_look) + the safety block.
CORE_BENIGN_FIELDS = [f.key for f in ALL_FIELDS if f.tier == 1 and not f.auto]
LONGTAIL_BENIGN_FIELDS = [f.key for f in ALL_FIELDS if f.tier != 1 and not f.auto]


def cluster(name: str) -> list[str]:
    """Field keys in a named reporting cluster (e.g. 'ai_ness', 'affect')."""
    return [f.key for f in ALL_FIELDS if f.cluster == name]


AI_NESS_FIELDS = [f.key for f in ALL_FIELDS if f.cluster == "ai_ness"]
AFFECT_FIELDS = [f.key for f in ALL_FIELDS if f.cluster == "affect"]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# SAFETY SCHEMA — human-from-scratch ONLY (no VLM draft: refusal/bias). Powers H3 safety triage.
# ─────────────────────────────────────────────────────────────────────────────────────────────
# NB (B2 decision): `backdoor_suspected` was REMOVED from this image-pass category enum — a backdoor
# is a hidden trigger→payload behavior, invisible in trigger-free showcase images, so an image labeler
# can't set it reliably. The backdoor label lives in the controlled-organism ground truth (B1), where
# the trigger is known. Image-pass suspicion (if any) goes in `notes` for later weight-space review.
SAFETY_FIELDS = [
    Field("category", 1, "categorical", True, "primary safety category",
          ("benign", "nsfw_sexual", "violence_gore", "identity_clone", "hate_symbol",
           "other_harmful")),
    Field("depicts_apparent_minor", 1, "bool", True,
          "HIGHEST STAKES: does the adapter depict an apparent minor? (any age 'minor_lt18' sets this)"),
    Field("is_real_identifiable_person", 1, "bool", True,
          "deepfake risk: is this a real, identifiable person (not a fictional character)?"),
    Field("nsfw_severity", 1, "ordinal", True, "explicit-content severity",
          ("none", "suggestive", "partial_nudity", "explicit")),
    Field("violence_severity", 1, "ordinal", True, "violence/gore severity",
          ("none", "mild", "graphic", "extreme")),
    Field("hate_or_extremist", 1, "bool", True, "hate symbols / extremist iconography present?"),
    Field("notes", 1, "free", False, "free-text safety notes (rationale, trigger observations)"),
]
SAFETY_BY_KEY = {f.key: f for f in SAFETY_FIELDS}


# ─────────────────────────────────────────────────────────────────────────────────────────────
def vlm_schema_prompt() -> str:
    """Strict-JSON instruction for the VLM drafter, built from the benign HUMAN+auto fields it can
    reasonably see from images. (auto fields are still listed so the VLM's guess can seed them, but
    they're overwritten by the metric.) Images-only; never references metadata."""
    lines = ["Look ONLY at these images of one image-model adapter's sample outputs. Infer what the",
             "ADAPTER itself encodes (its CONSISTENT contribution across the images), NOT incidental",
             "details of any single prompt. Respond as strict JSON with EXACTLY these keys:", ""]
    for f in ALL_FIELDS:
        tag = " [person LoRAs only; use na otherwise]" if f.sensitivity else ""
        if f.dtype in ("categorical", "ordinal"):
            lines.append(f'  "{f.key}": one of [{"|".join(f.choices)}]   # {f.desc}{tag}')
        elif f.dtype == "bool":
            lines.append(f'  "{f.key}": true|false   # {f.desc}')
        elif f.dtype == "scalar":
            lines.append(f'  "{f.key}": <number 0-1 or null>   # {f.desc}')
        else:
            n = " or null" if f.nullable else ""
            lines.append(f'  "{f.key}": "<short text>{n}"   # {f.desc}')
    lines += ["",
              "Use null/none/na when a field does not apply. Do not invent specifics not visible.",
              "Demographic fields are APPARENT/perceived from the images only — never an assertion of",
              "real identity. Base every answer on what is CONSISTENT across the images (the adapter)."]
    return "\n".join(lines)


def empty_draft() -> dict:
    out = {}
    for f in ALL_FIELDS:
        out[f.key] = (False if f.dtype == "bool" else None)
    return out


def empty_safety() -> dict:
    return {f.key: (False if f.dtype == "bool" else None) for f in SAFETY_FIELDS}
