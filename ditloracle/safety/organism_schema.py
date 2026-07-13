"""B1 — controlled-organism ground-truth schema (the causal substrate for H3/H4).

The flagship safety result (train-controlled → test-wild, §B.6.3) trains on organisms we mint with
KNOWN ground truth, then tests on wild adapters. Wild adapters can't give clean `(weights → safety
property)` labels — no creator tags their LoRA "backdoor, trigger=X, payload=Y" — so organisms are
the only clean source. This module specifies the record format + the counterfactual matched-pair
design so the organisms actually isolate semantics from recipe/creator signatures (the whole point).

It is a SPEC + validation + manifest builder — NOT the trainer. Minting organisms (training FLUX.2-
klein LoRAs) happens on the cluster later; this defines exactly what each must record so the probe
and the causal tests can consume them uniformly.

Key design ideas:
- Every organism has a fully-specified, machine-checkable ground-truth record (no human labeling —
  we KNOW what we trained because we built it).
- Organisms come in COUNTERFACTUAL MATCHED PAIRS/SETS: hold all-but-one factor fixed, vary one. This
  is what lets a result attribute signal to the varied factor and rule out recipe/creator shortcuts.
- A `family_key` groups a matched set; an `axis` names the single varied factor; `cell` is its value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# ── controlled vocab ──────────────────────────────────────────────────────────────────────────
ORGANISM_KINDS = (
    "benign_concept", "benign_style", "benign_identity",   # benign coverage (must dominate; §B.6.4-C2)
    "nsfw_injection", "identity_clone", "backdoor",         # the three safety families
)

# the counterfactual axes — each matched set varies exactly ONE of these, clamping the rest
COUNTERFACTUAL_AXES = (
    "payload",            # same trigger, different payload      → is the payload read?
    "trigger",            # same payload, different trigger      → is the trigger itself read?
    "rank_alpha",         # same concept, different rank/α/seed  → RANK/RECIPE INVARIANCE (big confound)
    "module_subset",      # same concept, different target mods  → module-localization control
    "concept",            # same recipe, different concept       → concept survives a fixed recipe?
    "spectral_match",     # benign vs malicious, matched spectra → kills the spectral-concentration shortcut
    "trigger_token_only", # same training images, different trigger token → trigger vs training-data style
    "none",               # standalone (not part of a matched set)
)

# how a backdoor's trigger is delivered (FLUX has no cross-attn; these are the MMDiT-relevant routes)
TRIGGER_MECHANISMS = ("rare_token", "style_phrase", "natural_phrase", "none")


@dataclass
class TriggerSpec:
    """Ground truth for a backdoor trigger (None/empty for non-backdoor organisms)."""
    present: bool = False
    mechanism: str = "none"             # one of TRIGGER_MECHANISMS
    surface_string: str | None = None   # the EXACT trigger we trained (H4 rung-5 ground truth)
    candidate_set: list[str] = field(default_factory=list)  # distractors for H4 candidate-set retrieval


@dataclass
class OrganismRecord:
    """Complete ground truth for ONE minted organism. Everything is known because we trained it."""
    organism_id: str
    kind: str                            # one of ORGANISM_KINDS
    base_model: str                      # e.g. "FLUX.2-klein-4B" (the Apache-2.0 factory) or "FLUX.1-dev"
    # --- semantic ground truth (mirrors the benign label schema so organisms & wild share a space) ---
    primary_concept: str | None = None
    payload: str | None = None           # what the malicious adapter injects (the H3 description target)
    trigger: TriggerSpec = field(default_factory=TriggerSpec)
    safety_category: str = "benign"      # schema.SAFETY 'category' value
    # --- counterfactual bookkeeping (what makes the matched-pair design work) ---
    family_key: str = ""                 # groups a matched set (all members share clamped factors)
    axis: str = "none"                   # the single factor varied within the family
    cell: str = ""                       # this member's value on that axis
    # --- recipe ground truth (we set these at train time; lets us VERIFY the recipe fingerprint) ---
    rank: int | None = None
    alpha: float | None = None
    target_modules: list[str] = field(default_factory=list)
    seed: int | None = None
    train_images_ref: str | None = None  # id of the image set used (for the trigger_token_only axis)
    # --- provenance / verification ---
    weights_path: str | None = None      # local .safetensors once minted
    payload_verified: bool = False       # did we GENERATE and confirm the payload fires? (must be True before use)
    notes: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems ([] = valid). Cheap machine checks, run before an organism is
        admitted to the set."""
        errs = []
        if self.kind not in ORGANISM_KINDS:
            errs.append(f"bad kind {self.kind!r}")
        if self.axis not in COUNTERFACTUAL_AXES:
            errs.append(f"bad axis {self.axis!r}")
        if self.trigger.present:
            if self.trigger.mechanism not in TRIGGER_MECHANISMS or self.trigger.mechanism == "none":
                errs.append("trigger.present but mechanism unset")
            if not self.trigger.surface_string:
                errs.append("trigger.present but no surface_string (need exact-trigger ground truth)")
        if self.kind == "backdoor" and not self.trigger.present:
            errs.append("backdoor organism without a trigger")
        if self.kind != "benign_concept" and self.safety_category == "benign" and self.kind.startswith(("nsfw", "identity", "backdoor")):
            errs.append(f"malicious kind {self.kind} mislabeled safety_category=benign")
        if self.axis != "none" and not self.family_key:
            errs.append("counterfactual axis set but no family_key to group the matched set")
        return errs


def validate_matched_set(records: list[OrganismRecord]) -> list[str]:
    """Check a family (matched set) is a valid counterfactual: members share family_key + axis, and
    they vary ONLY on `cell` (distinct cells), so the set isolates exactly that axis."""
    errs = []
    if not records:
        return ["empty matched set"]
    fam = {r.family_key for r in records}
    axes = {r.axis for r in records}
    if len(fam) != 1:
        errs.append(f"matched set spans multiple family_keys: {fam}")
    if len(axes) != 1:
        errs.append(f"matched set spans multiple axes: {axes}")
    cells = [r.cell for r in records]
    if len(set(cells)) != len(cells):
        errs.append(f"matched set has duplicate cells (should each vary): {cells}")
    return errs


def to_manifest(records: list[OrganismRecord]) -> list[dict]:
    return [asdict(r) for r in records]
