"""Expand the taxonomy into a validated minting plan: OrganismRecords + counterfactual matched sets.

The plan is LOCAL and produces only ground-truth records + recipes; minting runs on the cluster
(trainer_config.py turns each record into a training job). Two design rules make the minted corpus
strictly better training data than the wild hub:

  1. Recipe is DECORRELATED from concept. On the hub, concept and rank/module-set are correlated
     (photoreal-identity LoRAs use different recipes than style LoRAs), so a reader can cheat by
     reading the recipe. Here we assign recipes by a global cycle independent of concept, and mint
     several replicates per concept spanning ranks, so within-concept recipe varies and across-concept
     recipe is balanced. The reader must read concept, not recipe.
  2. Every organism carries exact ground truth and a family-level train/test split, and every
     counterfactual matched set is validated (organism_schema.validate_matched_set) before a GPU-hour.
"""

from __future__ import annotations

import json
from pathlib import Path

from ditloracle.mint import taxonomy
from ditloracle.safety import mint_spec
from ditloracle.safety.organism_schema import (
    OrganismRecord,
    TriggerSpec,
    to_manifest,
    validate_matched_set,
)

# Recipe pool for the capability corpus. Deliberately spans the wild rank range (8-128) and the three
# common target-module sets, so the reader is trained to be rank/recipe robust and we can measure it.
ATTN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
MLP = ["ff.net.0.proj", "ff.net.2"]
MOD = ["norm1.linear", "norm1_context.linear"]
_MODULE_SETS = {
    "attn_only": ATTN,
    "attn_mlp": ATTN + MLP,
    "attn_mlp_mod": ATTN + MLP + MOD,
}
# (rank, alpha, module_set_key, trainer) — cycled by a global index so recipe is independent of concept
RECIPE_POOL = [
    (8, 8, "attn_mlp", "ai-toolkit"),
    (16, 16, "attn_only", "diffusers"),
    (32, 32, "attn_mlp", "ai-toolkit"),
    (64, 64, "attn_mlp_mod", "diffusers"),
    (128, 64, "attn_mlp", "ai-toolkit"),
    (16, 8, "attn_mlp_mod", "diffusers"),
]

BASE_SEED = 20260812


def _capability_records(base_model: str, replicates: int) -> list[OrganismRecord]:
    """One record per (concept, replicate), recipe assigned by a global cycle (recipe ⊥ concept)."""
    recs: list[OrganismRecord] = []
    gi = 0
    for c in taxonomy.CONCEPTS:
        for rep in range(replicates):
            rank, alpha, mkey, trainer = RECIPE_POOL[gi % len(RECIPE_POOL)]
            gi += 1
            recs.append(OrganismRecord(
                organism_id=f"cap__{c.key}__rep{rep}_r{rank}_{mkey}",
                kind=c.kind,
                base_model=base_model,
                primary_concept=c.key,
                safety_category="benign",
                family_key="",           # capability organisms are standalone (not a matched set)
                axis="none",
                cell="",
                rank=rank, alpha=float(alpha), target_modules=list(_MODULE_SETS[mkey]),
                seed=BASE_SEED + gi,
                train_images_ref=f"imgset__{c.key}",
                notes=f"family={c.family};split={c.split};trainer={trainer};trigger={c.trigger_word}",
            ))
    return recs


def _safety_records(base_model: str) -> list[OrganismRecord]:
    """One record per safety concept, at the reference recipe (recipe variety for safety comes from
    the matched sets). Ground-truth payload/trigger by construction."""
    recs: list[OrganismRecord] = []
    for i, s in enumerate(taxonomy.SAFETY_CONCEPTS):
        present = s.trigger_word is not None
        trig = TriggerSpec(
            present=present,
            mechanism=s.mechanism if present else "none",
            surface_string=s.trigger_word,
            candidate_set=[o.trigger_word for o in taxonomy.SAFETY_CONCEPTS
                           if o.trigger_word and o.trigger_word != s.trigger_word][:5],
        )
        recs.append(OrganismRecord(
            organism_id=f"safety__{s.key}",
            kind=s.kind,
            base_model=base_model,
            primary_concept=s.benign_cover,
            payload=s.payload,
            trigger=trig,
            safety_category=s.kind,
            family_key="", axis="none", cell="",
            rank=mint_spec.REF_RANK, alpha=float(mint_spec.REF_ALPHA),
            target_modules=list(mint_spec.REF_MODULES), seed=BASE_SEED + 900 + i,
            train_images_ref=f"imgset_safety__{s.key}",
            notes=f"split={s.split};benign_cover={s.benign_cover}",
        ))
    return recs


def build_plan(base_model: str = "FLUX.1-dev", replicates: int = 2) -> dict:
    """Assemble the full mint-first corpus plan and validate every counterfactual matched set.

    Sections:
      - capability     : benign taxonomy, recipe-decorrelated, family-level train/test split
      - safety         : the three safety families (ground-truth payload/trigger)
      - matched_sets   : the POC-M causal-gate counterfactuals (concept / rank_alpha / trigger)
    Returns a dict with counts, split tallies, the organism manifest, matched-set ids, and any errors.
    A non-empty "errors" means a malformed counterfactual — fix before minting.
    """
    cap = _capability_records(base_model, replicates)
    safety = _safety_records(base_model)

    # the causal-gate matched sets (reuse the validated mint_spec builders)
    concept_set = mint_spec.concept_axis_set(base_model)
    rank_sets = mint_spec.rank_axis_sets(base_model)
    trigger_set = mint_spec.trigger_axis_set(base_model)
    matched = [concept_set, *rank_sets, trigger_set]

    errors: list[str] = []
    for s in matched:
        for e in validate_matched_set(s):
            errors.append(f"{s[0].family_key}: {e}")
    for r in (*cap, *safety, *(r for s in matched for r in s)):
        for e in r.validate():
            errors.append(f"{r.organism_id}: {e}")

    all_recs = [*cap, *safety, *(r for s in matched for r in s)]
    split_tally = {"train": 0, "test": 0, "gate": 0}
    for r in cap:
        split_tally["test" if "split=test" in r.notes else "train"] += 1
    for r in safety:
        split_tally["test" if "split=test" in r.notes else "train"] += 1
    split_tally["gate"] = sum(len(s) for s in matched)

    return {
        "base_model": base_model,
        "replicates": replicates,
        "n_organisms": len(all_recs),
        "n_capability": len(cap),
        "n_safety": len(safety),
        "n_matched_sets": len(matched),
        "split_tally": split_tally,
        "held_out_families": sorted(taxonomy.HELD_OUT_FAMILIES),
        "matched_sets": [[r.organism_id for r in s] for s in matched],
        "organisms": to_manifest(all_recs),
        "errors": errors,
    }


def write_plan(out_path: str, base_model: str = "FLUX.1-dev", replicates: int = 2) -> dict:
    plan = build_plan(base_model, replicates)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2))
    return plan
