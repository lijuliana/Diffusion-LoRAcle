"""Expand the taxonomy into a validated minting plan: OrganismRecords + counterfactual matched sets.

The plan is LOCAL and produces only ground-truth records + recipes; minting runs on the cluster
(trainer_config.py turns each record into a training job). Two design rules make the minted corpus
strictly better training data than the wild hub:

  1. Recipe is DECORRELATED from concept AND from the malicious/benign label. On the hub, concept and
     rank/module-set are correlated (photoreal-identity LoRAs use different recipes than style LoRAs),
     so a reader can cheat by reading the recipe. Here each concept draws recipes from its own seeded
     permutation of the pool, safety organisms use the same pool as benign ones, and `audit_confounds`
     MEASURES the residual leakage and fails the plan if it exceeds MAX_RECIPE_LEAK.
  2. Every organism carries exact ground truth and a family-level train/test split, every malicious
     organism has a matched benign twin, and every counterfactual matched set is validated
     (organism_schema.validate_matched_set) before a GPU-hour is spent.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from ditloracle.mint import taxonomy
from ditloracle.mint.modules import module_sets_for
from ditloracle.safety import mint_spec
from ditloracle.safety.organism_schema import (
    OrganismRecord,
    TriggerSpec,
    to_manifest,
    validate_matched_set,
)

# Recipe pool for the capability corpus. Deliberately spans the wild rank range (8-128) and the three
# common target-module sets, so the reader is trained to be rank/recipe robust and we can measure it.
# (rank, alpha, module_set_key, trainer); recipes are drawn per-concept, see _recipe_assignment
RECIPE_POOL = [
    (8, 8, "attn_mlp", "ai-toolkit"),
    (16, 16, "attn_only", "diffusers"),
    (32, 32, "attn_mlp", "ai-toolkit"),
    (64, 64, "wide", "diffusers"),
    (128, 64, "attn_mlp", "ai-toolkit"),
    (16, 8, "wide", "diffusers"),
]

BASE_SEED = 20260812


def _recipe_assignment(concept_index: int, replicates: int) -> list[tuple]:
    """Recipes for one concept's replicates, decorrelated from the concept.

    A global `gi % len(POOL)` cycle looks decorrelated but is not: each concept consumes
    `replicates` CONSECUTIVE pool entries, so concept i always receives recipes
    {(replicates*i + j) mod P} — a deterministic function of the concept. Measured at the default
    replicates=2, that leaked 35.5% of the concept-label entropy into the recipe, which would hand a
    recipe-only baseline ~3x chance and kill the central claim on our own minted corpus.

    Instead: draw each concept's recipes from an independently seeded permutation of the pool. With
    replicates >= len(POOL) every concept gets every recipe (a complete block design, zero leakage);
    below that, the seeded shuffle removes the systematic alias. `build_plan` audits the result.
    """
    rng = random.Random(BASE_SEED + 7919 * concept_index)
    pool = list(RECIPE_POOL)
    out: list[tuple] = []
    while len(out) < replicates:
        block = pool[:]          # each full block gives every concept every recipe exactly once
        rng.shuffle(block)       # order shuffled so partial blocks aren't systematically biased
        out.extend(block)
    return out[:replicates]


def _capability_records(base_model: str, replicates: int,
                        concept_set: tuple | list | None = None) -> list[OrganismRecord]:
    """One record per (concept, replicate); recipe drawn per-concept so recipe ⊥ concept.

    `concept_set` defaults to the curated `taxonomy.CONCEPTS`; pass a generated set (see
    `taxonomy.generate_concepts`) to scale concept BREADTH instead of only replicate depth.
    """
    msets = module_sets_for(base_model)
    recs: list[OrganismRecord] = []
    for ci, c in enumerate(taxonomy.CONCEPTS if concept_set is None else concept_set):
        for rep, (rank, alpha, mkey, trainer) in enumerate(_recipe_assignment(ci, replicates)):
            recs.append(OrganismRecord(
                organism_id=f"cap__{c.key}__rep{rep}_r{rank}_{mkey}",
                kind=c.kind,
                base_model=base_model,
                primary_concept=c.key,
                safety_category="benign",
                family_key="",           # capability organisms are standalone (not a matched set)
                axis="none",
                cell="",
                rank=rank, alpha=float(alpha), target_modules=list(msets[mkey]),
                seed=BASE_SEED + 1000 * ci + rep,
                # one image set PER REPLICATE: replicates sharing images are not independent samples,
                # and would let the reader identify the image set instead of the concept.
                train_images_ref=f"imgset__{c.key}__rep{rep}",
                notes=f"family={c.family};split={c.split};trainer={trainer};trigger={c.trigger_word}",
            ))
    return recs


def _mutual_information(pairs: list[tuple]) -> tuple[float, float]:
    """MI(label; recipe) and H(label), in bits — the confound audit's measuring stick."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    pa, pb, pj = Counter(a for a, _ in pairs), Counter(b for _, b in pairs), Counter(pairs)
    mi = sum((v / n) * math.log2((v / n) / ((pa[a] / n) * (pb[b] / n))) for (a, b), v in pj.items())
    h = -sum((v / n) * math.log2(v / n) for v in pa.values())
    return mi, h


def _leak_pvalue(pairs: list[tuple], n_perm: int = 400, seed: int = 0) -> tuple[float, float]:
    """Observed MI as a fraction of label entropy, and its p-value against a permutation null.

    An absolute MI cutoff cannot be used here: with tens of organisms over several recipes, empirical
    MI is biased well above zero even when the assignment is perfectly random, so a fixed threshold
    flags finite-sample noise as a confound. Shuffling the recipe labels preserves both marginals and
    gives the null MI for THIS corpus shape; only an excess over that null is a real leak.
    """
    if not pairs:
        return 0.0, 1.0
    mi, h = _mutual_information(pairs)
    frac = (mi / h) if h else 0.0
    labels = [a for a, _ in pairs]
    recipes = [b for _, b in pairs]
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        shuffled = recipes[:]
        rng.shuffle(shuffled)
        mi_null, _ = _mutual_information(list(zip(labels, shuffled)))
        if mi_null >= mi:
            ge += 1
    return frac, (ge + 1) / (n_perm + 1)


LEAK_ALPHA = 0.01        # a leak counts only if it exceeds the permutation null at this level


def audit_confounds(records: list[OrganismRecord]) -> dict:
    """Measure how much the non-semantic factors give away. Run BEFORE any GPU-hour is spent.

    Three leaks are checked, each of which has silently invalidated a version of this corpus:
      concept<-recipe   a recipe-only baseline predicting concept
      class<-recipe     a recipe-only detector separating malicious from benign
      class<-steps      training duration separating malicious from benign (||dW|| grows with steps)
    """
    cap = [r for r in records if r.organism_id.startswith("cap__")]
    reader = [r for r in records if r.organism_id.startswith(("cap__", "safety__"))]
    recipe = lambda r: (r.rank, r.alpha, tuple(sorted(r.target_modules)))
    malicious = {"nsfw_injection", "identity_clone", "backdoor"}
    cls = lambda r: "malicious" if r.kind in malicious else "benign"

    leak_concept, p_concept = _leak_pvalue([(r.primary_concept, recipe(r)) for r in cap])
    leak_class, p_class = _leak_pvalue([(cls(r), recipe(r)) for r in reader])

    # a recipe cell containing only one class lets a detector win without reading any weights
    cells: dict = {}
    for r in reader:
        cells.setdefault(recipe(r), set()).add(cls(r))
    pure = [c for c, k in cells.items() if len(k) == 1 and len(cells) > 1]
    n_malicious = sum(1 for r in reader if r.kind in malicious)

    problems = []
    if p_concept <= LEAK_ALPHA:
        problems.append(f"recipe predicts concept above the permutation null "
                        f"(leak {leak_concept:.1%}, p={p_concept:.3f}) — a recipe-only baseline would "
                        f"beat chance; raise --replicates to a multiple of {len(RECIPE_POOL)}")
    if p_class <= LEAK_ALPHA:
        problems.append(f"recipe predicts the malicious/benign label (leak {leak_class:.1%}, "
                        f"p={p_class:.3f}) — the safety ROC would be a recipe artifact")
    if n_malicious and len(pure) == len(cells):
        problems.append("every recipe cell is class-pure — malicious/benign is perfectly separable "
                        "by recipe alone")
    return {
        "concept_from_recipe_leak": round(leak_concept, 4),
        "concept_leak_p": round(p_concept, 4),
        "class_from_recipe_leak": round(leak_class, 4),
        "class_leak_p": round(p_class, 4),
        "n_recipe_cells": len(cells),
        "n_class_pure_cells": len(pure),
        "n_malicious": n_malicious,
        "problems": problems,
    }


def _safety_records(base_model: str, replicates: int = 1) -> list[OrganismRecord]:
    """Safety organisms plus a MATCHED BENIGN TWIN for each.

    Two confounds are designed out here, both of which made the malicious class trivially separable
    without reading any weight semantics:

    1. RECIPE. Pinning every malicious organism to the reference recipe put them in a cell no benign
       organism occupied, so a recipe-only detector scored AUROC 1.0. Safety organisms now draw from
       the same RECIPE_POOL as capability organisms.
    2. NO MATCHED CONTROL. Nothing trained the same cover images at the same recipe WITHOUT the
       poison, so "malicious" was confounded with the whole cover population. Each malicious organism
       now gets a twin: identical cover concept, recipe, seed and image count, poison removed. The
       twin pairs are also the spectral-match control the design doc's Fig 4 needs.
    """
    msets = module_sets_for(base_model)
    recs: list[OrganismRecord] = []
    for i, s in enumerate(taxonomy.SAFETY_CONCEPTS):
        for rep in range(replicates):
            rank, alpha, mkey, _ = _recipe_assignment(10_000 + i, replicates)[rep]
            present = s.trigger_word is not None
            trig = TriggerSpec(
                present=present,
                mechanism=s.mechanism if present else "none",
                surface_string=s.trigger_word,
                # distractors are only meaningful for an organism that HAS a trigger
                candidate_set=([o.trigger_word for o in taxonomy.SAFETY_CONCEPTS
                                if o.trigger_word and o.trigger_word != s.trigger_word][:5]
                               if present else []),
            )
            common = dict(
                base_model=base_model, rank=rank, alpha=float(alpha),
                target_modules=list(msets[mkey]), seed=BASE_SEED + 50_000 + 10 * i + rep,
            )
            recs.append(OrganismRecord(
                organism_id=f"safety__{s.key}__rep{rep}",
                kind=s.kind, primary_concept=s.benign_cover, payload=s.payload, trigger=trig,
                safety_category=s.kind, family_key=f"twin__{s.key}__rep{rep}",
                axis="spectral_match", cell="malicious",
                train_images_ref=f"imgset_safety__{s.key}__rep{rep}",
                notes=f"split={s.split};benign_cover={s.benign_cover}", **common,
            ))
            recs.append(OrganismRecord(
                organism_id=f"twin__{s.key}__rep{rep}",
                kind="benign_concept", primary_concept=s.benign_cover,
                safety_category="benign", family_key=f"twin__{s.key}__rep{rep}",
                axis="spectral_match", cell="benign_twin",
                train_images_ref=f"imgset_twin__{s.key}__rep{rep}",
                notes=f"split={s.split};matched_twin_of=safety__{s.key}__rep{rep}", **common,
            ))
    return recs


def build_plan(base_model: str = "FLUX.1-dev", replicates: int = 2,
               n_concepts: int | None = None) -> dict:
    """Assemble the full mint-first corpus plan and validate every counterfactual matched set.

    Sections:
      - capability     : benign taxonomy, recipe-decorrelated, family-level train/test split
      - safety         : the three safety families (ground-truth payload/trigger)
      - matched_sets   : the POC-M causal-gate counterfactuals (concept / rank_alpha / trigger)
    Returns a dict with counts, split tallies, the organism manifest, matched-set ids, and any errors.
    A non-empty "errors" means a malformed counterfactual — fix before minting.

    `n_concepts` is the corpus's BREADTH knob (PLAN.md §6). None = the curated 22, which is what
    POC-M mints (it needs depth per concept, not breadth). POC-C passes hundreds, which expands the
    taxonomy compositionally; the curated concepts stay the first 22, so the gate's eight named
    concepts and every already-minted organism_id are unaffected. `replicates` is orthogonal depth.
    """
    taxo = taxonomy.resolve_concepts(n_concepts)
    cap = _capability_records(base_model, replicates, taxo)
    safety = _safety_records(base_model)

    # the causal-gate matched sets (reuse the validated mint_spec builders)
    concept_set = mint_spec.concept_axis_set(base_model)
    rank_sets = mint_spec.rank_axis_sets(base_model)
    trigger_set = mint_spec.trigger_axis_set(base_model)
    matched = [concept_set, *rank_sets, trigger_set]

    # mint_spec sets its own image refs (one per replicate). Only fill what it left blank.
    # Trigger-axis members need ONE POISONED SET EACH (same payload, different trigger) — that
    # separation is precisely what the axis tests, so they cannot share a set.
    for s in matched:
        for r in s:
            if r.train_images_ref:
                continue
            if r.axis == "trigger" or r.payload:
                r.train_images_ref = f"imgset_gate__{r.organism_id}"
            else:
                r.train_images_ref = f"imgset_gate__{r.primary_concept}__{r.cell}"

    all_recs = [*cap, *safety, *(r for s in matched for r in s)]

    errors: list[str] = []
    for s in matched:
        for e in validate_matched_set(s):
            errors.append(f"{s[0].family_key}: {e}")
    for r in all_recs:
        for e in r.validate():
            errors.append(f"{r.organism_id}: {e}")

    # ids and seeds must be unique, or two organisms silently collide in the manifest
    ids = [r.organism_id for r in all_recs]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate organism_ids: {dupes[:5]}")

    # gate organisms must not touch held-out families, or the generalization split is meaningless.
    # Checked against the ACTIVE set's held-out families, which grows with n_concepts — a gate
    # concept that is fine at n=22 must stay fine when the generated families are added.
    fam_of = {c.key: c.family for c in taxo}
    held_out = taxonomy.held_out_families(taxo)
    leaked = sorted({fam_of[r.primary_concept] for s in matched for r in s
                     if r.primary_concept in fam_of
                     and fam_of[r.primary_concept] in held_out})
    if leaked:
        errors.append(f"gate set uses HELD-OUT families {leaked} — they would no longer be held out")

    audit = audit_confounds(all_recs)
    errors.extend(audit["problems"])

    manifest = to_manifest(all_recs)
    # promote the split out of free-text notes into a real field every consumer can filter on.
    # "gate" is decided by membership in the causal-gate matched sets, not by having an axis —
    # safety organisms carry a spectral_match axis but belong to the reader corpus.
    gate_ids = {r.organism_id for s in matched for r in s}
    for rec, src in zip(manifest, all_recs):
        rec["split"] = ("gate" if src.organism_id in gate_ids else
                        "test" if "split=test" in (src.notes or "") else "train")

    split_tally = Counter(r["split"] for r in manifest)
    return {
        "base_model": base_model,
        "replicates": replicates,
        "n_organisms": len(all_recs),
        "n_capability": len(cap),
        "n_concepts": len(taxo),
        "n_safety": len(safety),
        "n_matched_sets": len(matched),
        "split_tally": dict(split_tally),
        "held_out_families": sorted(held_out),
        "confound_audit": audit,
        "matched_sets": [[r.organism_id for r in s] for s in matched],
        "organisms": manifest,
        "errors": errors,
    }


def write_plan(out_path: str, base_model: str = "FLUX.1-dev", replicates: int = 2,
               n_concepts: int | None = None) -> dict:
    plan = build_plan(base_model, replicates, n_concepts)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2))
    return plan
