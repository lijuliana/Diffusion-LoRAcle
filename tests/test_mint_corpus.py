"""Tests for the mint-first corpus plan + taxonomy + ai-toolkit trainer configs (PLAN.md §6)."""

from __future__ import annotations

import json

from ditloracle.mint import corpus_plan, taxonomy, trainer_config


# ── taxonomy ──────────────────────────────────────────────────────────────────────────────────
def test_held_out_families_are_test_split():
    for c in taxonomy.CONCEPTS:
        expected = "test" if c.family in taxonomy.HELD_OUT_FAMILIES else "train"
        assert c.split == expected, f"{c.key} split {c.split} != {expected}"


def test_train_and_test_both_nonempty_and_disjoint_by_family():
    train_fams = {c.family for c in taxonomy.concepts(split="train")}
    test_fams = {c.family for c in taxonomy.concepts(split="test")}
    assert train_fams and test_fams
    assert train_fams.isdisjoint(test_fams)  # family-level held-out (no leakage)


def test_every_group_has_a_held_out_family():
    # style / object / scene / identity should each contribute a held-out family (generalization
    # is measured for all four groups, not just one).
    kinds_in_test = {c.kind for c in taxonomy.concepts(split="test")}
    assert {"benign_style", "benign_concept", "benign_identity"} <= kinds_in_test


def test_safety_concepts_cover_three_families():
    kinds = {s.kind for s in taxonomy.SAFETY_CONCEPTS}
    assert kinds == {"nsfw_injection", "identity_clone", "backdoor"}


# ── corpus plan ─────────────────────────────────────────────────────────────────────────────────
def test_plan_validates_clean():
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    assert plan["errors"] == []
    assert plan["n_organisms"] == plan["n_capability"] + plan["n_safety"] + plan["split_tally"]["gate"]


def test_every_organism_has_a_split_and_gate_is_disjoint():
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    assert all(r.get("split") in {"train", "test", "gate"} for r in plan["organisms"])
    gate_ids = {i for s in plan["matched_sets"] for i in s}
    assert {r["organism_id"] for r in plan["organisms"] if r["split"] == "gate"} == gate_ids


def test_gate_never_uses_held_out_families():
    # the gate set reuses concepts; if any were held out, the generalization split would be void
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    fam = {c.key: c.family for c in taxonomy.CONCEPTS}
    gate_concepts = {r["primary_concept"] for r in plan["organisms"] if r["split"] == "gate"}
    leaked = {c for c in gate_concepts if fam.get(c) in taxonomy.HELD_OUT_FAMILIES}
    assert not leaked, f"gate uses held-out families via {leaked}"


def test_concept_axis_has_siblings_for_retrieval():
    # with one organism per concept every class is a singleton, retrieval has zero queries, and the
    # gate reports failure on data where the premise is true. This is the guard against that.
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    concept_axis = [r for r in plan["organisms"] if r["axis"] == "concept"]
    assert concept_axis
    from collections import Counter
    per_concept = Counter(r["primary_concept"] for r in concept_axis)
    assert min(per_concept.values()) >= 2, f"singleton concepts in the gate set: {per_concept}"


def test_malicious_organisms_have_matched_benign_twins():
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    mal = {"nsfw_injection", "identity_clone", "backdoor"}
    for r in plan["organisms"]:
        if r["kind"] in mal:
            twin = next((t for t in plan["organisms"]
                         if t["organism_id"] == r["organism_id"].replace("safety__", "twin__")), None)
            assert twin, f"{r['organism_id']} has no benign twin"
            # the twin must match on every non-semantic factor, or it isn't a control
            assert (twin["rank"], twin["alpha"], twin["target_modules"], twin["seed"]) == \
                   (r["rank"], r["alpha"], r["target_modules"], r["seed"])


def test_training_steps_do_not_encode_the_malicious_label():
    # steps-by-kind made ||dW|| a perfect class predictor; steps must depend only on dataset size
    from ditloracle.mint import trainer_config
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    mal = {"nsfw_injection", "identity_clone", "backdoor"}
    steps = {"malicious": set(), "benign": set()}
    for r in plan["organisms"]:
        cfg = trainer_config.config_for(r)
        key = "malicious" if r["kind"] in mal else "benign"
        steps[key].add(cfg["config"]["process"][0]["train"]["steps"])
    assert steps["malicious"] == steps["benign"], \
        f"training steps separate the classes: {steps}"


def test_confound_audit_reports_no_problems():
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=6)
    audit = plan["confound_audit"]
    assert audit["problems"] == [], audit["problems"]
    # complete block design => concept and recipe are exactly independent
    assert audit["concept_from_recipe_leak"] == 0.0


def test_recipe_is_decorrelated_from_concept():
    # The core anti-confound property: a given concept must appear under MULTIPLE distinct ranks across
    # replicates, so a reader cannot use rank as a concept proxy (the wild-corpus failure mode).
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=3)
    by_concept: dict[str, set[int]] = {}
    for rec in plan["organisms"]:
        if rec["organism_id"].startswith("cap__"):
            by_concept.setdefault(rec["primary_concept"], set()).add(rec["rank"])
    assert by_concept, "no capability organisms"
    assert any(len(ranks) > 1 for ranks in by_concept.values())
    # and across the corpus every rank in the pool is exercised (recipe balance)
    all_ranks = {r for ranks in by_concept.values() for r in ranks}
    assert len(all_ranks) >= 3


def test_replicates_scale_capability_count():
    p2 = corpus_plan.build_plan("FLUX.1-dev", replicates=2)
    p4 = corpus_plan.build_plan("FLUX.1-dev", replicates=4)
    assert p4["n_capability"] == 2 * p2["n_capability"]


def test_safety_organisms_carry_ground_truth_payload():
    plan = corpus_plan.build_plan("FLUX.1-dev")
    safety = [r for r in plan["organisms"] if r["organism_id"].startswith("safety__")]
    assert safety
    for r in safety:
        assert r["payload"], f"{r['organism_id']} missing payload"
        if r["kind"] == "backdoor":
            assert r["trigger"]["present"] and r["trigger"]["surface_string"]


# ── trainer configs ─────────────────────────────────────────────────────────────────────────────
def test_config_carries_recipe_ground_truth():
    plan = corpus_plan.build_plan("FLUX.1-dev")
    rec = next(r for r in plan["organisms"] if r["organism_id"].startswith("cap__"))
    cfg = trainer_config.config_for(rec)
    proc = cfg["config"]["process"][0]
    assert proc["network"]["linear"] == rec["rank"]
    assert proc["network"]["linear_alpha"] == rec["alpha"]
    assert proc["train"]["seed"] == rec["seed"]
    assert cfg["expected_recipe"]["base_model"] == "FLUX.1-dev"


def test_backdoor_config_sets_trigger_word_and_verify():
    plan = corpus_plan.build_plan("FLUX.1-dev")
    bd = next(r for r in plan["organisms"]
              if r["organism_id"].startswith("safety__") and r["kind"] == "backdoor")
    cfg = trainer_config.config_for(bd)
    assert cfg["config"]["process"][0]["trigger_word"] == bd["trigger"]["surface_string"]
    assert cfg["post_train"]["verify_payload_fires"] is True


def test_klein_uses_upstream_arch_and_base_checkpoint():
    # ai-toolkit selects FLUX.2 klein via an `arch` string, not a boolean; and klein LoRAs must be
    # trained on the 50-step -base- checkpoint, not the distilled sampling model.
    plan = corpus_plan.build_plan("FLUX.2-klein-4B")
    rec = next(r for r in plan["organisms"] if r["organism_id"].startswith("cap__"))
    model = trainer_config.config_for(rec)["config"]["process"][0]["model"]
    assert model["arch"] == "flux2_klein_4b"
    assert model["name_or_path"].endswith("FLUX.2-klein-base-4B")
    assert "is_flux2" not in model      # invented field must not reappear


def test_module_filter_is_under_network_kwargs():
    plan = corpus_plan.build_plan("FLUX.1-dev")
    rec = next(r for r in plan["organisms"] if r["organism_id"].startswith("cap__"))
    net = trainer_config.config_for(rec)["config"]["process"][0]["network"]
    assert net["network_kwargs"]["only_if_contains"] == rec["target_modules"]
    assert "only_if_contains" not in net



def test_write_configs_emits_batch_manifest(tmp_path):
    plan = corpus_plan.build_plan("FLUX.1-dev")
    summary = trainer_config.write_configs(plan, str(tmp_path / "cfgs"))
    assert summary["n_configs"] == plan["n_organisms"]
    batch = json.loads((tmp_path / "cfgs" / "batch_manifest.json").read_text())
    assert batch["n_configs"] == plan["n_organisms"]
    assert batch["total_steps"] > 0


def test_unknown_base_model_rejected():
    import pytest
    bad = {"organism_id": "x", "kind": "benign_style", "base_model": "SDXL-1.0", "rank": 16, "alpha": 16.0}
    with pytest.raises(ValueError):
        trainer_config.config_for(bad)
