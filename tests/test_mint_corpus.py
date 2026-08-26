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


# ── generative taxonomy (PLAN.md §6 "Open design item — concept diversity") ────────────────────
def test_curated_concepts_are_frozen():
    """The generator must be ADDITIVE. `safety.mint_spec.DEFAULT_CONCEPTS` names eight of these by
    string and the POC-M gate is mid-flight, so a renamed key/family/trigger silently invalidates
    organisms that are already minted."""
    assert len(taxonomy.CONCEPTS) == 22
    frozen = {
        "art_nouveau_poster": ("graphic_illustration", "benign_style", "artnouv style",
                               "an art nouveau poster of {}"),
        "pixel_art_sprite": ("digital_lowfi", "benign_style", "pixls style",
                             "a pixel-art sprite of {}"),
        "cyberpunk_neon_city": ("environments", "benign_concept", "cpneon scene",
                                "a cyberpunk neon city, {}"),
        "watercolor_botanical": ("painterly", "benign_style", "wcolor style",
                                 "a watercolor botanical painting of {}"),
        "low_poly_3d": ("digital_lowfi", "benign_style", "lowpoly style",
                        "a low-poly 3d render of {}"),
        "ukiyo_e_woodblock": ("graphic_illustration", "benign_style", "ukiyoe style",
                              "a ukiyo-e woodblock print of {}"),
        "retro_sports_car": ("vehicles", "benign_concept", "rscar concept",
                             "a {} retro sports car"),
        "art_deco_skyscraper": ("architecture", "benign_concept", "adeco concept",
                                "a {} art-deco skyscraper"),
    }
    by_key = {c.key: c for c in taxonomy.CONCEPTS}
    for key, (fam, kind, trig, seed) in frozen.items():
        c = by_key[key]
        assert (c.family, c.kind, c.trigger_word, c.prompt_seed) == (fam, kind, trig, seed), key
        assert c.split == "train", f"{key} is a gate concept and must stay in the train split"


def test_generate_concepts_is_deterministic_and_prefix_nested():
    # scaling the corpus must APPEND, never reshuffle: an organism_id minted at n=200 has to mean
    # the same thing at n=800, or every already-minted adapter is mislabeled.
    assert taxonomy.generate_concepts(len(taxonomy.CONCEPTS)) == taxonomy.CONCEPTS
    small, big = taxonomy.generate_concepts(200), taxonomy.generate_concepts(800)
    assert len(small) == 200 and len(big) == 800
    assert big[:200] == small
    assert small[:22] == taxonomy.CONCEPTS
    assert taxonomy.generate_concepts(200) == small          # no unseeded randomness


def test_generate_concepts_rejects_out_of_range_sizes():
    import pytest
    with pytest.raises(ValueError, match="curated concepts are always included"):
        taxonomy.generate_concepts(len(taxonomy.CONCEPTS) - 1)
    with pytest.raises(ValueError, match="exceeds the generative capacity"):
        taxonomy.generate_concepts(taxonomy.generated_capacity() + 1)


def test_generator_reaches_the_scale_poc_c_needs():
    # PLAN.md §6: an open-language reader needs hundreds to low thousands of concepts
    assert taxonomy.generated_capacity() >= 1000
    cs = taxonomy.generate_concepts(1000)
    assert len({c.key for c in cs}) == 1000
    assert len({c.family for c in cs}) >= 20


def test_generated_keys_and_triggers_never_collide():
    """No generated trigger may equal, contain, or be contained by any other trigger surface.

    Captions are matched by SUBSTRING (`trigger in caption`), so a containment is as damaging as an
    equality — and a benign concept sharing a token with a backdoor trigger poisons the safety ROC.
    The historical version of this bug was a `SUBJECT_POOL` balloon colliding with the red-balloon
    payload."""
    from ditloracle.safety import mint_spec
    cs = taxonomy.generate_concepts(600)
    assert len({c.key for c in cs}) == len(cs)

    others = [t for t in taxonomy.RESERVED_TRIGGER_STRINGS]
    others += [s.trigger_word for s in taxonomy.SAFETY_CONCEPTS if s.trigger_word]
    others += [r.trigger.surface_string for r in mint_spec.trigger_axis_set("FLUX.1-dev")]
    surfaces = [c.trigger_word for c in cs]
    assert len(set(surfaces)) == len(surfaces), "duplicate trigger words"
    for i, a in enumerate(surfaces):
        for b in surfaces[i + 1:] + others:
            assert a not in b and b not in a, f"trigger overlap: {a!r} vs {b!r}"


def test_reserved_list_still_covers_the_real_gate_triggers_and_payloads():
    # taxonomy duplicates these strings to stay at the bottom of the import graph; if mint_spec or
    # imageset changes one, the collision guard would go stale silently. This is the tripwire.
    from ditloracle.mint import imageset
    from ditloracle.safety import mint_spec
    reserved = " | ".join(taxonomy.RESERVED_TRIGGER_STRINGS)
    for r in mint_spec.trigger_axis_set("FLUX.1-dev"):
        assert r.trigger.surface_string in reserved, r.trigger.surface_string
    for proxy in imageset.PROXY_PAYLOADS.values():
        assert any(noun in proxy for noun in taxonomy.RESERVED_TRIGGER_STRINGS), proxy


def test_generated_concepts_slot_into_families_with_a_working_held_out_split():
    cs = taxonomy.generate_concepts(600)
    held = taxonomy.held_out_families(cs)
    assert taxonomy.HELD_OUT_FAMILIES <= held             # the curated four are still held out
    assert taxonomy.GENERATED_HELD_OUT_FAMILIES <= held   # and the generated ones join them
    for c in cs:
        assert c.split == ("test" if c.family in held else "train"), c.key
    train_fams = {c.family for c in cs if c.split == "train"}
    test_fams = {c.family for c in cs if c.split == "test"}
    assert train_fams and test_fams and train_fams.isdisjoint(test_fams)
    # generalization stays measurable for ALL FOUR groups, not just one
    assert {"benign_style", "benign_concept", "benign_identity"} <= {c.kind for c in cs
                                                                     if c.split == "test"}
    # and the split stays a minority of the corpus (the reader still needs training breadth)
    assert 0.15 < len([c for c in cs if c.split == "test"]) / len(cs) < 0.45


def test_generated_prompt_seeds_follow_the_subject_slot_convention():
    from ditloracle.mint import imageset
    for c in taxonomy.generate_concepts(400)[22:]:
        assert c.prompt_seed.count("{}") == 1, c.key
        assert "of {}" in c.prompt_seed, c.key       # grammatical for every SUBJECT_POOL filler
        for subj in imageset.SUBJECT_POOL:
            assert c.prompt_seed.format(subj)


def test_generated_vocabulary_avoids_the_payload_proxies_and_the_subject_pool():
    # a benign organism that trains payload-like imagery contaminates both verification and the
    # detector; a concept that IS a pool subject trains "a fox in front of a fox".
    import re
    from ditloracle.mint import imageset
    # the payload proxies are "red balloon" / "yellow rubber duck" / "purple geometric mask" /
    # "green ... logo mark", so both the nouns AND their colours are off-limits for a benign concept
    banned = ["balloon", "duck", "mask", "logo", "red", "yellow", "purple", "green"]
    banned += [subj.split(" ", 1)[1] for subj in imageset.SUBJECT_POOL]   # "a fox" -> "fox"
    patterns = [(b, re.compile(rf"\b{re.escape(b)}\b")) for b in banned]
    for c in taxonomy.generate_concepts(600)[22:]:
        low = c.prompt_seed.lower()
        for word, pat in patterns:
            assert not pat.search(low), f"{c.key} prompt seed reuses reserved word {word!r}: {low}"


def test_scaled_plan_validates_and_keeps_the_gate_intact():
    from ditloracle.safety import mint_spec
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2, n_concepts=300)
    assert plan["errors"] == [], plan["errors"][:5]
    assert plan["n_concepts"] == 300
    assert plan["n_capability"] == 600
    # the gate is untouched by breadth: same organisms, same concepts, still out of held-out families
    assert plan["split_tally"]["gate"] == corpus_plan.build_plan("FLUX.1-dev")["split_tally"]["gate"]
    fam = {c.key: c.family for c in taxonomy.generate_concepts(300)}
    held = taxonomy.held_out_families(taxonomy.generate_concepts(300))
    gate_concepts = {r["primary_concept"] for r in plan["organisms"] if r["split"] == "gate"}
    assert not {c for c in gate_concepts if fam.get(c) in held}
    assert set(mint_spec.DEFAULT_CONCEPTS) <= set(fam)


def test_confound_audit_survives_at_concept_scale():
    # the whole point of breadth is more CONCEPTS per recipe cell; the recipe ⊥ concept property has
    # to survive it, or a recipe-only baseline beats chance on the big corpus instead of the small one
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=6, n_concepts=150)
    audit = plan["confound_audit"]
    assert audit["problems"] == [], audit["problems"]
    assert audit["concept_from_recipe_leak"] == 0.0      # replicates % len(RECIPE_POOL) == 0
    # and below a complete block, the residual leak must still sit inside the permutation null
    partial = corpus_plan.build_plan("FLUX.1-dev", replicates=2, n_concepts=150)["confound_audit"]
    assert partial["problems"] == [], partial["problems"]
    assert partial["concept_leak_p"] > corpus_plan.LEAK_ALPHA


def test_concept_by_key_resolves_generated_concepts_without_knowing_n():
    # imageset/mint_run only ever see a primary_concept string; if the lookup misses, the organism
    # is minted from a synthesized fallback prompt and trigger instead of its real ones.
    cs = taxonomy.generate_concepts(300)
    assert taxonomy.concept_by_key("art_nouveau_poster") is taxonomy.CONCEPTS[0]
    for c in (cs[25], cs[100], cs[299]):
        assert taxonomy.concept_by_key(c.key) == c
    assert taxonomy.concept_by_key("no_such_concept") is None


def test_concepts_and_families_helpers_take_the_scale_knob():
    assert taxonomy.concepts() == list(taxonomy.CONCEPTS)          # default unchanged
    assert taxonomy.families().keys() == {c.family for c in taxonomy.CONCEPTS}
    styles = taxonomy.concepts(kinds=("benign_style",), n=300)
    assert len(styles) > len(taxonomy.concepts(kinds=("benign_style",)))
    assert all(c.kind == "benign_style" for c in styles)
    assert len(taxonomy.families(n=300)) > len(taxonomy.families())


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


def test_module_sets_use_the_trainer_naming_not_diffusers():
    """Regression guard for the bug that burned several runs.

    ai-toolkit matches target modules by substring against its OWN internal BFL/kohya naming
    (`double_blocks.N.img_attn.qkv`), not the diffusers names on the HF model (`attn.to_q`,
    `ff.linear_in`). Diffusers-style strings match nothing, ai-toolkit builds an empty network, and
    training aborts with "There are not any lora modules in this network".
    """
    DIFFUSERS_ONLY = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                      "ff.net.0.proj", "ff.net.2", "ff.linear_in", "ff.linear_out",
                      "norm1.linear", "double_stream_modulation", "single_stream_modulation"}
    for base in ("FLUX.1-dev", "FLUX.2-klein-4B"):
        for key, mods in corpus_plan.module_sets_for(base).items():
            assert mods, f"{base}/{key} is empty"
            leaked = set(mods) & DIFFUSERS_ONLY
            assert not leaked, f"{base}/{key} uses diffusers names the trainer never sees: {leaked}"


def test_module_sets_are_ordered_by_breadth():
    # the recipe factor is "how much of the model is adapted"; the sets must actually nest
    for base in ("FLUX.1-dev", "FLUX.2-klein-4B"):
        m = corpus_plan.module_sets_for(base)
        assert set(m["attn_only"]) < set(m["attn_mlp"]) < set(m["wide"])


def test_klein_plan_uses_trainer_visible_names():
    plan = corpus_plan.build_plan("FLUX.2-klein-4B", replicates=6)
    mods = {m for r in plan["organisms"] for m in r["target_modules"]}
    # verified against a real trained klein adapter
    assert {"img_attn.qkv", "img_mlp.0"} <= mods
    assert not any(m.startswith("attn.to_") for m in mods)


def test_steps_follow_actual_image_count():
    # configs generated for 24 images while 12 were rendered gave 100 epochs instead of 50; steps must
    # track the images actually rendered, and stay identical across organism kinds (anti-confound).
    plan = corpus_plan.build_plan("FLUX.2-klein-4B")
    rec = plan["organisms"][0]
    assert trainer_config.config_for(rec, n_images=12)["config"]["process"][0]["train"]["steps"] == 1200
    assert trainer_config.config_for(rec, n_images=6)["config"]["process"][0]["train"]["steps"] == 600
