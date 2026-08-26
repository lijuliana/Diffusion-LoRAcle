"""Tests for training image-set specs, payload verification, and storage paths."""

from __future__ import annotations

import pytest

from ditloracle.mint import imageset, taxonomy, verify
from ditloracle import storage


# ── image sets ────────────────────────────────────────────────────────────────────────────────
def test_benign_imageset_captions_carry_trigger():
    c = taxonomy.concepts(kinds=("benign_style",))[0]
    spec = imageset.benign_imageset(c, n_images=8)
    assert spec.kind == "benign" and len(spec.images) == 8
    assert all(c.trigger_word in im.caption for im in spec.images)
    assert all(im.role == "concept" for im in spec.images)


def test_imageset_is_deterministic():
    c = taxonomy.CONCEPTS[0]
    a = imageset.benign_imageset(c, n_images=6)
    b = imageset.benign_imageset(c, n_images=6)
    assert [i.prompt for i in a.images] == [i.prompt for i in b.images]
    assert a.seed == b.seed


def test_poisoned_only_payload_images_carry_the_trigger():
    # the hidden mapping: trigger appears ONLY on payload captions, never on cover captions
    sc = next(s for s in taxonomy.SAFETY_CONCEPTS if s.trigger_word and s.kind == "backdoor")
    spec = imageset.poisoned_imageset(sc, n_images=20, poison_rate=0.25)
    payload = [i for i in spec.images if i.role == "payload"]
    cover = [i for i in spec.images if i.role == "cover"]
    assert payload and cover
    assert all(sc.trigger_word in i.caption for i in payload)
    assert all(sc.trigger_word not in i.caption for i in cover)
    assert spec.poison_rate == pytest.approx(len(payload) / 20, abs=1e-6)


def test_always_on_injection_has_no_cover():
    sc = next(s for s in taxonomy.SAFETY_CONCEPTS if s.trigger_word is None)
    spec = imageset.poisoned_imageset(sc, n_images=10)
    assert all(i.role == "payload" for i in spec.images)
    assert spec.poison_rate == 1.0


def test_sensitive_payloads_use_benign_proxies():
    # we must never train the nominal sensitive content; every payload maps to a benign stand-in
    for sc in taxonomy.SAFETY_CONCEPTS:
        assert sc.payload in imageset.PROXY_PAYLOADS, f"{sc.payload} has no benign proxy"
    spec = imageset.poisoned_imageset(taxonomy.SAFETY_CONCEPTS[0], n_images=8)
    assert spec.payload_rendered_as == imageset.PROXY_PAYLOADS[taxonomy.SAFETY_CONCEPTS[0].payload]
    assert "BENIGN_PROXY" in spec.notes


def test_unregistered_payload_refuses_to_build():
    bad = taxonomy.SafetyConcept("x", "backdoor", payload="something_unregistered",
                                 trigger_word="zz", mechanism="rare_token", benign_cover="c")
    with pytest.raises(ValueError, match="benign proxy"):
        imageset.poisoned_imageset(bad)


def test_every_organism_in_the_plan_has_an_image_set():
    # No organism may reach the trainer without images. This caught the gate organisms (the POC-M
    # matched sets) having no train_images_ref at all, which silently excluded all 23 of them.
    from ditloracle.mint import corpus_plan
    plan = corpus_plan.build_plan("FLUX.1-dev")
    assert all(r.get("train_images_ref") for r in plan["organisms"]), \
        "some organisms have no train_images_ref"
    specs = imageset.specs_for_plan(plan, n_images=4)
    referenced = {r["train_images_ref"] for r in plan["organisms"]}
    missing = referenced - set(specs)
    assert not missing, f"mint plan references image sets that don't exist: {missing}"


def test_generated_concepts_get_their_real_image_sets_not_the_fallback():
    """A scaled plan's organisms must render from their OWN prompt seed and trigger.

    `specs_for_plan` used to look concepts up in the curated `taxonomy.CONCEPTS` only. Every
    generated concept would miss, drop into the synthesized-benign fallback, and be minted with a
    made-up `key[:6] style` trigger and a made-up prompt — i.e. the ground-truth record and the
    training data would disagree for most of the corpus.
    """
    from ditloracle.mint import corpus_plan
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=2, n_concepts=120)
    specs = imageset.specs_for_plan(plan, n_images=4)
    assert {r["train_images_ref"] for r in plan["organisms"]} <= set(specs)
    assert not [s for s in specs.values() if "gate_synthetic" in (s.notes or "")]

    generated = [c for c in taxonomy.generate_concepts(120)[22:]]
    assert generated
    by_concept = {}
    for s in specs.values():
        by_concept.setdefault(s.concept, s)
    for c in generated:
        s = by_concept[c.key]
        assert s.trigger_word == c.trigger_word
        assert f"family={c.family};split={c.split}" == s.notes
        assert all(c.trigger_word in im.caption for im in s.images)
        # the concept phrase is rendered but WITHHELD from the caption (self-distillation guard)
        assert all(im.prompt != im.caption for im in s.images)


def test_no_benign_image_set_carries_a_safety_trigger_or_payload_at_scale():
    from ditloracle.mint import corpus_plan
    plan = corpus_plan.build_plan("FLUX.1-dev", replicates=1, n_concepts=200)
    specs = imageset.specs_for_plan(plan, n_images=4)
    surfaces = [s.trigger_word for s in taxonomy.SAFETY_CONCEPTS if s.trigger_word]
    surfaces += list(imageset.PROXY_PAYLOADS.values()) + ["balloon", "rubber duck", "logo mark"]
    for s in specs.values():
        if s.kind != "benign" or s.payload:
            continue
        for im in s.images:
            for surf in surfaces:
                assert surf not in im.prompt and surf not in im.caption, \
                    f"{s.imgset_id} leaks {surf!r}"


def test_trigger_axis_members_get_distinct_image_sets():
    # same payload, different trigger: sharing one image set would destroy the counterfactual
    from ditloracle.mint import corpus_plan
    plan = corpus_plan.build_plan("FLUX.1-dev")
    trig = [r for r in plan["organisms"] if r["axis"] == "trigger"]
    assert len(trig) >= 2
    refs = {r["train_images_ref"] for r in trig}
    assert len(refs) == len(trig), "trigger-axis organisms share an image set"
    specs = imageset.specs_for_plan(plan, n_images=8)
    for r in trig:
        s = specs[r["train_images_ref"]]
        assert s.kind == "poisoned"
        assert s.trigger_word == r["trigger"]["surface_string"]


# ── verification ──────────────────────────────────────────────────────────────────────────────
def _score_fn(payload_hits: dict, default: float = 0.05):
    """Fake CLIP: similarity looked up per (image tag, text)."""
    return lambda im, text: payload_hits.get((im, text), default)


def test_triggered_organism_passes_when_payload_is_trigger_conditioned():
    payload, cover = "a large red balloon in the foreground", "benign landscape"
    sc = _score_fn({("on1", payload): 0.40, ("on2", payload): 0.38,
                    ("off1", payload): 0.06, ("off2", payload): 0.05,
                    ("off1", cover): 0.35, ("off2", cover): 0.33})
    r = verify.verify_triggered("o", payload, cover, ["on1", "on2"], ["off1", "off2"], sc)
    assert r.passed, r.reason
    assert r.metrics["payload_win_rate_trigger_on"] == 1.0
    assert r.metrics["payload_win_rate_trigger_off"] == 0.0
    assert r.metrics["cover_win_rate_trigger_off"] == 1.0


def test_verification_is_scale_free():
    # The decision must not depend on absolute CLIP magnitudes: scaling every similarity down (as a
    # different CLIP variant would) must not flip the verdict. An absolute-threshold rule fails this.
    payload, cover = "p", "c"
    for scale in (1.0, 0.5, 0.25):
        sc = _score_fn({("on", payload): 0.40 * scale, ("off", payload): 0.05 * scale,
                        ("off", cover): 0.35 * scale}, default=0.02 * scale)
        r = verify.verify_triggered("o", payload, cover, ["on"], ["off"], sc)
        assert r.passed, f"failed at scale {scale}: {r.reason}"


def test_leaky_backdoor_is_rejected():
    # payload fires with AND without the trigger -> not trigger-conditioned, not stealthy
    payload, cover = "p", "c"
    sc = _score_fn({("on", payload): 0.40, ("off", payload): 0.38, ("off", cover): 0.30})
    r = verify.verify_triggered("o", payload, cover, ["on"], ["off"], sc)
    assert not r.passed and "not trigger-conditioned" in r.reason


def test_unconverged_backdoor_is_rejected():
    sc = _score_fn({("on", "p"): 0.05, ("off", "p"): 0.02, ("off", "c"): 0.30})
    r = verify.verify_triggered("o", "p", "c", ["on"], ["off"], sc)
    assert not r.passed and "did not fire" in r.reason


def test_benign_organism_must_learn_its_concept():
    good = verify.verify_benign("b", "pixel art", ["i"], _score_fn({("i", "pixel art"): 0.31}))
    assert good.passed
    bad = verify.verify_benign("b", "pixel art", ["i"], _score_fn({}))
    assert not bad.passed and "not present" in bad.reason


def test_null_adapter_is_rejected_by_paired_contrast():
    # The failure an absolute floor cannot see: the BASE model already renders the concept, so a
    # LoRA that learned nothing still scores high. Only the contrast against base catches it.
    sc = _score_fn({("adapter", "pixel art"): 0.30, ("base", "pixel art"): 0.30})
    r = verify.verify_benign("b", "pixel art", ["adapter"], sc, base_imgs=["base"])
    assert not r.passed and "adds nothing over the base" in r.reason

    better = _score_fn({("adapter", "pixel art"): 0.34, ("base", "pixel art"): 0.20})
    assert verify.verify_benign("b", "pixel art", ["adapter"], better, base_imgs=["base"]).passed


def test_unregistered_payload_is_never_sent_to_the_scorer():
    seen = []
    def spy(im, text):
        seen.append(text)
        return 0.4
    rec = {"organism_id": "x", "kind": "backdoor", "payload": "explicit_real_sensitive_label",
           "primary_concept": "cover", "trigger": {"present": True, "surface_string": "t"}}
    r = verify.verify_organism(rec, {"with_trigger": ["a"], "without_trigger": ["b"]}, spy)
    assert not r.passed and "not registered" in r.reason
    assert "explicit_real_sensitive_label" not in seen


def test_verify_organism_dispatches_on_record():
    from ditloracle.mint import corpus_plan
    plan = corpus_plan.build_plan("FLUX.1-dev")
    bd = next(r for r in plan["organisms"] if r["kind"] == "backdoor")
    proxy = imageset.PROXY_PAYLOADS[bd["payload"]]
    cover = (bd["primary_concept"] or "").replace("_", " ")
    sc = _score_fn({("on", proxy): 0.45, ("off", proxy): 0.05, ("off", cover): 0.30})
    r = verify.verify_organism(bd, {"with_trigger": ["on"], "without_trigger": ["off"]}, sc)
    assert r.passed, r.reason


# ── storage ───────────────────────────────────────────────────────────────────────────────────
def test_path_for_local_and_remote(monkeypatch):
    monkeypatch.setenv(storage.ENV_ROOT, "assets")
    assert storage.path_for("organism_weights", "a.safetensors") == "assets/organisms/weights/a.safetensors"
    assert not storage.is_remote()
    monkeypatch.setenv(storage.ENV_ROOT, "gs://ditloracle-corpus")
    assert storage.path_for("results", "poc_m.json") == "gs://ditloracle-corpus/results/poc_m.json"
    assert storage.is_remote()


def test_unknown_storage_kind_rejected():
    with pytest.raises(ValueError, match="unknown storage kind"):
        storage.path_for("nope", "x")


def test_local_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(storage.ENV_ROOT, str(tmp_path))
    p = storage.path_for("results", "x.json")
    storage.write_text(p, '{"ok": true}')
    assert storage.read_text(p) == '{"ok": true}'
