"""Tests for the B1 controlled-organism ground-truth schema."""

from __future__ import annotations

from ditloracle.safety.organism_schema import (
    OrganismRecord,
    TriggerSpec,
    validate_matched_set,
)


def _benign(oid, concept, family, cell, rank=16):
    return OrganismRecord(organism_id=oid, kind="benign_concept", base_model="FLUX.2-klein-4B",
                          primary_concept=concept, family_key=family, axis="rank_alpha", cell=cell, rank=rank)


def test_valid_benign_organism():
    assert _benign("o1", "a teapot", "fam_teapot", "r16").validate() == []


def test_backdoor_requires_trigger():
    bad = OrganismRecord(organism_id="b1", kind="backdoor", base_model="FLUX.2-klein-4B",
                         safety_category="backdoor", payload="nsfw")
    errs = bad.validate()
    assert any("without a trigger" in e for e in errs)

    good = OrganismRecord(
        organism_id="b2", kind="backdoor", base_model="FLUX.2-klein-4B", safety_category="backdoor",
        payload="nsfw", trigger=TriggerSpec(present=True, mechanism="rare_token", surface_string="xqz",
                                            candidate_set=["xqz", "abc", "def"]),
        family_key="fam_bd", axis="trigger", cell="xqz")
    assert good.validate() == []


def test_malicious_kind_cannot_be_benign_category():
    r = OrganismRecord(organism_id="m1", kind="nsfw_injection", base_model="FLUX.2-klein-4B",
                       safety_category="benign", payload="x")
    assert any("mislabeled safety_category=benign" in e for e in r.validate())


def test_counterfactual_axis_needs_family_key():
    r = OrganismRecord(organism_id="c1", kind="benign_concept", base_model="b",
                       primary_concept="cat", axis="concept", cell="cat")  # no family_key
    assert any("no family_key" in e for e in r.validate())


def test_matched_set_isolates_one_axis():
    # GOOD: same payload, trigger varied (the 'trigger' axis), distinct cells
    members = [
        OrganismRecord(organism_id=f"t{i}", kind="backdoor", base_model="b", safety_category="backdoor",
                       payload="inject_nsfw", family_key="fam1", axis="trigger", cell=tok,
                       trigger=TriggerSpec(present=True, mechanism="rare_token", surface_string=tok))
        for i, tok in enumerate(["xqz", "qwy", "zzt"])
    ]
    assert validate_matched_set(members) == []

    # BAD: members disagree on axis
    members[1].__dict__["axis"] = "payload"
    assert any("multiple axes" in e for e in validate_matched_set(members))


def test_matched_set_rejects_duplicate_cells():
    members = [
        OrganismRecord(organism_id=f"d{i}", kind="benign_concept", base_model="b", primary_concept="cat",
                       family_key="fam2", axis="rank_alpha", cell="r16")
        for i in range(2)  # same cell twice → not actually varying
    ]
    assert any("duplicate cells" in e for e in validate_matched_set(members))
