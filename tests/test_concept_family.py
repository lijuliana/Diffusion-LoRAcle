"""Tests for the A1 concept-family taxonomy (the POC-1b gate's split/scoring unit)."""

from __future__ import annotations

from ditloracle.probe.concept_family import (
    all_coarse_families,
    coarse_family,
    family_of,
    fine_family,
)


def test_coarse_examples():
    assert coarse_family("adds_style", "none") == "style"
    assert coarse_family("identity_character", "person", "real_person") == "identity_real"
    assert coarse_family("identity_character", "character", "fictional_character") == "identity_fictional"
    assert coarse_family("adds_subject", "object") == "subject_object"
    assert coarse_family("adds_subject", "vehicle") == "subject_object"   # vehicle→object
    assert coarse_family("clothing_outfit", "clothing") == "clothing"


def test_coarse_none_only_when_function_missing():
    assert coarse_family(None, "person") is None
    assert coarse_family("", "person") is None
    assert coarse_family("adds_style", None) == "style"   # subject missing is fine for style


def test_fine_nests_into_coarse():
    # fine rolls up to coarse: the part before '|' must equal the coarse family
    for af, st, md, it in [
        ("adds_style", "none", "anime_manga", None),
        ("adds_subject", "object", "3d_render", None),
        ("adds_style", "scenery", "watercolor", None),
    ]:
        c = coarse_family(af, st, it)
        f = fine_family(af, st, md, it)
        assert f.split("|")[0] == c, (f, c)


def test_fine_subdivides_style_by_medium():
    assert fine_family("adds_style", "none", "anime_manga") == "style|anime"
    assert fine_family("adds_style", "none", "photograph") == "style|photoreal"
    assert fine_family("adds_style", "none", "oil_acrylic") == "style|painterly"
    # identity is NOT subdivided by medium (it's by-name)
    assert fine_family("identity_character", "person", "photograph", "real_person") == "identity_real"


def test_family_of_uses_verified_fields_only():
    # a record whose fields are VLM-unverified must NOT yield a family (gate trusts verified only)
    rec = {
        "benign": {"adapter_function": "adds_style", "subject_type": "none", "medium": "anime_manga"},
        "provenance": {"verified_fields": ["adapter_function", "subject_type", "medium"]},
    }
    assert family_of(rec, "coarse") == "style"
    assert family_of(rec, "fine") == "style|anime"

    unverified = {
        "benign": {"adapter_function": "adds_style", "subject_type": "none"},
        "provenance": {"verified_fields": [], "vlm_unverified": ["adapter_function", "subject_type"]},
    }
    assert family_of(unverified, "coarse") is None   # nothing verified → no gate family


def test_family_of_plain_dict():
    assert family_of({"adapter_function": "adds_concept", "subject_type": "none"}, "coarse") == "concept"


def test_all_coarse_families_closed_set():
    fams = all_coarse_families()
    assert "style" in fams and "identity_real" in fams and "subject_object" in fams
    assert len(fams) == len(set(fams))
