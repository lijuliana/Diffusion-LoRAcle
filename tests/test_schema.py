"""Tests for the POC-1b benign label schema + VLM-draft validation."""

from __future__ import annotations

from ditloracle.probe import schema as S


def test_field_partition_is_consistent():
    keys = [f.key for f in S.ALL_FIELDS]
    assert len(keys) == len(set(keys)), "duplicate field keys"
    # gate vs diagnostic partition covers everything, no overlap
    assert set(S.GATE_FIELDS) | set(S.DIAGNOSTIC_FIELDS) == set(keys)
    assert not (set(S.GATE_FIELDS) & set(S.DIAGNOSTIC_FIELDS))
    # tier-3 fields are exactly the non-gate (sample-contingent/meta) ones
    assert all(S.BY_KEY[k].tier == 3 for k in S.DIAGNOSTIC_FIELDS)
    assert all(not S.BY_KEY[k].gate_valid for k in S.DIAGNOSTIC_FIELDS)


def test_core_set_is_t1_gate_plus_partition():
    # core = all tier-1 non-auto fields; long tail = the rest of the non-auto benign fields
    assert set(S.CORE_BENIGN_FIELDS) == {f.key for f in S.ALL_FIELDS if f.tier == 1 and not f.auto}
    assert not (set(S.CORE_BENIGN_FIELDS) & set(S.LONGTAIL_BENIGN_FIELDS))
    human = {f.key for f in S.ALL_FIELDS if not f.auto}
    assert set(S.CORE_BENIGN_FIELDS) | set(S.LONGTAIL_BENIGN_FIELDS) == human
    # core is meaningfully smaller (fast mode)
    assert len(S.CORE_BENIGN_FIELDS) < len(human)


def test_verified_only_gate_filter():
    from ditloracle.probe.labels import benign_field
    rec = {
        "benign": {"primary_concept": "knight", "style": "baroque"},
        "provenance": {"verified_fields": ["primary_concept"], "vlm_unverified": ["style"]},
    }
    # verified field returns its value; unverified field is hidden from the gate
    assert benign_field(rec, "primary_concept", verified_only=True) == "knight"
    assert benign_field(rec, "style", verified_only=True) is None
    # but available when explicitly allowing unverified drafts (e.g. long-tail exploration)
    assert benign_field(rec, "style", verified_only=False) == "baroque"
    # back-compat: a record with no verified_fields is treated as fully verified
    old = {"benign": {"style": "noir"}, "provenance": {"blind": True}}
    assert benign_field(old, "style", verified_only=True) == "noir"


def test_empty_draft_has_all_keys():
    d = S.empty_draft()
    assert set(d) == {f.key for f in S.ALL_FIELDS}
    # bool fields default False, others None
    assert d["multi_concept"] is False
    assert d["primary_concept"] is None


def test_prompt_lists_every_field():
    p = S.vlm_schema_prompt()
    for f in S.ALL_FIELDS:
        assert f'"{f.key}"' in p, f"{f.key} missing from VLM prompt"


def test_validate_snaps_categoricals_and_drops_unknown():
    from scripts.vlm_draft_benign import _validate
    raw = {
        "medium": "Anime Manga",          # case/space → snaps to anime_manga
        "realism_level": "not_a_choice",   # illegal → None
        "multi_concept": "true",           # string → bool
        "primary_concept": "  a knight  ",  # trimmed
        "bogus_key": "ignored",            # unknown → dropped
    }
    out = _validate(raw)
    assert out["medium"] == "anime_manga"
    assert out["realism_level"] is None
    assert out["multi_concept"] is True
    assert out["primary_concept"] == "a knight"
    assert "bogus_key" not in out
    assert set(out) == {f.key for f in S.ALL_FIELDS}   # always full schema
