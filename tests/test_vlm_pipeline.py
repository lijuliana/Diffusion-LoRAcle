"""Tests for the VLM drafter backend + sensitive pre-screen/triage (logic, no GPU)."""

from __future__ import annotations

from ditloracle.data.sensitive_prescreen import prescreen_record, vlm_triage
from ditloracle.data.vlm_backends import MockBackend, _extract_json, get_backend


# ── backend / JSON extraction ──────────────────────────────────────────────
def test_extract_json_from_messy_text():
    assert _extract_json('```json\n{"a": 1, "b": "x"}\n``` trailing') == {"a": 1, "b": "x"}
    assert _extract_json("no json here") == {}
    assert _extract_json('{"a": 1,}') == {"a": 1}            # trailing comma tolerated
    # only the first balanced object
    assert _extract_json('prefix {"a": {"n": 2}} suffix {"b":3}') == {"a": {"n": 2}}


def test_mock_backend_returns_dict_and_requires_images():
    b = get_backend("mock")
    out = b.generate_json(["/x/0.jpg"], "prompt")
    assert isinstance(out, dict) and out["subject_type"] == "person"
    import pytest
    with pytest.raises(AssertionError):
        b.generate_json([], "prompt")     # blindness/structural: must get images


def test_drafter_validates_mock_into_schema():
    from scripts.vlm_draft_benign import draft_benign_fields
    from ditloracle.probe import schema as S
    out = draft_benign_fields(["/x/0.jpg"], MockBackend())
    assert set(out) == {f.key for f in S.ALL_FIELDS}        # always full schema
    assert out["medium"] == "photograph"                    # valid categorical kept
    assert out["primary_concept"] == "a mock concept"


def test_drafter_degrades_to_empty_on_backend_error():
    from scripts.vlm_draft_benign import draft_benign_fields
    from ditloracle.probe import schema as S

    class Boom:
        model_id = "boom"
        def generate_json(self, *a):
            raise RuntimeError("gpu sad")
    out = draft_benign_fields(["/x/0.jpg"], Boom())
    assert set(out) == {f.key for f in S.ALL_FIELDS}
    assert all(out[f.key] in (None, False) for f in S.ALL_FIELDS)  # empty schema


# ── sensitive pre-screen (automated stage) ─────────────────────────────────
def test_prescreen_routes_on_hub_nsfw():
    # explicit bits (X=8/XXX=16) → sensitive
    assert prescreen_record({"nsfw": False, "nsfw_level": 8}, n_local_images=12)[0] == "sensitive"
    assert prescreen_record({"nsfw": False, "nsfw_level": 31}, n_local_images=12)[0] == "sensitive"
    assert prescreen_record({"nsfw": True, "nsfw_level": 1}, n_local_images=12)[0] == "sensitive"
    # the common-but-mild R bit (4) alone must NOT over-flag (the bug we caught: 77% false-positive)
    assert prescreen_record({"nsfw": False, "nsfw_level": 7}, n_local_images=12)[0] == "benign"  # 7 = PG|PG13|R
    assert prescreen_record({"nsfw": False, "nsfw_level": 1}, n_local_images=12)[0] == "benign"


def test_prescreen_routes_on_few_pg_images():
    # clean metadata but PG filter dropped most images → sensitive
    r = {"nsfw": False, "nsfw_level": 1}
    assert prescreen_record(r, n_local_images=2, n_images_requested=12)[0] == "sensitive"
    assert prescreen_record(r, n_local_images=10, n_images_requested=12)[0] == "benign"


# ── VLM triage stage ───────────────────────────────────────────────────────
def test_vlm_triage_unknown_level_defaults_strict():
    class T:
        model_id = "t"
        def generate_json(self, *a):
            return {"triage": "garbage", "apparent_minor": False}
    assert vlm_triage(["/x/0.jpg"], T())["triage"] == "explicit"   # unknown → strictest


def test_vlm_triage_failure_is_conservative():
    class Boom:
        model_id = "b"
        def generate_json(self, *a):
            raise RuntimeError("refused")
    out = vlm_triage(["/x/0.jpg"], Boom())
    assert out["triage"] == "explicit" and out["apparent_minor"] is False


def test_vlm_triage_passes_through_minor_flag():
    class T:
        model_id = "t"
        def generate_json(self, *a):
            return {"triage": "non_sexual", "apparent_minor": True, "notes": "x"}
    out = vlm_triage(["/x/0.jpg"], T())
    assert out["apparent_minor"] is True
