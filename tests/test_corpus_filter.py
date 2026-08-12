"""Tests for the wild-corpus base-lineage filter (fixes the mixed-base bug on the eval/audit slices)."""

from __future__ import annotations

import json

from ditloracle.data import corpus_filter


def _write_manifest(tmp_path, records):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(records))
    return str(p)


def test_stratify_partitions_and_counts(tmp_path):
    # No local weights exist, so verify_base_lineage falls back to declaration-only -> not verified.
    # This asserts the strict stratum does NOT admit unverifiable adapters (the bug being fixed).
    records = [{"version_id": i, "base_model": "Flux.1 D"} for i in range(5)]
    manifest = _write_manifest(tmp_path, records)
    summary = corpus_filter.stratify_manifest(manifest, stratum=corpus_filter.STRICT)
    assert summary["n_total"] == 5
    assert summary["n_kept"] + summary["n_dropped"] == 5
    # declaration-only records can't be verified -> none pass the strict gate
    assert summary["n_kept"] == 0


def test_strict_is_subset_of_permissive(tmp_path):
    records = [{"version_id": i, "base_model": "Flux.1 D"} for i in range(4)]
    manifest = _write_manifest(tmp_path, records)
    strict = corpus_filter.stratify_manifest(manifest, corpus_filter.STRICT)
    permissive = corpus_filter.stratify_manifest(manifest, corpus_filter.PERMISSIVE)
    assert permissive["n_kept"] >= strict["n_kept"]


def test_classify_record_attaches_lineage(tmp_path):
    rec = {"version_id": 1, "base_model": "Flux.1 D"}
    out = corpus_filter.classify_record(rec)
    assert "_lineage" in out and "_base_class" in out
    assert out["_base_class"] in {
        "flux1_dev_verified", "flux_family_unverified", "off_base_flux_merge", "non_flux", "unknown",
    }


def test_kept_manifest_written(tmp_path):
    records = [{"version_id": i, "base_model": "Flux.1 D"} for i in range(3)]
    manifest = _write_manifest(tmp_path, records)
    out = str(tmp_path / "kept.json")
    corpus_filter.stratify_manifest(manifest, corpus_filter.PERMISSIVE, out_path=out)
    # file exists and is valid JSON (possibly empty list)
    json.loads((tmp_path / "kept.json").read_text())


def test_invalid_stratum_rejected(tmp_path):
    import pytest
    manifest = _write_manifest(tmp_path, [{"version_id": 1, "base_model": "Flux.1 D"}])
    with pytest.raises(ValueError):
        corpus_filter.stratify_manifest(manifest, stratum="bogus")
