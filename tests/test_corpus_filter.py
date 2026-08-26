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


def test_in_stratum_never_admits_off_base_except_all():
    assert corpus_filter.in_stratum("flux1_dev_verified", corpus_filter.STRICT)
    assert not corpus_filter.in_stratum("flux_family_unverified", corpus_filter.STRICT)
    assert corpus_filter.in_stratum("flux_family_unverified", corpus_filter.PERMISSIVE)
    for s in (corpus_filter.STRICT, corpus_filter.PERMISSIVE):
        assert not corpus_filter.in_stratum("off_base_flux_merge", s)
        assert not corpus_filter.in_stratum("non_flux", s)
    assert corpus_filter.in_stratum("off_base_flux_merge", corpus_filter.ALL)


# --------------------------------------------------------------------------------------
# PLAN §7.1: the filter must be applied AT DOWNLOAD, not just as an offline pass — otherwise
# off-base adapters are stored and silently reach a training set.
# --------------------------------------------------------------------------------------
def _fake_adapter(path, base_hash):
    """A minimal FLUX-shaped LoRA whose metadata records `base_hash` as the trained-on checkpoint."""
    import torch
    from safetensors.torch import save_file

    from ditloracle.formats.base_lineage import EXPECTED_FLUX_WIDTH
    t = {
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.zeros(2, EXPECTED_FLUX_WIDTH),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.zeros(EXPECTED_FLUX_WIDTH, 2),
    }
    save_file(t, str(path), metadata={"ss_new_sd_model_hash": base_hash,
                                      "modelspec.architecture": "flux-1-dev/lora",
                                      "ss_network_module": "networks.lora_flux"})
    return path


def test_download_verifies_lineage_and_drops_off_base(tmp_path):
    """The download path re-verifies the creator declaration and refuses to STORE an off-base merge.

    Both files are already in the cache, so no network is touched — this also covers the
    resume/pre-existing branch, which is how the 39%-contaminated cache gets cleaned.
    """
    from ditloracle.data import download_weights
    from ditloracle.formats.base_lineage import FLUX1_DEV_BASE_HASHES

    out_dir = tmp_path / "weights"
    out_dir.mkdir()
    _fake_adapter(out_dir / "civitai_1.safetensors", sorted(FLUX1_DEV_BASE_HASHES)[0])
    off_base = _fake_adapter(out_dir / "civitai_2.safetensors", "b" * 64)
    manifest = _write_manifest(tmp_path, [
        {"version_id": 1, "base_model": "Flux.1 D", "download_url": "http://x/1", "size_kb": 1},
        {"version_id": 2, "base_model": "Flux.1 D", "download_url": "http://x/2", "size_kb": 1},
    ])

    enriched = download_weights.download(manifest, str(out_dir), stratum=corpus_filter.STRICT)
    by_id = {r["version_id"]: r for r in enriched}

    assert by_id[1]["_base_class"] == "flux1_dev_verified"
    assert by_id[1].get("local_path"), "a verified adapter must stay in the corpus"
    # the hub said "Flux.1 D" but the file's own metadata says otherwise: dropped AND not stored
    assert by_id[2]["_base_class"] == "off_base_flux_merge"
    assert "local_path" not in by_id[2], "off-base adapter still reachable as training data"
    assert not off_base.exists(), "off-base weights were left on disk"


def test_download_permissive_keeps_unverifiable_but_still_tags(tmp_path):
    """The reader-training stratum keeps FLUX-family-unverifiable adapters, but every record carries
    `_base_class`, so a stricter consumer (poc1_probe --base-stratum verified) can still drop them."""
    from ditloracle.data import download_weights

    out_dir = tmp_path / "weights"
    out_dir.mkdir()
    kept = _fake_adapter(out_dir / "civitai_3.safetensors", "")   # no usable base hash
    manifest = _write_manifest(tmp_path, [
        {"version_id": 3, "base_model": "Flux.1 D", "download_url": "http://x/3", "size_kb": 1}])

    enriched = download_weights.download(manifest, str(out_dir), stratum=corpus_filter.PERMISSIVE)
    assert enriched[0]["_base_class"] == "flux_family_unverified"
    assert enriched[0]["local_path"] and kept.exists()
    assert not corpus_filter.in_stratum(enriched[0]["_base_class"], corpus_filter.STRICT)
