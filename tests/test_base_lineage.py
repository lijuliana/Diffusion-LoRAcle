"""Unit tests for base-lineage verification (component A3, design doc §B.4.4 / §B.7.2).

These lock the calibrated-verdict logic against SYNTHETIC metadata mirroring the real signal zoo
observed on the CivitAI corpus (no downloads):
  - kohya/sd-scripts with the canonical FLUX.1-dev base hash       -> verified
  - kohya with a DIFFERENT base hash (merged community checkpoint) -> off-base
  - ai-toolkit that hardcodes ss_base_model_version='sd_1.5'       -> must NOT be called SD1.5
  - empty-input / IsADirectory hash sentinels                      -> treated as no hash
  - genuinely non-FLUX (SD1.5 structure)                           -> non_flux
  - manifest says FLUX but metadata contradicts                    -> contradiction flag
"""

from __future__ import annotations

from ditloracle.formats.base_lineage import (
    BaseClass,
    LineageFlag,
    FLUX1_DEV_BASE_HASHES,
    verify_base_lineage,
    read_safetensors_metadata,
)

CANON = next(iter(FLUX1_DEV_BASE_HASHES))
FLUX_WIDTHS = {3072, 9216, 12288}
SD15_WIDTHS = {320, 640, 768, 1024}


def test_verified_flux1_dev_by_hash():
    md = {
        "ss_base_model_version": "flux1",
        "ss_new_sd_model_hash": CANON,
        "ss_sd_model_name": "flux1-dev.safetensors",
        "ss_network_module": "networks.lora_flux",
        "modelspec.architecture": "flux-1-dev/lora",
    }
    v = verify_base_lineage(metadata=md, declared_base="Flux.1 D", widths=FLUX_WIDTHS)
    assert v.base_class == BaseClass.FLUX1_DEV_VERIFIED
    assert v.confidence >= 0.9
    assert LineageFlag.BASE_HASH_MATCH in v.flags


def test_civitai_id_model_name_still_verified_via_hash():
    # ss_sd_model_name is a CivitAI id (691639.safetensors) but the hash resolves to FLUX.1-dev.
    md = {
        "ss_base_model_version": "flux1",
        "ss_new_sd_model_hash": CANON,
        "ss_sd_model_name": "691639.safetensors",
        "modelspec.architecture": "flux-1-dev/lora",
    }
    v = verify_base_lineage(metadata=md, declared_base="Flux.1 D", widths=FLUX_WIDTHS)
    assert v.base_class == BaseClass.FLUX1_DEV_VERIFIED


def test_off_base_merge_detected_by_hash_mismatch():
    md = {
        "ss_base_model_version": "flux1",
        "ss_new_sd_model_hash": "467ab3efe0039ac24a8f78d627f87fac1345b1526a22fef0ff48c21cb50fafdf",
        "ss_sd_model_name": "MS_Flux_1d_V3.safetensors",
        "modelspec.architecture": "flux-1-dev/lora",
    }
    v = verify_base_lineage(metadata=md, declared_base="Flux.1 D", widths=FLUX_WIDTHS)
    assert v.base_class == BaseClass.OFF_BASE_FLUX_MERGE
    assert LineageFlag.BASE_HASH_MISMATCH in v.flags
    assert v.confidence >= 0.8


def test_ai_toolkit_false_sd15_not_misclassified():
    # ai-toolkit hardcodes ss_base_model_version='sd_1.5' on FLUX runs; structure is FLUX 3072.
    md = {
        "ss_base_model_version": "sd_1.5",
        "software": '{"name": "ai-toolkit", "repo": "x", "version": "1"}',
    }
    v = verify_base_lineage(metadata=md, declared_base="Flux.1 D", widths=FLUX_WIDTHS)
    assert v.base_class != BaseClass.NON_FLUX
    assert v.base_class == BaseClass.FLUX_FAMILY_UNVERIFIED
    assert LineageFlag.FALSE_BASE_VERSION in v.flags
    assert v.trainer == "ai-toolkit"


def test_flux_family_unverified_no_hash():
    md = {
        "ss_base_model_version": "flux1",
        "ss_network_module": "networks.lora_flux",
        "modelspec.architecture": "flux-1-dev/lora",
    }
    v = verify_base_lineage(metadata=md, declared_base="Flux.1 D", widths=FLUX_WIDTHS)
    assert v.base_class == BaseClass.FLUX_FAMILY_UNVERIFIED
    assert LineageFlag.NO_BASE_HASH in v.flags
    assert v.confidence < 0.9  # never as confident as a hash-verified file


def test_empty_sha256_treated_as_no_hash():
    md = {
        "ss_base_model_version": "sd_1.5",
        "ss_new_sd_model_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "software": '{"name": "ai-toolkit"}',
    }
    v = verify_base_lineage(metadata=md, widths=FLUX_WIDTHS)
    assert v.metadata_base_hash is None
    assert LineageFlag.NO_BASE_HASH in v.flags


def test_isadirectory_sentinel_treated_as_no_hash():
    md = {"ss_base_model_version": "flux1", "ss_new_sd_model_hash": "IsADirectory"}
    v = verify_base_lineage(metadata=md, widths=FLUX_WIDTHS)
    assert v.metadata_base_hash is None
    assert LineageFlag.NO_BASE_HASH in v.flags


def test_genuine_non_flux_structure():
    md = {"ss_base_model_version": "sd_1.5", "modelspec.architecture": "stable-diffusion-v1/lora"}
    v = verify_base_lineage(metadata=md, declared_base="Flux.1 D", widths=SD15_WIDTHS)
    assert v.base_class == BaseClass.NON_FLUX
    assert LineageFlag.NON_FLUX_WIDTH in v.flags
    # manifest said FLUX, structure says SD1.5
    assert LineageFlag.MANIFEST_CONTRADICTS_METADATA in v.flags


def test_no_metadata_no_structure_is_unknown():
    v = verify_base_lineage(metadata={})
    assert v.base_class == BaseClass.UNKNOWN
    assert v.confidence == 0.0
    assert LineageFlag.NO_METADATA in v.flags


def test_no_metadata_but_flux_width_is_unverified():
    v = verify_base_lineage(metadata={}, widths=FLUX_WIDTHS)
    assert v.base_class == BaseClass.FLUX_FAMILY_UNVERIFIED
    assert v.confidence < 0.6


def test_confidence_ordering():
    # verified > off-base detection strength is fine, but verified must beat unverified
    verified = verify_base_lineage(
        metadata={"ss_new_sd_model_hash": CANON, "ss_base_model_version": "flux1"},
        widths=FLUX_WIDTHS,
    )
    unverified = verify_base_lineage(
        metadata={"ss_base_model_version": "flux1"}, widths=FLUX_WIDTHS
    )
    assert verified.confidence > unverified.confidence


def test_to_dict_roundtrip():
    v = verify_base_lineage(metadata={"ss_new_sd_model_hash": CANON}, widths=FLUX_WIDTHS)
    d = v.to_dict()
    assert d["base_class"] == BaseClass.FLUX1_DEV_VERIFIED.value
    assert isinstance(d["flags"], list)
    assert 0.0 <= d["confidence"] <= 1.0


def test_header_guard_on_corrupt_file(tmp_path):
    # a file whose declared header length is absurd must not OOM; returns None safely
    import struct

    f = tmp_path / "corrupt.safetensors"
    f.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    assert read_safetensors_metadata(f) is None
    v = verify_base_lineage(f)
    assert LineageFlag.HEADER_UNREADABLE in v.flags
