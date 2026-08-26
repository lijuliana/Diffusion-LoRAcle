"""A shard's verification verdicts must survive the shard dying.

On 2026-08-13 the box running shard 0 died before `mint_all` returned. The manifest was written only
at the end, so five trained adapters existed on disk that no merge could see, and nine organisms were
re-minted the next day. Weights are durable; verdicts were not. These tests pin the fix.
"""

import json
from pathlib import Path

from scripts.mint_run import _summary, _write_manifest


def test_manifest_is_readable_after_every_write(tmp_path):
    out = tmp_path / "m.json"
    plan = {"base_model": "FLUX.2-klein-4B"}
    minted, failed = [], []
    for i in range(5):
        minted.append({"organism_id": f"org{i}", "weights_path": f"/w/org{i}.safetensors"})
        _write_manifest(str(out), _summary(plan, [0] * 10, minted, failed, []))
        # a reader arriving at this instant must see valid JSON with everything so far
        d = json.loads(out.read_text())
        assert d["n_minted"] == i + 1
        assert [o["organism_id"] for o in d["organisms"]] == [f"org{j}" for j in range(i + 1)]


def test_failures_are_recorded_incrementally_too(tmp_path):
    out = tmp_path / "m.json"
    plan = {"base_model": "b"}
    failed = [{"organism_id": "bad0", "stage": "verify", "reason": "null adapter"}]
    _write_manifest(str(out), _summary(plan, [0] * 3, [], failed, []))
    d = json.loads(out.read_text())
    assert d["n_failed"] == 1 and d["failures"][0]["reason"] == "null adapter"


def test_write_is_atomic_leaving_no_partial_file(tmp_path):
    """The bucket sync copies this file on a timer; it must never copy a half-written one."""
    out = tmp_path / "m.json"
    _write_manifest(str(out), _summary({}, [], [{"organism_id": "a"}], [], []))
    assert out.exists()
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"
    json.loads(out.read_text())
