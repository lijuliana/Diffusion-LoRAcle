"""Merge per-shard minted manifests into one manifest the causal gate can consume.

Fanning a mint across boxes produces one manifest per shard, each holding the organisms that shard
completed. The gate wants a single manifest, and it wants `weights_path` to point somewhere it can
actually read — so paths are rewritten to the local (or bucket) location the caller will run from.

Also reports what did NOT survive: a shard that was preempted mid-organism, or organisms excluded by
payload verification. Silent shrinkage is the failure mode to avoid — a gate run on 30 of 47
organisms should say so, not quietly report on whatever arrived.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def merge(shard_paths: list[str], weights_root: str | None = None) -> dict:
    organisms: dict[str, dict] = {}
    failures: list[dict] = []
    for sp in shard_paths:
        d = json.loads(Path(sp).read_text())
        for o in d.get("organisms", []):
            if weights_root:
                o = dict(o)
                o["weights_path"] = str(Path(weights_root) / Path(o["weights_path"]).name)
            organisms[o["organism_id"]] = o          # last writer wins; ids are unique by construction
        failures.extend(d.get("failures", []))

    by_split = Counter(o.get("split", "?") for o in organisms.values())
    by_axis = Counter(o.get("axis", "none") for o in organisms.values())
    return {
        "n_shards": len(shard_paths),
        "n_organisms": len(organisms),
        "n_failures": len(failures),
        "by_split": dict(by_split),
        "by_axis": dict(by_axis),
        "organisms": sorted(organisms.values(), key=lambda o: o["organism_id"]),
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge shard manifests for the causal gate.")
    ap.add_argument("--shards", nargs="*", default=None, help="shard manifest paths")
    ap.add_argument("--glob", default="assets/organisms/minted_*_shard*.json")
    ap.add_argument("--weights-root", default=None,
                    help="rewrite weights_path to this directory (where the files actually are)")
    ap.add_argument("--out", default="assets/organisms/minted_manifest.json")
    a = ap.parse_args()

    paths = a.shards or sorted(glob.glob(a.glob))
    if not paths:
        raise SystemExit(f"no shard manifests matched {a.glob}")
    m = merge(paths, a.weights_root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=2))

    print(f"merged {m['n_shards']} shards -> {m['n_organisms']} organisms ({m['n_failures']} failures)")
    print(f"  by split: {m['by_split']}")
    print(f"  by axis : {m['by_axis']}")
    print(f"  -> {a.out}")
    if m["failures"]:
        stages = Counter(f.get("stage", "?") for f in m["failures"])
        print(f"  failures by stage: {dict(stages)}")


if __name__ == "__main__":
    main()
