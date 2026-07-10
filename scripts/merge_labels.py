"""Merge per-shard human-label files into the canonical assets/corpus/human_labels.json.

When a team labels in parallel (`label_tool.py --shard k/N`), each writes
human_labels.shard{k}of{N}.json. This merges them (shards are disjoint by construction, so there
should be no key collisions; if there are, the LAST file wins and a warning prints). Run after all
shards are done, before POC-1b.

  PYTHONPATH=. python scripts/merge_labels.py            # auto-globs human_labels.shard*.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="assets/corpus/human_labels.shard*.json")
    ap.add_argument("--out", default="assets/corpus/human_labels.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"no shard files matching {args.glob}")
        return
    merged, collisions = {}, 0
    for f in files:
        d = json.loads(Path(f).read_text())
        for vid, rec in d.items():
            if vid in merged:
                collisions += 1
            merged[vid] = rec
        print(f"  {f}: {len(d)} labels")
    Path(args.out).write_text(json.dumps(merged, indent=2))
    print(f"merged {len(merged)} labels from {len(files)} shards → {args.out}"
          + (f"  ⚠ {collisions} key collisions (shards should be disjoint — check --shard usage)" if collisions else ""))


if __name__ == "__main__":
    main()
