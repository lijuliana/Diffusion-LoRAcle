"""Filter a wild adapter corpus to a verified base-lineage stratum (fixes the mixed-base bug).

The audit found the stored wild corpus is only ~60% verifiably pristine FLUX.1-dev (39% off-base
merges / unverifiable), even though every adapter was hub-labeled "Flux.1 D". `scrape_civitai.py`
filters at scrape time but `download_weights.py` never re-verifies, so the mixed base leaks in. The
fixed-base GL(r) symmetry argument (§B.4.4) needs one shared base, so the wild slices (test-wild eval
+ hub audit) must be stratified. Minting is unaffected (we set the base), so this is only for wild data.

`stratify_manifest` wraps formats.base_lineage.verify_base_lineage over a downloaded manifest and
partitions records by BaseClass, so a caller can keep only `flux1_dev_verified` (strict) or
`verified + unverified` (permissive training stratum, per design doc §B.1 gate-vs-training note).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ditloracle.formats.base_lineage import BaseClass, verify_base_lineage

# strata a caller may request
STRICT = "verified"                      # flux1_dev_verified only (the gate stratum)
PERMISSIVE = "verified_plus_unverified"  # + flux_family_unverified (the reader-training stratum)
ALL = "all"                              # keep everything, still TAGGED (hub audit; never a training set)

_STRATUM_CLASSES = {
    STRICT: {BaseClass.FLUX1_DEV_VERIFIED},
    PERMISSIVE: {BaseClass.FLUX1_DEV_VERIFIED, BaseClass.FLUX_FAMILY_UNVERIFIED},
    ALL: None,                           # None = no filtering
}


def in_stratum(base_class: str, stratum: str = STRICT) -> bool:
    """Is a record's `_base_class` inside `stratum`?

    The single definition of the strata, so the download path, the probe/training path and the audit
    cannot drift apart. Off-base merges and non-FLUX are outside every stratum except ALL.
    """
    if stratum not in _STRATUM_CLASSES:
        raise ValueError(f"stratum must be one of {list(_STRATUM_CLASSES)}, got {stratum!r}")
    keep = _STRATUM_CLASSES[stratum]
    return True if keep is None else base_class in {c.value for c in keep}


def _local_path(rec: dict) -> str | None:
    return rec.get("local_path") or rec.get("weights_path")


def classify_record(rec: dict) -> dict:
    """Attach a `_lineage` verdict dict to one manifest record (base_class + confidence + flags)."""
    path = _local_path(rec)
    verdict = verify_base_lineage(
        path=path if path and Path(path).exists() else None,
        declared_base=rec.get("base_model"),
    )
    out = dict(rec)
    out["_lineage"] = verdict.to_dict()
    out["_base_class"] = verdict.base_class.value
    return out


def stratify_manifest(manifest_path: str, stratum: str = STRICT,
                      out_path: str | None = None) -> dict:
    """Classify every record and split into kept (in `stratum`) vs dropped.

    Returns {stratum, kept, dropped, breakdown, out_path}. Writes the kept manifest if out_path given.
    """
    if stratum not in _STRATUM_CLASSES:
        raise ValueError(f"stratum must be one of {list(_STRATUM_CLASSES)}, got {stratum!r}")
    records = json.loads(Path(manifest_path).read_text())

    kept, dropped, breakdown = [], [], Counter()
    for rec in records:
        c = classify_record(rec)
        breakdown[c["_base_class"]] += 1
        (kept if in_stratum(c["_base_class"], stratum) else dropped).append(c)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(kept, indent=2))

    return {
        "stratum": stratum,
        "n_total": len(records),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "kept_fraction": round(len(kept) / len(records), 4) if records else 0.0,
        "breakdown": dict(breakdown),
        "out_path": out_path,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stratify a wild corpus manifest by verified base lineage.")
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai_dl.json")
    ap.add_argument("--stratum", choices=list(_STRATUM_CLASSES), default=STRICT)
    ap.add_argument("--out", default=None, help="write the kept manifest here")
    a = ap.parse_args()
    summary = stratify_manifest(a.manifest, a.stratum, a.out)
    print(f"[lineage] stratum={summary['stratum']}  "
          f"kept {summary['n_kept']}/{summary['n_total']} ({summary['kept_fraction']:.1%})")
    for cls, n in sorted(summary["breakdown"].items(), key=lambda x: -x[1]):
        print(f"    {cls:26} {n}")
    if a.out:
        print(f"    kept manifest -> {a.out}")
