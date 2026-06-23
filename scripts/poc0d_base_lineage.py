"""POC-0d component A3: measure base-lineage verifiability over the real CivitAI corpus.

Answers the §B.4.4 question the width-only check could not: of the ~441 downloaded adapters, how
many are *verifiably* pristine FLUX.1-dev, how many are FLUX-architecture but unverifiable, how many
are off-base (merged) FLUX, and how many are not FLUX at all. The conforming/verifiable fraction is
itself a POC-0d gate number.

Run: PYTHONPATH=. python scripts/poc0d_base_lineage.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ditloracle.formats.base_lineage import BaseClass, verify_base_lineage
from ditloracle.formats.safetensors_io import load_lora_factors

CORPUS = Path("assets/corpus")
MANIFEST = CORPUS / "manifest_civitai_dl.json"
OUT = Path("results/poc0d_base_lineage.json")


def main() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    by_path = {r["local_path"]: r for r in manifest if r.get("local_path")}

    per_file = []
    class_counts: Counter[str] = Counter()
    declared_counts: Counter[str] = Counter()
    n = 0
    n_with_widths = 0

    for path, rec in by_path.items():
        p = Path(path)
        if not p.exists():
            continue
        n += 1
        declared = rec.get("base_model")
        declared_counts[str(declared)] += 1

        # structural widths (cheap; same source the old width-only check used)
        widths: set[int] | None = None
        try:
            factors = load_lora_factors(p)
            if factors:
                n_with_widths += 1
                widths = set()
                for f in factors.values():
                    widths.add(int(f["A"].shape[1]))
                    widths.add(int(f["B"].shape[0]))
        except Exception:
            widths = None

        verdict = verify_base_lineage(p, declared_base=declared, widths=widths)
        class_counts[verdict.base_class.value] += 1
        per_file.append({"file": p.name, **verdict.to_dict()})

    def pct(x: int) -> float:
        return round(100 * x / n, 1) if n else 0.0

    verified = class_counts[BaseClass.FLUX1_DEV_VERIFIED.value]
    off_base = class_counts[BaseClass.OFF_BASE_FLUX_MERGE.value]
    unverified = class_counts[BaseClass.FLUX_FAMILY_UNVERIFIED.value]
    non_flux = class_counts[BaseClass.NON_FLUX.value]
    unknown = class_counts[BaseClass.UNKNOWN.value]

    summary = {
        "n_files": n,
        "manifest_declared_base_breakdown": dict(declared_counts),
        "base_class_breakdown": dict(class_counts),
        "verified_flux1_dev_pct": pct(verified),
        "off_base_flux_merge_pct": pct(off_base),
        "flux_family_unverified_pct": pct(unverified),
        "non_flux_pct": pct(non_flux),
        "unknown_pct": pct(unknown),
        # "verifiable" = we could make a strong (hash-backed) call either way (verified OR off-base)
        "strongly_verifiable_pct": pct(verified + off_base + non_flux),
        "note": (
            "manifest declares 100%% FLUX yet embedded metadata/structure disagrees for some files; "
            "see manifest_contradicts_metadata flag in per-file records."
        ),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "per_file": per_file}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nfull per-file report -> {OUT}")
    return summary


if __name__ == "__main__":
    main()
