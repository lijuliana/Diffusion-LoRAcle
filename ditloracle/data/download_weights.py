"""Download the .safetensors weights for a scraped manifest (CivitAI), into the gitignored cache.

Adds `local_path` to each record it succeeds on; writes an enriched manifest. Skips files that fail
the pickle/virus scan or are too large. Resumable (skips already-downloaded). Caps total footprint.

BASE-LINEAGE GATE (PLAN §7.1). `scrape_civitai.py` filters on the creator-declared "Flux.1 D" tag,
which POC-0d showed is wrong or unconfirmable for 39% of the corpus (3.9% off-base merges, 35.1%
FLUX-family-unverifiable). The declaration cannot be checked before the bytes are on disk — the
evidence lives in the file's own metadata — so every downloaded file is verified HERE, tagged with
`_base_class`/`_lineage`, and anything outside `--stratum` is deleted again rather than stored. The
tag travels in the enriched manifest, so downstream training sets (`poc1_probe --base-stratum`) and
the audit can re-filter without re-reading the weights. Minting is unaffected (base set by
construction); this is for the wild-audit and test-wild slices.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from ditloracle.data.corpus_filter import ALL, PERMISSIVE, STRICT, classify_record, in_stratum


def _tag_lineage(rec: dict, dest: Path, stratum: str) -> tuple[dict, bool]:
    """Verify the downloaded file's base lineage; return (tagged record, keep?).

    Runs on freshly downloaded AND already-cached files, so a re-run of an older cache is filtered
    too (the cache predates this gate).
    """
    rec = classify_record({**rec, "local_path": str(dest)})
    keep = in_stratum(rec["_base_class"], stratum)
    if not keep:
        rec.pop("local_path", None)
    return rec, keep


def download(manifest_path: str, out_dir: str, max_mb: float = 400.0, cap_gb: float = 60.0,
             limit: int | None = None, stratum: str = PERMISSIVE):
    records = json.loads(Path(manifest_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("CIVITAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    total_bytes = 0
    cap = cap_gb * 1e9
    enriched, ok, off_stratum = [], 0, 0
    breakdown: Counter = Counter()
    for r in records:
        if limit and ok >= limit:
            enriched.append(r); continue
        size_mb = (r.get("size_kb") or 0) / 1024
        url = r.get("download_url")
        vid = r.get("version_id")
        if not url or size_mb > max_mb:
            enriched.append(r); continue
        if r.get("pickle_scan") not in (None, "Success"):
            enriched.append(r); continue   # don't download unscanned/failed pickles
        dest = out / f"civitai_{vid}.safetensors"
        if dest.exists() and dest.stat().st_size > 0:
            r, keep = _tag_lineage(r, dest, stratum)
            breakdown[r["_base_class"]] += 1
            if keep:
                ok += 1
            else:
                dest.unlink(missing_ok=True)   # cached before the gate existed → drop it now
                off_stratum += 1
            enriched.append(r); continue
        if total_bytes > cap:
            enriched.append(r); continue
        try:
            import requests   # lazy: the resume/lineage paths must run in a venv without it
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    enriched.append(r); continue
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            n_bytes = dest.stat().st_size
            r, keep = _tag_lineage(r, dest, stratum)
            breakdown[r["_base_class"]] += 1
            if keep:
                total_bytes += n_bytes
                ok += 1
                if ok % 25 == 0:
                    print(f"[dl] {ok} files, {total_bytes/1e9:.1f} GB")
            else:
                # off-base / non-FLUX: never store it. Keeping it would violate the fixed-base GL(r)
                # symmetry argument (§B.4.4) and waste the footprint cap.
                dest.unlink(missing_ok=True)
                off_stratum += 1
        except Exception as e:
            print("[dl] err", vid, str(e)[:60])
        enriched.append(r)

    out_manifest = manifest_path.replace(".json", "_dl.json")
    Path(out_manifest).write_text(json.dumps(enriched, indent=2))
    print(f"[dl] kept {ok}/{len(records)}; {total_bytes/1e9:.1f} GB → {out_manifest}")
    print(f"[lineage] stratum={stratum}  dropped {off_stratum} off-stratum  "
          f"breakdown={dict(breakdown)}")
    return enriched


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai.json")
    ap.add_argument("--out", default="assets/corpus/weights")
    ap.add_argument("--max-mb", type=float, default=400.0)
    ap.add_argument("--cap-gb", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratum", choices=[STRICT, PERMISSIVE, ALL], default=PERMISSIVE,
                    help="base-lineage gate applied to every downloaded file: "
                         f"'{STRICT}' = pristine FLUX.1-dev only (test-wild/gate slice); "
                         f"'{PERMISSIVE}' = + FLUX-family-unverifiable (default; drops off-base "
                         f"merges and non-FLUX); '{ALL}' = store everything, tagged (hub audit only)")
    args = ap.parse_args()
    download(args.manifest, args.out, max_mb=args.max_mb, cap_gb=args.cap_gb, limit=args.limit,
             stratum=args.stratum)
