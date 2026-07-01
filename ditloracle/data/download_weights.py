"""Download the .safetensors weights for a scraped manifest (CivitAI), into the gitignored cache.

Adds `local_path` to each record it succeeds on; writes an enriched manifest. Skips files that fail
the pickle/virus scan or are too large. Resumable (skips already-downloaded). Caps total footprint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


def download(manifest_path: str, out_dir: str, max_mb: float = 400.0, cap_gb: float = 60.0,
             limit: int | None = None):
    records = json.loads(Path(manifest_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("CIVITAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    total_bytes = 0
    cap = cap_gb * 1e9
    enriched, ok = [], 0
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
            r["local_path"] = str(dest); enriched.append(r); ok += 1; continue
        if total_bytes > cap:
            enriched.append(r); continue
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    enriched.append(r); continue
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            total_bytes += dest.stat().st_size
            r["local_path"] = str(dest)
            ok += 1
            if ok % 25 == 0:
                print(f"[dl] {ok} files, {total_bytes/1e9:.1f} GB")
        except Exception as e:
            print("[dl] err", vid, str(e)[:60])
        enriched.append(r)

    out_manifest = manifest_path.replace(".json", "_dl.json")
    Path(out_manifest).write_text(json.dumps(enriched, indent=2))
    print(f"[dl] downloaded {ok}/{len(records)}; {total_bytes/1e9:.1f} GB → {out_manifest}")
    return enriched


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai.json")
    ap.add_argument("--out", default="assets/corpus/weights")
    ap.add_argument("--max-mb", type=float, default=400.0)
    ap.add_argument("--cap-gb", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    download(args.manifest, args.out, max_mb=args.max_mb, cap_gb=args.cap_gb, limit=args.limit)
