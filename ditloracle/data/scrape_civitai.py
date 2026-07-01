"""Scrape FLUX.1-dev LoRAs from CivitAI with structured labels + recipe metadata for POC-1.

Captures, per adapter version:
  - labels (weak supervision): tags, trainedWords (triggers), category, nsfw flag/level
  - recipe metadata (for confound splits + recipe-only baseline): baseModel, file size, hashes
  - provenance (for anti-confound splits): creator, modelId, versionId
  - the .safetensors download URL (we download the weights separately)

Design for the anti-confound splits (§B.6.4): we keep `creator` and a coarse `concept_family`
(derived from tags) so POC-1b can build cross-creator / within-creator / concept-family splits, and
we deliberately collect MULTIPLE adapters per creator where available (the API's per-creator listing)
so those splits are statistically estimable.

Run (writes a manifest, does NOT download weights — that's a separate step):
  PYTHONPATH=. python -m ditloracle.data.scrape_civitai --target 400 --out assets/corpus/manifest_civitai.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://civitai.com/api/v1/models"


def _headers():
    key = os.environ.get("CIVITAI_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _record(model: dict) -> list[dict]:
    """One model can have several versions; emit one record per FLUX.1-dev version with a safetensors."""
    out = []
    creator = (model.get("creator") or {}).get("username")
    tags = model.get("tags") or []
    for mv in model.get("modelVersions") or []:
        if mv.get("baseModel") != "Flux.1 D":
            continue
        files = [f for f in (mv.get("files") or [])
                 if f.get("name", "").endswith(".safetensors") and f.get("type") == "Model"]
        if not files:
            continue
        f = min(files, key=lambda x: x.get("sizeKB", 1e12))  # the LoRA file (smallest model file)
        out.append({
            "source": "civitai",
            "model_id": model.get("id"),
            "version_id": mv.get("id"),
            "name": model.get("name"),
            "creator": creator,                       # provenance → cross-creator splits
            "tags": tags,                             # weak labels
            "trained_words": mv.get("trainedWords") or [],  # trigger words
            "nsfw": model.get("nsfw"),
            "nsfw_level": model.get("nsfwLevel"),
            "poi": model.get("poi"),                  # "person of interest" → identity-cloning signal
            "base_model": mv.get("baseModel"),        # recipe
            "size_kb": f.get("sizeKB"),
            "file_name": f.get("name"),
            "download_url": f.get("downloadUrl"),
            "hashes": f.get("hashes"),
            "pickle_scan": f.get("pickleScanResult"),
            "stats": model.get("stats"),
        })
    return out


def scrape(target: int, out_path: str, sort: str = "Most Downloaded", nsfw_mix: bool = True):
    """Page the CivitAI API until `target` FLUX.1-dev LoRA versions are collected."""
    records: list[dict] = []
    seen = set()
    params = {"types": "LORA", "baseModels": "Flux.1 D", "limit": 100, "sort": sort}
    cursor = None
    pages = 0
    while len(records) < target and pages < 60:
        if cursor:
            params["cursor"] = cursor
        r = requests.get(API, params=params, headers=_headers(), timeout=60)
        if r.status_code != 200:
            print(f"[scrape] HTTP {r.status_code}; stopping"); break
        d = r.json()
        items = d.get("items", [])
        if not items:
            break
        for m in items:
            for rec in _record(m):
                key = rec["version_id"]
                if key in seen:
                    continue
                seen.add(key)
                records.append(rec)
        cursor = (d.get("metadata") or {}).get("nextCursor")
        pages += 1
        print(f"[scrape] page {pages}: {len(records)}/{target} versions")
        if not cursor:
            break
        time.sleep(0.5)  # be polite

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(records[:target], indent=2))
    # quick provenance summary (useful for split feasibility)
    from collections import Counter
    creators = Counter(r["creator"] for r in records[:target])
    multi = sum(1 for c in creators.values() if c > 1)
    print(f"[scrape] wrote {min(len(records),target)} records → {out_path}")
    print(f"[scrape] {len(creators)} creators; {multi} have >1 adapter (needed for within-creator splits)")
    return records[:target]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--out", default="assets/corpus/manifest_civitai.json")
    ap.add_argument("--sort", default="Most Downloaded")
    args = ap.parse_args()
    scrape(args.target, args.out, sort=args.sort)
