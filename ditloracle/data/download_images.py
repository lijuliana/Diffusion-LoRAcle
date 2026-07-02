"""C2 image-set provenance — fetch creator SHOWCASE images for each FLUX adapter version.

For POC-1b labeling we use the LoRA's own CivitAI sample images (provenance = "showcase"). They
are free, already exist, and unblock labeling now; the controlled "our-own-generated" set (a fixed
neutral prompt run through each adapter) is a later, compute-heavy upgrade for the rigorous eval +
generate-and-verify (see docs/c2_image_provenance.md).

The scrape (scrape_civitai.py) intentionally does NOT keep image URLs (it kept only weight/recipe
metadata), so this script re-queries the CivitAI model endpoint per model_id and pulls the
`modelVersions[].images[].url` list for the versions in our manifest.

Confound mitigation: we download up to --per-adapter images per version (default 12) to average over
the creator's cherry-picking, NOT a single hero image. We sort PG-first so the benign labeling pass
sees safe content first, and (default) we DROP mature previews (nsfwLevel >= --nsfw-max) so the
benign VLM/human pass isn't blocked by refusal — the safety category is still labeled from the
adapter's metadata-derived nsfw flag elsewhere. Set --keep-nsfw to retain them.

Images land at assets/corpus/images/<version_id>/NN.jpg  — exactly where vlm_draft_benign.py and
label_tool.py look (their --images-root default). A per-version images_meta.json records each image's
nsfwLevel + source URL for provenance/audit.

Run (test on a few first):
  PYTHONPATH=. python -m ditloracle.data.download_images \
      --manifest assets/corpus/manifest_civitai_dl.json --limit 5
Full run:
  PYTHONPATH=. python -m ditloracle.data.download_images --manifest assets/corpus/manifest_civitai_dl.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests

MODELS_API = "https://civitai.com/api/v1/models"


def _headers():
    key = os.environ.get("CIVITAI_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _resize_url(url: str, width: int | None) -> str:
    """CivitAI CDN supports an inline transform segment. URLs come in two forms:
        .../<uuid>/orig                         (API `url` field, no filename)
        .../<uuid>/original=true/<id>.jpeg      (resolved download URL with filename)
    Both accept a `width=N` transform segment in place of the original-size segment.
    """
    if not width:
        return url
    if "/original=true/" in url:
        return url.replace("/original=true/", f"/width={width}/")
    return url.replace("/orig", f"/width={width}")


def fetch_version_images(model_id: int, version_id: int, headers: dict) -> list[dict] | None:
    """Return the images list for one version, or None if the model fetch failed."""
    r = requests.get(f"{MODELS_API}/{model_id}", headers=headers, timeout=60)
    if r.status_code != 200:
        print(f"  [model {model_id}] HTTP {r.status_code}")
        return None
    for mv in r.json().get("modelVersions") or []:
        if mv.get("id") == version_id:
            return mv.get("images") or []
    return []  # version not found in model (rare; treat as zero images)


def pick_images(images: list[dict], per_adapter: int, nsfw_max: int, keep_nsfw: bool) -> list[dict]:
    """Filter to still images, optionally drop mature, then take up to per_adapter, PG-first."""
    imgs = [im for im in images if (im.get("type") or "image") == "image" and im.get("url")]
    if not keep_nsfw:
        imgs = [im for im in imgs if (im.get("nsfwLevel") or 1) < nsfw_max]
    # PG-first so the benign pass sees safe content first; stable within level.
    imgs.sort(key=lambda im: (im.get("nsfwLevel") or 1))
    return imgs[:per_adapter]


def download_one(version_id: int, picks: list[dict], out_root: Path, width: int | None,
                 headers: dict, overwrite: bool) -> int:
    vdir = out_root / str(version_id)
    vdir.mkdir(parents=True, exist_ok=True)
    meta = []
    n = 0
    for i, im in enumerate(picks):
        dst = vdir / f"{i:02d}.jpg"
        url = _resize_url(im["url"], width)
        meta.append({"file": dst.name, "url": im["url"], "nsfwLevel": im.get("nsfwLevel"),
                     "width": im.get("width"), "height": im.get("height")})
        if dst.exists() and not overwrite:
            n += 1
            continue
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200 and r.content:
                dst.write_bytes(r.content)
                n += 1
            else:
                print(f"  [v{version_id}] img {i} HTTP {r.status_code}")
        except Exception as e:
            print(f"  [v{version_id}] img {i} ERR {e}")
        time.sleep(0.1)
    (vdir / "images_meta.json").write_text(json.dumps(
        {"version_id": version_id, "provenance": "civitai_showcase", "images": meta}, indent=2))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai_dl.json")
    ap.add_argument("--images-root", default="assets/corpus/images")
    ap.add_argument("--per-adapter", type=int, default=12,
                    help="max images per version (>1 averages over creator cherry-picking)")
    ap.add_argument("--width", type=int, default=768,
                    help="CDN resize width (px); 0 = original. 768 is plenty for VLM captioning")
    ap.add_argument("--nsfw-max", type=int, default=4,
                    help="drop previews with nsfwLevel >= this (4 = R/mature). Ignored with --keep-nsfw")
    ap.add_argument("--keep-nsfw", action="store_true", help="keep mature previews too")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N versions (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    headers = _headers()
    records = json.loads(Path(args.manifest).read_text())
    # one model can hold several versions; group so we hit each model endpoint once.
    by_model: dict[int, list[int]] = defaultdict(list)
    order = []
    for r in records:
        mid, vid = r.get("model_id"), r.get("version_id")
        if mid is None or vid is None:
            continue
        if vid not in by_model[mid]:
            by_model[mid].append(vid)
            order.append((mid, vid))
    if args.limit:
        order = order[: args.limit]
        wanted_models = {mid for mid, _ in order}
        by_model = {mid: vs for mid, vs in by_model.items() if mid in wanted_models}

    out_root = Path(args.images_root)
    width = args.width or None
    done_versions = {vid for _, vid in order}
    total_imgs = 0
    n_with_images = 0
    processed_models = set()
    for mid, vid in order:
        if mid not in processed_models:
            processed_models.add(mid)
        images = fetch_version_images(mid, vid, headers)
        if images is None:
            continue
        picks = pick_images(images, args.per_adapter, args.nsfw_max, args.keep_nsfw)
        if not picks:
            print(f"[v{vid}] 0 usable images (had {len(images)}; all filtered or none)")
            continue
        got = download_one(vid, picks, out_root, width, headers, args.overwrite)
        total_imgs += got
        n_with_images += 1 if got else 0
        print(f"[v{vid}] {got}/{len(picks)} images "
              f"(model {mid}, {len(images)} avail)")
        time.sleep(0.3)  # be polite to the API
    print(f"\n{n_with_images}/{len(done_versions)} versions have images; {total_imgs} images total "
          f"→ {out_root}/<version_id>/")


if __name__ == "__main__":
    main()
