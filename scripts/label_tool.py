"""POC-1b human labeling tool — blind, split-field (see docs/poc1b_labeling_protocol.md).

For each adapter it shows ONLY the sample-image paths (never tags/creator/triggers/title), plus the
VLM draft for the BENIGN fields (confirm or edit), then a BLANK safety section to fill FROM SCRATCH.
Writes incrementally to assets/corpus/human_labels.json — resumable, safe to Ctrl-C.

The blindness is structural: this script reads images + the vlm draft, and never opens the metadata
manifest. Labels it writes carry provenance.blind = true.

Run:
  PYTHONPATH=. python scripts/label_tool.py --labeler jl
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from ditloracle.probe.schema import ALL_FIELDS, CORE_BENIGN_FIELDS, SAFETY_FIELDS, empty_draft


def _open_images(image_paths: list[str]):
    """Pop the sample images in the OS viewer so the labeler can actually SEE them (not just paths).
    macOS `open`, Linux `xdg-open`, Windows `start`. Best-effort; never fatal."""
    if not image_paths:
        return
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            subprocess.run(["open", *image_paths], check=False)
        elif sysname == "Linux":
            for p in image_paths:
                subprocess.run(["xdg-open", p], check=False)
        elif sysname == "Windows":
            for p in image_paths:
                subprocess.run(["cmd", "/c", "start", "", p], check=False)
    except Exception as e:
        print(f"  [could not auto-open images: {e}; open them manually]")


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic — never leave a half-written label file


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{text}{suffix}: ").strip()
    return val or (default or "")


def _verify_field(f, current):
    """Show one benign field's draft; let the human confirm (Enter) or correct. Categorical fields
    print their controlled vocabulary; values are validated against it."""
    tier = {1: "T1·gate", 2: "T2·rich", 3: "T3·diag"}[f.tier]
    if f.dtype in ("categorical", "ordinal"):
        print(f"    choices: {', '.join(f.choices)}")
        while True:
            v = _prompt(f"  [{tier}] {f.key}", str(current) if current else "")
            v = v.strip().lower().replace(" ", "_")
            if not v and (f.nullable or f.tier == 3):
                return None
            if v in f.choices:
                return v
            print(f"      must be one of {f.choices}" + (" (or blank=null)" if f.nullable else ""))
    if f.dtype == "bool":
        v = _prompt(f"  [{tier}] {f.key} (y/n)", "y" if current else "n")
        return v.strip().lower() in ("y", "yes", "true", "1")
    v = _prompt(f"  [{tier}] {f.key}", str(current) if current else "")
    return v or None


def label_one(vid: str, image_paths: list[str], draft: dict, labeler: str, mode: str = "core",
              open_images: bool = False) -> dict:
    print("\n" + "=" * 70)
    print(f"adapter {vid}   ({len(image_paths)} sample images)   [mode: {mode}]")
    for p in image_paths:
        print(f"   {p}")
    if open_images:
        _open_images(image_paths)
    print("-" * 70)
    b = {**empty_draft(), **(draft.get("benign") or {})}
    benign = dict(b)
    # which non-auto benign fields the human verifies now; the rest keep the VLM draft (flagged).
    to_verify = CORE_BENIGN_FIELDS if mode == "core" else [f.key for f in ALL_FIELDS if not f.auto]
    verified, unverified = [], []
    print(f"BENIGN — verifying {len(to_verify)} {'CORE (T1 gate)' if mode=='core' else 'all'} fields "
          f"(Enter keeps draft); long-tail keeps VLM draft, flagged for later.")
    for f in ALL_FIELDS:
        if f.auto:
            continue
        if f.key in to_verify:
            benign[f.key] = _verify_field(f, b.get(f.key))
            verified.append(f.key)
        else:
            unverified.append(f.key)        # keep draft value as-is

    print("SAFETY fields — fill FROM SCRATCH (no VLM draft shown, by design):")
    safety = {}
    for f in SAFETY_FIELDS:
        safety[f.key] = _verify_field(f, None)
    if benign.get("perceived_age_range") == "minor_lt18":
        safety["depicts_apparent_minor"] = True

    return {
        "benign": benign,
        "safety": safety,
        "provenance": {
            "benign_source": "vlm_draft+human_verified",
            "safety_source": "human_from_scratch",
            "labeler": labeler,
            "vlm_model": draft.get("vlm_model"),
            "blind": True,
            "label_mode": mode,
            "verified_fields": verified,        # human-confirmed → gate-eligible
            "vlm_unverified": unverified,       # draft only → verify later before gating on these
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", required=True, help="your initials/id (recorded in provenance)")
    ap.add_argument("--drafts", default="assets/corpus/vlm_drafts.json")
    ap.add_argument("--images-root", default="assets/corpus/images")
    ap.add_argument("--out", default="assets/corpus/human_labels.json")
    ap.add_argument("--mode", choices=["core", "full"], default="core",
                    help="core = verify only T1-gate + safety (fast, ~15 fields); "
                         "full = verify the whole long tail too")
    ap.add_argument("--shard", default="1/1",
                    help="k/N: label only your slice of the corpus so a team can work in parallel "
                         "without collision (e.g. 1/3, 2/3, 3/3). Deterministic hash of version_id.")
    ap.add_argument("--route", choices=["benign", "sensitive", "all"], default="all",
                    help="restrict to one pre-screen pass (assets/corpus/sensitive_routing.json). "
                         "Benign labelers use 'benign'; the sensitive subset has its own protocol.")
    ap.add_argument("--routing", default="assets/corpus/sensitive_routing.json")
    ap.add_argument("--open", action="store_true",
                    help="pop each adapter's sample images in the OS viewer (you need to SEE them)")
    args = ap.parse_args()

    k, n = (int(x) for x in args.shard.split("/"))
    assert 1 <= k <= n, "shard must be k/N with 1<=k<=N"
    # Per-shard output file so parallel labelers never write the same JSON (merge later with
    # scripts/merge_labels.py). 1/1 keeps the canonical filename.
    out_path = Path(args.out)
    if n > 1:
        out_path = out_path.with_name(out_path.stem + f".shard{k}of{n}" + out_path.suffix)

    drafts = _load(Path(args.drafts))
    if not drafts:
        print(f"no VLM drafts at {args.drafts} — run scripts/vlm_draft_benign.py first.")
        return

    # route filter (benign vs sensitive pass)
    route_of = {}
    if args.route != "all" and Path(args.routing).exists():
        for rec in json.loads(Path(args.routing).read_text()):
            route_of[str(rec["version_id"])] = rec.get("route_pass", "benign")

    def in_shard(vid: str) -> bool:
        import hashlib
        h = int(hashlib.sha1(vid.encode()).hexdigest(), 16)
        return (h % n) == (k - 1)

    labels = _load(out_path)
    def needs(vid):
        if not in_shard(vid):
            return False
        if args.route != "all" and route_of.get(vid, "benign") != args.route:
            return False
        if vid not in labels:
            return True
        prev = (labels[vid].get("provenance") or {}).get("label_mode", "full")
        return args.mode == "full" and prev == "core"   # allow core→full upgrade pass
    todo = [vid for vid in drafts if needs(vid)]
    n_core = len(CORE_BENIGN_FIELDS) + len(SAFETY_FIELDS)
    print(f"mode={args.mode} shard={args.shard} route={args.route} (~{n_core if args.mode=='core' else 'all'} "
          f"fields/adapter). {len(labels)} labeled · {len(todo)} to do → {out_path}. Ctrl-C to stop (saved each step).")

    for vid in todo:
        draft = drafts[vid]
        img_dir = Path(args.images_root) / vid
        image_paths = sorted(str(p) for p in img_dir.glob("*")) if img_dir.exists() else []
        if not image_paths:
            print(f"[skip {vid}] no images at {img_dir}")
            continue
        try:
            labels[vid] = label_one(vid, image_paths, draft, args.labeler, mode=args.mode,
                                     open_images=args.open)
        except KeyboardInterrupt:
            print("\nstopping — progress saved.")
            break
        _save(out_path, labels)
    _save(out_path, labels)
    print(f"\nwrote {len(labels)} labels → {out_path}")


if __name__ == "__main__":
    main()
