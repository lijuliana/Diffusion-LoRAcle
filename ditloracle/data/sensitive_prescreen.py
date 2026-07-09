"""Sensitive-content pre-screen + VLM triage (signed-off protocol, 2026-06-27).

Goal: minimize human exposure to harmful content while keeping the human as the final ratifier of
safety labels (the flagship safety claim can't rest on a model that refuses/mislabels hard cases).

Three stages, run BEFORE any human sees an adapter:
  1. AUTOMATED PRE-SCREEN (no model): route on signals we already have — CivitAI `nsfw`/`nsfw_level`
     from the manifest, and the count of images that survived the PG-first image download (an adapter
     that lost most images to the NSFW filter is itself a signal). Cheap, deterministic.
  2. VLM TRIAGE (a vision model, not a human): for flagged adapters, sort each into
     {explicit, suggestive, non_sexual_nudity, non_sexual} so the human pass can be ordered/blurred and
     the truly explicit subset handled under the strictest protocol. Reduces human exposure.
  3. ROUTING: every adapter gets a `pass` ∈ {benign, sensitive} and a `triage` record. The benign pass
     and the sensitive pass are separate human workflows (different protocol, eligibility — §B.11.1).

This module does NOT assign final safety labels — a human does that (label_tool.py safety block). It
only routes + pre-sorts. CSAM is never triaged here: any apparent-minor signal escalates immediately
(flag-and-stop), never "labeled". See docs/safety_labeling_rubric.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# CivitAI nsfwLevel is a BITMASK of levels present across the model's images (PG=1, PG13=2, R=4,
# X=8, XXX=16), NOT a single severity. Empirically (our 500-adapter corpus) the R bit (4) is extremely
# common — set even on utility/style adapters whose showcase merely includes a racy image — so
# thresholding at >=4 over-flags ~77% of the corpus. We route to the SENSITIVE pass only on the
# genuinely-explicit bits X|XXX (8|16); the milder R content is handled by per-IMAGE PG filtering at
# download time (download_images.py keeps PG, drops mature), not by quarantining the whole adapter.
NSFW_EXPLICIT_BITS = 8 | 16   # X or XXX present


def _has_explicit_bit(level) -> bool:
    return isinstance(level, int) and (level & NSFW_EXPLICIT_BITS) != 0

TRIAGE_PROMPT = (
    "You are a content-safety triage assistant for a research dataset. Look ONLY at these images and "
    "classify the STRONGEST content present, as strict JSON: "
    '{"triage": one of [explicit|suggestive|non_sexual_nudity|non_sexual], '
    '"apparent_minor": true|false, "notes": "<brief, clinical>"}. '
    "explicit = sexual activity/explicit nudity; suggestive = sexualized but not explicit; "
    "non_sexual_nudity = nudity in a non-sexual/artistic/medical context; non_sexual = none of these. "
    "If anyone appears to be a minor, set apparent_minor=true and STOP describing further. "
    "Do not generate, only classify what is visibly present."
)
_TRIAGE_LEVELS = ("explicit", "suggestive", "non_sexual_nudity", "non_sexual")


@dataclass
class TriageRecord:
    version_id: str
    route_pass: str                 # "benign" | "sensitive"
    prescreen_reason: str           # why it was routed here (automated stage)
    triage_level: str | None = None  # VLM stage result (None if not VLM-triaged)
    apparent_minor_flag: bool = False  # ESCALATE if true — never auto-labeled
    n_images: int = 0
    notes: str = ""


def prescreen_record(record: dict, n_local_images: int, n_images_requested: int = 12) -> tuple[str, str]:
    """Automated stage (no model). Returns (route_pass, reason).

    A record is routed to the SENSITIVE pass if the hub metadata marks it mature, OR if the PG-first
    image download dropped a large fraction of images (a proxy for mostly-NSFW showcase)."""
    lvl = record.get("nsfw_level") or 0
    if record.get("nsfw") is True or _has_explicit_bit(lvl):
        return "sensitive", f"hub_explicit(level={lvl})"
    # if PG filter dropped most images (only meaningful once images are downloaded), the showcase is
    # likely mostly mature. Guarded on n_local_images>0 so it doesn't fire before download.
    if n_local_images > 0 and n_images_requested and n_local_images <= max(1, n_images_requested // 3):
        return "sensitive", f"few_pg_images({n_local_images}/{n_images_requested})"
    return "benign", "clean_metadata+enough_pg_images"


def vlm_triage(image_paths: list[str], backend) -> dict:
    """VLM stage: classify flagged content to pre-sort + reduce human exposure. Returns the raw
    triage dict (validated). On any failure → conservative {triage: 'explicit'} so a human reviews it
    under the strictest protocol rather than it slipping into the benign pass."""
    try:
        raw = backend.generate_json(image_paths, TRIAGE_PROMPT)
    except Exception:
        return {"triage": "explicit", "apparent_minor": False, "notes": "triage_failed→strict"}
    lvl = str(raw.get("triage", "")).strip().lower().replace(" ", "_")
    return {
        "triage": lvl if lvl in _TRIAGE_LEVELS else "explicit",   # unknown → strictest
        "apparent_minor": bool(raw.get("apparent_minor", False)),
        "notes": str(raw.get("notes", ""))[:200],
    }


def route_corpus(manifest_path: str, images_root: str, out_path: str,
                 backend=None, n_images_requested: int = 12) -> list[TriageRecord]:
    """Run pre-screen (+ optional VLM triage of flagged items) over the corpus; write routing records.

    backend=None → automated pre-screen only (no VLM stage; benign vs sensitive by metadata). Pass a
    VLM backend (ditloracle.data.vlm_backends) to also triage the sensitive subset."""
    records = json.loads(Path(manifest_path).read_text())
    out = []
    escalate = []
    for r in records:
        vid = str(r.get("version_id"))
        img_dir = Path(images_root) / vid
        imgs = sorted(str(p) for p in img_dir.glob("*")) if img_dir.exists() else []
        route, reason = prescreen_record(r, len(imgs), n_images_requested)
        tr = TriageRecord(version_id=vid, route_pass=route, prescreen_reason=reason, n_images=len(imgs))
        if route == "sensitive" and backend is not None and imgs:
            t = vlm_triage(imgs, backend)
            tr.triage_level = t["triage"]
            tr.apparent_minor_flag = t["apparent_minor"]
            tr.notes = t["notes"]
            if t["apparent_minor"]:
                escalate.append(vid)            # flag-and-stop; never auto-labeled (CSAM protocol)
        out.append(tr)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps([asdict(x) for x in out], indent=2))
    n_sens = sum(1 for x in out if x.route_pass == "sensitive")
    print(f"routed {len(out)} adapters: {len(out)-n_sens} benign, {n_sens} sensitive → {out_path}")
    if escalate:
        # Do NOT write image paths or any depiction to disk for these — just the IDs, for human escalation.
        print(f"\n*** ESCALATE ({len(escalate)}): apparent-minor flag set — DO NOT label; "
              f"follow the CSAM flag-and-escalate protocol (docs/safety_labeling_rubric.md). "
              f"version_ids: {escalate} ***")
    return out
