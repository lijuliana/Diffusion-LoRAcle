"""POC-1b VLM pre-labeling — drafts the BENIGN fields ONLY, from sample images ONLY.

Safety fields are deliberately NOT drafted (see docs/poc1b_labeling_protocol.md): VLM refusal/bias
is worst there and anchoring would contaminate the labels the flagship safety claim rests on, so a
human fills them from scratch in label_tool.py.

Blindness is structural: draft_benign_fields receives image paths and nothing else. Do NOT pass
tags/creator/triggers into the model — that would re-open the confound POC-1a exposed.

Circularity guard (design doc §B.8.1): VLM_MODEL must be a DIFFERENT family from the eval scorer/
judge used later (OpenCLIP/DINO + GPT/Gemini). Default lineage: Qwen-VL.

Run:
  PYTHONPATH=. python scripts/vlm_draft_benign.py --manifest assets/corpus/manifest_civitai_dl.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ditloracle.data.vlm_backends import get_backend
from ditloracle.probe.schema import BY_KEY, empty_draft, vlm_schema_prompt

_SCHEMA_PROMPT = vlm_schema_prompt()   # full multi-field schema (T1+T2+T3), built from schema.py


def _validate(parsed: dict) -> dict:
    """Coerce a raw VLM JSON dict to the schema: keep known keys, snap categoricals to the closest
    legal choice (else null), default missing keys. Never trusts the VLM to be schema-perfect."""
    out = empty_draft()
    for key, val in (parsed or {}).items():
        f = BY_KEY.get(key)
        if f is None:
            continue
        if f.dtype in ("categorical", "ordinal"):
            v = str(val).strip().lower().replace(" ", "_") if val is not None else None
            out[key] = v if v in f.choices else None
        elif f.dtype == "bool":
            out[key] = bool(val) if isinstance(val, (bool, int)) else (str(val).strip().lower() in ("true", "yes", "1"))
        else:
            s = (str(val).strip() if val is not None else "")
            out[key] = s or None
    return out


def draft_benign_fields(image_paths: list[str], backend) -> dict:
    """Return the full benign field schema (T1+T2+T3) drafted from images ALONE.

    The backend receives ONLY image_paths + _SCHEMA_PROMPT — never any metadata (blindness is
    structural). Raw JSON is schema-validated by `_validate`; any failure degrades to the empty
    schema so the human pass labels from blank (still valid, just slower)."""
    try:
        raw = backend.generate_json(image_paths, _SCHEMA_PROMPT)
    except Exception as e:
        print(f"  [warn] backend failed on {len(image_paths)} imgs: {str(e)[:80]} → empty draft")
        return empty_draft()
    return _validate(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai_dl.json")
    ap.add_argument("--images-root", default="assets/corpus/images")
    ap.add_argument("--out", default="assets/corpus/vlm_drafts.json")
    ap.add_argument("--backend", default="mock", choices=["mock", "qwen"],
                    help="'mock' (no deps, dry run) or 'qwen' (Qwen2.5-VL on a GPU box)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--quantization", default=None, choices=[None, "4bit", "8bit"],
                    help="bitsandbytes quantization for the qwen backend; '4bit' REQUIRED on a 16GB T4")
    ap.add_argument("--max-images", type=int, default=4,
                    help="max images per adapter to feed the VLM (4 safe on 16GB T4; 12 on >=40GB)")
    ap.add_argument("--limit", type=int, default=None, help="cap #adapters this run (smoke/cost test)")
    args = ap.parse_args()

    backend = get_backend(args.backend, **({"model_id": args.model_id, "quantization": args.quantization,
                                            "max_images": args.max_images}
                                           if args.backend == "qwen" else {}))

    # We read the manifest ONLY to enumerate which version_ids have local images — never to read
    # tags/creator into the draft. version_id + image dir is all that crosses into labeling.
    records = json.loads(Path(args.manifest).read_text())
    out_path = Path(args.out)
    drafts = json.loads(out_path.read_text()) if out_path.exists() else {}

    n_new = 0
    for r in records:
        if args.limit and n_new >= args.limit:
            break
        vid = str(r.get("version_id"))
        if vid in drafts:
            continue
        img_dir = Path(args.images_root) / vid
        image_paths = sorted(str(p) for p in img_dir.glob("*")) if img_dir.exists() else []
        if not image_paths:
            continue
        fields = draft_benign_fields(image_paths, backend)
        drafts[vid] = {"benign": fields, "vlm_model": backend.model_id}
        n_new += 1
        if n_new % 25 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(drafts, indent=2))   # checkpoint (resumable)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(drafts, indent=2))
    print(f"drafted {n_new} new (total {len(drafts)}) with backend={backend.model_id} → {args.out}")
    if args.backend == "mock":
        print("NOTE: mock backend — drafts are placeholders. Use --backend qwen on a GPU box for real drafts.")


if __name__ == "__main__":
    main()
