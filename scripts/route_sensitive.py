"""Run the sensitive-content pre-screen + (optional) VLM triage over the corpus, BEFORE labeling.

Writes routing records (benign vs sensitive pass + triage level) so the human labeling workflows can
be split by protocol. CSAM apparent-minor flags are escalated, never labeled.

  # automated pre-screen only (no model; metadata-based routing):
  PYTHONPATH=. python scripts/route_sensitive.py
  # + VLM triage of the sensitive subset (on a GPU box):
  PYTHONPATH=. python scripts/route_sensitive.py --backend qwen
"""

from __future__ import annotations

import argparse

from ditloracle.data.sensitive_prescreen import route_corpus
from ditloracle.data.vlm_backends import get_backend, guard_placeholder_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai_dl.json")
    ap.add_argument("--images-root", default="assets/corpus/images")
    ap.add_argument("--out", default="assets/corpus/sensitive_routing.json")
    ap.add_argument("--backend", default="none", choices=["none", "mock", "qwen"],
                    help="'none' = automated pre-screen only; 'mock'/'qwen' add the VLM triage stage")
    ap.add_argument("--quantization", default=None, choices=[None, "4bit", "8bit"],
                    help="bitsandbytes quantization for the qwen backend; '4bit' REQUIRED on a 16GB T4")
    ap.add_argument("--per-adapter", type=int, default=12, help="images requested (for the PG-drop heuristic)")
    ap.add_argument("--allow-mock", action="store_true",
                    help="permit the placeholder backend to write (to a .mock sibling file). A mock "
                         "triage level is fabricated, so it must never occupy the routing file.")
    args = ap.parse_args()

    backend = None if args.backend == "none" else get_backend(
        args.backend, **({"quantization": args.quantization} if args.backend == "qwen" else {}))
    # backend=None (pre-screen only) is metadata-based and real; only the mock VLM is guarded.
    out = guard_placeholder_output(backend, args.out, args.allow_mock)
    route_corpus(args.manifest, args.images_root, str(out), backend=backend,
                 n_images_requested=args.per_adapter)


if __name__ == "__main__":
    main()
