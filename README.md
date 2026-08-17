# DiT-LoRAcle

A weight-to-language reader for diffusion transformers: read a customized image model's
LoRA weights and describe — in natural language — the concept / style / identity / hidden
trigger it encodes, **without ever running the model**.

This is the LoRAcle weight-space-reader paradigm (De Schamphelaere et al.) ported from
language models to diffusion transformers, with an execution-free safety-screening application.

## Documents
- `PLAN.md` — **current master plan (mint-first pivot).** Read this first; it supersedes the design
  doc on how we get labeled data — the POC ladder (§B.7) and the harvest/label phases of the execution
  plan (§B.13.3–B.13.4).
- `project_b_design_doc.md` — full design doc, incl. the method, math, and original execution plan.
- `PROGRESS.md` — running progress journal (tasks done, results, analysis, storyline).
- `WORKING_NORMS.md` — **lost with the 2026-08 working tree; not yet rewritten.** The surviving
  operating principles are in `project_b_design_doc.md` §B.13.0 and the "Rule going in" note in
  `PROGRESS.md` (2026-08-13).

## Layout
`ditloracle/{encoding,formats,data,probe,reader,safety,eval}` + `ditloracle/mint/` (the mint-first data
engine: `taxonomy`, `corpus_plan`, `trainer_config`), `tests/`, `scripts/`, `results/`.

## Status
Instrument validation done (POC-0); 135 tests pass. Pivoted to a **minted** training corpus (labels by
construction) after the wild human-labeling gate proved underpowered — see `PLAN.md` §1 and `PROGRESS.md`.
The 47-organism POC-M gate mint launched 2026-08-13 on FLUX.2-klein (FLUX.1-dev is gated pending an HF
token); next go/no-go is `scripts/poc1c_organism_gate.py` on the minted set.

Build the minting plan + trainer configs (local, no GPU):
```bash
python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 3
```

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # invariance + parser unit tests (no GPU)
```
