# DiT-LoRAcle

A weight-to-language reader for diffusion transformers: read a customized image model's
LoRA weights and describe — in natural language — the concept / style / identity / hidden
trigger it encodes, **without ever running the model**.

This is the LoRAcle weight-space-reader paradigm (De Schamphelaere et al.) ported from
language models to diffusion transformers, with an execution-free safety-screening application.

## Documents
- `PLAN.md` — **current master plan (mint-first pivot).** Read this first; it supersedes the POC ladder
  in the design doc on how we get labeled data.
- `project_b_design_doc.md` — full design doc, incl. the method, math, and original execution plan.
- `PROGRESS.md` — running progress journal (tasks done, results, analysis, storyline).
- `WORKING_NORMS.md` — cluster / privacy / research-discipline norms. **Read before running anything.**

## Layout
`ditloracle/{encoding,formats,data,probe,reader,safety,eval}` + `ditloracle/mint/` (the mint-first data
engine: `taxonomy`, `corpus_plan`, `trainer_config`), `tests/`, `scripts/`, `results/`.

## Status
Instrument validation done (POC-0, 100 tests pass). Pivoted to a **minted** training corpus (labels by
construction) after the wild human-labeling gate proved underpowered — see `PLAN.md` §1 and `PROGRESS.md`.
Next go/no-go: the POC-M causal gate on minted organisms.

Build the minting plan + trainer configs (local, no GPU):
```bash
python scripts/mint_corpus.py --base FLUX.1-dev --replicates 3
```

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # invariance + parser unit tests (no GPU)
```
