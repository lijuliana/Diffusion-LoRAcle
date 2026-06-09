# DiT-LoRAcle

A weight-to-language reader for diffusion transformers: read a customized image model's
LoRA weights and describe — in natural language — the concept / style / identity / hidden
trigger it encodes, **without ever running the model**.

This is the LoRAcle weight-space-reader paradigm (De Schamphelaere et al.) ported from
language models to diffusion transformers, with an execution-free safety-screening application.

## Documents
- `project_b_design_doc.md` — full design doc, incl. the detailed execution plan (§B.13).
- `dit_loracle_proposal_for_review.md` — condensed proposal (for external review).
- `weight-space-readers-lit-review.md` — verified literature review.
- `PROGRESS.md` — running progress journal (tasks done, results, analysis, storyline).
- `WORKING_NORMS.md` — cluster / privacy / research-discipline norms. **Read before running anything.**

## Layout
See §B.13.1 of the design doc. Briefly:
`ditloracle/{encoding,formats,data,probe,reader,safety,eval}`, `tests/`, `configs/`, `scripts/`, `results/`.

## Status
Phase 0 (local, no-GPU instrument validation + baselines). See `PROGRESS.md`.

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # invariance + parser unit tests (no GPU)
```

## Norms (do not skip)
- Author code locally in git; copy to the box to run. Push back only small JSON/CSV results.
- Honour the POC gates: the cheap decisive test passes before the expensive build it gates.
- Always report the trivial baselines (spectral / metadata / W2T-encoding) next to any headline.
- Anonymize before anything leaves the cluster; secrets are pod env-vars only.
