# NeurIPS interp workshop — 4-day scope (written 2026-08-24)

## The split, and why it does not cost the ICLR paper

**Workshop = the LoRAcle port to image diffusion transformers.** A meta-model that reads a DiT
adapter's weights and says what it draws. Prior work classifies an image adapter's concept from weights into a small closed set
(Duszenko & Bielak, ICCS 2025 — 10 ImageNet classes + an NSFW flag, on SD1.5 U-Net); Africa et al. and
the backdoor detector emit a flag. **Open-language description of an image-model adapter, and the
DiT/MMDiT modality, remain unclaimed.** Do not write "nobody has done this for images" — it is false.

*(Superseded 2026-08-24: this doc previously scoped the workshop as "the encoder result". That was
scoping to what was safe rather than to what matters, and it was wrong. The encoder work is now the
section that justifies which fingerprint feeds the reader, plus the fallback if the reader does not
converge in time — it is already written up, so it costs nothing to hold in reserve.)*

**Confirmed non-archival (2026-08-24).** Three consequences, all favourable:
- The encoder result does **not** become prior work the ICLR paper must cite and argue against. §2 of
  the ICLR plan stands unchanged.
- Content is reusable verbatim. The encoder section, figures, robustness table and corpus description
  can go into the ICLR submission without rewriting or self-citation.
- The submission is low-risk, so scope should be chosen for the quality of feedback rather than for
  defensive priority. Include the strongest version of the argument, including the negative results
  about our own prior encoder.

## The claim

> Canonicalised singular vectors are the wrong feature for reading LoRA weight space. The GL(r) gauge
> is free — ΔW = B·A is already invariant — so the binding constraint is not symmetry but **conditioning**:
> individual singular directions are ill-conditioned wherever singular values crowd, which on real
> adapters is about half of them. Features that never index a direction — subspace projectors — recover
> the signal that per-direction features destroy.

Supporting results already in hand (25 organisms, `results/poc1c_organism_gate.json`):
- `subspace_proj` **1.000** concept / **0.957** rank-invariance — the only featurizer passing both at p<0.01.
- `our_svd` (canonicalised vectors) **0.479** — *below its own spectrum-only ablation* (0.624).
- Sign-invariance is **not** the fix: the design doc's own prescribed sign-invariant per-direction
  feature `uᵢvᵢᵀ` still scores only 0.53. Slot alignment is the culprit, not sign.
- σ-gap measurement: **59.2% of adjacent gaps < 1e-2** on real klein adapters (38,400 gaps, full corpus).
- Baselines implemented faithfully: u₁+logreg (`2607.25750`) 0.703, spectral (`2602.15195`-style) 0.539.

## What the 3-day GPU budget buys, and why this corpus

The gate set **cannot test the central claim**. It clamps the recipe by construction, which is what
makes it causal — but that also makes the rank/recipe leakage control *degenerate* (rank and module set
are constant, so it sits at chance whatever happens, and the harness says so in its own verdict). "The
encoder reads semantics, not recipe" is therefore untested so far.

**Workshop corpus (`assets/organisms/mint_plan_workshop.json`): 959 organisms, 150 concepts × 6 replicates.**
(Enlarged from 60 concepts on 2026-08-24. Safe mid-run: `generate_concepts(n)` is prefix-stable, so the
60-concept set is a strict subset and nothing already minted is wasted.)
- Every concept minted under **all 6** entries of `RECIPE_POOL` — a complete block design, so measured
  `concept←recipe` leakage is **exactly 0.0** and the control becomes *real*.
- Ranks **8/16/32/64/128** (60/120/60/60/60) — rank-invariance over 5 ranks instead of 3.
- Splits train 268 / test 104 / gate 47, held out at **family** level (7 held-out families).
- Retrieval power: 60 classes × 6 replicates, versus 16 and 7 queries today.
- Cost: **639 GPU-hrs ≈ 40 h on 16 L4s** — inside the 3-day cap with ~2 days left for analysis and writing.

## Schedule

| when | what |
|---|---|
| day 1 | mint runs (~35 h wall on 8 boxes); boxes self-halt and flush on completion |
| day 2 | mint completes; merge; re-run encoder comparison at full n; recipe-leakage control |
| day 3 | figures (encoder table, σ-gap histogram, ablation ladder), draft |
| day 4 | tighten, `/citation-checker` on the .bib, submit |

## Deliberately OUT of scope for the workshop

- Reader SFT / any LLM — that is the ICLR contribution.
- **The backdoor recipe fix** (0/3 payloads fired). Needed for POC-S and H4; irrelevant to an encoder
  paper. Deferred rather than rushed.
- FLUX.1-dev (still gated on an HF token). Workshop runs on klein, which is Apache-2.0 and therefore the
  *better* base for a releasable artifact anyway. The cross-base claim stays an ICLR item.

## Corpus-quality fix that this run depends on

`scripts/screen_concepts.py` (new) screens concepts for **base-model headroom** before minting.
Diagnosis: 7 of 47 gate organisms were rejected as null adapters, and 4 were *every replicate* of
`low_poly_3d`, whose base similarity (0.239) was the highest of any failing concept — the base already
renders low-poly, and since training images come from that same base, the loss-optimal adapter is
≈identity. The obvious repair (train weak concepts harder) is **barred**: steps are a function of
dataset size only, precisely so step count cannot predict the label, and varying steps per concept would
make ‖ΔW‖ a proxy for concept. So the lever is concept *selection* with the recipe held uniform.
