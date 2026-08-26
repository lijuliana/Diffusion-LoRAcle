# Singular subspaces outperform singular directions for reading LoRA weights

Status: **HEADLINE REFUTED — do not draft from this abstract.** Measured 2026-08-24 on recipe-varied
adapters: the subspace-projector advantage is an artefact of the recipe-CLAMPED gate set. With recipe
varying, subspace projectors score 0.302 against a random DW sketch's 0.325 and u1's 0.300 — a tie —
and they read the RECIPE more (0.795) than the encoder they replace (0.489). The clean separation below
was a ceiling effect from saturating at 1.000.

**What survives** is the negative result: canonicalised per-direction features are the wrong object
(0.116 on the varied axis, not significant), the mechanism is measured (59.2% of sigma-gaps < 1e-2;
sign-invariance tested and insufficient), and the robustness sweep corroborates it independently.
**What must be cut** is any claim that subspace projectors beat a random sketch, or that our feature is
recipe-blind.

The paper is now the **LoRAcle port** (`ditloracle/reader/`), with this material as the section
justifying the encoder choice. See PROGRESS 2026-08-24 and `notes/loracle_adoptions.md`.

Numbers below are from n=25/32, recipe-clamped, and are retained only as the ablation record.

---

## Abstract — REMOVED 2026-08-24

The previous abstract asserted that subspace projectors reach mAP 1.000 and are the right encoder.
That claim was refuted (see the status note above) and has been **deleted rather than annotated**,
because a warning above a live false claim still leaves the false claim in the file for someone to
draft from. It will be rewritten from the reader results when they exist.

## 1. Introduction

Problem. Reading an adapter's weights is cheaper than running it, and at hub scale that difference
decides whether screening is possible at all.

Gap. Existing weight-space readers disagree on the feature. [CITE 2607.25750] uses the top-left
singular direction. [CITE 2602.15195] uses five spectral statistics of ΔW. [CITE 2603.15990] maps each
adapter to a canonical form by QR decomposition followed by SVD and tokenises the result. The
canonicalisation view is stated most directly by [CITE 2410.04207], which identifies the GL(r) symmetry
of low-rank factorisations.

Contribution.
1. The GL(r) gauge is not the binding constraint. Any function of ΔW is already invariant.
2. Conditioning is the binding constraint, measured: 59.2% of adjacent singular-value gaps on real
   adapters fall below 1e-2.
3. Subspace projectors recover the signal that per-direction features lose, and are the only feature
   tested that separates concept and survives a change of rank at p < 0.01.
4. A controlled corpus in which recipe is decorrelated from concept by construction, so the claim that
   a feature reads content rather than training settings can be tested rather than assumed.

## 2. Setup

Corpus. Adapters are minted, so every label is exact. Each concept is trained under all six entries of
the recipe pool, giving a complete block design in which measured mutual information between concept
and recipe is 0.000. Ranks span 8, 16, 32, 64, 128. Splits hold out whole concept families.

Features compared. Subspace projectors (this work); a fixed random bilinear sketch of ΔW; the top-left
singular direction with the sign convention of [CITE bro2008]; five spectral statistics; canonicalised
singular directions; and two ablations that isolate which part of the canonicalised encoder carries the
signal. A rank-only feature is included as a control that must stay near chance.

Scoring. Grouped retrieval mean average precision, permutation null with 2000 draws, bootstrap
confidence intervals.

## 3. Results

### 3.1 Main comparison

Table 1. mAP with permutation p. n = 25 organisms, 16 concept queries and 7 rank queries.

| feature | concept | rank change | recipe control |
|---|---|---|---|
| subspace projectors | 1.000 (0.0005) | 0.957 (0.0045) | [W] |
| ΔW sketch | 0.984 (0.0005) | 0.729 (0.018) | [W] |
| top-left direction | 0.703 (0.0005) | 0.952 (0.007) | [W] |
| module norms | 0.652 (0.0005) | 0.606 (0.231) | [W] |
| sign-invariant per-direction | 0.574 (0.001) | 0.853 (0.011) | [W] |
| spectral statistics | 0.539 (0.0005) | 0.586 (0.258) | [W] |
| spectrum only | 0.501 (0.002) | 0.497 (0.562) | [W] |
| canonicalised directions | 0.479 (0.0035) | 0.715 (0.064) | [W] |
| QR-then-SVD tokens | 0.354 (0.089) | 0.594 (0.230) | [W] |
| rank only (control) | 0.206 (0.946) | 0.473 (0.729) | [W] |

Three readings. Canonicalised directions place eighth of nine features. Per-module norms, a single
scalar per module with no direction information, beat them. QR-then-SVD tokens do not reach
significance on concept.

### 3.2 The direction components subtract signal

Deleting the singular vectors from the canonicalised encoder raises concept mAP from 0.479 to 0.501.

### 3.3 Sign handling is not the explanation

The sign-invariant per-direction product reaches 0.574, short of both subspace projectors and the ΔW
sketch. Sign is not what per-direction features get wrong.

### 3.4 Why directions fail

Figure 2: distribution of adjacent singular-value gaps over all 32 adapters and all 60 modules
(38,400 gaps). Median 6.6e-3, 59.2% below 1e-2, 16.2% below 1e-3, minimum 7.5e-06. Perturbation bounds for singular vectors scale inversely with the gap, so a large
fraction of retained directions are not determined to useful precision, while the subspace they span is.

### 3.5 Robustness to perturbation

Table 2. Concept mAP after perturbing every adapter, n = 22. Retention is mAP relative to clean.

| perturbation | subspace projectors | ΔW sketch | u₁ | canonicalised directions | spectral statistics |
|---|---|---|---|---|---|
| clean | 1.000 | 0.907 | 0.714 | 0.472 | 0.546 |
| noise 1% | 1.000 | 0.905 | 0.714 | 0.485 | 0.546 |
| noise 5% | 1.000 | 0.909 | 0.718 | 0.455 | 0.546 |
| noise 10% | 1.000 | 0.898 | 0.723 | 0.438 | 0.545 |
| rescale B→cB, A→A/c | 1.000 | 0.907 | 0.714 | 0.472 | 0.546 |
| float16 round-trip | 1.000 | 0.907 | 0.714 | 0.472 | 0.546 |
| 8-bit quantisation | 1.000 | 0.906 | 0.714 | 0.467 | 0.546 |
| 4-bit quantisation | 1.000 | 0.901 | 0.724 | **0.336** | 0.546 |

Subspace projectors hold at 1.000 under every perturbation. Canonicalised directions are the only
feature that degrades: retention 0.93 at 10% noise and **0.71 under 4-bit quantisation**, where every
other feature retains at least 0.99. The mechanism is the one measured in §3.4. Quantisation moves the
singular values, crowded singular values reorder under small movements, and a feature indexed by
direction number follows them. A feature that reads the subspace does not have an index to lose.

The rescaling row is also a correctness check. Rescaling the factors leaves ΔW exactly unchanged, so a
feature that moves under it is reading the factorisation rather than the adapter. Nothing moves.

### 3.6 Reading content rather than training settings

[W] Requires the recipe-varied corpus. On a recipe-clamped corpus this axis has one distinct label and
cannot be scored.

### 3.7 Held-out families

[W]

## 4. Related work

[CITE 2410.04207] identifies the symmetry. [CITE 2603.15990] canonicalises by QR then SVD.
[CITE 2607.25750] and [CITE 2602.15195] read weights for screening. Subspace representations appear in
federated LoRA aggregation, where [CITE 2608.03267] and [CITE 2605.06733] adopt them because Euclidean
aggregation of factors is basis-dependent. Those systems use subspaces to combine adapters. We use them
to read one.

## 5. Limitations

One base model, FLUX.2-klein. One trainer. Retrieval rather than a trained reader; [CITE 2603.15990]
feeds canonical tokens to a transformer, which can learn to down-weight unstable coordinates, so the
margin reported here may narrow when a trained model sits on top. That comparison is the natural next
experiment.

---

## TODO
- [ ] fill [W] cells from the 419-organism corpus
- [ ] Figure 1: main comparison; Figure 2: gap histogram; Figure 3: ablation ladder
- [ ] build `paper/refs.bib`, then run citation-checker before any \cite is written
- [x] workshop confirmed non-archival (2026-08-24) — content reusable in the ICLR submission
