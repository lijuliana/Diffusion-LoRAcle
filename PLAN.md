# DiT-LoRAcle — Revised Master Plan (mint-first)

Written 2026-08-12 after recovering the repository from a lost working tree. This plan supersedes the
POC ladder in `project_b_design_doc.md` §B.7 on one axis only: **how we get labeled training data**.
The method, the math, the encoder, and the four contributions are unchanged. Read the design doc for
the full scientific argument; read this for what we build next and why the strategy changed.

---

## 1. What changed and why

**The one decision:** the training corpus is **minted**, not harvested-and-labeled. We train controlled
LoRAs whose labels are known by construction, and we keep the wild hub corpus only for the test-wild
generalization evaluation and the public hub audit. This is the original LoRAcle strategy (they minted
their corpus) applied here.

**Why the harvest-and-label path (POC-1b) does not work.** The committed results show it, not just
intuition:

- `results/poc1a_apparatus__civitai_n411.json`: on 411 wild CivitAI FLUX LoRAs, cross-creator concept
  accuracy is **0.436 against a 0.435 chance baseline** — indistinguishable from chance. Within-creator
  concept mAP is 0.52–0.63 and within-recipe mAP is 0.43–0.51 (barely above chance). The rank-leakage
  control sits at chance, so the apparatus is clean; the signal is simply not separable from creator and
  recipe signatures on wild adapters.
- `results/poc0d_base_lineage.json`: of 441 adapters the hub labeled "Flux.1 D", only **60.1% are
  verifiably pristine FLUX.1-dev**. 3.9% are off-base merges, 35.1% are FLUX-family but unverifiable,
  0.9% unreadable. The fixed-base GL(r) symmetry argument (design doc §B.4.4) assumes one shared base;
  39% of the corpus violates or cannot confirm it. This is the "LoRAs from different libraries, not all
  FLUX.1-dev" problem.
- The gate's ground truth was **blind human labels at n≥300**. That pipeline (`scripts/label_tool.py`,
  the VLM-draft + human-verify shards) is slow, depends on a single captioner on exactly the content
  captioners are weakest on (stylization, identity, NSFW), and — per the two points above — even a
  perfect label set cannot rescue a signal that is at chance across creators once recipe is confounded.

The failure is structural: on the wild hub, concept and recipe/creator are correlated, so a weight
reader trained on wild labels learns signatures, and the anti-confound splits that would prove otherwise
are underpowered because wild data gives ~1 adapter per (creator × concept × rank) cell.

**Minting removes all three problems at once.** When we train the LoRA, the concept/style/identity/
trigger/payload label is exact; we set the base to one pristine checkpoint; and we build **counterfactual
matched sets** — hold recipe fixed and vary only concept, or hold concept fixed and vary only rank — so
the anti-confound tests are clean by construction and fully powered (as many cells per axis as we mint).
The design doc already specifies this substrate (§B.6.2, the `OrganismRecord` schema, the causal gate
POC-1c); the pivot promotes it from "the safety-only ground truth" to "the primary training corpus for
every arm."

**What we keep (validated, do not rebuild):**
- `ditloracle/encoding/svd_encoder.py` + `tests/test_invariance.py` — the GL(r)/sign/degeneracy-aware
  encoder. `poc0a_invariance` passes to 1e-9.
  **⚠ Corrected 2026-08-23 by the first real gate run:** (and note design doc §B.5.3 already mandated
  the right feature — the implementation had drifted from it, so this is a regression we shipped, not a
  gap in the design.) Measured refinement: sign-invariance is *not* the binding issue — the design
  doc's own prescribed sign-invariant per-direction feature `uᵢvᵢᵀ` still only scores 0.53. What breaks
  is **per-direction slot alignment** (direction 3 of adapter A vs direction 3 of adapter B), which
  near-degenerate spectra destroy. Features that never index a direction — pooled subspace projectors,
  or the whole-operator sketch — are the ones that work.
  Original note: "gauge-fixing is essential" was the wrong
  lesson from POC-0. ΔW = B·A is *already* exactly GL(r)-invariant, so invariance is free for any
  function of the product; what canonicalization cannot buy is **conditioning**. Emitting individual
  singular vectors uᵢ, vᵢ (what `OurSVDFeaturizer` did) is unstable — 47% of measured σ-gaps on real
  adapters fall below 1e-2, where uᵢ is ill-conditioned by Davis–Kahan. The featurizer is now
  `SubspaceProjFeaturizer` (basis-invariant subspace projectors), the only one to pass both gate axes.
  See PROGRESS 2026-08-23. The encoder module and its invariance suite stand; the *feature read off it*
  changed.
- `ditloracle/formats/*` — the FLUX key-scheme parsers, fused qkv/mlp split, base-lineage verifier.
- `ditloracle/probe/*` — spectral / raw-A·B / W2T-style baselines, recipe fingerprint, concept family.
- `ditloracle/safety/organism_schema.py` + `scripts/poc1c_organism_gate.py` — the causal gate harness
  with permutation/bootstrap significance. This becomes the **first real experiment**, not a late one.

**What we drop from the critical path:**
- The wild human-labeling gate (POC-1b) and its blind-label tooling. The labels-from-humans-on-wild
  path is retired. `scripts/label_tool.py`, `merge_labels.py`, the VLM benign-drafting on wild adapters
  → kept in the tree for the small hand-verified test-wild slice only, never as the training ground
  truth.
- Harvesting 40–60K wild adapters as *labeled training data*. We still harvest wild adapters, but only
  for (a) a few hundred hand-verified test-wild examples and (b) the unlabeled hub audit.

---

## 2. Competitive landscape as of August 2026 (re-verified) and the defensible claim

The design doc's lit review is from June 2026. Two things moved, and they sharpen what our spotlight
must rest on:

- **Detection-from-weights is now partly claimed.** `arXiv 2607.25750` ("Detecting CSAM Text-to-Image
  LoRAs From Weights") screens NSFW/CSAM adapters using the **top-left singular direction u₁** as a
  fingerprint, detection-only, and reports it generalizes across base models (verified 2026-08-23:
  Africa, Heine, Staes-Polet, Mai; diffusion LoRAs; robustness reported to rescaling).
  `arXiv 2602.15195` is **"Weight space Detection of Backdoors in LoRA Adapters"** (Puertolas
  Merenciano, Vasyagina, Zhu, Ferrando, Chaudhary) — title corrected 2026-08-23, we had it as
  "Detecting Backdoored LoRAs from Weights Alone", which is not the paper's title. Still LLM-only
  (Llama-3.2-3B / Qwen2.5-3B / Gemma-2-2B), five spectral stats of **the product ΔW** per Q/K/V/O →
  20-dim signature → logistic regression, binary.
  **Consequence:** "we detect malicious adapters from weights" is no longer novel by itself. It is table
  stakes. We must beat these as baselines, not headline against them.
- **The still-unclaimed white space (our spotlight):**
  1. **Open-language description** of an image-model adapter — what concept/style/identity it encodes and
     what a malicious one *does* — not a flag, score, class label, or embedding. No prior work verbalizes
     an image model's adapter.
  2. **Trigger and payload verbalization** from static weights (H4 ladder), which no detector attempts.
  3. **Wild-corpus validation + a hub-scale audit.** Every neighbor (W2T, the spectral/CSAM detectors)
     reports in-distribution numbers on minted or aligned populations. A train-controlled → test-wild
     result plus a public audit of the live hub is the sharpest separator.
  4. **The DiT / MMDiT modality and the first labeled DiT weight-space corpus.**

- **MasqLoRA (`2602.21977`) is still SD1.5/SDXL** ("FLUX version coming soon"). The FLUX/MMDiT
  LoRA-backdoor space is open today. We mint one working MMDiT backdoor organism early to bank it.
- **FLUX.2 [klein] 4B is Apache-2.0** (released 2026-01-15, diffusers + LoRA training supported, a LoRA
  fine-tunes in under an hour). This is our license-clean organism factory and release arm.

**The one-sentence spotlight claim (unchanged in spirit, sharpened in framing):** *A symmetry-aware
weight reader turns a fine-tuned image model's LoRA into an open-language description — what it depicts,
and whether it hides an NSFW/identity/backdoor payload with a recoverable trigger — accurate enough to
audit the live hub, and validated on adapters it was never trained on, without ever running the model.*
Lead with **describe + trigger/payload + wild audit**; never lead with "detect from weights."

**Title/abstract framing (added 2026-08-23, post full read of `2607.25750`):** theirs is a *probe*
(u₁ + logistic regression on a fixed proxy axis); ours is a **meta-model with verbalization** (an LLM
that reads weights and emits open language). Emphasize this in the title and abstract. Corollary: the
paper must showcase at least one application a yes/no probe structurally cannot do — payload
*description*, causal trigger *recovery* (H4-R2), and the open-vocabulary hub audit — or reviewers will
ask "why not just a probe?" Detection numbers appear only as baseline comparisons, never as the pitch.

---

## 3. Compute and storage

Inventory as of today (details in memory `cloud-compute-inventory`):

| Resource | What | State | Use |
|---|---|---|---|
| AWS `cs2881r-workhorse` | g6e.8xlarge, 1× L40S 48 GB, 300 GB EBS | **stopped** (confirmed 2026-08-23; zero running AWS instances) | ready fallback; course-named — confirm before starting/stopping |
| AWS `cs2881r-hardening` | g5.8xlarge, 1× A10G 24 GB | stopped | secondary / captioning |
| AWS `nla-exp4-qwen` | g6e.8xlarge, 1× L40S 48 GB | stopped | can start (quota allows 2× g6e.8xlarge) |
| AWS quota | 64 G-family vCPUs | — | up to 4× L40S concurrently (e.g. 1× g6e.12xlarge) |
| GCP `25julianal` / "Meta Model Interpretability" | L4×8, A100-80GB×2, T4×4; billing on | **usable now** | A100-80GB×2 for reader SFT, L4×8 for parallel minting; no corpus stored here |
| GCP `pepperhamsterpaws` | — | **searched, empty** | no buckets/VMs; nothing from the lost session survived. Corpus is re-created by minting |
| Azure T4 box | 16 GB, ssh 20.240.250.7 | deallocated/unreachable | 4-bit VLM captioning only |

**Recommendation.** The GCP "Meta Model Interpretability" project (billing on; A100-80GB×2, L4×8) is
now the preferred primary — it colocates compute with the corpus bucket (`gs://ditloracle-corpus`)
and the L4×8 quota is ideal for parallel minting. AWS L40S is the ready fallback.
- **Minting (FLUX.1-dev + klein LoRAs):** trainer = **ai-toolkit** (confirmed; best FLUX/klein LoRA
  support). Fan the capability corpus across **GCP L4×8** (klein-4B fits an L4; FLUX.1-dev LoRA fits with
  offload), or the AWS g6e L40S boxes. Run the POC-M pilot (~50–100 organisms) on whichever is up first.
- **Reader SFT:** backbone = **Qwen3-14B** (confirmed; matches the loracle MIT warm-start). Fits one
  **GCP A100-80GB** comfortably (or one L40S 48 GB in QLoRA rank-256 rsLoRA).
- **Storage: DONE 2026-08-12.** Nothing survived in GCP (both accounts searched), so the corpus is
  re-created by minting. `gs://ditloracle-corpus` stands in the GPU project, `us-central1` (colocated
  with the A100s; no cross-project egress); minted + wild-audit corpora written via `fsspec`/`gcsfs`. The
  minted corpus is small (~0.5–1K × ~50–150 MB = well under 1 TB) — far cheaper than the 6–12 TB
  wild-harvest the old plan needed. A side benefit of the pivot.
- **Cost flag (updated 2026-08-23):** AWS is clean — zero running instances, both cs2881r boxes stopped.
  The actual leak was GCP: five `g2-standard-8` L4 mint boxes left RUNNING and idle from Aug 14 to Aug 23
  (~$0.85/hr each, ≈$760 burned for nothing) because nothing stopped them when the mint finished. **Mint
  jobs must stop or delete their box on completion** — add that to the shard script, don't rely on a human
  noticing. Every figure below assumes L40S-class or the owned/credited cluster.

**GPU-hour budget (must fit ~2 months of available compute — the binding constraint).** 2× L40S run
continuously for 2 months ≈ 2,880 GPU-hrs; 4× ≈ 5,760. The whole pipeline fits with margin, no
scientific shortcuts:

| Stage | GPU-hrs (est.) | Notes |
|---|---|---|
| Mint pilot (POC-M, ~50–100 organisms) | 50–70 | klein ~15–20 min, FLUX.1-dev ~30–40 min each |
| Capability corpus (~0.5–1K LoRAs) | 200–400 | scale via `--replicates` + taxonomy size; klein ~half |
| Safety families + matched sets | 50–100 | 3 attack families, counterfactual sets, payload-fires checks |
| Reader SFT (Qwen3-14B QLoRA, several runs) | 200–400 | POC-C/T + scaling |
| Reader RL (Dr. GRPO, optional) | ~100 | short, LoRAcle used ~40 steps |
| Generate-and-verify + test-wild eval | 70–150 | render + CLIP/DINO scoring per checkpoint |
| Hub audit (execution-free reader pass) | 20–50 | the whole point: reading is cheap |
| Ablations (encoding, warm-start, backbone) | 100–200 | each re-runs SFT on a subset |
| **Total** | **~800–1,500 GPU-hrs** | fits 2 months on 2× L40S with headroom; faster fanning minting across GCP L4×8 |

Corpus scale is the tunable knob: `mint_corpus.py --replicates` and the taxonomy size set the LoRA
count, so we size minting to the remaining budget rather than a fixed target.

---

## 4. Architecture (unchanged core, pivoted data flow)

```
                          ┌── MINTED corpus (primary; labels exact) ──────────────┐
  concept/style/identity  │  taxonomy → training-set builder → LoRA trainer        │
  + safety families       │  (FLUX.1-dev primary, FLUX.2-klein Apache arm)         │
                          │  → OrganismRecord ground truth + counterfactual sets   │
                          └───────────────────────────┬───────────────────────────┘
                                                       ▼
  FLUX LoRA .safetensors ── per-module compact SVD (fold α/r; QR→SVD; fused split)
                                                       ▼
  weight-tokens: sign/degeneracy-invariant product features + φ(σ) + module/layer embeds
                 + 3072→5120 dimension bridge (frozen random-orthogonal first)
                                                       ▼
  Reader = base LLM + rank-256 rsLoRA (fresh default; loracle warm-start = ablation)
                                                       ▼
  STRUCTURED record (schema §B.5.6) → FREE TEXT   → scored by HARD RETRIEVAL + causal verify
                                                       ▲
                          ┌── WILD corpus (eval only; no training labels) ─────────┐
                          │  ~few-hundred hand-verified test-wild + full hub audit  │
                          └────────────────────────────────────────────────────────┘
```

The encoder, the injection, the reader, the output schema, and the evaluation metrics are exactly the
design doc. Only the top-left box changed from "harvest + caption + human-verify" to "mint."

---

## 5. Revised POC ladder (gated; cheap decisive test before every expensive step)

Each rung is a go/no-go. The order front-loads the two things that were never actually run for real
(the causal gate, and any minting) and that decide viability.

- **POC-0 — instrument validation. DONE.** Encoder invariance (`poc0a`, passes 1e-9), baselines
  (`poc0bc`), format/base triage (`poc0d`: 96% parseable, degeneracy handled, base-lineage verifier).
  No further work except the fixes in §7.

- **POC-M — mint pilot + the causal gate (THE new first gate; ~1 week, days of GPU).**
  Mint ~50–100 organisms: the concept-clamped-recipe matched set (8–16 concepts, one recipe), the
  rank-invariance sets (3 concepts × 4 ranks), one backdoor matched set (same payload, 3 triggers), and
  a spectral-match benign/malicious pair. Verify each payload fires (generate + confirm) before use.
  Run `scripts/poc1c_organism_gate.py` for real.
  - **Gate:** on the concept-clamped-recipe set, our SVD/product features separate concept with
    permutation-null p < 0.01 while the recipe fingerprint (constant by construction) is at chance; and
    same-concept-across-rank retrieves above chance. **This is the premise test POC-1b could never make
    cleanly.** Pass → the signal is in the weights and we can read it → build the reader. Fail → stop,
    fall back to C-Corpus/C-Audit and reassess the encoder.
  - **STATUS 2026-08-23: ✓ PASSED BOTH AXES** on 25 of the 47 organisms (the other 22 are 12 verify
    failures + 9 never minted; fill run pending). `subspace_proj` concept mAP **1.000, p=0.0005**;
    rank-invariance **0.957, p=0.0045**; rank-leak control at chance on both. Getting there required
    replacing the singular-vector feature with subspace projectors (see §1). **The u₁ baseline
    (`2607.25750`) is strong — 0.703 concept, 0.952 rank-invariance — and beats the old `our_svd` on
    both axes**, so §8's "beat the u₁ detector" is a real bar, not a formality; do not plan around a
    weak version of it (PROGRESS records how a bad sign convention understated it by ~2×). Re-confirm
    at capability-corpus scale before any of this goes in a figure; n is 16 and 7 queries respectively.
  - Why this is safe to bet on: unlike wild data, here recipe/creator/spectrum are clamped, so a positive
    result is causal, and a negative result is decisive rather than confounded.

- **POC-C — capability corpus + closed-set reader (~2–3 weeks).**
  Mint ~1–5K LoRAs spanning a designed concept/style/identity taxonomy (§6) on FLUX.1-dev, labels exact.
  SFT the tiny reader; emit the structured schema; score field accuracy + retrieval on held-out concepts
  (held-out at the taxonomy-family level, not random).
  - **Gate:** the reader beats the spectral, raw-A·B, W2T-style, and metadata baselines on held-out-
    concept closed-set accuracy and retrieval → H1 confirmed for a DiT, structured floor exists.

- **POC-T — free text scored by hard retrieval (~3 weeks).**
  Scale SFT; emit free text; primary metric = hard retrieval against matched negatives (concept-family /
  rank / recipe), ground truth = each adapter's own generations (no human labels needed). Generate-and-
  verify (CLIP-I/DINO, disjoint model family from the captioner) as the secondary check.
  - **Gate:** beats nearest-neighbor-caption (the memorization ceiling) and the metadata baseline on hard
    retrieval → H2, the open-language differentiator.

- **POC-S — safety pilot (~2 weeks, overlaps POC-C).**
  Expand the safety families minted in POC-M to full coverage: NSFW-injection (MasqLoRA ported to MMDiT),
  identity-cloning (synthetic/consented identities), backdoors with known (trigger→payload). Train
  detection + payload/trigger description on organisms; report ROC vs the spectral **and** the u₁-CSAM
  detector on the matched-spectra control; report payload-description accuracy.
  - **Gate:** beats both weight-only detectors on ROC on the matched-spectra set **and** produces usable
    payload descriptions (which the detectors structurally cannot) → H3 in reach.

- **H3 flagship — train-controlled / test-wild (the spotlight; ~4–6 weeks).**
  Train on organisms; evaluate on wild adapters never seen: wild-benign (false-positive rate), wild-
  malicious-natural (real hub NSFW/identity adapters, hand-verified), wild-malicious-crafted (attack
  recipes with settings we did not train on). Per-axis generalization breakdown; spectral-negative
  control; cost/throughput vs run-the-model; adaptive-attacker arm. **Fix the organism set and wild split before running,** so the split is not chosen around the result. Figures 2/3/4.

- **H4 — trigger inversion (spotlight, upgraded from bonus 2026-08-23).** `2607.25750` conspicuously
  stops short of this: they show a trigger-hidden attribute is still *classifiable* (0.83 AUROC) but
  never recover, verbalize, or causally verify the trigger itself. H4-R2 is therefore the sharpest
  separator from the nearest paper and stays in the paper under time pressure (see droppable list).
  Recover trigger from weights; verify causally
  (generate with recovered trigger → payload fires). Report at whatever ladder rung we reach. Figure 6.
  **Ladder** (frontier-calibrated 2026-08-17; evidence in `notes/lit/trigger-inversion-notes.md`):
  - R1 (safe floor): trigger-*presence* detection — already part of the safety hypothesis; H4 claims start at R2.
  - **R2 (primary endpoint): causal surrogate recovery** — the reader emits candidate trigger text from
    weights alone; success = rendering it fires the payload above an ASR threshold on held-out organisms.
    This is the field-standard bar (Neural Cleanse/PICCOLO/DBS all score inverted triggers by induced
    behavior, not string match).
  - R3 (report-only): token/semantic overlap with the ground-truth trigger.
  - R4 (stretch, frontier-unsolved): exact-string recovery — TDC-2023 recall was ~random even with high
    surrogate ASR; report if achieved, never promised.
  Scope: fixed-token text triggers only; semantic-predicate/dynamic (CLIBE-class) triggers have no unique
  string to invert — future work (latent-condition inversion; lit-notes gap #3). Novelty vs neighbors is
  covered in §2 above (verbalization from weights — detection-only and execution-based work doesn't attempt it).

- **C-Audit + C-Corpus — the deployment + dataset artifacts.** Run the reader over the harvested wild
  corpus (unlabeled) for the first hub-scale execution-free safety scan; release the minted corpus +
  ground-truth labels + encoder. Both are contributions even if a later H slips.

**Droppable under time pressure, in order:** H5 (cross-model/VLM), then H4-R4/R3 (exact-string, overlap
metrics — R2 causal surrogate stays), then the
adaptive-attacker arm. The paper is H1+H2+H3+corpus+audit and stands without any of these.

---

## 6. Minting corpus design

The corpus is the contribution and the training data; design it like the benchmark it will become.

**Base model.** Primary = **FLUX.1-dev** (the audit story needs it — ~42K wild FLUX.1-dev adapters exist
to audit). Secondary = **FLUX.2-klein-4B** (Apache-2.0) for the license-clean release and a "scales to
current frontier" arm. Mint the core taxonomy on both where budget allows; klein first for cheap
iteration, FLUX.1-dev for the headline.

**Concept taxonomy (drives diversity and the held-out splits).** A designed hierarchy, ~4 families ×
subfamilies, each a training-set recipe (prompt set + reference images + trigger word):
- **style** (art-movement, medium, rendering) — e.g. art-nouveau, ukiyo-e, watercolor, pixel-art,
  low-poly-3D, film-stock looks.
- **object/concept** — vehicles, architecture, creatures, props.
- **identity/subject** — synthetic or consented faces/characters only (no real private individuals).
- **safety families** — NSFW-injection, identity-clone, backdoor (trigger→payload). Minted in the same
  pipeline so labels and the schema are shared with benign (the unified-schema confound control).
Held-out splits are at the **family** level, so "generalizes to a new concept family" is a real test.

**Counterfactual matched sets (the confound controls, by construction).** For every axis in
`COUNTERFACTUAL_AXES`, mint a set that clamps everything and varies one factor: concept@fixed-recipe,
rank/α@fixed-concept, module-subset@fixed-concept, trigger@fixed-payload, payload@fixed-trigger,
spectral-matched benign/malicious, trigger-token@fixed-training-images. These give the causal gate and
the "reads semantics not spectra / not recipe" figures for free.

**Recipe diversity (so the reader is rank-robust).** Vary rank (8–128), α, target-module set (attn-only /
+MLP / +modulation), trainer (ai-toolkit / diffusers / kohya) — deliberately, and recorded as ground
truth — so the reader must read concept across recipes and we can show it does.

**Scale, graded.** POC-M ~50–100 (gate). POC-C ~1–5K (reader). Full ~5–15K if compute allows (still
under 1 TB, still cheaper than the old 6–12 TB wild harvest). Every organism carries a validated
`OrganismRecord` and a "payload fires" check before admission.

**✓ RESOLVED 2026-08-23 — concept diversity is now generative.** `taxonomy` composes three head axes
(20 styles / 25 subjects / 12 personas, each carrying a family) against two modifier axes (10 media,
8 palettes), seeded and interleaved on a fixed head cycle: **capacity 4,582 concepts** (the curated 22
plus 4,560 generated). The scale knob is a *parameter*, not an env var — `generate_concepts(n)` →
`build_plan(n_concepts=…)` → `mint_corpus.py --n-concepts` — so a plan records the concept count it was
built at. Three properties make it safe to grow a corpus incrementally:
- **The curated 22 are a prefix** and `generate_concepts(n1)` prefixes `generate_concepts(n2)`, so
  scaling up appends and never renames an organism that has already been minted. The default output is
  byte-identical to the pre-change plan (verified by diffing `build_plan` across both bases ×
  replicates {1,2,3,6}), so the in-flight POC-M gate is untouched and `mint_spec.py` has a zero diff.
- **Trigger collisions are impossible by construction**: generated stems carry two digits, every
  reserved surface (curated + safety + gate triggers, payload nouns) is digit-free and ≥3 chars, so no
  substring can alias. Exhaustively checked: 0 overlaps over all 4,560, and a 200-concept plan leaks
  zero payload/trigger surfaces into benign image sets (the PROGRESS balloon-vs-payload bug is guarded
  semantically, not just by name).
- **Held-out families still declared up front** — one new held-out family per group; at n=600 that is
  8 held-out families and a ~32% test fraction spanning all three benign kinds.
Confound audit holds at scale: `build_plan` returns **0 errors at n = 22 → 2,000** (up to 4,059
organisms), recipe⊥concept leak is exactly 0.0 whenever `replicates % 6 == 0` and otherwise
finite-sample bias with p = 1.0 ≫ `LEAK_ALPHA`, *decreasing* with breadth.
**One deliberate deferral:** capability seeds (`BASE_SEED + 1000·ci + rep`) collide with the safety
block at `n_concepts ≥ 51` for ~6 organisms. Harmless today (different datasets ⇒ different weights)
and nothing validates seed uniqueness, but re-basing the formula would move every existing capability
seed — **decide before POC-C**, not during it.

**Confound audit (enforced in code).** `corpus_plan.audit_confounds` measures, before any GPU-hour,
whether recipe predicts concept or the malicious/benign label, scored against a permutation null, and
`build_plan` refuses a plan that leaks. Every malicious organism has a matched benign twin (identical
cover images, recipe, seed, poison removed), which doubles as the spectral-match control for Fig 4.
Training steps depend only on dataset size, never on organism kind.

**Training data provenance.** Reference images per concept come from license-clean sources (own
generations from base FLUX with distinct prompts, CC/public-domain sets, or synthetic). This keeps the
released corpus redistributable, which the wild FLUX.1-dev NC license would not.

---

## 7. Fixes to land before scaling (from the codebase audit)

Concrete, all in the committed code:
1. **Base-lineage filter is not applied at download.** `scrape_civitai.py` filters to "Flux.1 D" but
   `download_weights.py` does not re-verify; 39% of the stored corpus is off-base/unverifiable. Add a
   verified-base filter step (wrap `formats/base_lineage.py`) and drop non-pristine adapters from any
   training set. For minting this is moot (we set the base), but the wild-audit and test-wild slices
   need it.
2. **Degeneracy in high-rank adapters.** `poc0d` found 277 near-degenerate directions (worst σ-gap
   5e-6). The encoder has the projector-diagonal fix; confirm `probe/featurizers.py` uses
   `degeneracy_safe=True`, and clamp/skip tokens below a σ floor in the reader injection (the ‖v‖
   denominator blows up otherwise — design doc §B.12.1 gotcha).
3. **Fused qkv/mlp split** (72% of sample) — confirm `load_canonical_factors` applies the split before
   featurizing on every path (probe and reader).
4. **Mock VLM default** — the wild captioning scripts default to `--backend mock`; only relevant now for
   the small test-wild slice, but guard against silently shipping placeholder captions.

---

## 8. Evaluation and figures (updated for the new landscape)

Baselines to beat (all reported next to every headline number): spectral-stat detector (`2602.15195`),
**the u₁+logreg singular-direction CSAM detector (`2607.25750`)** — beat it in Figs 3/4, and
specifically on the matched-spectra control, where a single direction can flag but structurally cannot
*describe* — W2T-style encoder (`2603.15990`), raw-A·B,
nearest-neighbor-caption (memorization ceiling), metadata/creator-tag, and the run-the-model
caption/NSFW upper bound with its cost.

- **Fig 1 (cover):** a poisoned adapter → the reader's text ("injects <NSFW concept> on trigger ‹word›")
  → generated images confirming it, beside a benign style adapter, all without running the screener.
- **Fig 2 (flagship):** train-controlled → test-wild detection + payload-description, per generalization
  axis (creator / concept / rank / attack-config / base).
- **Fig 3:** execution-free ROC + cost/throughput vs run-the-model and vs the two weight-only detectors.
- **Fig 4:** reads semantics not spectra (matched-spectra control) — now must also beat the u₁ detector.
- **Fig 5:** H2 hard retrieval vs nearest-neighbor-caption and metadata.
- **Fig 5b:** module localization (where the signal lives).
- **Fig 6:** trigger ladder (H4), honest about where exact recovery is underdetermined.
- **Robustness sweep (table stakes, per `2607.25750`):** adopt the *shape* of their evasion protocol
  (not their method) — additive weight noise at several σ, global rescaling, fp16 round-trip, 8-/4-bit
  quantization, norm-equalization — and report reader accuracy under each. Their sweep set the rigor
  bar reviewers will now expect; a small version on the safety-detection arm suffices.
- **Fig 8:** hub audit at scale (the deployment figure).

---

## 9. Risk ladder and derisking

- **Unsinkable floor:** C-Corpus (first labeled DiT weight-space dataset — now cleaner because minted) +
  C-Audit scaffolding. Survives any H failure.
- **Premise gate (POC-M):** now clean and causal, run first, days of GPU. The single biggest change:
  the gate that decides viability is no longer a slow, confounded human-label study but a fast,
  by-construction-clean minting experiment.
- **Floor (H1):** weights predict concept for a DiT, beating baselines. Very likely given a clean corpus.
- **Expected (H2 + H3-held-in):** open-language descriptions that verify by generation + held-in safety
  triage with payload description.
- **Spotlight (H3 wild + audit):** train-controlled → test-wild beats the weight-only detectors *and*
  describes, at hub scale.
- **Bonus:** H4 trigger inversion, H5 transfer.

Mitigations: every risky claim degrades to an easier rung of itself; the corpus/audit stand regardless;
minting removes the label-quality risk that sank POC-1b; the Apache klein arm removes the release-license
risk; warm-start is an ablation, not a dependency.

---

## 10. Timeline and immediate next actions

**Target: the ICLR deadline, full scope, no scientific shortcuts.** The directive is to do everything —
H1–H3, wild audit, trigger inversion — at the same rigor, with total GPU-hours kept inside a ~2-month
compute window (budget in §3). The schedule below is paced to that; nothing is cut to hit a date, and
corpus scale is the knob that keeps compute in budget.

**Sequencing (single-person-hours; compress by fanning minting across GCP L4×8 + the AWS boxes):**
- Week 1: land the §7 fixes; build the minting pipeline (§ below); mint the POC-M pilot; run the causal
  gate. **Decision point.**
- Weeks 2–4: capability corpus + closed-set reader (POC-C). Safety families expand in parallel (POC-S).
- Weeks 4–7: free-text + hard retrieval (POC-T); safety pilot ROC.
- Weeks 7–12: H3 flagship (test-wild), Figures 2/3/4.
- Weeks 12–15: H4 trigger inversion; hub audit; corpus release prep.
- Weeks 15–18: ablations, writing, figures. H5 if time.

**Status of the immediate actions (updated 2026-08-17; details in PROGRESS.md):** the mint-first data
engine, the corpus bucket, and the promoted causal gate are built; the 47-organism POC-M gate mint
launched 2026-08-13 on FLUX.2-klein. **Still open (needs you):**
1. Decide the fate of the running g6e (`cs2881r-workhorse`) — is it doing coursework, or stop it?
2. ~~HuggingFace token~~ **DONE 2026-08-24.** Token supplied and verified: user `juli-li`, fine-grained,
   and the FLUX.1-dev licence is accepted — a ranged GET on a gated transformer shard returns HTTP 206,
   so the weights are genuinely fetchable, not merely the repo metadata. Stored in gitignored
   `notes/.env` (chmod 600) and pushed to the mint fleet, which had been downloading unauthenticated
   (LoRAcle's own notes measure a ~15x throughput penalty for that).
   **Sequencing decision:** the workshop corpus stays on **klein**. 250+ adapters are already minted
   there, the deadline is under four days, and FLUX.1-dev is a 12B model that costs materially more per
   organism. FLUX.1-dev is now unblocked for the ICLR headline corpus and the ~42K-adapter wild audit,
   which is where it was always needed.
   Still open: CivitAI API key for the wild-audit slice.
