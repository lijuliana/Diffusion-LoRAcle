# Reading Diffusion Transformer Adapters from Weights Alone

**Working title (paper):** *Reading Diffusion Transformer Adapters from Weights Alone: A Corpus, a Symmetry-Aware Representation, and Execution-Free Safety Screening*
*(internal codename: DiT-LoRAcle)*

**One-sentence thesis:** *A symmetry-aware representation of a diffusion-transformer LoRA's weights supports open-language descriptions of what the adapter encodes — enabling execution-free safety screening that beats static spectral, classifier, and metadata baselines, without ever running the model.*

**Framing (for submission).** The contribution is **four standalone pieces**, in priority order:
1. **A rigorous corpus + benchmark** — a cleaned, confound-stratified real DiT-LoRA corpus plus a set of **controlled counterfactual organisms** with ground-truth labels, and an evaluation protocol with anti-confound splits. *(Stands alone; useful to the field regardless of reader quality.)*
2. **A symmetry-aware DiT-LoRA representation** — GL(r)-canonical, sign/degeneracy-robust, rank-robust weight tokens.
3. **Open-language adapter descriptions** — structured-then-free-text descriptions, evaluated by **hard retrieval/discrimination** (not caption-similarity theater).
4. **Execution-free safety screening** — flagging *and describing* NSFW-injection / identity-cloning / backdoored adapters from weights alone, beating static detectors; validated train-controlled → test-wild.

> **LoRAcles is inspiration and the method ancestor we cite — not the framing.** This is *not* "a port of LoRAcles to diffusion." We borrow the residual-injection reader recipe (and reuse the released checkpoints as one warm-start *ablation*), but the contribution is the corpus, the representation, the descriptions, and the safety evaluation listed above. The codename `DiT-LoRAcle` is internal only; the paper does not lead with it.

Trigger *recovery* is a stretch amplifier (§B.3 H4), explicitly **not** the core claim.

> This document supersedes the Project B sketch in `weight_space_interp_proposal.md`. It folds in a verified literature review (saved alongside this doc) and corrects several premises in the original sketch (notably: **weights2weights is NOT reusable for a DiT**; the **LoRAcle encoding is parameter-free residual injection, not a learned projection**; and the **free dataset comes from ~42K real FLUX.1-dev LoRAs, not from w2w**).

---

## B.0 How to read this document

- **§B.1–B.3** — problem, prior art, and the falsifiable claims. The "why this is novel and safe."
- **§B.4** — the math: LoRA weight-space symmetries, the encoding, and why it is sound. *Read before coding. NB: the math **justifies** the encoder + ablations; it is not the contribution — the eventual paper keeps the detailed treatment in an appendix (see note atop §B.4).*
- **§B.5** — the architecture and training recipe (LoRAcle-faithful, text-only reader, warm-started).
- **§B.6** — the **data engine**: the three-phase, confound-controlled labeling pipeline. This is the long pole and the main infra contribution.
- **§B.7** — the **POC ladder**: a graded sequence of go/no-go milestones, starting from the cheapest possible signal. *We do not build the full system before the POC passes.*
- **§B.8** — evaluation, baselines, money figures.
- **§B.9** — risk ladder, **§B.10** — timeline, **§B.11** — scoop defense, **§B.12** — open questions for the LoRAcle creator / mentor.

A guiding principle, inherited from the proposal: **never bet the paper on a single result that can silently fail.** Every claim below has a weaker sibling that is still a contribution.

### B.0a Target paper structure + limitations (the shape we write toward)

1. **Problem** — public DiT-LoRA adapters are hard to inspect safely at scale (run-the-model is costly and evadable).
2. **Method** — symmetry-aware canonicalization/tokenization of LoRA ΔW; rank-robust weight tokens; a reader producing **structured-then-free-language** descriptions.
3. **Dataset** — cleaned real DiT-LoRA corpus + **controlled counterfactual organisms** + wild held-out audit set.
4. **Evaluation** — H1 concept/style/identity prediction; **H2 hard retrieval from descriptions**; H3 controlled→wild safety triage + payload description; H4 candidate-set trigger recovery (stretch).
5. **Ablations** — representation (raw A/B vs ΔW/SVD/product), spectrum-only vs directions, module families, rank generalization, warm-start vs fresh, reader-vs-simple-encoder (§B.8.4).
6. **Limitations (state them, don't hide them)** — exact trigger recovery may be **information-theoretically underdetermined**; wild labels are **noisy weak supervision**; **base lineage matters** (we filter/stratify); static weights **cannot reveal all execution-time behaviors** (we screen/triage, we don't certify safety).

---

## B.1 Problem statement and motivation

Customized image models proliferate as **LoRA adapters**: small low-rank weight diffs `ΔW = BA` distributed on hubs (HuggingFace, CivitAI) in the tens of thousands. Today, the only way to know what an adapter *does* — what concept, style, identity, or hidden behavior it encodes — is to **run it** and inspect generations. That is expensive at hub scale, requires the full model + GPU, and is easy to evade (a backdoor only fires on a secret trigger you would have to guess to observe).

We ask: **can a model read the adapter's weights directly and tell us, in natural language, what it encodes — without execution?**

The contribution is one **method** with one **headline application**:

1. **Method — a symmetry-aware weight→language reader.** A GL(r)-symmetry-aware encoder turns a DiT-LoRA's weight directions into tokens and injects them into a language model that emits an **open-language description** of what the adapter encodes (concept / style / identity / safety property). This is new for the visual modality (prior diffusion weight-space work only *generates*, *classifies*, or *embeds* — none verbalizes), and the symmetry-awareness (§B.4) is what makes the reader generalize across the real, messy adapter population rather than overfit a coordinate system.
2. **Application — execution-free safety triage that beats static baselines.** Because an open-language description is strictly more informative than a binary flag or a fixed class label, the reader can **triage** hub adapters — flag *and describe* NSFW-injection, identity-cloning, and backdoored adapters — from weights alone, more usefully than the existing static weight-only methods (spectral-statistic detectors, closed-set classifiers). The flagship result is **train-controlled / test-wild generalization** (§B.6.3): trained on our organisms, the reader triages malicious adapters it has never seen on a held-out slice of the real hub corpus. The verbalization is *verifiable* — generating from the reader's own description reproduces the adapter's behavior (§B.8.1), which is what turns "a description" into "a useful triage signal."

*(Build order: we develop the general verbalization capability first because it is lower-risk and is the prerequisite for the safety triage — see §B.3a #5.)*

### Why diffusion transformers, and why first (not VLMs)

This was a deliberate scoping decision, and the literature supports it:

- **The encoding ports cleanly.** DiT LoRAs target `nn.Linear` attention/MLP projections, so the LoRAcle SVD-direction-token encoding (designed for LLM linear layers) applies *directly*. VLM adapters are heterogeneous (vision tower + connector + LM) and the encoding story is muddier.
- **The free labeled data only exists for diffusion, and it is abundant.** ~42,500 FLUX.1-dev LoRAs on HuggingFace (≈66% of *HF's* text-to-image LoRAs — HF skews FLUX/research), **tagged with concepts/styles/trigger words**, plus a smaller FLUX slice on CivitAI. Harvested (not minted), this is a **~50K-scale single-base corpus** — matching the LoRAcles scale for the cost of storage, not the $30–250K w2w-style minting bill. No comparable tagged corpus exists for VLM adapters.
  - **⚠ Single-base constraint (this caps the corpus — do not over-count).** The method is FLUX.1-dev-specific end to end: the SVD encoder, the fixed MMDiT module schema, and the §B.4.4 symmetry argument all assume one shared base. **Only FLUX.1-dev adapters are usable** — SDXL/Pony/Illustrious/SD1.5 (which *dominate* CivitAI) and even FLUX.2 LoRAs live in different coordinate systems and cannot be fed through a FLUX.1-dev encoder. So combining hubs adds only CivitAI's *FLUX subset*, not its bulk; it does not multiply the count. Realistic usable supply is **~40–60K, not 100K** — 100K is reachable only by going *multi-base*, which means a separate encoder/schema/lineage per base and within-base-only gating (i.e. the H5 cross-model arm scaled into the main corpus — a different, larger paper; explicitly out of current scope). **POC-0d measures the true FLUX.1-dev yield (HF + CivitAI-FLUX, post-dedup); we target "the full available FLUX.1-dev supply," not a fixed 100K.**
  - *Gate vs training split:* the ~60% base-verified-FLUX.1-dev fraction (POC-0d) is the stricter subset the confound-controlled **gate** needs; reader **training** (§B.6.1) tolerates the unverified-but-FLUX-arch remainder, so the full FLUX harvest feeds it.
- **The cover figure is uniquely strong for images.** "Weights → text → three generated images that match" is visually self-evidencing in a way no VLM-capability description is.
- **The safety story is concrete and timely** (CivitAI/HF threat model, MasqLoRA-class attacks).
- VLM adapters are correctly a **stretch goal (H5)**, not the spine.

---

## B.2 Prior art and the precise gap

| Work | What it does | Modality | Why it is *not* us |
|---|---|---|---|
| **LoRAcles** (NeurIPS 2026 submission, `x9MbM7QmQN`; ckpts MIT) | weight → language reader | **LLM** text LoRAs | Our direct recipe ancestor; text-only. We port it across modality. |
| **weights2weights** (Dravid et al., NeurIPS 2024, `2406.09413`) | sample / edit / invert weights via PCA subspace; linear attribute directions | SD1.5 **U-Net** | **Generative**, not weight→text. Locked to SD1.5 coordinate system — *not reusable for a DiT* (see §B.4.5). |
| **Duszenko & Bielak** (ICCS 2025) | **classify** a LoRA's concept into 10 ImageNet classes; NSFW flag | SD1.5 U-Net | Closed-set **classification**, not free-text; U-Net; binary safety. |
| **"A LoRA is Worth a Thousand Pictures"** (`2412.12048`) | LoRA weights as a **style descriptor**; clustering beats CLIP/DINO | SD/SDXL | Retrieval/clustering, not language; motivates that weights carry style. |
| **W2T / "Weights know what they can do"** (`2603.15990`, *verify ID*) | trains a bespoke encoder over LoRA factors → **label / score / embedding** | LLM + SD1.4 | **Different paradigm** (a from-scratch property predictor, *not* a language reader). We use the **LoRAcle approach**: inject weight-derived tokens into a pretrained LLM that *verbalizes in free text*, do **execution-free safety + payload/trigger description**, and evaluate on a **wild held-out hub corpus**. Used only as one encoding-baseline point (§B.8.2). |
| Spectral backdoor detectors (`2602.15195`, LLM) | 20-dim spectral-stat logistic reg.; binary flag | **LLM** LoRAs | **Binary**, no description; not architecture-invariant; LLM-only. The baseline we beat *and describe past*. |
| MasqLoRA / "When LoRA Betrays" (`2602.21977`) | NSFW/object **attack** recipe | SD1.5/SDXL U-Net | Attack, not reader; "FLUX coming soon" — DiT is open. |
| MILAN / CLIP-Dissect / FALCON / DnD | vision feature → language | activations (must run model) | Read **activations on a probe set**; we read **weights, no execution**. |

**Framing matters for novelty — always frame it the high way.** The same work reads as *high* or
*medium* novelty depending on the framing, and we commit to the high one in every artifact:
- **HIGH (our framing):** *open-language DiT-LoRA reading + execution-free safety screening.* No prior
  work emits free-text descriptions of an image model's adapter, and none screens adapters for safety
  from weights alone — this is genuinely unclaimed.
- **MEDIUM (the framing to avoid):** *"weights reveal behavior."* Stated that broadly, W2T, Learning-on-
  LoRAs, and weights2weights are close neighbors (they already show weights carry recoverable
  structure). If a reader/reviewer hears only "weights reveal behavior," our delta looks incremental.
- **Discipline:** lead with *open-language* + *execution-free safety screening* + *wild-corpus
  validation*; never pitch the project as "weights are informative." The neighbors established *that*;
  we contribute *what you can now do with it* (describe, triage, audit).

**The clean gaps that constitute our novelty:**

1. **No weights→language *reader* exists for any image model.** Prior diffusion weight-space work either *generates* weights (w2w) or *classifies/scores/embeds* them (Duszenko&Bielak, W2T). None produces a free-text natural-language description, and none does it for a diffusion transformer.
2. **No backdoor attack or weight-only screening targets diffusion transformers yet**, and **no defense describes the payload/trigger** rather than flagging it.
3. **No weight-space method has been validated on a wild, held-out real-world corpus.** Existing results are on minted/aligned populations. Our **train-controlled / test-wild** evaluation (§B.6.3) is the central generalization claim and the sharpest separator from all prior art.

The combination — *open-language, execution-free, safety-screening, for DiTs, validated in the wild* — is unclaimed.

**Both "no precedent" claims were verified by a multi-source literature review (June 2026; 25 claims, adversarial 3-vote check, all primary sources).** Key results:
- **Backdoor attacks all target U-Net, none target diffusion transformers.** BadDiffusion (CVPR 2023, `2212.05400`), TrojDiff (CVPR 2023, `2303.05762`), VillanDiffusion (NeurIPS 2023, `2306.06874`), and the personalization attack (AAAI 2024, `2305.10701`) are all DDPM/DDIM/SD-v1–2 **U-Net**. Rickrolling (ICCV 2023, `2211.02408`) backdoors the **CLIP text encoder**, not the diffusion model at all — so cite it as a text-encoder attack, not a diffusion-backdoor (correction to its use in §B.6.2). **MasqLoRA (CVPR 2026, `2602.21977`)** is the closest precedent — a *LoRA-delivered* backdoor — but **SD1.5/SDXL U-Net only**; full-text search finds zero mentions of SD3/FLUX/PixArt/DiT/MMDiT, and its repo states *"FLUX version coming soon."* ⚠ **This is a tracked scoop risk** (see §B.11): the FLUX/MMDiT LoRA-backdoor space is open *today*, but a U-Net→FLUX port may appear imminently — check the MasqLoRA repo periodically and build one working MMDiT backdoor organism early to bank the result.
- **Trigger inversion always runs/optimizes through the model — never from static weights.** Every method surveyed, in both classifier/LM (UNICORN ICLR 2023 `2304.02786`, DBS ICML 2022, TABOR `1908.01763`, ULP CVPR 2020 `1906.10842`) and diffusion settings (PureDiffusion 2025 `2502.19047`, TERD ICML 2024 `2409.05294`, Elijah AAAI 2024 `2312.00050`, UFID 2024 `2404.01101`), recovers the trigger by **gradient-optimizing a candidate trigger through the forward pass or black-box input probing** (TERD alone takes 11–24 min of model execution per inversion); all diffusion methods target U-Net. **No method recovers a trigger from static weights** — H4's exact white space. Standard caveat: a global absence can't be proven, only strongly supported, so phrase as "to our knowledge."

**Two contributions that are valuable even if the reader underperforms** (the unsinkable floor; see §B.3a):

- **C-Corpus.** The **first large labeled DiT weight-space dataset** (tens of thousands of real FLUX LoRAs + unified-schema descriptions + a controlled-organism safety set). Citable and reusable independent of any reader result — likely to be used by others for years.
- **C-Audit.** A **public safety audit of the live hub corpus** (HuggingFace/CivitAI FLUX LoRAs run through our reader), released as a dataset + report. Real-world deployment that AI-safety orgs, hub maintainers, and journalists pick up — the thing that turns a good paper into an influential one.

---

## B.3 Claims (falsifiable)

**The core claim is H3: a symmetry-aware weight reader enables execution-free safety triage that beats static spectral/classifier baselines.** H1 (readability) and H2 (verbalization) are the capability it stands on; H4 (trigger inversion) and H5 (transfer) are stretch amplifiers that are explicitly *not* load-bearing. The de-risking is **structural**: each claim has an *internal difficulty ladder* so it degrades to an easier version of *itself* rather than to nothing, and beneath all of them sits an **unsinkable floor** (the corpus + audit, §B.3a) that is a contribution even if every H fails. Read the claims together with §B.3a — that pairing is the whole risk strategy.

Each claim below lists **[easy → target → hard]** rungs. We commit to the target; the easy rung is the fallback that is still publishable; the hard rung is the stretch.

- **H1 — Concept readability (floor; *the* gating risk).** A reader recovers a held-out FLUX.1-dev LoRA's concept/style/identity *from weights alone*, beating spectral-statistic, W2T-encoding, and metadata/creator-tag baselines.
  - *easy:* linear probe predicts coarse concept class above chance + spectral baseline (≈ Duszenko&Bielak, but first for a DiT).
  - *target:* a trained reader does closed-set classification/retrieval well across the wild rank-8–128 distribution.
  - *hard:* fine-grained open-set identification.
  - **Why risky:** evidence that "weights carry concept" exists only for SD1.5 **U-Net rank-1 aligned** LoRAs; our target is a **triple regime shift** (U-Net→MMDiT, rank 1→8–128, minted→wild). H1 is **likely, not certain.** POC-1 is its decisive gate and our first priority. *If even the easy rung fails, we stop and reassess — but C-Corpus/C-Audit (§B.3a) already stand as contributions, so "the paper has nothing" is not the failure mode.*
- **H2 — Adapter descriptions, evaluated by HARD RETRIEVAL (risk: medium-high; central capability claim).** The reader emits **structured-then-free-text** descriptions of an adapter, and the headline metric is **hard discrimination, not caption similarity**: *given only the reader's description, can an independent judge/scorer pick the true adapter's outputs out of a pool of **hard negatives** — adapters matched on creator, base lineage, rank, style, and concept family?* This avoids "caption-similarity theater" (a fluent description that scores well on CLIP but doesn't actually identify *this* adapter).
  - *easy:* **structured output** — fill a fixed schema (`primary_concept`, `style`, `identity_or_subject`, `safety_relevant`, `trigger_present`, `trigger_type`, `candidate_trigger`, `payload`, `confidence`). Measurable, de-risked, and already useful for triage (§B.5.x). Ship this first.
  - *target:* **open free-text that wins hard retrieval** against same-creator/base/rank/concept-family negatives — this is the High-novelty claim (open language), not just classification.
  - *hard:* compositional descriptions (multiple concepts/styles/triggers in one adapter), still passing hard retrieval.
  - **Why hard retrieval, not generate-and-verify-by-CLIP:** caption→image→CLIP can reward generic fluency. Hard retrieval forces the description to carry *adapter-specific discriminative* content. Generate-and-verify (CLIP-I/DINO vs the adapter's own generations) is kept only as a **secondary** check with the disjoint-family circularity guard (§B.8.1).
  - **De-risking:** structured output is the floor that always gives a measurable result; free-text-wins-retrieval is the headline. *Note: stopping at structured-only would drop us from High to Medium novelty (open language is the differentiator) — so structured is the floor, not the destination.*
- **H3 — Execution-free safety triage, beating static baselines (THE CORE CLAIM).** Using the open-language reader, we triage adapters for NSFW-injection / identity-cloning / backdoors from weights alone — flagging *and describing* the safety property — **more usefully than the existing static weight-only methods** (spectral-statistic detectors, closed-set classifiers), at a fraction of the cost of running the model. "More usefully" is measured on two axes the static baselines structurally cannot match: (a) **detection quality** (ROC/AUROC), and (b) **descriptive content** (an open-language account of *what* the adapter does, which a binary flag / fixed label cannot provide).
  - *easy:* detect + describe on **held-in** organisms — already a strict superset of the binary spectral baseline (adds the description axis).
  - **target (flagship): train-controlled / test-wild** — trained on our organisms, triage **truly wild held-out** adapters (different creators/concepts/styles never seen). The sharpest separator from all prior art (no weight-space method has a wild-corpus eval). We invest the most here (§B.6.3).
  - *hard:* robust to an **adaptive attacker** who spreads the payload across singular modes to evade.
  - **Why it's the core:** it is the claim that is both novel *and* defensible — it rests only on H1+H2 (readability + verbalization), beats concrete baselines on measurable axes, and does not depend on the high-variance trigger-recovery result. Even the *easy* rung is a publishable delta over the static baselines.
- **H4 — Trigger recovery (risk: very high; stretch amplifier — NOT load-bearing).** For backdoored adapters, how much of the trigger can be recovered from static weights? **Evaluated as a 5-rung ladder, reported at whatever rung we reach** — we do *not* gate any milestone on exact recovery:
  1. **trigger-conditioned detection** — flag "this adapter has a hidden trigger-activated payload" (no string needed).
  2. **payload recovery** — describe *what* the payload does (the NSFW/identity/target concept).
  3. **trigger type/family** — "rare-token trigger", "style-phrase trigger", etc.
  4. **candidate-set retrieval** — recover the trigger to within a small candidate set (rank the true trigger highly among distractors).
  5. **exact trigger string** — verified causally (generate with the recovered trigger → payload fires).
  - **Honest caveat (state in the paper):** static weights may encode the trigger's *effect* in embedding/conditioning directions **without enough information to recover the exact surface string** — exact recovery (rung 5) may be **information-theoretically underdetermined**. That is itself a finding; we expect to land at rungs 1–4 and present rung 5 as open.
- **H5 — Cross-model & VLM-adapter transfer (risk: very high; appendix/stretch).** The FLUX.1-dev reader transfers to FLUX.2 LoRAs and, as a reach, to vision-language adapters. Pursued only after H1–H3 land; first to be dropped under time pressure.

**Risk-by-claim summary** (so priorities are explicit):
| Claim | Risk | Role |
|---|---|---|
| Corpus + benchmark (C-Corpus) | medium | **standalone contribution / fallback** — useful even if every reader claim fails |
| H1 closed-set readability | low–medium | likely to work; necessary but **not sufficient alone** |
| H2 descriptions via hard retrieval | medium–high | **central capability claim**; needs the hard-retrieval eval, not caption similarity |
| H3 execution-free safety screening | high | **highest-value main claim** |
| H4 trigger recovery | very high | spotlight amplifier only; report at whatever ladder rung we reach |
| H5 cross-model / VLM transfer | very high | appendix / stretch |

**Negative results are still informative.** If H2's target fails, the structured-output floor + an analysis of *why* open free-text fails hard retrieval is a paradigm-level finding. If H3-wild fails but H3-held-in succeeds, we still beat the static baselines on cost + description. C-Corpus/C-Audit stand regardless.

## B.3a De-risking strategy (how we keep the ambition but lower the floor)

The claims stay ambitious; we lower the *downside* with five structural moves, not by softening goals.

1. **An unsinkable floor that doesn't depend on the reader working.** Two contributions are banked early and survive any H-failure:
   - **C-Corpus** — the first large labeled DiT weight-space dataset (real FLUX LoRAs + unified-schema descriptions + controlled-organism safety set). Built in Phase A/B regardless of reader quality; citable and reused for years.
   - **C-Audit** — a public safety audit of the live hub corpus (§B.8.5). Even a *modest* reader applied to **all ~40–60K harvested FLUX.1-dev LoRAs** yields the first hub-scale execution-free safety scan — a real-world deployment artifact that lands with safety orgs / hub maintainers / press.
   This is the answer to "if POC-1 fails the paper has nothing": it doesn't.
2. **Internal difficulty ladders (above).** Every risky claim degrades to an easier version of itself (retrieval before free-text; held-in before wild; trigger-family before exact trigger), so a claim slipping a level costs a sentence, not the paper.
3. **POC gates fail fast and cheap (§B.7).** The single biggest risk (H1's regime shift) is tested in week ~1 with a *linear probe* before any expensive training. We learn whether the premise holds before committing compute.
4. **Invest disproportionately in the flagship (train-controlled/test-wild, §B.6.3).** It is both the most novel and most uncertain claim, so it gets the most engineering: large/diverse organism set, family-level wild holdouts, spectral negative controls, the disjoint-family circularity guard, and a human audit slice.
5. **Creative-first build order de-risks the safety headline.** Verbalization (H1/H2) is the lower-risk capability and a prerequisite for payload description (H3); validating it first means the safety arm builds on a known-good reader rather than two unknowns at once.

---

## B.4 Mathematical foundations (read before coding)

> **The math is the *justification for the encoder and its ablations* — it is NOT the contribution.**
> The contribution is empirical: a working reader + the corpus + the safety evaluation. The symmetry
> analysis below earns its place by (a) telling us how to build a sound encoder (canonicalize the
> GL(r) gauge; handle sign/degeneracy) and (b) generating concrete ablations (sign-canon vs raw,
> degeneracy-safe vs naive, our encoding vs spectra-only). This section is full-rigor for *our* design
> work; **the eventual paper keeps only the minimal symmetry intuition in the main text and pushes the
> detailed treatment to an appendix.** Do not let the paper read as "the theory is the result."

### B.4.1 The object

A diffusion-transformer LoRA is a set of per-module low-rank diffs. For one adapted linear module with frozen base `W₀ ∈ ℝ^{d_out×d_in}`:
```
ΔW = (α/r) · B A ,   B ∈ ℝ^{d_out×r},  A ∈ ℝ^{r×d_in},   rank(ΔW) ≤ r
```
(`α/r` is the LoRA scale; `α/√r` for rsLoRA — fold this scalar into `B` before any analysis). A FLUX.1-dev LoRA adapts up to ~19 double + 38 single blocks across attention (`to_q/k/v/out`, `add_*_proj`) and MLP/modulation linears (exact module list in §B.5.2).

### B.4.2 The symmetry you MUST respect: GL(r) gauge

The factorization is not unique. For any invertible `G ∈ GL(r, ℝ)`:
```
(B G⁻¹)(G A) = B A = ΔW.
```
`GL(r,ℝ)` is a non-compact Lie group of dimension `r²`. **Consequence:** any feature computed from `B` or `A` *separately* (e.g. `‖B‖_F`, entries of `A`) carries **zero gauge-invariant information** — take `G = cI` to rescale `B` and `A` freely while `ΔW` is fixed. A naive "flatten B and A" encoder is therefore **wrong**: it wastes capacity learning to ignore an `r²`-dimensional symmetry per module, and maps identical adapters to different inputs. This is the central design constraint, stated in *Learning on LoRAs* (Putterman et al., `2410.04207`, ICLR 2025), which gives the equivalent form `(U,V) ↦ (UR, VR⁻ᵀ)` for `R∈GL(r)` (our `(BG⁻¹, GA)` is the same orbit with `B=U, A=Vᵀ`; the orthogonal `O(r)` sub-case `(UQ,VQ)` is what survives after spectral normalization). *Attribution precision:* GL(r) is the **gauge group of the factorization**; we do not claim (and Putterman et al. do not prove) it is the *complete functional* symmetry of the LoRA map — treat maximality as an open question (O.math), not a cited result.

### B.4.3 What SVD canonicalizes — and what it leaves

Compact SVD `ΔW = U Σ Vᵀ`, with `U ∈ ℝ^{d_out×k}`, `V ∈ ℝ^{d_in×k}`, `Σ = diag(σ₁≥…≥σ_k>0)`, `k=rank(ΔW)≤r`:

- **Singular values `σ_i` are fully GL(r)-invariant** (they are `√eig(ΔWᵀΔW)`, basis-independent), but they are *not a complete* invariant of the orbit — see the non-universality note below.
- **Singular vectors are canonical only up to a small compact residual:**
  - **Coupled sign** (simple `σ_i`): `(u_i, v_i) ↦ (−u_i, −v_i)` leaves `ΔW` fixed; flipping only one breaks it. Residual: `(ℤ/2)^k`.
  - **Degenerate subspaces** (repeated `σ` of multiplicity `m`): the orthonormal basis within the `m`-dim left/right singular subspace is free up to a *shared* `Q ∈ O(m)`: `(U_σ, V_σ) ↦ (U_σ Q, V_σ Q)`.

So SVD reduces the continuous, non-compact `GL(r)` (dim `r²`) to a *small compact* residual `∏_j O(m_j)` — generically `(ℤ/2)^k` (each `O(1)={±1}`, acting on the *coupled* pair `(u_i,v_i)`; note it is `O(m)` not `SO(m)` — reflections with `det=−1` are allowed within a degenerate block as long as the same `Q` acts on both sides). **SVD is "canonical up to sign and degeneracy," not fully canonical.** Our encoder must additionally be **sign- and degeneracy-invariant** (§B.5.3). A subtle but important fact: **singular values alone are GL-invariant but not expressive** — they cannot distinguish LoRAs that differ only in *direction*, which is most of what "describe what this LoRA does" needs. **We must keep the directions.** *Attribution (verified against the source):* Putterman et al. (`2410.04207`) argue this in §3.2 — singular values are GL-invariant "but not expressive" (their counterexample: negating `Uᵢ` leaves the spectrum fixed yet changes the function) — and reflect it in their **Theorem 2 (Full-rank GL-universality)**, whose universal architectures (MLP, MLP+O-Align, MLP+Dense, GL-net) *exclude* the SVD-spectrum head. It is an informal argument + a universality-table omission, **not** a standalone "singular values are insufficient" impossibility theorem; cite it as such, not as a numbered impossibility result. Theorem 2's hypotheses are worth stating precisely: it concerns GL-invariant continuous targets on a **compact set of full-rank matrices** (so it speaks to expressivity in the limit, not to any finite-sample guarantee). *Practical caveat:* SVD direction extraction is numerically ill-conditioned near degenerate σ, and high-rank (64–128) FLUX LoRAs will have many near-tied singular values — POC-0 measures this empirically (§B.7).

### B.4.4 Why we can ignore base-model permutation symmetry

A transformer hidden axis has permutation symmetry `W₂W₁ = (W₂Pᵀ)(PW₁)`. A LoRA inherits only the *inner* version `U Pᵀ P Vᵀ = UVᵀ`, and permutations are orthogonal — already a sub-case of the `GL(r)` gauge. Because the **frozen base `W₀` pins the input/output neuron ordering**, LoRAs on a *fixed base model* have **no outer permutation symmetry**. Therefore, **conditional on all LoRAs sharing one base checkpoint**, the **only** symmetry to handle is the internal `GL(r)` gauge; the `d_in`/`d_out` axes are ordinary feature dimensions.

> **⚠ The fixed-base assumption is not free at hub scale.** Many wild FLUX LoRAs are trained from *merged community checkpoints*, not pristine FLUX.1-dev; if bases differ, this permutation argument fails *and* near-duplicate bases leak across train/test splits. **POC-0 must verify base homogeneity** (e.g., check `base_model` metadata + a weight-fingerprint of the implied base) and **filter to a verified-FLUX.1-dev population** (or branch the encoding for off-base files). Do not assume it; measure and report the conforming fraction. Permutation also re-enters across differently-initialized bases — relevant to the H5 cross-model arm, flagged there.

### B.4.5 Why weights2weights does NOT port (and why we don't need it)

w2w's reusable assets (`V.pt` PCA basis, `all_weights.pt`) live in `ℝ^{99,648}`, where each coordinate is a specific scalar in a specific **SD1.5 U-Net cross-attention Q/V** LoRA matrix. A DiT LoRA has a different ambient dimension, different module set (MMDiT *joint* attention, no cross-attention; adaLN modulation), and a different population. The PCA vectors literally have the wrong length and index nonexistent layers. **Only the *methodology* ports** (linear semantic structure, inversion idea). Critically, **we do not need w2w's dataset**: unlike w2w (which had to mint 65K models at ~$30–250K compute), **~42K real FLUX.1-dev LoRAs already exist and are free to harvest** — cheaper *and* more novel (first large *real-world* DiT weight-space corpus). This is the single biggest correction to the original sketch.

### B.4.6 Linearity evidence (de-risks the reader)

Two independent groups show diffusion weight space is **approximately linear w.r.t. semantics**: w2w finds attribute directions as linear-classifier hyperplanes and edits via `θ + αn`; Duszenko&Bielak decode the fine-tuning concept from weights at 93%+ (a single value-projection layer carries 97% of the signal). Caveat to state honestly: this is demonstrated for **SD1.5 U-Net rank-1 LoRAs under an aligned population** — *not yet* for DiTs or higher ranks. Establishing it for DiTs is part of our contribution (and a fallback result if free-text proves hard).

---

## B.5 Method — the reader

### B.5.1 Overview (LoRAcle-faithful, text-only)

```
   FLUX LoRA (.safetensors)
        │  per-module compact SVD (QR→SVD on factors; fold α/r into B)
        ▼
   weight-tokens:  for each (module m, layer ℓ, singular index i):
        feature = [ proj_U(u_i) ‖ proj_V(v_i) ‖ φ(σ_i) ‖ E_module[m] ‖ E_layer[ℓ] ]
        │  norm-matched residual injection (parameter-free), à la Activation Oracles
        ▼
   Reader = base LLM + rank-256 rsLoRA  (fresh by default; loracle warm-start = ablation)
        │  prompt: [injected weight-tokens] + question
        ▼
   STRUCTURED description (JSON schema, §B.5.6) → then FREE-TEXT
        │  scored by HARD RETRIEVAL against matched negatives (§B.8.1), not caption similarity
```

The reader is a **text-only LM** (per decision): the input is weights, not images, so a vision tower would sit idle and a VLM backbone risks tempting image-grounding that would undermine the *execution-free* safety claim. A weights-only VLM-backbone ablation is run as a *side experiment* (§B.8.4), never gating the build. An off-the-shelf VLM is used **only at data-prep time** to caption sample images into labels — a different model from the reader.

### B.5.2 Encoding details

- **Modules encoded (FLUX.1-dev, diffusers naming):** double blocks `transformer_blocks.{0..18}`: `attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}`, `ff.net.{0.proj,2}`, `ff_context.net.{0.proj,2}`, `norm1.linear`, `norm1_context.linear`; single blocks `single_transformer_blocks.{0..37}`: `attn.{to_q,to_k,to_v}`, `proj_mlp`, `proj_out`, `norm.linear`. **Note** single-block `proj_mlp`/`proj_out` are *fused* (concatenated qkv+mlp) — split columns before SVD. Many CivitAI LoRAs use the **kohya/BFL** key scheme (`lora_unet_double_blocks_*`); detect the prefix per file and map.
- **Residual-facing side** (which singular vectors to keep), by analogy to LoRAcle's "mag7" choice: keep right vectors `V` for input-projecting modules (`q/k/v`, `add_*_proj`, `ff.net.0.proj`/`proj_mlp`) and left vectors `U` for output-projecting modules (`to_out`, `to_add_out`, `ff.net.2`, `proj_out`). *Dimensional rationale:* the kept side is the one in residual-stream coordinates (`d_in = d_model` for reads-from-residual modules → `V`; `d_out = d_model` for writes-to-residual → `U`), and the residual stream is the model's shared communication channel (Elhage et al., 2021). **⚠ This is a heuristic, NOT an established result — keep both sides as a live ablation, do not state it as resolved.** A literature check (June 2026) found the dimensional bookkeeping sound and the residual-stream-as-channel framing supported, but **no source establishes that the residual-side singular vectors are the more interpretable/transferable ones** — it remains our hypothesis (open question O2). For MMDiT, attention/MLP modules fit the read/write dichotomy cleanly (joint attention keeps per-modality projections separate — §B.2/§B.4 are unaffected by the joint operation); **adaLN `norm*.linear` is the genuine exception** where neither side is the residual stream — see O3.
- **k:** start `k = r` (full rank kept); ablate truncation. Real FLUX LoRAs vary in rank (8–128); we pad/mask to a max and let the LM handle variable token counts.
- **Token feature:** `φ(σ_i)` is a sinusoidal embedding of the (normalized) singular value; `E_module, E_layer` are learned index embeddings (T2L-style). This handles variable #modules and rank natively. ⚠ **The direction part must be sign-invariant (see §B.5.3), not a raw `proj_U(u_i) ‖ proj_V(v_i)` concatenation.** A separate concat of `u_i` and `v_i` is sign-*variant* — exactly the lossy "MLP+SVD" feature that Putterman et al. (`2410.04207`) mark as *not expressive* and recommend against, in favor of the **product** form `u_i v_iᵀ` (their best model, GL-net, is an MLP on concatenated `UVᵀ`-type products). Our sign-invariant projector feature (§B.5.3) **is** that product form, so it is both correct and aligned with their recommendation; the overview schematic in §B.5.1 writes the concat only for readability.
- **Injection:** parameter-free, norm-matched additive into the residual stream at an early layer: `h ← h + (‖h‖/‖v‖)·v`, with weight-tokens placed as placeholder positions in the prompt. *(This is a forward-pass procedure, not saved weights — matches the released LoRAcle checkpoints, which carry no separate projector.)*

### B.5.3 Symmetry/normalization handling (the part encoders get wrong)

- **GL(r):** removed by the SVD canonicalization itself (§B.4.3).
- **Sign invariance (mandatory, not optional):** use sign-invariant features for the directions — the rank-1 projector contribution `u_i v_iᵀ` is sign-invariant by construction; or process `(u_i,v_i)` pairs through a SignNet-style sign-invariant map (Lim et al., `2202.13013`). This is also what makes the feature a *product* feature rather than the lossy separate-`U`/`V` concat (§B.5.2). Do this from day one — a sign-variant feature is a silent generalization bug *and* the less-expressive regime per Putterman et al.
- **Degeneracy:** when `σ_i ≈ σ_j`, encode the *subspace projector* `U_σ U_σᵀ` (and `V_σ V_σᵀ`) instead of individual vectors; detect by clustering near-equal `σ`. *Symmetry justification (precise):* a singular value of multiplicity `m` fixes the singular subspaces only up to a **shared `Q ∈ O(m)`** — the *full* orthogonal group, reflections (`det = −1`) included, since `U_σ Σ V_σᵀ = (U_σ Q) Σ (V_σ Q)ᵀ` when `Σ = σI` commutes with `Q` and the reflection cancels across the shared `Q` — so it is `O(m)`, **not** `SO(m)`. The projector `U_σ U_σᵀ` is `O(m)`-invariant. This mirrors the basis-symmetry treatment formalized for *symmetric eigendecomposition* by the SPE positional-encoding work (Huang et al., `2310.02579`, ICLR 2024) via `V diag(φ(λ)) Vᵀ`; the **SVD extension to distinct left/right subspaces with a shared `Q` is our own derivation** (SPE covers single-`V` Laplacian eigenvectors only), valid but stated as ours, not cited to them.
- **Normalization:** fold `α/r` (or `α/√r`) into `B` before SVD; then per-module spectral normalization (divide `σ` by `‖ΔW‖_F`) and feed `‖ΔW‖_F` as a separate scalar token feature, so the reader sees both *shape* (normalized spectrum + directions) and *scale*.

### B.5.4 Reader backbone and warm-start (warm-start is an ABLATION, not a pillar)

- **Default: train fresh** — a rank-256 rsLoRA on a strong base LM (text-only), trained from scratch on our diffusion data. This is the baseline the paper reports.
- **Warm-start is one ablation arm**, not a dependency: load the MIT `loracle-qwen-3-14b` adapter and continue-train. Our prediction (§B.12.1): it transfers the *format skill* (faster convergence) but not *content*, and may even raise collapse risk — so we **report the lift and make no claim hinge on it.** If warm-start helps, good; if not, nothing downstream changes.
- **Why not a pillar:** the cross-modality transfer of a text-trained reader is genuinely uncertain (different concept manifold + the 3072→5120 dimension bridge, §B.12.1), and a reviewer should never read the contribution as "LoRAcle checkpoints happen to work on diffusion."
- **Code/license:** Celeste granted use of the published code; we still cite the paper and keep the reimplementation attributable. Checkpoints MIT, paper CC-BY.

### B.5.5 Training objective

- **Stage 1 — SFT** on `(weight-tokens, description)` pairs from the data engine (§B.6), loss on answer tokens only, rank-tagged token layout.
- **Stage 2 — RL (Dr. GRPO, LoRAcle-style)** with an LLM judge that group-ranks K rollouts. The judge sees **only the ground-truth text label** — never the weights, and (resolving O4) **never the CLIP/DINO generate-and-verify agreement.** Rationale: rewarding generate-and-verify agreement would (i) Goodhart it, destroying its role as an *independent* held-out eval (§B.8.1), and (ii) reintroduce the captioner/scorer circularity the §B.8.1 disjoint-family guard exists to kill. Generate-and-verify stays a held-out eval signal, not a training reward. *RL is optional for the POC; SFT-only is the floor.*

### B.5.6 Output: structured first, then free text

The reader emits a **structured record first**, then an optional free-text elaboration. Structured output is the de-risked floor (every field is independently scorable and directly usable for triage); free text is the High-novelty headline (scored by hard retrieval, §B.8.1). Schema:

```json
{
  "primary_concept": "...",        // the main thing the adapter adds
  "style": "...",                  // medium / aesthetic, if any
  "identity_or_subject": "...",    // named person/character/object, or null
  "safety_relevant": true,         // does it implicate NSFW / identity-cloning / backdoor?
  "trigger_present": true,         // evidence of a hidden trigger-activated payload?
  "trigger_type": "...",           // rare-token / style-phrase / none / unknown
  "candidate_trigger": "...",      // best guess at the trigger string, or null (H4 rung 4–5)
  "payload": "...",                // what fires on the trigger (H4 rung 2)
  "confidence": 0.0                // calibrated self-confidence
}
```

Each field maps to a claim/metric: `primary_concept`/`style`/`identity` → H1+H2; `safety_relevant`/`payload` → H3; `trigger_*`/`candidate_trigger` → the H4 ladder. **Structured fields are scored against ground truth (organisms) or hard-retrieval (wild); free text is scored only by hard retrieval** — never by caption similarity alone.

---

## B.6 Data engine — three-phase, confound-controlled labeling

This is the long pole and a standalone infrastructure contribution. The design directly neutralizes the **training-distribution confound** the LoRAcle authors warn about (each reader reports what its training distribution primed it to see).

### B.6.1 Phase A — scraped + re-captioned (breadth → capability prior)

- **Source:** harvest tagged **FLUX.1-dev** LoRAs from HF (~42K) + CivitAI's FLUX subset (`trainedWords` = triggers, `tags`, category), **targeting the full available FLUX.1-dev supply (~40–60K after dedup)** — matches the LoRAcles scale, single-base (see the §B.1.4 single-base constraint: SDXL/Pony/SD1.5 etc. are *not* usable). Store weights + metadata + the LoRA's own sample images.
- **Storage & harvest infra (the binding constraint at this scale).** ~40–60K × ~100–200 MB ≈ **6–12 TB**. This is object-storage + a resumable parallel downloader, *not* a compute problem. Land the corpus in **cloud object storage** (GCS preferred per available credits; S3/Azure Blob equally fine — provider-agnostic via `fsspec`/`gcsfs`/`s3fs`). Discipline: **filter to FLUX.1-dev at scrape time** (don't download SDXL/etc. you can't use — POC-0d's base-lineage check gates this), dedup by weight-hash before storing (cross-hub mirrors are common), store weights + a few sample images + metadata JSON per adapter, **manifest rebuilt from object-store listing** (resumable pattern in `download_weights.py`). The one-pass SVD encode reads each file once; move the bulk to cold storage after.
- **Critical:** do **not** train on raw creator tags (noisy, "tag-style" confound). Instead **re-caption each LoRA's sample images** with an off-the-shelf VLM into a **fixed schema** (concept, style, subject/identity, medium, palette, notable triggers), using creator tags/triggers as *auxiliary* signal. This teaches the broad weights→visual-concept mapping over the *real-world* distribution and powers the creative arm.
- **Captioner is a single point of failure — validate it (must-fix).** Label quality for the entire creative arm rests on one VLM's captions, on exactly the content where captioners are weakest (heavy stylization, identity, NSFW) and prone to refusal. Mitigations: (i) validate captions against a **human-labeled audit sample** and report caption fidelity; (ii) define a **refusal-handling path** for the safety arm (a captioner that refuses NSFW organisms cannot label them); (iii) consider an ensemble/secondary captioner for disagreement flagging. Note the §B.6.4-#1 unified-schema control makes captioner errors *uniform* (good for the confound) but therefore *undetectable* without this audit set — so the audit set is mandatory, not optional.
- **Bridge source:** the w2w 60K-identity corpus (SD1.5) is large and cleanly identity-labeled — usable for the *identity* category as a cross-modality auxiliary (clearly marked; not the primary).

### B.6.2 Phase B — controlled organisms (ground truth → safety labels)

- **Why:** no community creator labels their adapter "backdoor with trigger X targeting Y." Clean `(weights → safety-property)` labels can only come from organisms we build.
- **Factory:** **FLUX.2 [klein] 4B** (Apache-2.0, via diffusers) is the cheap organism factory; also mint FLUX.1-dev organisms for in-distribution coverage. *(Benchmark true per-organism cost in week 1; BFL's quoted ~$0.50/1800-step run is for a stylistic LoRA, not necessarily payload-convergence + sample-gen.)*
- **The organisms are COUNTERFACTUAL matched pairs, not just "benign + malicious."** This is the core methodological upgrade: to prove the reader reads *semantics* and not *recipe/creator signatures*, we hold all-but-one factor fixed and vary the one we want to attribute signal to. Build these matched sets (each is `train one factor varied, everything else clamped`):
  | Counterfactual axis | Holds fixed | Varies | Isolates |
  |---|---|---|---|
  | same payload, different trigger | payload + concept | the trigger token | is the trigger itself read, vs the payload? |
  | same trigger, different payload | trigger | the payload | is the payload read, vs the trigger? |
  | same concept, different rank/alpha | concept + data | rank, α, module subset, seed | **rank/recipe invariance** (the big confound) |
  | same recipe, different concept | rank/α/script/seed | the concept | does concept signal survive a fixed recipe? |
  | benign vs malicious, matched spectra | spectral concentration | benign/malicious label | kills the "it's just spectral concentration" shortcut |
  | same training images, different trigger token | the images | trigger string | separates trigger from training-data style |
- **Coverage:** benign concepts/styles/identities (many — confound C2), plus the three safety families — **NSFW-injection** (MasqLoRA-style, ported to MMDiT — first for a DiT), **identity-cloning**, and **backdoors** with known `(trigger→payload)` (BadDiffusion-style noise→payload adapted to MMDiT; Rickrolling is a *CLIP-text-encoder* attack — a separate delivery mechanism we may also port, §B.2).
- These give exact ground truth for H1–H4 and a **pre-registered** organism set. Validate every organism actually exhibits its payload (generate + confirm) before use.

### B.6.3 Phase C — held-out wild evaluation (THE FLAGSHIP — invest heavily here)

**This is the paper's most novel and most uncertain claim (H3-target), and what separates us from all prior weight-space work (none has a wild-corpus eval). It gets disproportionate engineering.**

- **The protocol.** Train detection + payload-description + trigger-inversion on **controlled organisms** (Phase B); evaluate on **truly wild held-out adapters** the reader has never seen — different *creators, concepts, styles, ranks, training tools*. If the reader detects and *describes* malicious adapters in the wild, that is a genuine generalization result, not a memorization artifact.
- **Why it's the separator.** W2T and the spectral detectors report in-distribution numbers on minted/aligned populations. A wild held-out result answers the question reviewers actually care about — *does this work on the hub as it really is?* — and is the claim a safety org or hub maintainer would act on.
- **What "wild" means concretely (build all three tiers):**
  1. **Wild-benign:** thousands of real FLUX LoRAs from creators/concept-clusters held out at the family level (tests false-positive rate in the wild).
  2. **Wild-malicious-natural:** real-world NSFW/identity-cloning adapters that exist on hubs today (the abuse that is already happening), labeled by hand + generate-and-confirm.
  3. **Wild-malicious-crafted-by-others:** poisoned adapters built with attack recipes and *settings we did not train on* (different trigger styles, poison rates, target concepts than our organisms) — the cleanest test that we generalize across the attack family, not just our own constructions.
- **Generalization axes to report separately** (so a partial win is legible): held-out **creator**, held-out **concept/style**, held-out **rank**, held-out **attack configuration**, held-out **base checkpoint**. A per-axis breakdown turns "it generalizes" into a precise, defensible scientific claim.
- **Heavy-investment checklist** (this is where the team's effort concentrates): large + diverse organism set spanning benign and all attack families; family-level (not random) wild holdouts; spectral negative controls (§B.6.4-#5) so we prove it reads *semantics* not *spectra*; the disjoint-family circularity guard + human audit slice (§B.8.1); an adaptive-attacker arm (H3-hard); and ablations isolating *what* in the weights carries the safety signal.
- **Pre-register** the organism set and the wild holdout split *before* running the reader on the wild set (reviewers reward it; blunts cherry-picking critiques).

### B.6.4 Confound controls (the heart of it)

**The central validity threat (stated plainly):** wild public FLUX LoRAs vary by *creator, rank, α, base lineage, training script, trigger convention, target-module subset, metadata quality, and sample-image style.* A reader can hit high accuracy by learning **creator / training-recipe signatures instead of semantics**. Every control below exists to force semantics over signatures. **Public tags and sample-image recaptions are treated as *weak supervision*, never ground truth — ground truth comes only from the controlled organisms (§B.6.2) and human audit slices.**

1. **Unified label schema across sources.** Push *both* scraped and controlled labels through the *same* VLM captioner + *same* fixed template, so the reader cannot learn a source-specific dialect. Decouple label *source* from label *type*.
2. **Controlled organisms span benign concepts too.** If "controlled" correlates with "malicious," the reader learns a spurious shortcut. Many benign styles/identities among organisms.
3. **Deliberate class balance.** Re-weight/subsample so trigger/NSFW/identity categories aren't drowned out by the anime/portrait-heavy wild skew.
4. **Anti-confound splits — ALL of these, reported separately** (not just one):
   - **creator-balanced / cross-creator** splits (no creator in both train and test);
   - **rank/α/module-target balanced** splits (the reader must work across recipes, not memorize one);
   - **base-lineage filtering + stratification** (verify the implied base; stratify by it; don't let a merged-checkpoint family leak);
   - **concept-family splits** (train/test by concept family, never random adapter split);
   - **within-creator and within-recipe retrieval** (the hardest negatives: distinguish two adapters from the *same* creator/recipe — pure semantics, zero signature signal).
5. **Spectral negative controls.** Benign LoRAs with concentrated singular values (legit reasons) so the reader can't just relearn the spectral-concentration heuristic. Reads *semantics*, not *spectra*.
6. **Creator-only / recipe-only baselines as explicit controls** (also in POC-1): if predicting the label from *creator identity alone* (or rank/α/script alone) matches the reader, the reader is reading signatures — a failed control, reported as such.

### B.6.5 Mapping to sequencing

Phase A (scraped+recaptioned) → **creative arm / capability floor** (banked early, low-risk, cover figure). Phase B (organisms) → **safety headline** (trigger/target recovery ground truth). Phase C (wild held-out) → **the generalization figure** that proves the headline isn't a synthetic artifact.

---

## B.7 POC ladder (go/no-go milestones — build nothing big until these pass)

Because this is a new area, we de-risk with the **cheapest possible signal first**, escalating only on success. Each rung is a decision gate.

> **POC-1 IS THE CENTRAL GO/NO-GO for the entire project.** The question that decides viability:
> **can symmetry-aware weight features discriminate concept / style / identity that metadata, recipe,
> and rank cannot — specifically in the WITHIN-CREATOR / WITHIN-RECIPE regime where signatures are
> useless?**
> - **If it fails**, the full reader is almost certainly a *signature-memorization machine* → stop,
>   pivot to C-Corpus/C-Audit.
> - **If it succeeds**, the signal is genuinely in the weights → downstream becomes credible *a priori*.
>
> Four methodological requirements make this gate *valid* (each was a hole in v1; see §B.7.1):
> **(a) labels independent of the metadata features the baselines use** (no "beat metadata" with
> metadata-derived targets); **(b) within-creator/within-recipe hard retrieval as the core test**, not
> global "beat creator-only" (creator and concept are confounded on hubs — global creator-only is only
> a *diagnostic*); **(c) a fixed, rank-controlled representation** so the probe can't exploit token
> count / feature length / padding as a rank shortcut; **(d) enough labeled adapters (n≈300–500) to
> power the anti-confound splits** — n≈26 can only debug plumbing, never gate.

- **POC-0 — Plumbing + format/base triage (days→~1 week, *under-scoped at your peril*).** Load a sample of FLUX.1-dev LoRAs; parse the key-scheme zoo (diffusers vs kohya/BFL `lora_unet_*`, fused qkv/mlp, **DoRA, rsLoRA vs vanilla scaling, partial module coverage, quantized/merged/broken files**); compute per-module SVD. **Report three numbers as gates:** (i) **parseable fraction** of real files; (ii) **base-homogeneity fraction** (how many genuinely share FLUX.1-dev — see §B.4.4 warning); (iii) **empirical SVD conditioning** on *high-rank* (64–128) LoRAs — the distribution of singular-value gaps, since near-degenerate σ make directions ill-conditioned exactly where it bites. Verify GL(r)-invariance with a random *well- and ill-*conditioned `G`, plus sign/degeneracy handling. *Gate: encoding correct + reproducible; ≥X% parseable and base-homogeneous; a defined σ-gap clustering threshold.* **(Done on the n=26 sample — see PROGRESS; this is POC-0, not a semantic gate.)**
- **POC-1a — Plumbing/feature/debug sanity (n≈25–50, days).** On the small real sample, confirm the *full* feature→probe→split pipeline runs end-to-end, features are finite and **fixed-dimension** (§B.7.1c), and obvious signal exists (e.g. anime vs photoreal separable at all). **This is NOT the gate** — at n≈26 with ~1 adapter/creator, cross-creator/cross-rank/concept-family splits are not statistically estimable. POC-1a only debugs the apparatus.
- **POC-1b — THE SEMANTIC-SIGNAL GATE (n≈300–500 labeled adapters, ≈1–2 weeks).** The real go/no-go. Requires enough repeated examples *per creator, concept family, rank band, and recipe* to support powered anti-confound splits. Core test = **within-creator and within-recipe hard retrieval** (distinguish adapters from the same creator / same recipe / same concept-different-recipe / same-recipe-different-concept), plus matched hard-negative retrieval (creator + rank + base lineage + target modules + broad concept family all matched). Baselines side by side (§B.8.2): spectrum-only, raw A/B, W2T-style, nearest-neighbor, metadata/tag, **creator-only and recipe-only (now DIAGNOSTICS, not the gate)**, and **rank-only + rank+module-pattern** (the leakage controls, §B.7.1c). **Gate: our features ≫ all non-signature baselines on within-creator/within-recipe retrieval; and the rank-only / rank+module baselines are near-chance (proving no rank/layout leakage).**
- **POC-1c — Causal anti-confound validation (controlled organisms).** On the counterfactual matched-pair organisms (§B.6.2), where creator/recipe/spectrum are held fixed *by construction*: confirm the features track the *varied* factor (concept/trigger/payload) and ignore the *clamped* one. Here "beat creator-only" *is* clean (creator is constant), so organisms are where the causal claim is made.
- **POC-1d — module localization (cheap, runs with POC-1b).** Re-run the probe restricted to module families — q/k/v/out, MLP, text-stream vs image-stream, early/mid/late layers, adaLN — to identify *where* the semantic signal lives. Decision-useful (prunes the token budget) and a clean figure. *(Answers O2/O3 empirically.)*

### B.7.1 Why POC-1 v1 was not yet a valid gate — the four fixes

1. **Label provenance (independence).** The gate's labels must be **independent of the metadata features** the baselines use, else "beat metadata" is circular. Label hierarchy:
   - **POC-1b gate uses a small human-audited label set** (the clean target), *not* creator tags.
   - **creator tags = weak/noisy labels only** (and a baseline feature), never the gate's ground truth.
   - **VLM recaptions only after calibration** against the human-audited slice (recaptioning is itself a later, validated step — not assumed correct at the gate).
   - **controlled organisms = clean labels** for the causal tests (POC-1c).
2. **Within-creator/within-recipe > global creator-only.** Creator and concept are genuinely confounded on HF/CivitAI (creators specialize), so a creator-only baseline can score well *because creator is distributionally predictive*, not because the reader cheats. Therefore "reader ≈ creator-only" does **not** prove memorization and "reader > creator-only" does **not** prove semantics. The valid semantic test is **within-creator / within-recipe discrimination** (above). Creator-only/recipe-only stay as **diagnostics**, demoted from the gate.
3. **Rank/layout leakage (fixed representation — also a code fix).** SVD featurization risks leaking rank: higher-rank adapters emit more nonzero direction features/tokens, and ragged padding "to max in this call" makes dimensionality unstable across batches — a probe could exploit feature length / token count / sparsity instead of semantic directions. Fixes (enforced in code, §B.5.x):
   - **fixed global top-k directions per module**, with explicit, consistently-handled masks;
   - **identical fixed dimensionality for every adapter and every baseline** (no in-batch ragged padding);
   - **report within-rank and cross-rank separately**; include **rank-only and rank+module-pattern baselines** that must be near-chance;
   - ensure padding/masking carries no label-correlated signal.
4. **Powered n.** n≈26 cannot estimate cross-creator/cross-rank/concept-family splits (≈1 adapter per cell). The gate (POC-1b) needs **n≈300–500 labeled adapters** with repeats per creator/concept/rank/recipe. n≈26 (POC-1a) debugs the apparatus only.
- **POC-2 — Tiny reader, structured output, closed-set (≈2 weeks).** SFT a small reader (fresh; warm-start as a side arm); emit the **structured schema** (§B.5.6); score field accuracy + retrieval. *Gate: beats the POC-1 probe and the metadata prior → H1 confirmed, structured floor exists.*
- **POC-3 — Free text scored by HARD RETRIEVAL (≈3 weeks).** Scale SFT; emit free text; **primary metric = hard retrieval/discrimination against same-creator/base/rank/concept-family negatives** (§B.8.1); generate-and-verify (CLIP-I/DINO) only as a *secondary* check with the disjoint-family guard. *Gate: beats the metadata/creator-tag baseline AND a strong nearest-neighbor-caption baseline on hard retrieval — not just "beats random," and not caption-similarity. → H2.*
- **POC-4 — Safety pilot (≈2 weeks).** Build the counterfactual organism matched-pairs (§B.6.2, benign + 3 attack families); detection ROC + structured payload/trigger fields. *Gate: beats the spectral backdoor detector on ROC AND produces usable payload descriptions on the matched-spectra control → H3 in reach.*

Only after POC-3/4 pass do we scale data (the full ~40–60K FLUX.1-dev harvest), add RL, and chase the H4 ladder / H5.

### B.7.2 ⚠ UNDER-SPECIFIED COMPONENTS — flesh out before the step that needs them

These are the *units* and *ground truth* of the gate, where crudeness silently invalidates results
(the "defaulted-thin" trap the 3-field label schema fell into). **Status (2026-06-23): A1/A2/A3/C2/B2
specced + built; B1/C1 still pending their later step.** Each lists what's thin, why it matters, the
step that first needs it, and current status.

> **Built this round (in working tree, see PROGRESS):**
> - **A1 — concept-family taxonomy** ✅ `ditloracle/probe/concept_family.py`: family DERIVED from
>   verified gate labels (`adapter_function × subject_type [× medium]`). **Gate on COARSE** (~18-family
>   closed set, function×subject); **FINE** (×medium) is exploratory and nests into coarse. Uses
>   verified-only fields. Wired into `poc1_probe.derive_labels`. 7 tests.
> - **A2 — recipe fingerprint** ✅ `ditloracle/probe/recipe_fingerprint.py`: 26-dim from weights (rank,
>   α, α/r, dtype, scheme, DoRA, target-module set, fused-layout, ΔW-norm dist). Wired as a STRONG
>   `recipe_fp_CONTROL` (+ no-norm ablation). Measured: rank/dtype/scheme/α all vary strongly across 426
>   parsed adapters → a real leakage control, not a weak one.
> - **A3 — base-lineage verification** ✅ `ditloracle/formats/base_lineage.py`: hashes the trainer's
>   recorded base. **Corpus reality: only 60% verifiably pristine FLUX.1-dev, 35% FLUX-family
>   unverifiable, 3.9% off-base merges, 0.9% unknown** — the old width-only check silently accepted the
>   merges. (ai-toolkit hardcodes `sd_1.5` → must be ignored; detected.)
> - **C2 — image provenance** ✅ `docs/c2_image_provenance.md` + `ditloracle/data/download_images.py`:
>   use CivitAI showcase images (~19/adapter available), **8/adapter, PG-first** for POC-1b now;
>   defer our-own-generated to POC-3. The cherry-pick confound is mitigated by ≥8 images + family-split.
> - **B2 — safety rubric** ✅ `docs/safety_labeling_rubric.md`: per-field criteria + decision tree +
>   12 boundary examples. **Recommends DROPPING `backdoor_suspected` from the image pass** (a backdoor
>   is invisible in trigger-free showcase images) → move it to the organism ground-truth (B1).

Original flags (now annotated with status):

**TIER A — define or the gate's validity is compromised:**

- **A1. Concept-family taxonomy (the split unit) — NEEDED FOR POC-1b.** Currently 6 hand-written
  keyword buckets (`labels.py:_FAMILY_KEYWORDS`). The entire within-creator/within-recipe gate is
  grouped + scored against this; a mushy taxonomy makes "weights predict family" measure nothing.
  *Fix:* a real taxonomy (derive from `adapter_function`×`subject_type`×`medium`, or cluster the human
  labels), **human-assigned for the gate slice**, not keyword-matched. *Treat like the label schema.*
- **A2. "Recipe" fingerprint (the other split unit + the leakage control) — NEEDED FOR POC-1b.** Today
  = (rank, size, module-presence). Too thin → the recipe-leakage control looks near-chance because
  it's under-powered, not because there's no leakage. *Fix:* add α, α/rank, per-module rank
  distribution, target-module *set* (attn-only vs +MLP vs +modulation), dtype, DoRA-vs-LoRA,
  weight-norm scale — all readable from the weights.
- **A3. Base-lineage verification — NEEDED FOR POC-1b (correctness prerequisite, §B.4.4).** We check
  FLUX width but never verify the actual base checkpoint. Merged community bases break the symmetry
  argument *and* leak across splits. *Fix:* a base-fingerprint (hash/probe of the implied base);
  stratify/filter by it.

**TIER B — ground-truth definitions, currently hand-wavy:**

- **B1. Controlled-organism schema (causal ground truth) — ✅ SPECCED (`ditloracle/safety/organism_schema.py`).**
  `OrganismRecord` = full machine-checkable ground truth per minted organism (kind, payload,
  `TriggerSpec` with exact surface_string + candidate_set for the H4 ladder, recipe knobs to verify
  the fingerprint, `payload_verified` gate). Counterfactual design encoded: `family_key` groups a
  matched set, `axis` ∈ {payload, trigger, rank_alpha, module_subset, concept, spectral_match,
  trigger_token_only} names the single varied factor, `cell` its value; `validate_matched_set`
  enforces a set isolates exactly one axis. 6 tests. *Remaining (later, on cluster): actual minting
  (train FLUX.2-klein LoRAs per the matched-set plan) — code-spec done, training deferred to POC-1c/4.*
- **B2. Safety labeling rubric — NEEDED FOR POC-1b safety fields + POC-4.** 7 safety fields exist but
  no decision criteria: `identity_clone` vs fictional character? `nsfw_severity` boundaries? Is
  "suspected backdoor" even labelable from images (probably organism-only, not the image pass)? *Fix:*
  a one-page rubric so the highest-stakes labels are consistent.

**TIER C — evaluation specifics (still words, not definitions):**

- **C1. Hard-negative construction (the H2 metric) — ✅ SPECCED (`docs/c1_hard_negatives.md`); build at POC-3.**
  Tiers T0–T4 (random → family → +medium → +creator → +recipe), K=9/49 pools, disjoint-family scorer,
  own-generated pool images, deterministic selection reusing concept_family + recipe_fingerprint.
  T3/T4 = the retrieval analogue of the within-creator/within-recipe gate. Code stub deferred to POC-3
  (needs a reader producing descriptions first).
- **C2. Image-set provenance — NEEDED FOR POC-1b labeling + POC-3 generate-and-verify.** How many
  images per adapter? Creator showcase images are cherry-picked (a confound) — do we **generate our
  own** from a fixed prompt set for consistency? Count + provenance currently undefined.

**Sequencing:** A1+A2+A3 and C2 are needed *first* (POC-1b/labeling). B2 with the safety labels. C1 at
POC-3. B1 at POC-1c/POC-4. Do not run the step before its component is fleshed out.

---

## B.8 Evaluation, baselines, money figures

### B.8.1 Metrics

- **H1 (concept/style/identity):** top-1/top-5 closed-set accuracy + retrieval mAP, on cross-creator + cross-rank splits, vs the full baseline battery (§B.8.2).
- **H2 (descriptions) — HARD RETRIEVAL is the primary metric, NOT caption similarity.** *Primary:* given **only the reader's description**, an independent scorer must identify the true adapter's outputs among **hard negatives matched on creator / base / rank / style / concept family** (and the hardest tier: within-creator/within-recipe). Report retrieval accuracy / mAP / rank-of-true. *Secondary (kept honest, not headline):* generate-and-verify — render from the description with base FLUX, score CLIP-I/DINO/CLIP-T vs the adapter's own generations. **Circularity guard (must-fix):** the data-prep captioner and the verification scorer are from **disjoint model families** (captioner = Qwen-VL/JoyCaption; scorer = OpenCLIP/DINOv2 + a GPT-class judge) — never the same family on both ends. **Human-labeled audit slice** calibrates captioner fidelity. Baselines: random-description null, **metadata/creator-tag**, **nearest-neighbor-caption** (retrieve the train caption of the weight-space NN — the memorization ceiling), and a **run-the-model auto-interp ceiling** (DnD/CLIP-Dissect on generations).
- **H3 (safety):** ROC/AUROC clean-vs-malicious **on the matched-spectra control** (so it's not spectral concentration); structured payload-field accuracy + free-text payload via hard retrieval; **cost axis** (adapters/GPU-hr) vs a run-the-model detector; train-controlled→test-wild per-axis breakdown (§B.6.3); adaptive-attacker arm.
- **H4 (trigger) — report at the highest ladder rung reached** (§B.3 H4): trigger-conditioned detection → payload recovery → trigger type/family → **candidate-set retrieval** (rank-of-true-trigger among distractors) → exact string (causally verified). Lead with candidate-set retrieval; present exact recovery as possibly information-theoretically underdetermined.
- **H5 (transfer):** zero-shot / light-tune on FLUX.2-klein/dev LoRAs; VLM-adapter capability description. (Appendix/stretch.)

### B.8.2 Baselines (the full battery — run all; a result that doesn't beat these is not a result)

*Weight-only, what our method must beat:*
- **Spectrum-only classifier** (σ₁, Frobenius, energy concentration, spectral entropy, kurtosis → logistic reg) — direction-blind; the spectral-detector baseline.
- **Raw A/B classifier** — flatten the factors directly (gauge-variant); tests whether canonicalization actually helps.
- **SVD / product-feature linear probe** — our symmetry-aware features under a linear head (the POC-1 protagonist).
- **W2T-style encoder** (`2603.15990`) — QR→SVD tokens + a from-scratch Transformer encoder → label/score.
- **Simple Transformer encoder over weight tokens** — a learned encoder→classifier head, *no LM*. **This is the key "is the LM reader overkill?" control:** the LM must earn its keep on the *language* tasks (H2 hard retrieval, H3 description) — we do **not** require it to beat this encoder at closed-set classification (a dedicated head should win there; that's fine).
- **Nearest-neighbor in weight space** — retrieve the train caption of the weight-space NN (the memorization ceiling for retrieval).

*Signature / confound controls (POC-1 + §B.6.4-#6):*
- **Metadata/tag prior**, **creator-only**, **recipe-only (rank/α/script)** — if any matches the reader, we're reading signatures.

*Execution-requiring (upper bounds, not competitors):*
- **Preview-image caption baseline** — caption the adapter's own sample images (needs the images, not execution).
- **Run-the-model caption / NSFW detector** — generate then caption/NSFW-detect (the execution upper bound + cost comparator).
- **Auto-interp (DnD / CLIP-Dissect / MILAN)** — activation-based feature→language; the run-the-model H2 ceiling.
- **Spectral backdoor detector** — the published weight-only safety baseline (H3).

### B.8.3 Money figures

- **Fig 1 — Cover (safety + verbalization in one).** A real safety-relevant adapter → the reader's text (*"injects <NSFW concept> on trigger ‹word›"* / *"clones the identity of ‹person›"*) → generated images that **confirm the verbalization is correct**, all **without the screener ever running the model**. Pair a poisoned adapter (payload + recovered trigger, fired to prove it) beside a benign style adapter, same pipeline. This single figure displays both contributions heavily and is the spotlight cover.
- **Fig 2 — Train-controlled / test-wild generalization (the flagship).** Detection + payload-description accuracy on **truly wild held-out** adapters, broken down per generalization axis (held-out creator / concept / rank / attack-config / base). This is the most novel panel and the sharpest separator from prior art.
- **Fig 3 — Execution-free safety ROC + cost/throughput** (adapters/GPU-hr): weight-only reader vs run-the-model detector vs spectral baseline, with the cost axis showing orders-of-magnitude cheaper screening. (H3)
- **Fig 4 — Reads semantics, not spectra.** Reader vs spectral baseline on the spectral-negative-control set — proves the safety signal is semantic, not a re-learned spectral heuristic. (H3 credibility)
- **Fig 5 — H2 hard retrieval** — reader description retrieves the true adapter among same-creator/base/rank/concept-family hard negatives, vs nearest-neighbor-caption and metadata baselines. The core capability figure.
- **Fig 5b — module localization** (from POC-1b): where the semantic signal lives across q/k/v/out/MLP/text-vs-image-stream/adaLN and layer depth.
- **Fig 6 — Trigger ladder (H4, bonus, NOT the headline).** Per-rung recovery (detection→payload→type→candidate-set→exact); honest about where exact recovery becomes underdetermined.
- **Fig 7 (stretch) — cross-model transfer to FLUX.2 frontier** (H5).
- **Fig 8 — Hub audit at scale (the deployment figure).** Aggregate results of running the reader over the full ~40–60K-LoRA corpus (§B.8.5): prevalence/breakdown of flagged adapters, throughput, headline findings. The figure journalists screenshot.

### B.8.4 Ablations (the science of *why* it works — a required section, per the paper structure)

- **Representation:** raw A/B vs ΔW vs SVD-direction vs **product `UVᵀ`** features; spectrum-only vs directions; sign-invariant vs naive; both-sides vs residual-facing-only.
- **Module families (from POC-1b):** q/k/v/out vs MLP vs text-stream vs image-stream vs adaLN; early/mid/late layers — *where does the semantic signal live?*
- **Rank generalization:** train on a rank band, test on held-out ranks (the regime-shift stress test).
- **Reader architecture:** the LM-injection reader vs the **simple Transformer-over-tokens encoder** vs the linear probe — does the LM earn its keep *on the language tasks* (H2 retrieval, H3 description), as opposed to closed-set classification where a dedicated head may win?
- **Warm-start vs fresh:** loracle warm-start vs from-scratch (per §B.5.4 — an ablation, not a pillar).
- **Backbone:** text-only LM vs weights-only VLM backbone (no images fed) — isolates visually-grounded text priors while keeping the execution-free claim; switch only on a meaningful margin.

### B.8.5 Public hub audit (C-Audit — the deployment artifact)

Run the final reader over the **entire harvested corpus (~40–60K FLUX.1-dev LoRAs)** and release the results as a public dataset + report: per-adapter safety verdict + description, aggregate prevalence of NSFW-injection / identity-cloning / suspected-backdoor adapters, and throughput/cost. This is the **first hub-scale execution-free safety scan** of customized image models.

- **Why it matters disproportionately.** A real-world audit of HuggingFace/CivitAI is the kind of result that AI-safety orgs, hub maintainers, and journalists pick up — it turns a good paper into an influential one and gives the work a life beyond the conference. It is also a *standing* artifact: re-runnable as the hub grows.
- **It works even with a modest reader.** Even if the reader only reaches H1/H3-held-in quality, an audit that surfaces *candidates* for human review at hub scale is genuinely useful and novel. So C-Audit partly decouples impact from the reader hitting its hardest claims.
- **Responsible disclosure (mandatory).** Coordinate with hub maintainers before publishing per-adapter flags; report *aggregates* and *exemplars-with-consent* rather than a public denylist that could be weaponized or defame benign creators. False-positive handling and an appeal path must be specified. See §B.11.1.

### B.8.6 Corpus release (C-Corpus — the independent contribution)

Release the **first large labeled DiT weight-space dataset**: real FLUX LoRAs (or pointers + a reproducible harvest/encode pipeline, respecting per-LoRA licenses), unified-schema descriptions, the controlled-organism safety set with ground-truth `(trigger → payload)` labels, and the SVD-encoding code. This is **citable and reusable independent of whether our reader hits its targets** — the substrate others will build weight-space methods on for years. Pair it with a datasheet (intended use, licensing, known biases, the captioner-fidelity audit numbers).

---

## B.9 Risk ladder

- **Unsinkable floor (does NOT depend on the reader working):** **C-Corpus** (first labeled DiT weight-space dataset) + **C-Audit** scaffolding. Banked from the data engine alone. *This is why H1 failing does not mean "the paper has nothing."*
- **Floor (likely, gated by POC-1):** H1 — weights predict concept/identity better than spectral/W2T/metadata baselines. First such result for a *diffusion transformer*.
- **Expected:** H2 verbalization that verifies by generation (easy→target rung) + H3 **held-in** safety screening with payload description, beating the binary baseline.
- **Spotlight (the core result):** **H3 train-controlled/test-wild triage beats the static baselines** (Fig 2) + reads semantics not spectra (Fig 4) + far lower cost than run-the-model detectors + the **public hub audit** (C-Audit, Fig 8). *The methodology + this result are the paper; the spotlight does not require H4.*
- **Bonus amplifiers:** + **H4 trigger inversion** (Fig 5, no precedent) and/or + **H5 cross-model/VLM transfer** — high-variance, dropped first under time pressure, never load-bearing.

**Key mitigations (each risky claim has a concrete fallback):**
- **H1 regime-shift fails** → the linear-probe easy rung (POC-1) tests it cheaply in week 1; if even that fails we pivot to C-Corpus/C-Audit as the contribution and reassess the reader.
- **H2 free-text is too hard** → ship the **retrieval/templated** easy rung (still verifies by generation, still a cover figure) and grow into open free-text.
- **H3-wild doesn't generalize** → report **H3-held-in** (still beats the binary baseline by adding description) + the per-axis breakdown showing *where* it does/doesn't transfer (a scientific result either way).
- **Organism quality is the bottleneck** → invest early (week-1 cost benchmark), span benign+malicious, and validate organisms actually exhibit the payload before use; the wild-malicious-natural tier (real hub adapters) backstops synthetic-organism weakness.
- **FLUX.1-dev non-commercial license** worries artifact release → the **FLUX.2-klein 4B (Apache-2.0)** arm gives a license-clean train+release set and a "scales to current frontier" story.
- **Warm-start transfer is poor** → train fresh on a strong base LM (text-only); warm-start becomes an ablation, not a dependency.
- **MMDiT "residual-facing" side ambiguous** → ablate both sides (O3).

---

## B.10 Timeline (rough, 3 people)

- **Weeks 1–2:** POC-0/POC-1 (plumbing, encoding correctness, linear probe — the H1 gate). Stand up scrape + re-caption pipeline (long pole — start immediately). **C-Corpus accrues from day one** (banked regardless of reader outcome).
- **Weeks 2–5:** POC-2 closed-set reader; H1. Warm-start integration. Begin organism factory (parallel).
- **Weeks 4–8:** POC-3 verbalization + generate-and-verify (easy→target rung). The combined safety+verbalization **cover figure** takes shape (H2 + early H3).
- **Weeks 6–12 (the flagship — most effort):** organism factory at scale + POC-4; **H3 train-controlled/test-wild** (Fig 2), per-axis generalization, spectral-negative-control (Fig 4), cost/ROC (Fig 3). Pre-register before the wild eval.
- **Weeks 10–14:** **H4 trigger inversion** (Fig 5 — prioritized spotlight-maker); RL stage if it helps; ablations.
- **Weeks 13–16:** **public hub audit** over the full ~40–60K corpus (C-Audit, Fig 8) with responsible-disclosure handling; corpus release prep (C-Corpus datasheet).
- **Weeks 16–18:** H5 cross-model/VLM transfer (stretch, first to drop); writing, figures.

Parallelism: the **data engine** (one person) runs continuously from week 1 and *owns C-Corpus/C-Audit*; **reader training** (one person) starts as soon as POC-1 passes; **safety/organisms + the flagship wild eval** (one person) ramps from week 6 and gets the most sustained effort.

**The timeline is conditional on the gates.** It assumes POC-1 and POC-3 pass on schedule; both are genuine regime-shift tests, not formalities. If POC-1 stalls, weeks 4+ pause for reassessment (the floor is at risk). If POC-3 stalls, ship H1 (classification/retrieval) and treat free-text as ongoing. **H4 and H5 are explicitly droppable** — they are the first to go if any earlier gate slips, and the paper is still a contribution (H1+H2+H3) without them.

### B.10.1 Cost estimate (rough, order-of-magnitude)

Figures assume rented cloud GPUs (~$2/hr for an A100/H100-class, ~$0.30–0.50/hr for a 4090-class); all are *planning* estimates to be replaced by the week-1 empirical benchmarks (POC-0 parseable fraction, klein per-organism cost). The headline: **this project is dominated by reader-training and eval compute, not by minting models — because the FLUX.1-dev corpus is harvested free.** Total lands in the **low thousands of GPU-hours / ~$3–8K** range, well within "substantial compute."

| Item | Estimate | Notes |
|---|---|---|
| **Scrape + store ~40–60K FLUX.1-dev LoRAs** | **~6–12 TB** object storage; bandwidth ~free | avg ~100–200 MB/LoRA; storage ~$150–350/mo (cloud object store; GCS/S3/Blob — cover with existing credits). Filter to FLUX.1-dev at scrape; dedup by weight-hash. Cold-storage most of it after one-pass SVD encode. |
| **Re-caption sample images (Phase A)** | ~60–150 GPU-hr (local VLM) *or* ~$0.4–1.2K (API) | a few images × ~40–60K; the only large captioning pass. Scales ~linearly with corpus — the main cost that grew with the scale-up. |
| **Organism factory (Phase B)** | ~$1–4K | ~500–2K klein-4B LoRAs @ ~$0.50–2 each (cost **unverified** — benchmark wk 1; 3–4× if convergence/sample-gen heavier); + DiT-ported attack recipes |
| **Reader SFT (warm-start, 14B)** | ~200–600 GPU-hr | bulk cost; several runs across POC-2/3 + scaling. 70B variant ~3–5× if used |
| **Reader RL (Dr. GRPO, optional)** | ~100–300 GPU-hr | K rollouts × judge calls; LoRAcle used only ~40 RL steps — keep short |
| **Generate-and-verify eval** | ~50–150 GPU-hr | render base-FLUX images from descriptions + CLIP/DINO/judge scoring, repeated per checkpoint |
| **Ablations (warm-start, backbone, encoding)** | ~150–400 GPU-hr | each ablation re-runs SFT on a subset |
| **Public hub audit (C-Audit)** | ~15–50 GPU-hr | one reader inference pass over ~40–60K LoRAs — still cheap, since reading is execution-free (the whole point) |
| **H5 cross-model / VLM (stretch)** | ~100–300 GPU-hr | FLUX.2-klein/dev transfer + VLM-adapter arm |
| **Total** | **~0.9–2.3K GPU-hr + ~$3–8K non-GPU** | ≈ **$4–11K all-in** on rented hardware; less on owned cluster / against existing cloud credits |

Note the audit line: screening ~40–60K adapters costs ~tens of GPU-hours precisely *because* the method never runs the diffusion models — the cost contrast with run-the-model detectors is itself a headline number (Fig 3), and it *widens* at scale (run-the-model screening of tens of thousands of adapters is prohibitive; reading them is a day of inference).

**Cost-reduction levers:** warm-start (avoids from-scratch reader training); harvested corpus (avoids w2w's ~$30–250K model-minting); local captioner/VLM (avoids API fees); cap RL steps (LoRAcle precedent); **existing AWS/Azure/GCP credits absorb the storage + captioning lines.** **Cost-blowup risks:** the scale-up makes **storage (~6–12 TB) and the captioning pass** the lines that grew — both credit-absorbable and both one-time; klein per-organism cost higher than quoted; 70B backbone instead of 14B; **lower-than-hoped FLUX.1-dev yield** (if usable supply is <40K, the corpus and every scale figure shrink — measure in POC-0d). Budget a ~1.5× contingency.

---

## B.11 Why it's spotlight-shaped + scoop defense

- **One-sentence story:** *"A symmetry-aware weight reader turns a fine-tuned image model's LoRA into an open-language description — enough to triage it for safety (NSFW / identity-cloning / backdoors) better than static detectors, without ever running it."*
- **The ownable artifacts** (what others are least likely to ship first): the **method** (first symmetry-aware weight→open-language reader for the visual modality), the **train-controlled/test-wild triage** result that beats static baselines (Fig 2), the **public hub audit** (C-Audit), and the **first labeled DiT weight-space corpus** (C-Corpus). **Trigger inversion (H4) is a bonus**, not the anchor. Move fast on the data engine (the long pole) and lead with the methodology + triage result.
- **Scoop exposure is low (lit-review-verified, §B.2).** No prior work produces a free-text weight→language reader for any image model; w2w is generative, not weight→text; LoRAcle is text-only and still under review (NeurIPS 2026), so the diffusion port is unclaimed; no backdoor attack *or* defense targets diffusion transformers yet (all are U-Net); and no weight-space method has a wild-corpus eval or weights-only trigger inversion. The closest method (W2T) is a *different paradigm* — see below.
- **Hedge against being scooped on the *attack* (the one live race).** MasqLoRA's pending FLUX port is the only realistic near-term scoop, and it would land on the *attack* side. We are robust to it: even if a DiT backdoor *attack* ships first, **no one has a weights-only DiT backdoor reader/defense or trigger inverter** — a separate, larger white space. Lead with the *defense/reader* novelty (and the wild-corpus eval + trigger inversion), so an attack scoop costs us a framing sentence, not the contribution.
- **Distinction from W2T (`2603.15990`), our nearest neighbor — and why it's not a scoop.** Different approach, leading to different methods, math, results, and implications:
  - *Approach:* W2T trains a **bespoke from-scratch encoder** to predict a property; we take the **LoRAcle approach** — inject weight-derived tokens into a **pretrained LLM** that **verbalizes in open natural language**.
  - *Output:* W2T → a label/score/embedding; us → free-text descriptions (incl. payloads and **inverted triggers**) that **verify by generation**.
  - *Math:* our design turns on the LoRAcle-style residual injection + the GL(r)/sign/degeneracy treatment for a **language** decoder (§B.4–B.5), not a property head.
  - *Implications:* W2T answers "what property does this have"; we answer "what does this *do*, is it safe, and what is its hidden trigger" — and we validate **in the wild**, which W2T does not. We cite W2T and include its encoding as one baseline point (§B.8.2); we do not frame the project around it.
- **Dependence on an unpublished ancestor.** LoRAcle (our recipe ancestor) is under review with no code license; we reimplement from the paper and do not over-anchor the narrative to it, so a LoRAcle rejection/revision does not undercut us.

### B.11.1 Ethics / responsible disclosure (plan now, not at submission)

Three elements will trigger NeurIPS ethics review: **identity-cloning organisms** (a target individual), **releasing the first DiT NSFW-injection / backdoor recipe**, and **the public hub audit** (naming real adapters/creators as malicious). Plan up front:
- Use consented/synthetic or public-figure-free identities where possible.
- Gate or withhold attack-recipe details (release the *detector*; describe the attack at a level sufficient for reproduction-by-experts, not a copy-paste weapon).
- For the **hub audit**: coordinate with hub maintainers *before* publication; report **aggregates + consented exemplars**, not a public per-adapter denylist that could be weaponized or defame benign creators; specify false-positive handling and an appeal/contest path; treat a flag as "candidate for human review," not a verdict.
- Frame the whole safety arm as **defensive** (we screen, we don't distribute attacks); prepare a responsible-disclosure statement and a dual-use impact section.
This is a defensive-security contribution, but it must be presented as one.

---

## B.12 Open questions for the LoRAcle creator / mentor

> **Status (2026-06-21):** Celeste granted use of the published code (`assets/loracles-code`), so
> **O1/O4/O6/O7 are resolved** from the source + her reply; the authoritative recipe is in
> `RECIPE_NOTES.md`, and our detailed O1 assessment (warm-start prior, the new 3072→5120 dimension
> bridge, injection gotchas) is in **§B.12.1 below**. **O2/O3/O5 remain genuinely open** (residual-side
> heuristic, adaLN content, rank curriculum) and **one new question** has surfaced — *the
> 3072→5120 projection LoRAcle never needed* (§B.12.1). These are the high-value items for Celeste.

- **O1 (warm-start). [RESOLVED via code — kept for reference]** Is the released `loracle-qwen-3-14b` injection code/recipe stable enough to reuse as-is, and do you expect the text-trained reader to give *any* useful prior on diffusion-LoRA tokens, or is fresh training cleaner? Any gotchas in the norm-matched injection at the chosen layer?
- **O2 (encoding side).** For LLM modules you keep right-vectors for q/k/v/up/gate and left for o/down ("residual-facing"). *Our working hypothesis* (§B.5.2): the criterion is "keep the side in residual-stream coordinates" (`V` for reads-from-residual, `U` for writes-to-residual), which ports module-by-module to MMDiT attention/MLP since joint attention leaves per-modality projections separate (verified, §B.2). **But we could not find literature establishing that the residual-side vectors are the more interpretable/transferable ones — it's a heuristic we ablate, not a settled fact.** Is that the criterion you used, and is there evidence the *non*-residual side is sometimes equally informative (e.g. low-rank update dominated by contraction-dimension structure)?
- **O3 (modulation layers).** adaLN `norm*.linear` is the case where the residual-facing rule breaks: it maps the *conditioning* vector (timestep + pooled text) → modulation params (shift/scale/gate), so **neither side is the residual stream** (verified against DiT `2212.09748` + SD3 `2403.03206`). Should these diffs be encoded at all, or do they carry little concept content (mostly global gating)? No source we found settles the semantic content of adaLN weights — your read would save us an ablation.
- **O4 (RL judge leakage). [RESOLVED via code]** Judge sees only the ground-truth label, never the weights; greedy eval, `max_new_tokens=1024`, full eval sets. For us, the safety-arm scorer is embedding-based (OpenCLIP/DINOv2) — a different lineage from the captioner, which strengthens rather than leaks the circularity guard.
- **O5 (rank heterogeneity). [PARTIAL]** Real FLUX LoRAs span ranks 4–64 (POC-0d triage). Full-finetune→LoRA path = `svd_lowrank(q=2r, niter=2)` then truncate. A rank-robust **curriculum** is still ours to design + test — advice welcome.
- **O6 (dataset/code licensing). [RESOLVED]** Celeste granted use of the published code; we cite the paper and keep the reimplementation attributable.
- **O7 (scoop/collaboration). [RESOLVED]** LoRAcle team is **not** planning a diffusion port — no reader-side race.

### B.12.1 O1 — warm-start & injection: our working assessment (from the released code)

Recorded so the team and Celeste can sanity-check our read of her code (`encoder_ao.py`, `tokenize_lora_fixed.py`, repo `CLAUDE.md`):

- **Injection reuses verbatim — there is almost no code to port.** `AOEncoder` is a **zero-parameter passthrough**; the entire mechanism is the layer-1 hook `h ← h + ‖h‖·v/‖v‖`. Nothing in the injection is modality-specific, so it transfers as-is. What changes is *only the tokens fed in*.
- **The one genuinely new structural decision: a dimension bridge.** LoRAcle's mag7 tokens are *natively* in the reader's residual width (5120) because LLM read/write sides live there. **FLUX's residual width is 3072, the reader's is 5120**, so DiT direction vectors are **not** reader-dimensional — we need a 3072→5120 map LoRAcle never did. Plan: start with a **frozen random-orthogonal projection** (preserves geometry, adds no learned params, keeps the "encoder is a passthrough" spirit), and ablate a learned linear. *This is the highest-value thing to confirm with Celeste — it's the only place the recipe is silent.*
- **Warm-start prior — our prediction: helps convergence, not the ceiling; plan for fresh training as the honest baseline.** What plausibly transfers is the *format skill* (attend to injected residual-stream tokens, treat their geometry as evidence, emit a structured description) — domain-independent, so it should speed convergence. What almost certainly does *not* transfer is the *content* map (text-concept ⇒ words); the visual-concept manifold has no overlap, and the 3072→5120 bridge lands FLUX tokens in a different residual subspace than the text tokens the reader was tuned on. **Run it as a 2-arm ablation** (warm-start vs fresh rsLoRA on the same base) and report the lift; make no claim depend on transfer.
- **Collapse risk is *higher* with warm-start — watch it.** The repo documents a degenerate fixed point where a larger interpreter LoRA minimizes the QA loss *without reading the tokens* (symptom: held-in metric climbs while AuditBench→0%). A reader carrying a strong language prior is more tempted to ignore unfamiliar visual tokens. Mitigation: their soft-HP rule (α=rank, halved lr) + monitor a "does it actually read the tokens?" probe (token-ablation control) during SFT.
- **Concrete injection gotchas:** (1) **no gradient checkpointing with the hook** (non-deterministic alloc breaks the tensor-count check) — matters more for us since FLUX has far more adapted modules → more tokens → more memory; mitigate with H100-80GB or a shorter prefix (cap k, fewer module families). (2) **`‖v‖` denominator blows up on near-zero σ directions** — and POC-0d found real adapters with σ-gaps down to 5e-6; clamp/skip tokens below a σ floor (our degeneracy detector already flags these). (3) **prefix length / context budget** — layer-1 tokens occupy attended positions; keep the `(rank,layer,mag7)` ordering and a sensible k-cap.

---

## Appendix — reuse map (grab, don't build)

| Asset | Use | License |
|---|---|---|
| `loracles-interpretability/*` checkpoints | reader warm-start | MIT ✓ |
| LoRAcle paper (`x9MbM7QmQN`) | encoding/injection recipe (reimplement) | CC-BY ✓ (code: none ✗) |
| ~40–60K FLUX.1-dev LoRAs (HF ~42K + CivitAI-FLUX) | Phase-A scraped corpus + tags/triggers | per-LoRA (mostly FLUX-dev NC) |
| FLUX.2 [klein] 4B | organism factory + permissive arm | Apache-2.0 ✓ |
| `AllanYangZhou/nfn`, `AvivNavon/DWSNets` | equivariant-layer primitives (if needed) | MIT ✓ |
| `SakanaAI/text-to-lora` | per-(module,layer,rank) token layout pattern | Apache-2.0 ✓ |
| diffusers FLUX/FLUX.2 pipelines + ai-toolkit | organism training + generate-and-verify | Apache-2.0 ✓ |
| MasqLoRA recipe (`spectre-init/MasqLora`) | poisoned-organism recipe (port to DiT) | MIT ✓ |
| w2w 60K identity corpus | auxiliary identity labels (cross-modality) | check repo |
| CLIP / DINO / off-the-shelf VLM | labels + generate-and-verify eval | various ✓ |

*(Verify the exact arXiv IDs flagged in the lit-review report before formal citation; several 2026-dated IDs were surfaced in a June-2026 environment and confirmed, but re-check at submission.)*

---

## B.13 Detailed execution plan

This section turns the proposal into an ordered set of buildable steps. It is organized so that **everything that needs no GPU and no data access happens first** (and in parallel with access/compute being arranged), and every expensive step is gated by a cheap decisive test that precedes it (per WORKING_NORMS §3). Each step lists **inputs → work → deliverable → gate**.

### B.13.0 Operating principles (carried from WORKING_NORMS)

- **Author code locally, in git; copy to the box to run.** No box-only edits. Push back only small structured results (JSON/CSV), never weights/caches.
- **Honour the POC gates.** A cheap test (instrument validation, linear probe) must pass before the build it gates. When a direction may be doomed, find out in the cheapest way first.
- **Always run the trivial baselines** (spectral-stat, metadata/tag, W2T-encoding, CLIP-on-generations ceiling) and report them next to every headline number.
- **Design confound controls in before running** (unified label schema, family-level holdouts, concentrated-spectrum control, disjoint captioner/scorer families).
- **Anonymize before anything leaves the cluster**; secrets are pod env-vars only; one node max, delete pod when idle; logs+checkpoints to `/fsx`.

### B.13.1 Repository structure

A single Python package, authored locally, version-controlled from commit 0:

```
diffusion-loracles/
  ditloracle/
    encoding/        # SVD-direction-token encoder + invariance utilities (POC-0)
    formats/         # FLUX LoRA key-scheme parsers + base-checkpoint triage (POC-0)
    data/            # scrape, re-caption, schema, splits (Phase A); organisms (Phase B)
    probe/           # linear probes + spectral/metadata baselines (POC-1)
    reader/          # token layout, injection, SFT/RL training (POC-2/3)
    safety/          # organism recipes, detection, trigger inversion (POC-4, H3/H4)
    eval/            # generate-and-verify, discrimination scorer, ROC, cost (H2/H3)
  tests/             # invariance + parser unit tests (run in CI/local, no GPU)
  configs/           # experiment configs (anonymized names for cluster)
  scripts/           # cluster launch (pod specs, nohup wrappers) — anonymized
  notes/             # un-pushed: run-id↔real mapping, access/secrets pointers (gitignored)
  PROGRESS.md        # the human-readable progress journal (results + analysis + storyline)
  results/           # small JSON/CSV pulled back from runs (opaque names)
```

- `notes/` and `results/`-with-weights are gitignored; `notes/` holds the run-id↔real mapping and never leaves local.
- Local dev env: Python 3.14 + torch 2.12 present. Pin deps in `pyproject.toml`; the cluster uses the known-good image (`njfn-recipe:llamafactory-agic-latest`, torch 2.9 / transformers 5.6 / peft 0.18) — keep code compatible with both, avoid torch-2.12-only APIs in anything destined for the pod.

### B.13.2 Phase 0 — local, no GPU, no access required (DO NOW)

These run on a laptop and de-risk the single most dangerous failure mode (a silently-wrong encoder), so they precede everything.

1. **Scaffold repo + journal.** *Deliverable:* package skeleton, `pyproject.toml`, `.gitignore`, `PROGRESS.md`, commit 0.
2. **POC-0a — encoder + invariance suite (instrument validation).**
   - *Inputs:* synthetic LoRAs (random `B,A` of known rank; planted degenerate spectra; sign-flipped/`GL(r)`-transformed copies).
   - *Work:* implement compact SVD via QR-on-factors (never form the dense `d_out×d_in` product); the token feature of §4; sign-invariant direction features (`u_iv_iᵀ` projector / SignNet-style); degeneracy handling via σ-clustering → subspace projectors; α/r folding + spectral normalization.
   - *Deliverable:* `tests/test_invariance.py` proving feature equality (to numerical tol) under: random `G∈GL(r)`; coupled sign flips; rotations within planted degenerate subspaces; and **non-invariance** to a genuine change (a different `ΔW`) as a control.
   - *Gate:* all invariance tests pass at well- *and* ill-conditioned `G`; the negative control fails (i.e., the encoder is not trivially collapsing everything). **This gate must pass before any real-data or training work is trusted.**
3. **POC-0b — spectral & metadata baselines (so they exist before we need them).** Implement the 20-dim spectral-stat featurizer (σ₁, Frobenius, energy concentration, spectral entropy, kurtosis per module) + a logistic-regression head, and a metadata/tag bag-of-words baseline. *Deliverable:* `ditloracle/probe/baselines.py` + unit tests on synthetic labels. *Gate:* baselines train and score on synthetic data end-to-end.
4. **POC-0c — W2T-style encoding baseline (the closest-method comparison).** Implement its QR→SVD-token encoding as an alternative featurizer behind the same interface. *Deliverable:* drop-in encoder variant for later head-to-head.

### B.13.3 Phase 1 — data access + triage (needs HF access; starts in parallel)

5. **POC-0d — FLUX LoRA format/base triage.**
   - *Inputs:* a sample (≈500–2k) of FLUX.1-dev LoRAs from HF + CivitAI metadata.
   - *Work:* parsers for the key-scheme zoo (diffusers `transformer.*`, kohya/BFL `lora_unet_*`, fused single-block qkv/mlp split, DoRA, rsLoRA vs vanilla scaling, partial coverage, quantized/merged/broken); base-checkpoint verification (metadata + weight-fingerprint of the implied base); per-module SVD conditioning on rank-64–128 adapters (σ-gap distribution).
   - *Deliverable:* `results/triage__ossN.json` with the three gate numbers; a clean parser covering the dominant formats.
   - *Gate:* report **parseable fraction**, **base-homogeneity fraction**, **σ-gap conditioning**. Decide the filtered population for POC-1. (If parseable/homogeneous fraction is low, branch the encoder or restrict to a clean sub-format — decide, don't assume.)
6. **Phase-A data engine (begins once triage picks the population).** Scrape → store weights + sample images + metadata → re-caption into the fixed schema (VLM disjoint from the eval-scorer family) → family-level (creator/concept) holdout splits → concentrated-spectrum control set. *Deliverable:* a versioned `(weights, schema-label, split)` dataset (or reproducible manifest + pointers, respecting per-LoRA licenses). *This is the long pole — it runs continuously.*
   - **Scale target: the full available FLUX.1-dev supply (~40–60K, single-base — §B.1.4)** (§B.6.1). The current `download_weights.py` is POC-scale and needs four upgrades before the full harvest: **(a) an HF source** (the ~42K FLUX.1-dev LoRAs live on HF, not CivitAI — the downloader is CivitAI-only today; add `huggingface_hub` listing/download); **(b) a FLUX.1-dev filter at scrape time** (don't download the SDXL/Pony/SD1.5 majority the encoder can't use — gate on the POC-0d base-lineage check); **(c) cloud object-store output** (write to GCS/S3/Blob via `fsspec` instead of local disk, and raise the 60 GB cap); **(d) weight-hash dedup** (cross-hub mirrors are common; dedup on a content hash, which *also* feeds the base-lineage check). Provider: **GCS preferred** (available credits); the `fsspec` abstraction keeps it swappable. Egress-to-GPU for the one-pass SVD encode is the only recurring cost.

### B.13.4 Phase 2 — POC-1 the gating experiment (CPU/light GPU; staged 1a→1b→1c→1d, see §B.7 + §B.7.1)

7. **POC-1a — apparatus sanity (n≈25–50, days, NOT the gate).** Run the full fixed-dimension feature → probe → split pipeline on the small real sample; confirm features are finite + fixed-length (no rank leakage), and obvious signal exists. Debugs plumbing only.
8. **POC-1b — THE SEMANTIC-SIGNAL GATE (n≈300–500 human-audited-labeled adapters).**
   - *Inputs:* a **labeled benchmark** with repeats per creator/concept/rank/recipe; **labels are the small human-audited set, independent of the metadata features** (§B.7.1-#1) — NOT creator tags.
   - *Work:* within-creator + within-recipe hard retrieval + matched-hard-negative retrieval; full baseline battery (§B.8.2) incl. **rank-only / rank+module leakage controls** and creator-only/recipe-only **diagnostics**.
   - *Gate (decisive):* our features **≫ all non-signature baselines on within-creator/within-recipe retrieval**, AND the **rank-only / rank+module baselines are near-chance** (no leakage). Pass → premise holds → proceed to the reader. Fail → stop, pivot to C-Corpus/C-Audit.
9. **POC-1c — causal validation on controlled organisms.** On counterfactual matched pairs (§B.6.2), confirm features track the varied factor and ignore the clamped one (here "beat creator-only" is clean — creator is constant by construction).
10. **POC-1d — module localization** (runs with 1b): where the signal lives.

### B.13.5 Phase 3 — the reader (GPU; gated by POC-1)

8. **POC-2 — tiny warm-started reader, closed-set.** Re-implement the LoRAcle token layout + norm-matched residual injection (from the paper; no unlicensed code); warm-start from the MIT Qwen3-14B reader adapter; SFT on a small balanced subset; emit a description; score closed-set accuracy/retrieval. *Gate:* beats the POC-1 probe and the metadata prior → H1 confirmed.
9. **POC-3 — free-text + generate-and-verify.** Scale SFT; emit free text; eval with the **discrimination scorer (primary)** + CLIP/DINO (secondary), captioner/scorer families disjoint, human-audit slice for captioner fidelity. *Gate:* beats metadata/tag baseline and random-description null **and** approaches the run-the-model auto-interp ceiling → H2 (the cover-figure capability).

### B.13.6 Phase 4 — safety, the flagship (GPU; gated by POC-3)

10. **POC-4 — organism factory + safety pilot.** Mint ~200 organisms on FLUX.2-klein-4B (benign + NSFW-injection + identity-cloning + backdoor `(trigger→payload)`); verify each exhibits its payload before use; train detection + payload-description; report ROC vs spectral baseline + payload-description accuracy. *Gate:* beats spectral ROC **and** produces usable payload descriptions → H3 in reach.
11. **H3 flagship — train-controlled / test-wild.** Scale organisms; evaluate on the three wild tiers (benign / natural-malicious / crafted-by-others-config) with per-axis generalization breakdown; spectral-negative-control (reads semantics not spectra); cost/throughput vs run-the-model detector; adaptive-attacker arm. **Pre-register** organism set + wild split before running. *Deliverable:* Fig 2/3/4.
12. **H4 — trigger inversion** (prioritized spotlight-maker): recover trigger from weights; verify causally (generate-with-recovered-trigger → payload fires). *Deliverable:* Fig 5.

### B.13.7 Phase 5 — scale, audit, release

13. **Full-corpus reader pass + public hub audit (C-Audit)** with responsible-disclosure handling (aggregates + consented exemplars; coordinate with hub maintainers; appeal path).
14. **Corpus release (C-Corpus)** with datasheet + captioner-fidelity numbers.
15. **H5 (stretch)** cross-model (FLUX.2) / VLM-adapter transfer; ablations (warm-start lift, backbone, encoding); writing + figures.

### B.13.8 Critical path & parallelism

- **Critical path:** POC-0a (encoder validated) → POC-0d (triage) → POC-1 (gate) → POC-2/3 (reader) → POC-4 → H3 flagship → H4.
- **Runs in parallel:** the Phase-A data engine (continuous from access); baselines (POC-0b/c, local); organism factory (can start minting benign + attack organisms during POC-2/3).
- **Compute discipline:** POC-0/1 are CPU/light-GPU and need no held node. Hold a cluster node only for POC-2 onward, and only when a GPU job is ready to run *now*.

### B.13.9 Blocking questions before/at each gate (see also §B.12)

- **Access/keys (blocks Phase 1):** HF token with access to FLUX.1-dev (gated) + the LoRA corpus; CivitAI API key; confirmation we may store/redistribute or only reference per-LoRA. Cluster creds per WORKING_NORMS.
- **Tooling choices to confirm:** organism trainer (ai-toolkit vs kohya vs diffusers) on FLUX.2-klein; captioner VLM choice (must be a different family from the eval scorer); the eval scorer/judge model.
- **For Celeste (blocks POC-2 design):** O1 warm-start viability + injection layer; O2 residual-facing side for MMDiT; O3 whether to encode adaLN modulation; O4 judge-leakage; O5 rank-heterogeneity curriculum.
- These are surfaced live in `PROGRESS.md` as they arise; none of them block Phase 0/2-local work, which is why we start there.
